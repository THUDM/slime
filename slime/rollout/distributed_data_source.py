"""Global index allocation with node-local datasets and partial-rollout buffers."""

import copy
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.rollout.base_types import iter_samples
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.tensor_store import DiskTensorRef
from slime.utils.types import Sample


class RolloutIndexAllocator:
    """Run as a serial Ray actor. No prompts or tensors cross this boundary."""

    def __init__(self):
        self.position = 0
        self.claims = {}

    def claim(self, worker_id, request_id, count):
        if count <= 0:
            raise ValueError("Index claim size must be positive")
        previous = self.claims.get(worker_id)
        if previous is not None:
            last_request, start, stop = previous
            if request_id == last_request:
                return start, stop
            if request_id != last_request + 1:
                raise ValueError("Index claims must be issued in order")
        elif request_id != 0:
            raise ValueError("First index claim must have request_id=0")
        start = self.position
        self.position += count
        self.claims[worker_id] = (request_id, start, self.position)
        return start, self.position

    def state_dict(self):
        return {"position": self.position, "claims": dict(self.claims)}

    def load_state_dict(self, state):
        self.position = state["position"]
        self.claims = dict(state["claims"])


class _DistributedReader(RolloutDataSourceWithBuffer):
    """Preserve the synchronous DataSource API; prefetch only integer ranges.

    ``claim`` is a blocking callback executed by one dedicated transport thread.
    The common path consumes a local range. A cold/empty range may wait for the
    allocator; built-in async rollouts use ``get_samples_async`` to keep serving
    other tasks during that wait.
    """

    def __init__(self, args, claim, *, prefetch_size):
        super().__init__(args)
        if self.dataset is not None and not len(self.dataset):
            raise ValueError("Distributed rollout dataset is empty after filtering")
        self._claim = claim
        self._prefetch_size = max(1, prefetch_size)
        self._ranges = deque()
        self._request_id = 0
        self._prefetch = None
        self._transport = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-indices")
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-data")
        self._lock = threading.RLock()
        self._closed = False
        self._buffer_files = set()

    def _start_prefetch(self):
        if self._prefetch is None and not self._closed:
            self._prefetch = self._transport.submit(self._claim, self._request_id, self._prefetch_size)

    def _collect_prefetch(self):
        start, stop = self._prefetch.result()
        self._ranges.append((start, stop))
        self._request_id += 1
        self._prefetch = None

    def get_samples(self, num_samples):
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        with self._lock:
            if self._closed:
                raise RuntimeError("Distributed data source is closed")
            samples = self._get_samples_from_buffer(num_samples)
            while len(samples) < num_samples:
                self._start_prefetch()
                if not self._ranges:
                    self._collect_prefetch()
                position, stop = self._ranges.popleft()
                if position + 1 < stop:
                    self._ranges.appendleft((position + 1, stop))
                if sum(end - start for start, end in self._ranges) <= self._prefetch_size // 2:
                    self._start_prefetch()

                if self.dataset is None:
                    prompt = Sample()
                else:
                    epoch, offset = divmod(position, len(self.dataset))
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(epoch)
                    prompt = self.dataset[offset]
                    self.epoch_id, self.sample_offset = epoch, offset + 1
                group = []
                for i in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt)
                    sample.group_index = position
                    sample.index = position * self.args.n_samples_per_prompt + i
                    group.append(sample)
                samples.append(group)
            return samples

    async def get_samples_async(self, num_samples):
        import asyncio

        return await asyncio.get_running_loop().run_in_executor(self._reader, self.get_samples, num_samples)

    def add_samples(self, samples):
        with self._lock:
            # A completed surplus group can outlive its original rollout's
            # spill directory. Retain its immutable files beside the buffer.
            for sample in iter_samples(samples):
                for key in ("rollout_routed_experts", "rollout_topk_token_ids", "rollout_topk_log_probs"):
                    value = getattr(sample, key)
                    if isinstance(value, DiskTensorRef) and not self.args.keep_rollout_routed_experts_files:
                        directory = Path(self.args.rollout_routed_experts_store_dir) / "buffer"
                        if Path(value.path).parent != directory:
                            value = value.link(directory / f"{uuid.uuid4().hex}.safetensors")
                            setattr(sample, key, value)
                            self._buffer_files.add(value.path)
            super().add_samples(samples)

    def state_dict(self):
        with self._lock:
            if self._prefetch is not None:
                self._collect_prefetch()
            return {
                "ranges": list(self._ranges),
                "request_id": self._request_id,
                "buffer": self.materialize_samples(list(self.buffer), release_files=False),
                "metadata": dict(self.metadata),
            }

    def load_state_dict(self, state):
        with self._lock:
            if self._prefetch is not None:
                raise RuntimeError("Restore the data source before starting generation")
            self._ranges = deque(state["ranges"])
            self._request_id = state["request_id"]
            self.buffer = list(state["buffer"])
            self.metadata = dict(state["metadata"])

    def materialize_samples(self, samples, *, release_files=True):
        """Transfer tensor values, never paths into this worker's local storage."""
        files = set()

        def materialize(value):
            if isinstance(value, list):
                return [materialize(child) for child in value]
            sample = copy.copy(value)
            for key, tensor in vars(sample).items():
                if isinstance(tensor, DiskTensorRef):
                    setattr(sample, key, tensor.load())
                    files.add(tensor.path)
            return sample

        transported = materialize(samples)
        # A group is fully materialized before releasing its own spill files.
        # Never delete a whole rollout directory: other groups may still use it.
        if (
            release_files
            and self.args.rollout_routed_experts_store_dir
            and not self.args.keep_rollout_routed_experts_files
        ):
            root = Path(self.args.rollout_routed_experts_store_dir).resolve()
            with self._lock:
                for name in files:
                    path = Path(name)
                    if root in path.resolve().parents:
                        path.unlink(missing_ok=True)
                        self._buffer_files.discard(name)
        return transported

    def save(self, rollout_id):
        raise RuntimeError("Checkpoint local readers through their owning DistributedDataSourceWithBuffer")

    def load(self, rollout_id=None):
        raise RuntimeError("Restore local readers through their owning DistributedDataSourceWithBuffer")

    def close(self):
        with self._lock:
            self._closed = True
        self._reader.shutdown(wait=True)
        self._transport.shutdown(wait=True)
        for path in self._buffer_files:
            Path(path).unlink(missing_ok=True)


@dataclass(frozen=True)
class ReaderConfig:
    """Small, serializable description; open it in the process that will read data."""

    args: object
    allocator: object
    reader_id: str
    prefetch_size: int

    def open(self, args=None):
        return _DistributedReader(
            self.args if args is None else args,
            lambda request_id, count: ray.get(self.allocator.claim.remote(self.reader_id, request_id, count)),
            prefetch_size=self.prefetch_size,
        )


class DistributedDataSourceWithBuffer(_DistributedReader):
    """Replicated local readers sharing an allocator, with coordinated checkpoints.

    The owner also supports ordinary get_samples/add_samples calls. Rollouts
    pass reader_config() to their own remote processes instead of serializing
    the loaded dataset. Registered consumers own execution and prefetch state;
    the data source only coordinates their pause/snapshot/resume lifecycle.
    """

    def __init__(self, args):
        super().__init__(args, None, prefetch_size=max(1, args.rollout_batch_size))
        # Keep shared index ownership with the manager, so losing a generation
        # node cannot also take down every surviving reader's allocator.
        self.allocator = (
            ray.remote(RolloutIndexAllocator)
            .options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    ray.get_runtime_context().get_node_id(), soft=False
                ),
            )
            .remote()
        )
        self._claim = lambda request_id, count: ray.get(self.allocator.claim.remote("owner", request_id, count))
        self.consumers = {}
        self._restored_consumers = {}
        self.data_config = {
            "dataset_size": len(self),
            "n_samples_per_prompt": args.n_samples_per_prompt,
            "rollout_seed": args.rollout_seed,
            "rollout_shuffle": args.rollout_shuffle,
        }

    def reader_config(self, reader_id, *, prefetch_size=128):
        if reader_id == "owner":
            raise ValueError("Reader ID 'owner' is reserved for the data-source owner")
        return ReaderConfig(self.args, self.allocator, reader_id, prefetch_size)

    def register_consumer(self, name, consumer):
        """Consumers provide pause/resume, state_dict/load_state_dict and close."""
        if name in self.consumers:
            raise ValueError(f"Data consumer {name!r} is already registered")
        try:
            if name in self._restored_consumers:
                consumer.load_state_dict(self._restored_consumers[name])
        except Exception:
            consumer.close()
            raise
        self.consumers[name] = consumer
        self._restored_consumers.pop(name, None)

    def save(self, rollout_id):
        paused = []
        try:
            for consumer in self.consumers.values():
                paused.append((consumer, consumer.pause()))
            state = {
                "data_config": self.data_config,
                "reader": self.state_dict(),
                "consumers": dict(self._restored_consumers),
            }
            state["consumers"].update({name: consumer.state_dict() for name, consumer in self.consumers.items()})
            # Collect reader prefetch claims before snapshotting the global cursor.
            state["allocator"] = ray.get(self.allocator.state_dict.remote())
            path = Path(self.args.save) / "rollout" / f"distributed_data_source_{rollout_id}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
        finally:
            for consumer, was_paused in paused:
                if not was_paused:
                    consumer.resume()

    def load(self, rollout_id=None):
        if not self.args.load:
            return
        if self.consumers:
            raise RuntimeError("Restore the data source before starting rollout consumers")
        path = Path(self.args.load) / "rollout" / f"distributed_data_source_{rollout_id}.pt"
        if not path.exists():
            return
        state = torch.load(path, weights_only=False)
        if state["data_config"] != self.data_config:
            raise ValueError("Restore with the same dataset size, samples per prompt, seed and shuffle settings")
        self.load_state_dict(state["reader"])
        ray.get(self.allocator.load_state_dict.remote(state["allocator"]))
        self._restored_consumers = state["consumers"]

    def close(self):
        if self._closed:
            return
        try:
            for consumer in self.consumers.values():
                consumer.close()
            super().close()
        finally:
            ray.kill(self.allocator)
