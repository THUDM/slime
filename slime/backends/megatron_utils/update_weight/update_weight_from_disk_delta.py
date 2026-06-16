from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import time
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import safetensors.numpy
import torch
import torch.distributed as dist
import zstandard
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader, overwrite_encode
from slime.utils.distributed_utils import get_gloo_group

from .update_weight_from_distributed import UpdateWeightFromDistributed

logger = logging.getLogger(__name__)


class UpdateWeightFromDiskDelta(UpdateWeightFromDistributed):
    """
    Delta weight sync over a shared filesystem. PP-src ranks diff each gathered HF tensor against
    a CPU snapshot of the previous sync and publish the changes as a canonical HF checkpoint dir;
    every rollout host applies the delta into its local checkpoint and reloads via the ordinary
    update_weights_from_disk path, so sglang needs no delta support.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        super().__init__(args, model, weights_getter, model_name=model_name, quantization_config=quantization_config)
        self.delta_dir = args.update_weight_disk_dir
        os.makedirs(self.delta_dir, exist_ok=True)
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        self._commit_hook: Callable | None = None
        if args.custom_delta_pre_push_path:
            from slime.utils.misc import load_function

            self._commit_hook = load_function(args.custom_delta_pre_push_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        all_engine_actors: Sequence[ActorHandle] | None = None,
    ) -> None:
        # The local checkpoint is host-local, so every host applies its own copy:
        # all_engine_actors is one actor per host, vs rollout_engines (node 0 only). The
        # rollout_engine_lock the NCCL path uses isn't needed — a per-host flock serializes applies.
        self.rollout_engines = rollout_engines
        self.all_engine_actors = list(all_engine_actors or rollout_engines)
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )

    def disconnect_rollout_engines(self) -> None:
        pass  # no NCCL groups to tear down

    @torch.no_grad()
    def update_weights(self) -> None:
        # The first call only captures the baseline snapshot the next sync diffs against.
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return

        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        t0 = time.perf_counter()
        self._publish()
        publish_s = time.perf_counter() - t0
        self._reload_engines()
        self._record_metrics(publish_s, time.perf_counter() - t0)

    def _capture_baseline(self) -> None:
        """Capture the baseline snapshot the first delta diffs against (no publish), and clear any
        stale stream from a prior run. Seeds from hf_checkpoint — what each host materializes its
        base from — so the invariant ``snapshot == engine base`` holds even where the megatron->HF
        round-trip trims vocab-padding rows (embed/lm_head). A tensor absent there (rare) falls back
        to the gathered value."""
        # a prior run's versions would apply against the wrong base; start the dir clean
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._commit_hook is not None:
                self._commit_hook(self.args, self.delta_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())

        read_hf = make_tensor_reader(self.args.hf_checkpoint)  # index the HF headers once
        for name, tensor in self._iter_hf_tensors():
            try:
                self._snapshot[name] = read_hf(name)
            except KeyError:
                self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
                logger.warning("seed: %s absent from hf_checkpoint; seeding from current weights", name)
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta] captured baseline snapshot of %d tensors from %s",
                len(self._snapshot),
                self.args.hf_checkpoint,
            )

    def _publish(self) -> None:
        """Encode this version's changed tensors (PP-src ranks), then write it as a canonical HF dir."""
        self._encode_delta()
        dist.barrier(group=get_gloo_group())
        self._write_delta_files()

    def _write_delta_files(self) -> None:
        """Write this rank's changed tensors as one canonical model-NNNNN.safetensors, and on rank
        0 the HF index. The sequential file numbers and the index are coordinated over gloo (small
        object gathers), not the filesystem — a shared volume may not surface one rank's writes to
        another until commit."""
        group = get_gloo_group()
        world, rank = dist.get_world_size(), dist.get_rank()

        # number the files sequentially across only the ranks that have one (no gaps)
        counts: list = [None] * world
        dist.all_gather_object(counts, int(bool(self._delta)), group=group)
        offset, total = sum(counts[:rank]), sum(counts)

        fname = None
        self.wire_bytes = 0
        if self._delta:
            fname = f"model-{offset:05d}-of-{total:05d}.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, fname), blob)

        maps: list = [None] * world
        dist.all_gather_object(maps, {name: fname for name in self._delta}, group=group)
        if rank == 0:
            index = {
                "metadata": {
                    "version": f"{self.weight_version:06d}",
                    "base_version": f"{self.weight_version - 1:06d}",
                    "delta_encoding": self.delta_encoding,
                    "compression_format": "zstd",
                    "checksum_format": self.checksum_algorithm,
                },
                "weight_map": {name: f for m in maps for name, f in m.items()},
            }
            _atomic_write(os.path.join(self._version_dir, "model.safetensors.index.json"), json.dumps(index).encode())
        dist.barrier(group=group)

    def _reload_engines(self) -> None:
        """Commit the published files, have each host apply the delta, then reload the engines."""
        if self._commit_hook is not None:
            self._commit_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([actor.sync_local_checkpoint.remote(self.weight_version) for actor in self.all_engine_actors])
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self.args.update_weight_local_checkpoint_dir,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _iter_hf_tensors(self):
        """Yield (name, gathered HF tensor) for every param: base-class TP then EP gather passes."""
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                yield from hf_chunk
            dist.barrier(group=get_gloo_group())

    def _encode_delta(self) -> None:
        """Diff each gathered HF tensor against the snapshot, keeping the changed ones (compressed)
        in self._delta with their checksums. The GPU->CPU gather is pipelined into a compute pool:
        the main loop copies one tensor to a pinned buffer and submits it; pool workers diff and
        compress in parallel (each is a few big GIL-releasing numpy/zstd calls)."""
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
        if self._is_pp_src_rank:
            os.makedirs(self._version_dir, exist_ok=True)
        snapshot = self._snapshot
        self._delta: dict[str, np.ndarray] = {}  # changed tensor name -> compressed diff
        self._checksums: dict[str, str] = {}  # changed tensor name -> new-state checksum
        self.changed_bytes = self.total_bytes = 0

        # Pinned host-buffer pool: a pinned non_blocking GPU->CPU copy is far faster than .cpu().
        max_bytes = max((int(v.nbytes) for v in snapshot.values()), default=0)
        free_q: queue.Queue = queue.Queue()
        use_pinned = True
        try:
            for _ in range(max(4, min(2 * NUM_WORKERS, (32 << 30) // max(max_bytes, 1)))):
                free_q.put(torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True))
        except RuntimeError as e:  # low memlock limit
            logger.warning("pinned host buffers unavailable (%s); using pageable .cpu()", e)
            use_pinned = False

        def diff_and_compress(name, buf, nbytes, pinned):
            if pinned:  # copy out and free the pinned buffer before the heavy diff/compress
                new = np.empty(nbytes, dtype=np.uint8)
                np.copyto(new, buf.numpy()[:nbytes])
                free_q.put(buf)
            else:
                new = buf
            old = snapshot[name]
            if self.delta_encoding == "xor":
                diff = new ^ old
                changed = int(np.count_nonzero(diff))
            elif self.delta_encoding == "overwrite":
                mask = new != old
                changed = int(np.count_nonzero(mask))
                diff = overwrite_encode(new, mask)
            else:
                raise ValueError(f"unknown delta encoding {self.delta_encoding!r}")
            if not changed:
                return name, new, None, None, 0
            compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
            return name, new, compressed, checksum(self.checksum_algorithm, new), changed

        def collect(fut):
            name, new, compressed, digest, changed = fut.result()
            snapshot[name] = new  # becomes the next sync's base
            if changed:
                self.changed_bytes += changed
                self._delta[name] = compressed
                self._checksums[name] = digest

        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        inflight: deque = deque()
        try:
            for name, tensor in self._iter_hf_tensors():
                flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                nbytes = int(flat.numel())
                if use_pinned and nbytes <= max_bytes:
                    buf = free_q.get()  # blocks when all buffers are in flight -> backpressures the gather
                    buf[:nbytes].copy_(flat, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    payload, pinned = buf, True
                else:
                    payload, pinned = flat.cpu().numpy(), False
                self.total_bytes += nbytes
                inflight.append(pool.submit(diff_and_compress, name, payload, nbytes, pinned))
                if len(inflight) >= 2 * NUM_WORKERS:
                    collect(inflight.popleft())
            while inflight:
                collect(inflight.popleft())
        finally:
            pool.shutdown()

    def _record_metrics(self, publish_s: float, total_s: float) -> None:
        """All-reduce the byte counts and record changed-fraction / wire size / sync time; the
        actor drains update_weight_metrics onto the step log."""
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        m = self.update_weight_metrics
        m["perf/update_weights_density"] = changed / max(total, 1)
        m["perf/update_weights_wire_bytes"] = wire
        m["perf/update_weights_total_s"] = total_s
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta v=%s] update %.1fs (publish %.1fs reload %.1fs) | density=%.2f%% wire=%.2f GB",
                self.weight_version,
                total_s,
                publish_s,
                total_s - publish_s,
                100.0 * changed / max(total, 1),
                wire / 1e9,
            )


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
