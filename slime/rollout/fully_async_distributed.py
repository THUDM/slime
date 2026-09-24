"""Process execution and bounded prefetch owned by the fully-async rollout function."""

import asyncio
import copy
import logging
import threading
from collections import Counter, deque
from pathlib import Path

import ray
from ray.exceptions import RayActorError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slime.observability import logging_utils
from slime.observability.rollout_data_utils import validate_rollout_id_annotated
from slime.ray.utils import add_default_ray_env_vars
from slime.rollout.base_types import RolloutFnTrainOutput, iter_samples
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.sample_hooks import rollout_context
from slime.utils.async_utils import get_async_loop
from slime.utils.http_utils import get_rollout_num_engines, init_http_client
from slime.utils.misc import load_function
from slime.utils.types import Sample


class RolloutScheduler:
    """Collect results from autonomous workers into global training batches.

    Each worker maintains its own generation concurrency. This collector bounds
    completed prefetch, so slow training applies backpressure to local pools.
    """

    def __init__(self, args, workers, capacities):
        self.args = args
        self.workers = workers
        self.capacities = capacities
        self.capacity = sum(capacities)
        self.ready = deque()
        self.pending = {}
        self.controls = {}
        self.running_workers = set()
        self.demand = self.prefetch = 0
        self.rollout_id = 0
        self.paused = True
        self.closed = False
        self.error = None
        self.condition = threading.Condition()
        self.thread = None

    def _retire_worker(self, worker, reason):
        """Drop unavailable workers without replaying their unreturned groups.

        Called under the condition lock. Checkpoint calls here run after
        pause() has drained all pending result requests.
        """
        if not self.capacities[worker]:
            return
        for ref, owner in list(self.pending.items()):
            if owner == worker:
                ray.cancel(ref)
                del self.pending[ref]
        for ref, (owner, _) in list(self.controls.items()):
            if owner == worker:
                ray.cancel(ref)
                del self.controls[ref]
        self.running_workers.discard(worker)
        self.capacities[worker] = 0
        self.capacity = sum(self.capacities)
        self.prefetch = min(self.prefetch, self.capacity)
        ray.kill(self.workers[worker], no_restart=True)
        logging.getLogger(__name__).warning(
            "Retired rollout worker %d: %s. Unreturned groups are discarded; remaining capacity=%d",
            worker,
            reason,
            self.capacity,
        )
        self.condition.notify_all()
        if not self.capacity:
            raise RuntimeError("All rollout workers are unavailable")

    def _run(self):
        configurations = {}
        try:
            while True:
                with self.condition:
                    if self.closed:
                        return
                    desired = (self.paused, self.rollout_id)
                    configuring = {worker for worker, _ in self.controls.values()}
                    for i, worker in enumerate(self.workers):
                        if not self.capacities[i] or i in configuring or configurations.get(i) == desired:
                            continue
                        if self.paused and i not in self.running_workers:
                            configurations[i] = desired
                            continue
                        # Control RPCs share the wait loop with results. Never
                        # hold the condition lock waiting for a remote worker.
                        ref = worker.pause.remote() if self.paused else worker.start.remote(self.rollout_id)
                        self.controls[ref] = (i, desired)
                    # One outstanding result request per worker. These requests
                    # only consume results; workers refill their pools locally.
                    waiting = set(self.pending.values())
                    for i in self.running_workers - waiting:
                        if not self.paused and len(self.pending) + len(self.ready) >= self.demand + self.prefetch:
                            break
                        self.pending[self.workers[i].next.remote()] = i
                    refs = [*self.controls, *self.pending]
                    if not refs:
                        self.condition.notify_all()
                        self.condition.wait(timeout=0.05)
                        continue
                done, _ = ray.wait(refs, num_returns=1, timeout=0.01)
                with self.condition:
                    for ref in done:
                        control = self.controls.pop(ref, None)
                        worker = control[0] if control is not None else self.pending.pop(ref)
                        try:
                            output = ray.get(ref)
                        except RayActorError as error:
                            self._retire_worker(worker, error)
                            continue
                        if control is not None:
                            configurations[worker] = control[1]
                            if not control[1][0]:
                                self.running_workers.add(worker)
                        elif output is None:  # The paused worker has drained its pool.
                            self.running_workers.remove(worker)
                        else:
                            self.ready.append(output)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            with self.condition:
                for ref in (*self.controls, *self.pending):
                    ray.cancel(ref)
                self.controls.clear()
                self.pending.clear()
                self.running_workers.clear()
                self.condition.notify_all()

    def generate(self, rollout_id, *, prefetch):
        groups = []
        dropped_count = 0
        drop_reasons = Counter()
        with self.condition:
            if self.error is not None:
                raise self.error
            if self.demand:
                raise RuntimeError("A rollout batch is already being collected")
            self.rollout_id = rollout_id
            self.demand = self.args.rollout_batch_size
            self.prefetch = min(prefetch, self.capacity)
            self.paused = False
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="rollout-collect", daemon=True)
                self.thread.start()
            self.condition.notify_all()
            while self.demand:
                if self.error is not None:
                    raise self.error
                if not self.ready:
                    self.condition.wait()
                    continue
                group, verdict = self.ready.popleft()
                # Fully async continuously replaces rejected groups. The
                # finite-batch keep_when_insufficient fallback does not apply.
                if not verdict.keep:
                    reason = verdict.reason or "dynamic_filter"
                    dropped_count += 1
                    drop_reasons[reason] += 1
                else:
                    groups.append(group)
                    self.demand -= 1
                self.condition.notify_all()

        def first_index(group):
            return next(iter_samples(group)).index

        groups.sort(key=first_index)
        if self.args.rollout_sample_filter_path is not None:
            load_function(self.args.rollout_sample_filter_path)(self.args, groups)
        metrics = {f"rollout/dynamic_filter/drop_{reason}": count for reason, count in drop_reasons.items()}
        metrics["rollout/dynamic_filter/dropped_groups"] = dropped_count
        metrics["rollout/dynamic_filter/dropped_ratio"] = dropped_count / (len(groups) + dropped_count)
        return RolloutFnTrainOutput(samples=groups, metrics=metrics)

    def pause(self):
        with self.condition:
            was_paused = self.paused
            self.paused = True
            self.condition.notify_all()
            while (self.controls or self.pending or self.running_workers) and self.error is None:
                self.condition.wait()
            if self.error is not None:
                raise self.error
            return was_paused

    def resume(self):
        with self.condition:
            self.paused = False
            self.condition.notify_all()

    def state_dict(self):
        with self.condition:
            if self.controls or self.pending or self.running_workers or not self.paused:
                raise RuntimeError("Pause scheduling before saving rollout state")
            return {"ready": list(self.ready)}

    def load_state_dict(self, state):
        with self.condition:
            if self.thread is not None:
                raise RuntimeError("Restore rollout state before starting generation")
            self.ready = deque(state["ready"])

    def close(self):
        with self.condition:
            self.closed = True
            # Unblock consumers and pause() callers waiting for in-flight work.
            if self.error is None:
                self.error = RuntimeError("Rollout scheduler is closed")
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join()


class _GenerationActor:
    """Maintain a local generation pool and expose completed groups to the collector."""

    def __init__(self, reader_config, concurrency):
        from slime.rollout.sglang_rollout import GenerateState

        logging_utils.configure_logger(prefix=f" fully-async-{reader_config.reader_id}")
        self.args = copy.copy(reader_config.args)
        self.args.use_distributed_post = False
        self.args.use_wandb = self.args.use_tensorboard = False
        self.args.dump_details = None
        if self.args.rollout_routed_experts_store_dir:
            self.args.rollout_routed_experts_store_dir = str(
                Path(self.args.rollout_routed_experts_store_dir) / reader_config.reader_id
            )
        init_http_client(self.args, concurrency=concurrency)
        self.state = GenerateState(self.args, concurrency=concurrency)
        self.data_source = reader_config.open(self.args)
        self.dynamic_filter = (
            load_function(self.args.dynamic_sampling_filter_path) if self.args.dynamic_sampling_filter_path else None
        )

        self.capacity = concurrency // self.args.n_samples_per_prompt
        self.running = False
        self.rollout_id = 0
        self.producer = None

    def dataset_size(self):
        return len(self.data_source)

    async def start(self, rollout_id):
        future = asyncio.run_coroutine_threadsafe(self._start(rollout_id), get_async_loop().loop)
        await asyncio.wrap_future(future)

    async def _start(self, rollout_id):
        self.rollout_id = rollout_id
        self.running = True
        if self.producer is None or self.producer.done():
            if self.producer is not None:
                self.producer.result()
            # Keep completed results bounded even when training stops consuming.
            # At most one queued result plus one result per pool slot stays local.
            self.outputs = asyncio.Queue(maxsize=1)
            self.producer = asyncio.create_task(self._produce())

    async def pause(self):
        future = asyncio.run_coroutine_threadsafe(self._pause(), get_async_loop().loop)
        await asyncio.wrap_future(future)

    async def _pause(self):
        self.running = False

    async def next(self):
        future = asyncio.run_coroutine_threadsafe(self._next(), get_async_loop().loop)
        return await asyncio.wrap_future(future)

    async def _next(self):
        take = asyncio.create_task(self.outputs.get())
        try:
            await asyncio.wait((take, self.producer), return_when=asyncio.FIRST_COMPLETED)
            if take.done():
                return take.result()
            self.producer.result()  # Propagate custom generation errors.
            if not self.outputs.empty():
                return await take
            return None
        finally:
            take.cancel()
            await asyncio.gather(take, return_exceptions=True)

    async def _produce(self):
        async def generate_slot():
            while self.running:
                output = await self._generate_group(self.rollout_id)
                if output is not None:  # Aborted groups are already buffered locally.
                    await self.outputs.put(output)
                del output

        tasks = []
        try:
            for offset in range(0, self.capacity, 64):
                if not self.running:
                    break
                tasks.extend(asyncio.create_task(generate_slot()) for _ in range(min(64, self.capacity - offset)))
                # Spread startup callbacks across loop turns to reduce asyncio
                # pressure. Each slot then pulls its own next group immediately.
                await asyncio.sleep(0.001)
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_group(self, rollout_id):
        from slime.rollout.sglang_rollout import generate_and_rm_group

        group = (await self.data_source.get_samples_async(1))[0]
        # Completed local-buffer groups keep their rewards and loss masks.
        if not all(
            sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) and sample.reward is not None
            for sample in iter_samples(group)
        ):
            with rollout_context(rollout_id):
                group = await generate_and_rm_group(self.args, group, self.state.sampling_params.copy())
        validate_rollout_id_annotated([group])
        samples = list(iter_samples(group))
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            self.data_source.add_samples([group])
            return None
        for sample in samples:
            if sample.index is None:
                raise ValueError("Distributed generation must preserve the source sample index")
            if sample.rollout_id is None:
                sample.rollout_id = sample.index
        verdict = call_dynamic_filter(self.dynamic_filter, self.args, group)
        group = await asyncio.to_thread(self.data_source.materialize_samples, group)
        return group, verdict

    def state_dict(self):
        return self.data_source.state_dict()

    def load_state_dict(self, state):
        self.data_source.load_state_dict(state)


class DistributedRollout(RolloutScheduler):
    """Persistent fully-async execution; the data source coordinates checkpointing."""

    def __init__(self, args, data_source):
        if args.rollout_all_samples_process_path is not None:
            raise ValueError("--rollout-all-samples-process-path is not supported by distributed fully-async rollout")
        nodes = sorted(
            (node for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) >= 1),
            key=lambda node: (node["NodeManagerAddress"], node["NodeID"]),
        )
        total_concurrency = args.sglang_server_concurrency * get_rollout_num_engines(args)
        if not nodes or total_concurrency < len(nodes) * args.n_samples_per_prompt:
            raise ValueError("Distributed rollout needs concurrency for at least one prompt group per node")
        groups_per_worker, remainder = divmod(total_concurrency // args.n_samples_per_prompt, len(nodes))
        workers, capacities = [], []
        try:
            for i, node in enumerate(nodes):
                capacity = groups_per_worker + (i < remainder)
                capacities.append(capacity)
                reader = data_source.reader_config(f"fully_async_{i}", prefetch_size=capacity)
                workers.append(
                    ray.remote(_GenerationActor)
                    .options(
                        num_cpus=1,
                        max_concurrency=3,  # Result consumption, lifecycle control, and checkpoint access.
                        max_restarts=0,
                        max_task_retries=0,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
                        runtime_env={"env_vars": add_default_ray_env_vars()},
                    )
                    .remote(reader, capacity * args.n_samples_per_prompt)
                )
            sizes = ray.get([worker.dataset_size.remote() for worker in workers])
            if any(size != len(data_source) for size in sizes):
                raise ValueError(f"Replicated datasets differ in size: owner={len(data_source)}, workers={sizes}")
        except Exception:
            for worker in workers:
                ray.kill(worker)
            raise
        super().__init__(args, workers, capacities)
        logging.getLogger(__name__).info(
            "Fully-async rollout: %d nodes, dataset=%d, concurrency=%d",
            len(nodes),
            len(data_source),
            total_concurrency,
        )

    def state_dict(self):
        scheduler = super().state_dict()
        readers = []
        for i, worker in enumerate(self.workers):
            reader = None
            if self.capacities[i]:
                try:
                    reader = ray.get(worker.state_dict.remote())
                except RayActorError as error:
                    with self.condition:
                        self._retire_worker(i, error)
            readers.append(reader)
        return {"scheduler": scheduler, "readers": readers}

    def load_state_dict(self, state):
        if len(state["readers"]) != len(self.workers):
            raise ValueError("Restore fully-async rollout with the same number of worker nodes")
        super().load_state_dict(state["scheduler"])
        for i, (worker, reader) in enumerate(zip(self.workers, state["readers"], strict=True)):
            with self.condition:
                if reader is None:
                    self._retire_worker(i, "disabled in checkpoint")
                    continue
                try:
                    ray.get(worker.load_state_dict.remote(reader))
                except RayActorError as error:
                    self._retire_worker(i, error)

    def close(self):
        super().close()
        for worker in self.workers:
            ray.kill(worker, no_restart=True)
