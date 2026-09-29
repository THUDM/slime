"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.

A weight update pauses the engines, which aborts every in-flight request: the
sample keeps the tokens and per-token rollout log-probs generated so far. When
the requeued group is picked up again, ``generate`` resends prompt plus partial
response with the remaining token budget, so the trajectory continues under the
new weights and its ``weight_versions`` lists every policy that produced it.
Correct those tokens with ``--use-tis`` (e.g. FlashREINFORCE's
``binary_kl_trust_region_function``) or mask them with ``--partial-rollout
--mask-offpolicy-in-partial-rollout``.

Evaluation runs on the worker's event loop, next to the in-flight trajectories,
because ``GenerateState``'s semaphore and pacer are bound to one event loop.

``--dynamic-sampling-filter-path`` is honoured DAPO-style: rejected groups do
not count toward ``rollout_batch_size`` and are replaced from the warm queue.

``--fully-async-pool-size`` selects :class:`PooledRolloutWorker` instead, which
forms batches the way molt's asynchronous trainer does (see its docstring).
"""

from __future__ import annotations

import asyncio
import atexit
import heapq
import logging
import queue
import random
import threading
import time

from slime.rollout.base_types import RolloutFnTrainOutput, finalize_rollout_groups, iter_samples
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, eval_rollout, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.rollout_transport import discard_rollout_group, publish_rollout_async
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "PooledRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | PooledRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker | PooledRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            if getattr(args, "fully_async_pool_size", None) is not None:
                _global_worker = PooledRolloutWorker(args, data_buffer)
            else:
                _global_worker = AsyncRolloutWorker(
                    args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
                )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = max(1, concurrency // getattr(args, "n_samples_per_prompt", 1))
        self.running = True
        # Unbounded on purpose: put() runs inside the event-loop thread (task
        # done-callback), so a bounded queue that fills up would block the loop
        # and freeze every in-flight generation. Backpressure lives in _loop()
        # instead, which stops topping up while a full pool of completed groups
        # is already waiting to be consumed.
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue()
        self.poll_interval = 1.0
        self.worker_thread: threading.Thread | None = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self._event_loop_ready = threading.Event()
        self.state = GenerateState(args)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self, limit: int | None = None) -> list[tuple[int, list[Sample]]]:
        """Pop up to ``limit`` completed groups (all of them when ``None``).

        Callers that only need a fixed number of groups must pass ``limit`` —
        anything popped beyond it would otherwise have to be thrown away, and
        these groups are fully generated and reward-scored, with their prompts
        already consumed from ``data_buffer``.
        """
        completed: list[tuple[int, list[Sample]]] = []
        while limit is None or len(completed) < limit:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def run_coroutine(self, coro):
        """Run ``coro`` on the worker's event loop and block until it finishes.

        ``GenerateState``'s semaphore and pacer are bound to the event loop that
        first waits on them, so any other generation that shares them with the
        in-flight trajectories (e.g. evaluation) must run on this loop too.
        """
        self._event_loop_ready.wait()
        return asyncio.run_coroutine_threadsafe(coro, self.event_loop).result()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        self.event_loop = asyncio.get_running_loop()
        self._event_loop_ready.set()
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done
                    if done:
                        # Done callbacks requeue ABORTED groups. Let them run
                        # before asking the data source for replacement work.
                        await asyncio.sleep(0)

                # Top up. The qsize gate is the queue's backpressure: once a
                # full pool of completed groups is waiting, stop pulling new
                # prompts until the training side drains some.
                while (
                    len(active_tasks) < max_concurrent and self.output_queue.qsize() < max_concurrent and self.running
                ):
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid))
                        active_tasks.add(task)

                await asyncio.sleep(self.poll_interval)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(self.poll_interval)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


# A weight update aborts the requests still generating and SGLang holds a resent request
# until the engines continue; the delay only bounds the retries of any other abort.
_ABORT_RETRY_DELAY = 1.0


class PooledRolloutWorker:
    """Fully-async worker that forms training batches the way molt's asynchronous trainer does.

    Selected by ``--fully-async-pool-size``. That many rollouts stay in a pool,
    counting the ones still generating and the finished ones not yet placed in a
    batch. A batch is formed only while one of ``--fully-async-max-queued-batches``
    slots is free: until it holds ``rollout_batch_size`` groups, the worker tops the
    pool up to its size and takes one finished rollout. When several have finished,
    it takes the one first in a random order fixed at dispatch, as molt's
    ``ray.wait`` does. Training frees a slot when it takes a batch, before it trains
    on it, so at most that many batches are formed ahead of the trainer.

    A weight update aborts the rollouts still generating. Each is resent at once and
    keeps its place in the pool: SGLang holds the request until the engines continue,
    and generation resumes from the partial response under the new weights, as after
    molt's keep-mode pause, which re-prefills the prefix.

    With ``--fully-async-drain-each-epoch`` the pool stops refilling once every
    prompt of the epoch is dispatched, and the epoch ends with a smaller batch of the
    remaining rollouts (molt's episodes); the next batch starts the next epoch.
    Batch formation pauses while an evaluation runs.
    """

    def __init__(self, args, data_buffer):
        if args.fully_async_drain_each_epoch and not (
            hasattr(data_buffer, "dataset") and hasattr(data_buffer, "sample_offset") and len(data_buffer) > 0
        ):
            raise ValueError(
                "--fully-async-drain-each-epoch needs prompt data read by slime.rollout.data_source.RolloutDataSource, "
                "which tracks the epoch position."
            )
        self.args = args
        self.data_buffer = data_buffer
        self.pool_size = args.fully_async_pool_size
        self.max_queued_batches = args.fully_async_max_queued_batches
        self.drain_each_epoch = args.fully_async_drain_each_epoch
        self.dynamic_filter = (
            load_function(args.dynamic_sampling_filter_path)
            if getattr(args, "dynamic_sampling_filter_path", None) is not None
            else None
        )
        model_parallel = (
            getattr(args, "tensor_model_parallel_size", 1)
            * getattr(args, "pipeline_model_parallel_size", 1)
            * getattr(args, "context_parallel_size", 1)
        )
        self.dp_size = max(
            1, getattr(args, "actor_num_nodes", 1) * getattr(args, "actor_num_gpus_per_node", 1) // model_parallel
        )
        self.state = GenerateState(args)
        self.running = True
        # Formed batches waiting for training. The slots bound it, so put() never blocks the loop.
        self.formed: queue.Queue[tuple[list[list[Sample]], dict]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self._event_loop_ready = threading.Event()
        self._main_task: asyncio.Task | None = None
        self._rng = random.Random(args.rollout_seed)
        self._pool: dict[int, asyncio.Task] = {}  # dispatch key -> rollout, generating or finished
        self._finished: list[tuple[float, int]] = []  # heap of (collection order, dispatch key)
        self._next_key = 0
        self._failed = 0
        self._draining = False
        self._next_epoch = False

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.event_loop is not None and self._main_task is not None:
            try:
                self.event_loop.call_soon_threadsafe(self._main_task.cancel)
            except RuntimeError:  # the loop already closed
                pass
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def next_batch(self) -> tuple[list[list[Sample]], dict]:
        """Block until a batch is formed, then take it and free its slot."""
        while True:
            try:
                groups, metrics = self.formed.get(timeout=5)
                break
            except queue.Empty:
                if not self.worker_thread.is_alive():
                    raise RuntimeError("the fully-async rollout worker stopped") from None
        metrics["rollout/fully_async/queued_batches"] = self.formed.qsize()
        # As in molt, taking a batch frees its slot before training on it.
        self.event_loop.call_soon_threadsafe(self._slots.release)
        return groups, metrics

    def run_coroutine(self, coro):
        """Run ``coro`` on the worker's event loop, where ``GenerateState``'s primitives live."""
        self._event_loop_ready.wait()
        return asyncio.run_coroutine_threadsafe(coro, self.event_loop).result()

    def evaluate(self, coro):
        """Run an evaluation coroutine on the worker's event loop, pausing batch formation meanwhile."""

        async def paused():
            self._forming.clear()
            try:
                return await coro
            finally:
                self._forming.set()

        return self.run_coroutine(paused())

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._loop())
        except asyncio.CancelledError:
            pass

    async def _loop(self) -> None:
        self.event_loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        self._slots = asyncio.Semaphore(self.max_queued_batches)
        self._forming = asyncio.Event()
        self._forming.set()
        self._wakeup = asyncio.Event()
        self._event_loop_ready.set()
        while self.running:
            await self._slots.acquire()
            groups, metrics = await self._form_batch()
            if groups:
                self.formed.put((groups, metrics))
            else:
                # Nothing to train on (the prompt data ran dry, or a tail too small to spread over the
                # data-parallel ranks was dropped): keep the slot and try again.
                self._slots.release()
                await asyncio.sleep(0.1)

    async def _form_batch(self) -> tuple[list[list[Sample]], dict]:
        if self._draining and not self._pool:
            # The previous batch drained an epoch; this one starts the next.
            self._draining = False
            self._next_epoch = True
        groups: list[list[Sample]] = []
        drop_reasons: dict[str, int] = {}
        while len(groups) < self.args.rollout_batch_size:
            await self._forming.wait()
            self._refill()
            group = await self._take_finished()
            if group is None:
                break  # the pool is empty: the epoch is drained
            verdict = call_dynamic_filter(self.dynamic_filter, self.args, group)
            if verdict.keep:
                groups.append(group)
                continue
            reason = verdict.reason or "dynamic_filter"
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            await asyncio.to_thread(discard_rollout_group, group, self.args, reason)

        tail = self._draining and not self._pool
        metrics = {
            "rollout/fully_async/pool_finished": len(self._finished),
            "rollout/fully_async/epoch_tail": float(tail),
            "rollout/fully_async/failed_rollouts": self._failed,
        }
        self._failed = 0
        if self.dynamic_filter is not None:
            dropped = sum(drop_reasons.values())
            metrics |= {f"rollout/dynamic_filter/drop_{reason}": count for reason, count in drop_reasons.items()}
            metrics["rollout/dynamic_filter/dropped_groups"] = dropped
            metrics["rollout/dynamic_filter/dropped_ratio"] = dropped / max(len(groups) + dropped, 1)
        if tail:
            num_samples = sum(1 for _ in iter_samples(groups))
            logger.info("fully-async: epoch drained, its last batch holds %d groups", len(groups))
            if 0 < num_samples < self.dp_size:
                # Like molt, drop an epoch tail that cannot give every data-parallel rank a sample.
                logger.warning(
                    "fully-async: dropping the %d-sample epoch tail, fewer than %d DP ranks", num_samples, self.dp_size
                )
                groups = []
        return groups, metrics

    def _refill(self) -> None:
        free = self.pool_size - len(self._pool)
        if self._draining or free <= 0:
            return
        if self.drain_each_epoch:
            remaining = len(self.data_buffer.dataset) - self.data_buffer.sample_offset
            if self._next_epoch:
                # get_samples starts the next epoch once the current one is used up.
                self._next_epoch = False
                remaining = remaining or len(self.data_buffer.dataset)
            if remaining == 0:
                self._draining = True
                return
            free = min(free, remaining)
        for group in self.data_buffer.get_samples(free):
            self._dispatch(group)

    def _dispatch(self, group: list[Sample]) -> None:
        key = self._next_key
        self._next_key += 1
        order = self._rng.random()
        task = asyncio.create_task(self._run(group))
        self._pool[key] = task
        task.add_done_callback(lambda task: self._on_done(key, order, task))

    async def _run(self, group: list[Sample]) -> list[Sample]:
        while True:
            group = await generate_and_rm_group(
                self.args, group, sampling_params=self.state.sampling_params.copy(), evaluation=False
            )
            if not any(sample.status == Sample.Status.ABORTED for sample in iter_samples(group)):
                return group
            if any(isinstance(item, list) for item in group):
                raise RuntimeError("cannot resume ABORTED samples of a custom generate function that fans out")
            # A weight update aborted the request; resend it, and it resumes under the new weights.
            await asyncio.sleep(_ABORT_RETRY_DELAY)

    def _on_done(self, key: int, order: float, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        if task.exception() is not None:
            logger.error("fully-async: dropping a failed rollout", exc_info=task.exception())
            self._pool.pop(key, None)
            self._failed += 1
        else:
            heapq.heappush(self._finished, (order, key))
        self._wakeup.set()

    async def _take_finished(self) -> list[Sample] | None:
        """Take the finished rollout first in collection order, waiting for one if none has finished."""
        while not self._finished:
            if not self._pool:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()
        _, key = heapq.heappop(self._finished)
        return self._pool.pop(key).result()


def _take_pooled_batch(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    worker = _get_global_worker(args, data_buffer)
    started = time.time()
    groups, metrics = worker.next_batch()
    logger.info(
        "fully-async rollout %d: took %d groups after %.1fs; %d batches formed ahead, %d finished rollouts pooled",
        rollout_id,
        len(groups),
        time.time() - started,
        metrics["rollout/fully_async/queued_batches"],
        metrics["rollout/fully_async/pool_finished"],
    )
    return finalize_rollout_groups(args, rollout_id, groups, metrics)


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput | list[list[Sample]]:
    filters_enabled = bool(
        getattr(args, "dynamic_sampling_filter_path", None) or getattr(args, "rollout_sample_filter_path", None)
    )
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected = []
    dropped_count = 0
    drop_reasons: dict[str, int] = {}
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull only what this rollout still needs; the surplus stays queued for
        # the next rollout (that is the "queue stays warm" contract).
        drained = 0
        for _gid, group in worker.get_completed_groups(limit=target - len(collected)):
            drained += 1
            verdict = call_dynamic_filter(dynamic_filter, args, group)
            if verdict.keep:
                if args.rollout_data_transport == "straw":
                    group = await publish_rollout_async(group, args, rollout_id, group=True)
                collected.append(group)
                continue

            await asyncio.to_thread(discard_rollout_group, group, args, verdict.reason or "dynamic_filter")
            reason = verdict.reason or "dynamic_filter"
            dropped_count += 1
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d (dropped %d), queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                dropped_count,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    logger.info(
        "fully-async rollout %d: done in %.1fs, kept=%d dropped=%d (%s), queue_left=%d",
        rollout_id,
        time.time() - started,
        len(collected),
        dropped_count,
        drop_reasons,
        worker.queue_size(),
    )
    metrics = {f"rollout/dynamic_filter/drop_{reason}": count for reason, count in drop_reasons.items()}
    metrics["rollout/dynamic_filter/dropped_groups"] = dropped_count
    metrics["rollout/dynamic_filter/dropped_ratio"] = dropped_count / (len(collected) + dropped_count)
    if args.rollout_sample_filter_path is not None:
        # Preserve the calling thread/context of custom batch hooks.
        output = finalize_rollout_groups(args, rollout_id, collected, metrics)
    else:
        output = await asyncio.to_thread(
            finalize_rollout_groups, args, rollout_id, collected, metrics if filters_enabled else None
        )
    return output if filters_enabled or args.rollout_data_transport == "straw" else output.samples


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if getattr(args, "rollout_data_transport", "object-store") == "straw":
        from slime.rollout.queue_data_source import QueueDataSource

        if isinstance(data_buffer, QueueDataSource):
            if evaluation:
                raise ValueError("distributed fully-async rollout doesn't support evaluation mode")
            from slime.rollout.fully_async_distributed import DistributedRollout

            worker = data_buffer.consumers.get("fully_async")
            if worker is None:
                worker = DistributedRollout(args, data_buffer)
                data_buffer.register_consumer("fully_async", worker)
            return worker.generate(rollout_id, prefetch=worker.capacity)
    if evaluation:
        # Evaluate next to the in-flight trajectories: they share GenerateState's loop-bound primitives.
        worker = _get_global_worker(args, data_buffer)
        execute = worker.evaluate if isinstance(worker, PooledRolloutWorker) else worker.run_coroutine
        output, _ = execute(eval_rollout(args, rollout_id))
        return output
    if getattr(args, "fully_async_pool_size", None) is not None:
        return _take_pooled_batch(args, rollout_id, data_buffer)
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
