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

The worker snapshots the current training-side ``policy_version`` when each
group is admitted and preserves that version on completion, even if a weight
update finishes while generation is still running. It remains oblivious to
pause / abort policy: each in-flight generation surfaces
:data:`Sample.Status.ABORTED` on its own, and the worker redirects those groups
back to ``data_buffer`` instead of shipping them to training.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from dataclasses import dataclass

from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "CompletedSampleRecord",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()
_published_policy_version = 0


@dataclass(frozen=True)
class CompletedSampleRecord:
    """A completed group and the immutable policy version captured at admission."""

    gid: int
    group: list[Sample]
    policy_version: int


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args,
                data_buffer,
                concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args),
                policy_version=max(getattr(args, "policy_version", 0), _published_policy_version),
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


def before_weight_update(*, policy_version: int) -> None:
    """Lifecycle hook called before training publishes a new actor policy."""

    logger.debug("fully-async: preparing to update policy version %d", policy_version)


def after_weight_update(*, policy_version: int, succeeded: bool) -> None:
    """Publish ``policy_version`` after a successful actor weight update."""

    if not succeeded:
        logger.warning("fully-async: weight update failed; policy remains unpublished")
        return

    global _published_policy_version
    with _worker_lock:
        if policy_version < _published_policy_version:
            raise ValueError(
                f"policy version must be monotonic: current={_published_policy_version}, new={policy_version}"
            )
        _published_policy_version = policy_version
        if _global_worker is not None:
            _global_worker.publish_policy_version(policy_version)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10, policy_version: int = 0):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        # Unbounded on purpose: put() runs inside the event-loop thread (task
        # done-callback), so a bounded queue that fills up would block the loop
        # and freeze every in-flight generation. Backpressure lives in _loop()
        # instead, which stops topping up while a full pool of completed groups
        # is already waiting to be consumed.
        self.output_queue: queue.Queue[CompletedSampleRecord] = queue.Queue()
        self.poll_interval = 1.0
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self._policy_version = policy_version
        self._policy_version_lock = threading.Lock()

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self, limit: int | None = None) -> list[CompletedSampleRecord]:
        """Pop up to ``limit`` completed groups (all of them when ``None``).

        Callers that only need a fixed number of groups must pass ``limit`` —
        anything popped beyond it would otherwise have to be thrown away, and
        these groups are fully generated and reward-scored, with their prompts
        already consumed from ``data_buffer``.
        """
        completed: list[CompletedSampleRecord] = []
        while limit is None or len(completed) < limit:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    @property
    def policy_version(self) -> int:
        """Return the latest successfully published policy version."""

        with self._policy_version_lock:
            return self._policy_version

    def publish_policy_version(self, policy_version: int) -> None:
        """Publish a monotonic policy version for future admissions."""

        with self._policy_version_lock:
            if policy_version < self._policy_version:
                raise ValueError(
                    f"policy version must be monotonic: current={self._policy_version}, new={policy_version}"
                )
            self._policy_version = policy_version

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
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
                        policy_version = self.policy_version
                        for sample in group:
                            sample.policy_version = policy_version
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid, policy_version))
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

    def _make_done_cb(self, gid: int, policy_version: int):
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
            # Custom generators may replace the admitted Sample objects. Stamp
            # their outputs from the admission snapshot, never from mutable
            # worker state observed at completion time.
            for sample in result:
                sample.policy_version = policy_version
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put(CompletedSampleRecord(gid=gid, group=result, policy_version=policy_version))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull only what this rollout still needs; the surplus stays queued for
        # the next rollout (that is the "queue stays warm" contract).
        drained = 0
        for record in worker.get_completed_groups(limit=target - len(collected)):
            collected[record.gid] = record.group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return out


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
