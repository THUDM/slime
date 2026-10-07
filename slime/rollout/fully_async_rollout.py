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

``--dynamic-sampling-filter-path`` is honoured DAPO-style: rejected groups do
not count toward ``rollout_batch_size`` and are replaced from the warm queue.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

from slime.data.transport import discard_rollout_group, publish_rollout_async
from slime.rollout.base_types import RolloutFnTrainOutput, finalize_rollout_groups
from slime.rollout.filter_hub.base_types import call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


def _get_worker(args, data_buffer):
    # The data source owns execution across calls and checkpoint boundaries.
    # Storage only decides how results are published.
    if not hasattr(data_buffer, "consumers"):
        data_buffer.consumers = {}
    worker = data_buffer.consumers.get("fully_async")
    if worker is None:
        if hasattr(data_buffer, "reader_config"):
            from slime.rollout.fully_async_distributed import DistributedRollout

            worker = DistributedRollout(args, data_buffer)
        else:
            worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
        if hasattr(data_buffer, "register_consumer"):
            data_buffer.register_consumer("fully_async", worker)
        else:
            data_buffer.consumers["fully_async"] = worker
        if isinstance(worker, AsyncRolloutWorker):
            worker.start()
    return worker


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = max(1, concurrency // getattr(args, "n_samples_per_prompt", 1))
        self.running = True
        self.paused = False
        self.condition = threading.Condition()
        self.active = 0
        self.admission_paused = False
        self.error = None
        # Unbounded on purpose: put() runs inside the event-loop thread (task
        # done-callback), so a bounded queue that fills up would block the loop
        # and freeze every in-flight generation. Backpressure lives in _loop()
        # instead, which stops topping up while a full pool of completed groups
        # is already waiting to be consumed.
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue()
        self.poll_interval = 0.01
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def pause(self, *, drain=True):
        with self.condition:
            previous = self.paused
            self.paused = True
            while self.running and self.error is None and (not self.admission_paused or (drain and self.active)):
                self.condition.wait()
            if self.error is not None:
                raise self.error
            return previous

    def resume(self):
        with self.condition:
            self.paused = False

    def state_dict(self):
        if not self.admission_paused or self.active:
            raise RuntimeError("Pause generation before saving worker state")
        with self.output_queue.mutex:
            return {"ready": list(self.output_queue.queue)}

    def load_state_dict(self, state):
        if self.worker_thread is not None:
            raise RuntimeError("Restore worker state before starting generation")
        for result in state["ready"]:
            self.output_queue.put(result)

    def close(self):
        self.running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=35)
            if self.worker_thread.is_alive():
                raise TimeoutError("Generation did not stop within 35 seconds")

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

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception as error:
            self.error = error
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()

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
                    if done:
                        # Done callbacks requeue ABORTED groups. Let them run
                        # before asking the data source for replacement work.
                        await asyncio.sleep(0)

                # Top up. The qsize gate is the queue's backpressure: once a
                # full pool of completed groups is waiting, stop pulling new
                # prompts until the training side drains some.
                while (
                    len(active_tasks) < max_concurrent
                    and self.output_queue.qsize() < max_concurrent
                    and self.running
                    and not self.paused
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

                with self.condition:
                    self.active = len(active_tasks)
                    self.admission_paused = self.paused
                    self.condition.notify_all()
                if self.error is not None:
                    raise self.error
                await asyncio.sleep(self.poll_interval)
            except Exception as e:  # noqa: BLE001
                self.error = e
                raise

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
            except asyncio.CancelledError:
                return
            except Exception as error:
                self.error = error
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


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput | list[list[Sample]]:
    filters_enabled = bool(
        getattr(args, "dynamic_sampling_filter_path", None) or getattr(args, "rollout_sample_filter_path", None)
    )
    worker = _get_worker(args, data_buffer)
    worker.resume()

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
        if worker.error is not None:
            raise worker.error
        # Pull only what this rollout still needs; the surplus stays queued for
        # the next rollout (that is the "queue stays warm" contract).
        drained = 0
        for _gid, group in worker.get_completed_groups(limit=target - len(collected)):
            drained += 1
            verdict = call_dynamic_filter(dynamic_filter, args, group)
            if verdict.keep:
                if args.rollout_data_transport == "straw":
                    group = await publish_rollout_async(
                        group, args, rollout_id, group=True, controller=getattr(data_buffer, "controller", None)
                    )
                collected.append(group)
                continue

            await asyncio.to_thread(
                discard_rollout_group,
                group,
                args,
                verdict.reason or "dynamic_filter",
                controller=getattr(data_buffer, "controller", None),
            )
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
        output = finalize_rollout_groups(
            args, rollout_id, collected, metrics, controller=getattr(data_buffer, "controller", None)
        )
    else:
        output = await asyncio.to_thread(
            finalize_rollout_groups,
            args,
            rollout_id,
            collected,
            metrics if filters_enabled else None,
            controller=getattr(data_buffer, "controller", None),
        )
    return output if filters_enabled or args.rollout_data_transport == "straw" else output.samples


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    worker = _get_worker(args, data_buffer)
    if not isinstance(worker, AsyncRolloutWorker):
        return worker.generate(rollout_id, prefetch=worker.capacity)
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
