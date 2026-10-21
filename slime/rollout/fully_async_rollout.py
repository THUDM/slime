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
``--fully-async-group-prefetch-factor`` can enlarge the bounded group pool
so slow group members do not leave sample slots idle. It does not enlarge
the per-sample semaphore.

The worker snapshots the current training-side ``policy_version`` when each
group is admitted and preserves it on completion.

Admission pause leaves in-flight tasks alive across training rounds. Agent
adapters with ``--partial-rollout`` continue interrupted model turns within
those tasks, retaining the live sandbox and CLI. A generator that does return
an ABORTED/PENDING result is requeued instead of sent to training. Flat samples
retain their partial response; segmented agent failures retry their original
inputs because their closed sandbox sessions cannot be restored from tokens.
Completed trajectories in the same group retain their results across retries.

``--dynamic-sampling-filter-path`` is honoured DAPO-style: rejected groups do
not count toward ``rollout_batch_size`` and are replaced from the warm queue.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace

from slime.data.transport import discard_rollout_group, publish_rollout_async
from slime.rollout.base_types import RolloutFnTrainOutput, finalize_rollout_groups, iter_samples, prepare_retry_group
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "CompletedSampleRecord",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


def _new_metrics_state() -> dict[str, int]:
    return {
        "lag_zero": 0,
        "lag_one": 0,
        "lag_two_to_three": 0,
        "lag_four_plus": 0,
        "lag_unknown": 0,
        "stale_rejected": 0,
        "stale_requeued": 0,
        "aborted_requeued": 0,
    }


@dataclass(frozen=True)
class CompletedSampleRecord:
    """A completed group and the immutable policy version captured at admission."""

    gid: int
    group: list[Sample]
    policy_version: int | None
    retry_group: list[Sample] | None = field(default=None, repr=False, compare=False)


class _RetryGroup(list):
    """Keep pristine inputs across partial retries and data-source checkpoints."""

    def __init__(self, samples, *, original_inputs, policy_version):
        super().__init__(samples)
        self.original_inputs = original_inputs
        self.policy_version = policy_version


def _source_policy_version(data_source, default=0):
    get_metadata = getattr(data_source, "get_metadata", None)
    metadata = get_metadata() if get_metadata is not None else getattr(data_source, "metadata", {})
    return metadata.get("fully_async_policy_version", getattr(data_source, "policy_version", default))


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
                args,
                data_buffer,
                concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args),
                policy_version=_source_policy_version(data_buffer, getattr(args, "policy_version", 0)),
            )
        if hasattr(data_buffer, "register_consumer"):
            data_buffer.register_consumer("fully_async", worker)
        else:
            data_buffer.consumers["fully_async"] = worker
        worker.publish_policy_version(_source_policy_version(data_buffer, getattr(args, "policy_version", 0)))
        if isinstance(worker, AsyncRolloutWorker):
            worker.start()
    return worker


def before_weight_update(*, data_source, policy_version: int) -> int:
    """Lifecycle hook called before training publishes a new actor policy."""

    policy_version = max(policy_version, _source_policy_version(data_source))
    logger.debug("fully-async: preparing to update policy version %d", policy_version)
    return policy_version


def after_weight_update(*, data_source, policy_version: int, succeeded: bool) -> None:
    """Publish ``policy_version`` after a successful actor weight update."""

    if not succeeded:
        logger.warning("fully-async: weight update failed; policy remains unpublished")
        return

    current = _source_policy_version(data_source)
    if policy_version < current:
        raise ValueError(f"policy version must be monotonic: current={current}, new={policy_version}")
    worker = getattr(data_source, "consumers", {}).get("fully_async")
    if worker is not None:
        worker.publish_policy_version(policy_version)
    # Publication can precede the first generate call. Keep it with this source,
    # so independent managers and closed/replaced sources cannot share a worker.
    data_source.policy_version = policy_version
    # Built-in source checkpoints already persist metadata, including manager
    # replacement and checkpoint rewind, before a consumer is reconstructed.
    metadata = {"fully_async_policy_version": policy_version}
    if update_metadata := getattr(data_source, "update_metadata", None):
        update_metadata(metadata)
    else:
        if not hasattr(data_source, "metadata"):
            data_source.metadata = {}
        data_source.metadata.update(metadata)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10, policy_version: int = 0):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = max(1, concurrency // getattr(args, "n_samples_per_prompt", 1))
        self.concurrency *= getattr(args, "fully_async_group_prefetch_factor", 1)
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
        self.output_queue: queue.Queue[CompletedSampleRecord] = queue.Queue()
        self.poll_interval = 0.01
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self._policy_version = policy_version
        self._policy_version_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics_state = _new_metrics_state()

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
            return {"ready": list(self.output_queue.queue), "policy_version": self.policy_version}

    def load_state_dict(self, state):
        if self.worker_thread is not None:
            raise RuntimeError("Restore worker state before starting generation")
        self.publish_policy_version(max(self.policy_version, state.get("policy_version", 0)))
        for result in state["ready"]:
            if isinstance(result, tuple):
                # Older source checkpoints contain (gid, group) without policy
                # provenance. Keep it unknown rather than assigning new weights.
                result = CompletedSampleRecord(*result, policy_version=None)
            self.output_queue.put(result)

    def close(self):
        self.running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=35)
            if self.worker_thread.is_alive():
                raise TimeoutError("Generation did not stop within 35 seconds")

    def get_completed_groups(
        self,
        limit: int | None = None,
        *,
        max_policy_version_lag: int | None = None,
    ) -> list[CompletedSampleRecord]:
        """Pop up to ``limit`` trainable completed groups.

        Callers that only need a fixed number of groups must pass ``limit`` —
        anything popped beyond it would otherwise have to be thrown away, and
        these groups are fully generated and reward-scored, with their prompts
        already consumed from ``data_buffer``. When a lag budget is configured,
        records beyond it are deterministically rejected and their admission-
        time prompt snapshots are requeued for regeneration.
        """
        if max_policy_version_lag is not None and max_policy_version_lag < 0:
            raise ValueError(f"max_policy_version_lag must be non-negative, got {max_policy_version_lag}")

        completed: list[CompletedSampleRecord] = []
        stale_count = 0
        current_policy_version = self.policy_version
        while limit is None or len(completed) < limit:
            try:
                record = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if record.policy_version is None:
                if max_policy_version_lag is not None:
                    raise RuntimeError("Cannot enforce a policy lag budget on a record with unknown policy version")
                completed.append(record)
                continue
            version_lag = current_policy_version - record.policy_version
            if version_lag < 0:
                raise ValueError(
                    "completed record is newer than the worker policy: "
                    f"current={current_policy_version}, record={record.policy_version}"
                )
            if max_policy_version_lag is not None and version_lag > max_policy_version_lag:
                if record.retry_group is None:
                    raise RuntimeError(f"stale completed record {record.gid} has no admission snapshot to requeue")
                self.data_buffer.add_samples([record.retry_group])
                stale_count += 1
                continue
            completed.append(record)
        if stale_count:
            with self._metrics_lock:
                self._metrics_state["stale_rejected"] += stale_count
                self._metrics_state["stale_requeued"] += stale_count
            logger.info(
                "fully-async: rejected and requeued %d stale group(s) at policy version %d (max lag=%d)",
                stale_count,
                current_policy_version,
                max_policy_version_lag,
            )
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def _record_version_lag_locked(self, version_lag: int | None) -> None:
        if version_lag is None or version_lag < 0:
            self._metrics_state["lag_unknown"] += 1
        elif version_lag == 0:
            self._metrics_state["lag_zero"] += 1
        elif version_lag == 1:
            self._metrics_state["lag_one"] += 1
        elif version_lag <= 3:
            self._metrics_state["lag_two_to_three"] += 1
        else:
            self._metrics_state["lag_four_plus"] += 1

    def record_processed_groups(self, groups: list[list[Sample]]) -> None:
        """Accumulate a fixed-bucket lag histogram for shipped samples."""

        current_policy_version = self.policy_version
        with self._metrics_lock:
            for group in groups:
                for sample in iter_samples(group):
                    sample_version = sample.policy_version
                    version_lag = current_policy_version - sample_version if isinstance(sample_version, int) else None
                    self._record_version_lag_locked(version_lag)

    def snapshot_metrics(self, *, reset: bool) -> dict[str, int]:
        """Return only bounded-cardinality keys and optionally start a new interval."""

        with self._metrics_lock:
            state = dict(self._metrics_state)
            if reset:
                self._metrics_state = _new_metrics_state()

        return {
            "fully_async/version_lag/count_0": state["lag_zero"],
            "fully_async/version_lag/count_1": state["lag_one"],
            "fully_async/version_lag/count_2_to_3": state["lag_two_to_three"],
            "fully_async/version_lag/count_4_plus": state["lag_four_plus"],
            "fully_async/version_lag/count_unknown": state["lag_unknown"],
            "fully_async/count/stale_rejected": state["stale_rejected"],
            "fully_async/count/stale_requeued": state["stale_requeued"],
            "fully_async/count/aborted_requeued": state["aborted_requeued"],
            "fully_async/queue/completed_groups": self.queue_size(),
            "fully_async/policy/current_version": self.policy_version,
        }

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
                        policy_version = self.policy_version
                        callback = self._make_done_cb(gid, group, policy_version)
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(callback)
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

    def _make_done_cb(self, gid: int, input_group: list[Sample], policy_version: int):
        # The lag bound covers the oldest admission contributing to this group.
        # An interrupted retry can keep completed siblings or partial tokens, so
        # preserve the first inputs until the entire group is accepted/rejected.
        admission_inputs = None
        group_policy_version = policy_version
        if getattr(self.args, "max_policy_version_lag", None) is not None:
            if isinstance(input_group, _RetryGroup):
                admission_inputs = input_group.original_inputs
                group_policy_version = min(policy_version, input_group.policy_version)
            else:
                admission_inputs = copy.deepcopy(input_group)
        # Interrupted segments cannot reopen their closed agent sessions.
        # Save their inputs before generation mutates them; completed siblings
        # will retain their results when the group is retried.
        retry_group = [sample if isinstance(sample, list) else copy.deepcopy(sample) for sample in input_group]
        # Retries may include already completed siblings. Their original policy
        # remains valid because the generator reuses them without new inference.
        retained_versions = {
            id(sample): sample.policy_version
            for sample in iter_samples(input_group)
            if sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) and sample.reward is not None
        }
        for sample in iter_samples(input_group):
            sample.policy_version = retained_versions.get(id(sample), policy_version)

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
            # Custom generators may replace the admitted Sample objects. Stamp
            # their outputs from the admission snapshot, never from mutable
            # worker state observed at completion time.
            for sample in iter_samples(result):
                sample.policy_version = retained_versions.get(id(sample), policy_version)
            # The early weight-update exit may return untouched PENDING inputs.
            if any(s.status in (Sample.Status.ABORTED, Sample.Status.PENDING) for s in iter_samples(result)):
                try:
                    retry = prepare_retry_group(retry_group, result)
                    if admission_inputs is not None:
                        retry = _RetryGroup(
                            retry, original_inputs=admission_inputs, policy_version=group_policy_version
                        )
                    self.data_buffer.add_samples([retry])
                    with self._metrics_lock:
                        self._metrics_state["aborted_requeued"] += 1
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put(
                CompletedSampleRecord(
                    gid=gid, group=result, policy_version=group_policy_version, retry_group=admission_inputs
                )
            )

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
    metric_gatherer = MetricGatherer()
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
        for record in worker.get_completed_groups(
            limit=target - len(collected), max_policy_version_lag=getattr(args, "max_policy_version_lag", None)
        ):
            group = record.group
            drained += 1
            verdict = call_dynamic_filter(dynamic_filter, args, group)
            metric_gatherer.on_group_completed(verdict)
            if verdict.keep:
                worker.record_processed_groups([group])
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
    metrics.update(metric_gatherer.collect())
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
    output = run(_generate_rollout_async(args, rollout_id, data_buffer))
    metrics = worker.snapshot_metrics(reset=True)
    if isinstance(output, RolloutFnTrainOutput):
        # Keep acceptance callbacks and per-sample transport references intact.
        return replace(output, metrics={**(output.metrics or {}), **metrics})
    return RolloutFnTrainOutput(samples=output, metrics=metrics)
