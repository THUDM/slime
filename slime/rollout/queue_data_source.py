"""Persistent prompt tasks and reader-local continuation buffers.

One job-owned Ray actor hosts the coordinator and a serialized dataset producer.
Readers load task inputs directly from the shared store; datasets and generated
Samples never travel through the coordinator RPC. Producer writes run outside
the coordinator lock, allowing existing workers to complete during refill.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from straw.coordinator import Coordinator
from straw.errors import LeaseExpired, StaleAttempt
from straw.protocol import Lease, Limits, Record, RecordSetRef, TaskSpec, encode

from slime.rollout.base_types import iter_samples
from slime.rollout.data_source import DataSource, RolloutDataSource, pop_first, pop_oldest
from slime.utils.misc import load_function
from slime.utils.rollout_transport import (
    DiskPayloadRef,
    group_lease,
    pack_rollout_payload,
    record_generation_provenance,
    release_rollout_publications,
    resolve_rollout_data_dir,
    rollout_store,
    unpack_rollout_payload,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class RolloutQueueController:
    """Plain class for unit testing; the host wraps it in one concurrent Ray actor."""

    def __init__(self, args, *, producer=None):
        self.args = args
        self.store, self.codec, self._writer_lock = rollout_store(args)
        self._gc_error = None
        self._queue = Coordinator(
            self.store,
            exclusive_owner="job owns this non-restarting actor; explicit resume confirms prior owner stopped",
            recover=getattr(args, "rollout_queue_resume", False),
            lease_seconds=getattr(args, "rollout_queue_lease_seconds", 300),
            limits=Limits(
                pending_tasks=getattr(args, "rollout_queue_max_pending", 65536),
                inflight_tasks=getattr(args, "rollout_queue_max_inflight", 65536),
            ),
        )
        self.producer = producer
        self._initial_cursor = None
        self._producer_lock = threading.Lock()
        self._reader_tokens = {}
        self._retired_readers = set()
        self.branch_id = uuid.uuid4().hex
        self._training_token = None
        self._training_lock = threading.RLock()
        self._gc_stop = threading.Event()
        self._gc_thread = None
        if getattr(args, "rollout_queue_resume", False):
            # This flag explicitly requires the entire prior job to have stopped;
            # unlike a lease timeout, that supervisor assertion ends old reads.
            readers = self.queue.outstanding_reads()
            for offset in range(0, len(readers), 128):
                self.queue.release_task_reads(readers[offset : offset + 128])
        # Joint restore can rewind the training cursor. Reconcile its durable
        # storage ownership before GC inspects restored live result roots.
        if not getattr(args, "_joint_resume", None):
            self._start_gc()
        # Recovery invalidates leases; unfinished collections from the stopped
        # manager cannot produce a new branch's batch. Accepted facts stay intact.
        for task_id, task in list(self.queue.tasks.items()):
            if task_id.startswith("collection:") and task["state"] == "pending":
                self.queue.cancel_task(task_id, request_id=f"abandon:{self.branch_id}:{task_id}")

    def _check_gc_error(self):
        if self._gc_error is not None:
            raise RuntimeError("Straw online GC failed; queue work has stopped") from self._gc_error

    def _start_gc(self):
        if getattr(self.args, "rollout_queue_online_gc", False) and self._gc_thread is None:
            self._gc_thread = threading.Thread(target=self._gc_loop, name="straw-gc", daemon=True)
            self._gc_thread.start()

    @property
    def queue(self):
        # Check every native queue access, including a call already in flight
        # that reaches its next queue operation after the background failure.
        self._check_gc_error()
        return self._queue

    def _collect_storage(self):
        result = self.queue.collect_garbage()
        print(f"straw_online_gc: {result}", flush=True)
        return result

    def _gc_loop(self):
        while not self._gc_stop.wait(60):
            try:
                self._collect_storage()
            except Exception as error:
                self._gc_error = error
                self._gc_stop.set()
                logger.exception("straw online GC failed; retained storage must be inspected")
                return

    def _source(self):
        self._check_gc_error()
        if self.producer is None:
            self.producer = RolloutDataSource(self.args)
        if self.producer.dataset is not None and not len(self.producer.dataset):
            raise ValueError("Queue rollout dataset is empty after filtering")
        if self._initial_cursor is None:
            self._initial_cursor = {
                key: copy.deepcopy(getattr(self.producer, key))
                for key in (
                    "sample_offset",
                    "epoch_id",
                    "sample_group_index",
                    "sample_index",
                    "metadata",
                )
            }
        return self.producer

    def configuration(self):
        with self._producer_lock:
            source = self._source()
            return {
                "dataset_size": len(source),
                "n_samples_per_prompt": self.args.n_samples_per_prompt,
                "seed": self.args.rollout_seed,
                "shuffle": self.args.rollout_shuffle,
                "run_id": self.store.run_id,
                "branch_id": self.branch_id,
            }

    def identity(self):
        self._check_gc_error()
        return {"run_id": self.store.run_id, "branch_id": self.branch_id}

    def take(self, reader_id, count):
        if reader_id in self._retired_readers:
            raise StaleAttempt("Reader was retired by the scheduler")
        requested = count
        page = self.queue.acquire(reader_id, count, task_prefix="prompt:")
        if page.status != "empty":
            return page
        with self._producer_lock:
            # Another reader may have refilled while this one waited.
            page = self.queue.acquire(reader_id, count, task_prefix="prompt:")
            if page.status != "empty":
                return page
            source = self._source()
            state = self.queue.producer_state("dataset")
            for key, value in (state["cursor"] if state else self._initial_cursor).items():
                setattr(source, key, copy.deepcopy(value))
            if source.dataset is not None and self.args.rollout_shuffle:
                source.dataset.shuffle(source.epoch_id)
            start = source.sample_group_index
            # Bounded refill, without unused index reservations held in readers.
            count = min(max(count, 8), 64, self.queue.limits.pending_tasks)
            groups = source.get_samples(count)
            group_ids = [next(iter_samples(group)).group_index for group in groups]
            task_ids = [f"prompt:{group_id}" for group_id in group_ids]
            with self._writer_lock:
                refs = self.codec.publish_many(
                    groups,
                    submission_ids=[f"input:{task_id}" for task_id in task_ids],
                    metadata=[{"task_id": task_id} for task_id in task_ids],
                )
            tasks = [
                TaskSpec(
                    task_id,
                    ref,
                    metadata={"group_index": group_id},
                    estimated_tokens=ref.tokens,
                )
                for task_id, group_id, ref in zip(task_ids, group_ids, refs, strict=True)
            ]
            cursor = {
                key: copy.deepcopy(getattr(source, key))
                for key in (
                    "sample_offset",
                    "epoch_id",
                    "sample_group_index",
                    "sample_index",
                    "metadata",
                )
            }
            self.queue.submit_tasks(
                f"dataset:{start}",
                tasks,
                producer_id="dataset",
                producer_state={"version": 1, "cursor": cursor},
            )
        return self.queue.acquire(reader_id, requested, task_prefix="prompt:")

    def heartbeat(self, leases):
        return self.queue.heartbeat(leases)

    def complete(self, lease, result_ref):
        if lease.worker_id in self._retired_readers and self.result(lease) is None:
            raise StaleAttempt("Reader was retired by the scheduler")
        return self.queue.complete_task(
            lease,
            submission_id=f"result:{lease.attempt_id}",
            result_ref=result_ref,
            result_digest=result_ref.digest,
        )

    def reject(self, lease, reason):
        return self.queue.fail_task(
            lease,
            request_id=f"reject:{lease.attempt_id}",
            failure={"category": "Filtered", "reason": reason},
            retryable=False,
        )

    def result(self, lease):
        return self.queue.lookup_submission(f"result:{lease.attempt_id}")

    def results(self, leases):
        return [self.result(lease) if lease else None for lease in leases]

    def progress(self, lease, input_ref):
        return self.queue.save_task_progress(
            lease,
            request_id=f"progress:{lease.attempt_id}:{input_ref.digest}",
            input_ref=input_ref,
        )

    def progress_many(self, updates):
        from straw.protocol import digest

        return self.queue.save_task_progress_many(updates, request_id="progress:" + digest(updates))

    def release(self, leases):
        from straw.protocol import digest

        for offset in range(0, len(leases), 128):
            batch = leases[offset : offset + 128]
            self.queue.release_tasks(
                batch,
                request_id="release:" + digest([asdict(lease) for lease in batch]),
            )

    def recover_reader_results(self, reader_id, delivered_ref):
        """Fence a stopped reader and recover accepted results whose reply was lost."""
        delivered = set(self.codec.load(delivered_ref))
        with self._producer_lock:
            self._retired_readers.add(reader_id)
            accepted = self.queue.retire_worker(reader_id)
            state = self._training_state()
            processed = set(state["processed_positions"])
            receipts = [
                receipt
                for receipt in accepted
                if receipt["position"] >= state["processed_cursor"]
                and receipt["position"] not in delivered | processed
            ]
            with self._writer_lock:
                return self.codec.publish(receipts, submission_id=f"recover-reader:{uuid.uuid4().hex}")

    def recover_pending_rollout(self):
        """Replay accepted groups after a whole-job failure before the first batch.

        Once conversion/training has started, queue progress alone cannot identify
        the model/optimizer version. That recovery requires a joint checkpoint.
        --rollout-queue-resume asserts that all prior readers have stopped.
        """
        if not getattr(self.args, "rollout_queue_resume", False):
            raise RuntimeError("Whole-job recovery requires --rollout-queue-resume")
        with self._producer_lock, self._training_lock:
            state = self._training_state()
            if self.queue.batches or state["processed_cursor"] or state["processed_positions"]:
                raise RuntimeError(
                    "Queue-only recovery is limited to an interrupted first rollout; "
                    "restore matching model/optimizer and rollout checkpoints after batch conversion begins"
                )
            receipts = [receipt for receipt in self.queue.commits if receipt["task_id"].startswith("prompt:")]
            with self._writer_lock:
                return self.codec.publish(receipts, submission_id=f"recover-job:{self.branch_id}")

    def begin_collection(self, request_id):
        task_id = f"collection:{self.branch_id}:{request_id}"
        self.queue.submit_tasks(
            task_id,
            [TaskSpec(task_id, allow_empty=True, control=True, estimated_records=0)],
        )
        task = self.queue.task_status(task_id)
        if task["state"] in {"leased", "completed"} and task["lease"]["worker_id"] == "manager":
            return Lease(**task["lease"])
        page = self.queue.acquire("manager", 1, task_ids=[task_id], control=True)
        if not page.assignments:
            raise RuntimeError(f"Collection task is not available: {task_id} ({page.status})")
        return page.assignments[0].lease

    def accepted(self, receipt):
        actual = self.queue.lookup_submission(receipt.submission_id)
        if actual != receipt:
            raise ValueError("Raw rollout reference is not in this run's accepted log")
        self.store.validate(receipt.result_ref)
        return actual

    def batch(self, batch_id):
        return self.queue.get_batch(batch_id)

    def plan_batch(self, batch_id, positions, plan_ref):
        with self._training_lock:
            if self._training_token is None:
                self._training_token = self.queue.open_consumer(
                    "training", exclusive_owner="job owns one BatchBuilder"
                )
            return self.queue.plan_batch(
                "training",
                token=self._training_token,
                batch_id=batch_id,
                input_positions=positions,
                plan_ref=plan_ref,
            )

    def _training_state(self):
        previous = self.queue.load_consumer_state("training")
        return (
            self.codec.load(RecordSetRef.from_dict(previous["state_ref"]))
            if previous
            else {
                "version": 1,
                "processed_cursor": 0,
                "fetch_cursor": 0,
                "processed_positions": [],
                "batches": [],
                "finished_batches": [],
            }
        )

    @staticmethod
    def _advance(state, positions):
        positions = set(state["processed_positions"]) | set(positions)
        cursor = state["processed_cursor"]
        end = max(positions, default=-1) + 1
        while cursor in positions:
            positions.remove(cursor)
            cursor += 1
        state.update(
            processed_cursor=cursor,
            fetch_cursor=max(state["fetch_cursor"], end),
            processed_positions=sorted(position for position in positions if position >= cursor),
        )

    def _save_training_state(self, state, *, batch_id=None, ready_ref=None):
        self._check_gc_error()
        if self._training_token is None:
            self._training_token = self.queue.open_consumer("training", exclusive_owner="job owns one BatchBuilder")
        with self._writer_lock:
            state_ref = self.codec.publish(state, submission_id=f"training-state:{uuid.uuid4().hex}")
            progress_ref = self.store.publish(
                [
                    Record(
                        "progress",
                        encode(
                            {
                                "version": 1,
                                "processed_positions": state["processed_positions"],
                                "finished_batches": state.get("finished_batches", []),
                            }
                        ),
                        codec="json.v1",
                    )
                ],
                submission_id=f"progress:{state_ref.digest}",
                dependencies=[state_ref],
            )
        kwargs = dict(
            token=self._training_token,
            state_ref=state_ref,
            progress_ref=progress_ref,
            fetch_cursor=state["fetch_cursor"],
            processed_cursor=state["processed_cursor"],
        )
        if batch_id is not None:
            self.queue.batch_ready("training", batch_id=batch_id, ready_ref=ready_ref, **kwargs)
        else:
            self.queue.save_consumer_state(
                "training",
                request_id=f"state:{self.branch_id}:{state_ref.digest}",
                **kwargs,
            )
        return state_ref

    def ready_batch(self, batch_id, ready_ref):
        with self._training_lock:
            batch = self.queue.get_batch(batch_id)
            if batch["ready"]:
                if batch["ready_ref"] != asdict(ready_ref):
                    raise ValueError("Batch was already published with another ready reference")
                return RecordSetRef.from_dict(self.training_state()["state_ref"])
            state = self._training_state()
            self._advance(state, batch["input_positions"])
            state["batches"].append(DiskPayloadRef(ready_ref, str(self.store.backend.root)))
            return self._save_training_state(state, batch_id=batch_id, ready_ref=ready_ref)

    def finish_batch(self, batch_id):
        """Acknowledge completed reads on every training rank; checkpoints retain their own roots."""
        with self._training_lock:
            batch = self.queue.get_batch(batch_id)
            if not batch or not batch["ready"]:
                raise ValueError("Cannot finish an unknown or unready training batch")
            state = self._training_state()
            finished = state.setdefault("finished_batches", [])
            if batch_id not in finished:
                finished.append(batch_id)
                if getattr(self.args, "rollout_queue_online_gc", False):
                    state["batches"] = [ref for ref in state["batches"] if asdict(ref.manifest) != batch["ready_ref"]]
                self._save_training_state(state)
            if getattr(self.args, "rollout_queue_online_gc", False):
                self._collect_storage()

    def record_dispositions(self, ref):
        """Journal why accepted outputs will not be selected by this training view."""
        with self._training_lock:
            decision = self.codec.load(ref)
            positions = decision["positions"]
            if not decision.get("reason") or any(not 0 <= p < len(self.queue.commits) for p in positions):
                raise ValueError("Invalid accepted-output disposition")
            state = self._training_state()
            self._advance(state, positions)
            state["disposition"] = DiskPayloadRef(ref, str(self.store.backend.root))
            self._save_training_state(state)

    def training_state(self):
        return self.queue.load_consumer_state("training")

    def handoff_restored_source(self, source_ref):
        """Move restore ownership after all current buffers have been snapshotted.

        The caller retains the complete replacement source before this call,
        with its readers paused. Old checkpoint owners remain independent until
        a successor joint commit allows their normal retirement.
        """
        with self._training_lock:
            state = self._training_state()
            if "restored_source" in state:
                state["restored_source"] = DiskPayloadRef(source_ref, str(self.store.backend.root))
                self._save_training_state(state)

    def restore_training_state(self, state_ref, fetch_cursor, processed_cursor, retained_ref=None):
        with self._training_lock:
            state = self.codec.load(state_ref)
            if (state["fetch_cursor"], state["processed_cursor"]) != (
                fetch_cursor,
                processed_cursor,
            ):
                raise ValueError("Consumer checkpoint cursors differ from its state")
            if retained_ref is not None:
                from slime.utils.rollout_transport import RolloutGroupRef

                retained, visited = set(), set()

                def visit(value):
                    if isinstance(value, RolloutGroupRef) and value.receipt:
                        retained.add(value.receipt.position)
                    elif isinstance(value, DiskPayloadRef):
                        if value.manifest not in visited:
                            visited.add(value.manifest)
                            visit(self.codec.load(value.manifest))
                    elif isinstance(value, Sample):
                        if receipt := getattr(value, "_queue_receipt", None):
                            retained.add(receipt["position"])
                        elif lease := getattr(value, "_queue_lease", None):
                            receipt = self.result(Lease(**lease))
                            if receipt is None:
                                task = self.queue.task_status(lease["task_id"])
                                if task["state"] == "completed":
                                    receipt = self.result(Lease(**task["lease"]))
                            if receipt is not None:
                                retained.add(receipt.position)
                    elif isinstance(value, dict):
                        for item in value.values():
                            visit(item)
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            visit(item)

                visit(DiskPayloadRef(retained_ref, str(self.store.backend.root)))
                processed = set(state["processed_positions"])
                abandoned = [
                    commit["position"]
                    for commit in self.queue.commits[processed_cursor:]
                    if commit["position"] not in retained | processed
                ]
                old_batches = {ref.load()["batch_id"] for ref in state["batches"]}
                abandoned_batches = [
                    batch_id
                    for batch_id, batch in self.queue.batches.items()
                    if batch["ready"] and batch_id not in old_batches
                ]
                with self._writer_lock:
                    decision = self.codec.publish(
                        {
                            "reason": "checkpoint branch excludes outputs absent from its warm buffers",
                            "positions": abandoned,
                            "batch_ids": abandoned_batches,
                            "checkpoint_state": asdict(state_ref),
                            "retained_source": asdict(retained_ref),
                            "branch_id": self.branch_id,
                        },
                        submission_id=f"restore-decision:{uuid.uuid4().hex}",
                    )
                self._advance(state, abandoned)
                state["finished_batches"] = sorted(set(state.get("finished_batches", [])) | set(abandoned_batches))
                state["disposition"] = DiskPayloadRef(decision, str(self.store.backend.root))
                # Filter dispositions may replace the audit record, but cannot
                # release old reader prefixes before a complete snapshot exists.
                state["restored_source"] = DiskPayloadRef(retained_ref, str(self.store.backend.root))
            restored = self._save_training_state(state)
            self._start_gc()
            return restored

    def reader_state(self, reader_id, ref=None):
        consumer_id = f"reader:{reader_id}"
        if ref is not None:
            if reader_id not in self._reader_tokens:
                self._reader_tokens[reader_id] = self.queue.open_consumer(
                    consumer_id, exclusive_owner="registered reader"
                )
            self.queue.save_consumer_state(
                consumer_id,
                token=self._reader_tokens[reader_id],
                request_id=ref.digest,
                state_ref=ref,
                fetch_cursor=0,
                processed_cursor=0,
            )
        return self.queue.load_consumer_state(consumer_id)

    def status(self, task_id):
        return self.queue.task_status(task_id)

    def restore_lease(self, reader_id, task_id):
        return self.queue.acquire(reader_id, 1, task_ids=[task_id])

    def metrics(self):
        return self.queue.metrics()

    def close(self):
        self._gc_stop.set()
        if self._gc_thread is not None:
            self._gc_thread.join()
        try:
            self._queue.close()
        finally:
            with self._writer_lock:
                self.store.seal()
        self._check_gc_error()


def create_queue_controller(args):
    resolve_rollout_data_dir(args)
    if getattr(args, "_rollout_queue_controller", None) is None:
        args._rollout_queue_controller = (
            ray.remote(RolloutQueueController)
            .options(
                num_cpus=0,
                max_concurrency=4,
                max_restarts=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    ray.get_runtime_context().get_node_id(), soft=False
                ),
            )
            .remote(copy.copy(args))
        )
        args._rollout_queue_branch = ray.get(args._rollout_queue_controller.identity.remote())["branch_id"]
    return args._rollout_queue_controller


@dataclass(frozen=True)
class QueueReaderConfig:
    args: object
    controller: object
    reader_id: str
    dataset_size: int

    def open(self, args=None):
        return QueueReader(
            self.args if args is None else args,
            self.controller,
            self.reader_id,
            self.dataset_size,
        )


class QueueReader(DataSource):
    def __init__(self, args, controller, reader_id, dataset_size):
        self.args, self.controller, self.reader_id = args, controller, reader_id
        self.args._rollout_queue_controller = controller
        self.dataset_size = dataset_size
        self.buffer, self.metadata = [], {}
        self._leases, self._leased_inputs = {}, {}
        self._closed = False
        self._lock = threading.RLock()
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="queue-input")
        self._stop = threading.Event()
        self._heartbeat_error = None
        self.buffer_filter = (
            load_function(args.buffer_filter_path)
            if args.buffer_filter_path
            else (pop_oldest if getattr(args, "buffer_sort_by_staleness", False) else pop_first)
        )
        self._heartbeats = threading.Thread(target=self._heartbeat_loop, name="queue-leases", daemon=True)
        self._heartbeats.start()

    def _heartbeat_loop(self):
        interval = max(0.1, getattr(self.args, "rollout_queue_lease_seconds", 300) / 3)
        while not self._stop.wait(interval):
            with self._lock:
                leases = list(self._leases.values())
            try:
                for offset in range(0, len(leases), 128):
                    batch = leases[offset : offset + 128]
                    outcomes = ray.get(self.controller.heartbeat.remote(batch))
                    with self._lock:
                        for lease, outcome in zip(batch, outcomes, strict=True):
                            if self._leases.get(lease.task_id) != lease:
                                continue  # A released attempt may already have been reacquired.
                            if outcome == "StaleAttempt":
                                self._leases.pop(lease.task_id, None)
                                self._leased_inputs.pop(lease.task_id, None)
                            elif outcome != "extended":
                                raise LeaseExpired(f"Reader lost lease for {lease.task_id}: {outcome}")
            except Exception as error:
                self._heartbeat_error = error
                return

    def get_samples(self, num_samples):
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        if self._closed:
            raise RuntimeError("Queue reader is closed")
        if self._heartbeat_error:
            raise RuntimeError("Queue lease heartbeat failed") from self._heartbeat_error
        with self._lock:
            samples = (
                self.buffer_filter(self.args, None, self.buffer, num_samples) if self.buffer and num_samples else []
            )
        leases = [group_lease(group) for group in samples]
        receipts = ray.get(self.controller.results.remote(leases)) if samples else []
        for group, receipt in zip(samples, receipts, strict=True):
            if receipt:
                for sample in iter_samples(group):
                    del sample._queue_lease
                    sample._queue_receipt = asdict(receipt)
        while len(samples) < num_samples:
            if self._closed:
                raise RuntimeError("Queue reader is closed")
            page = ray.get(self.controller.take.remote(self.reader_id, min(num_samples - len(samples), 64)))
            if page.status in {"end_of_input", "draining"}:
                break
            if not page.assignments:
                if page.status in {"backpressured", "empty"}:
                    if self._heartbeat_error:
                        raise RuntimeError("Queue lease heartbeat failed") from self._heartbeat_error
                    time.sleep(0.05)
                    continue
                raise RuntimeError(f"Queue cannot supply rollout inputs: {page.status}")
            store, codec, _ = rollout_store(self.args)
            with store.read_session() as reader:
                for assignment in page.assignments:
                    group = codec.load(assignment.task.input_ref, reader=reader)
                    for sample in iter_samples(group):
                        sample._queue_lease = asdict(assignment.lease)
                        sample._queue_generation_start = len(sample.tokens)
                    with self._lock:
                        self._leases[assignment.lease.task_id] = assignment.lease
                        self._leased_inputs[assignment.lease.task_id] = assignment.task.input_ref
                    samples.append(group)
        return samples

    async def get_samples_async(self, num_samples):
        return await asyncio.get_running_loop().run_in_executor(self._reader, self.get_samples, num_samples)

    def add_samples(self, groups):
        if len(groups) > 64:
            for offset in range(0, len(groups), 64):
                self.add_samples(groups[offset : offset + 64])
            return
        if not groups:
            return
        valid, retry, incomplete = [], [], []
        for group in groups:
            missing = []
            for sample in iter_samples(group):
                if not sample.response_length:
                    continue
                if getattr(self.args, "use_rollout_routing_replay", False):
                    rows = sample.get_rollout_routed_experts_length()
                    if rows != len(sample.tokens) - 1:
                        missing.append((sample.index, "routes", rows, len(sample.tokens) - 1))
                if getattr(self.args, "use_score_centering", False) and getattr(self.args, "rollout_top_p", 1) >= 1:
                    for name in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
                        value = getattr(sample, name)
                        rows = len(value) if value is not None else 0
                        if rows != sample.response_length:
                            missing.append((sample.index, name, rows, sample.response_length))
            if not missing:
                valid.append(group)
                continue
            lease = group_lease(group)
            if lease is None:
                raise ValueError(f"Incomplete continuation has no durable task input to retry: {missing}")
            release_rollout_publications(group, self.args)
            retry.append(lease)
            incomplete.extend(missing)
        if retry:
            # A server abort can return tokens without matching route/sampler
            # metadata. Do not replace the last complete durable input with it.
            with self._lock:
                for lease in retry:
                    if self._leases.get(lease.task_id) == lease:
                        self._leases.pop(lease.task_id)
                        self._leased_inputs.pop(lease.task_id, None)
            ray.get(self.controller.release.remote(retry))
            logger.warning(
                "Requeued %d incomplete continuations from their last durable inputs; (sample, field, actual, expected): %s",
                len(retry),
                incomplete[:8],
            )
        groups = valid
        if not groups:
            return
        leases = [group_lease(group) for group in groups]
        receipts = ray.get(self.controller.results.remote(leases))
        for i, (group, receipt) in enumerate(zip(groups, receipts, strict=True)):
            record_generation_provenance(group, self.args)
            if receipt:
                for sample in iter_samples(group):
                    del sample._queue_lease
                    sample._queue_receipt = asdict(receipt)
                leases[i] = None
        store, codec, writer_lock = rollout_store(self.args)
        with writer_lock:
            refs = codec.publish_many(
                groups,
                submission_ids=[f"continuation:{uuid.uuid4().hex}" for _ in groups],
            )
        updates = [
            {"lease": asdict(lease), "input_ref": asdict(ref)}
            for lease, ref in zip(leases, refs, strict=True)
            if lease
        ]
        if updates:
            ray.get(self.controller.progress_many.remote(updates))
        with self._lock, store.read_session() as reader:
            for group, lease, ref in zip(groups, leases, refs, strict=True):
                if lease:
                    self._leased_inputs[lease.task_id] = ref
                durable = codec.load(ref, reader=reader)
                for original, saved in zip(iter_samples(group), iter_samples(durable), strict=True):
                    original.__dict__.clear()
                    original.__dict__.update(vars(saved))
                self.buffer.append(group)
        # Continuations are already durable. Snapshot the complete buffer only
        # at checkpoints/close, avoiding quadratic writes during mass aborts.

    def _save_buffer(self):
        state = self.state_dict()
        ray.get(self.controller.reader_state.remote(self.reader_id, state.manifest))

    def state_dict(self):
        with self._lock:
            buffered = []
            for group in self.buffer:
                lease = group_lease(group)
                ref = self._leased_inputs.get(lease.task_id) if lease else None
                buffered.append((group, lease, ref))
            value = {
                "version": 2,
                "buffer": [],
                "metadata": dict(self.metadata),
            }
        for group, lease, ref in buffered:
            # add_samples already persisted the complete continuation. Reusing
            # that publication avoids encoding every token again on close.
            # Lease renewal/recovery can change authorization without changing
            # the immutable input, so snapshot the current lease separately.
            value["buffer"].append(
                {
                    "group": (
                        DiskPayloadRef(ref, self.args.rollout_data_dir)
                        if ref is not None
                        else pack_rollout_payload(group, self.args, -1)
                    ),
                    "lease": asdict(lease) if lease else None,
                }
            )
        return pack_rollout_payload(value, self.args, -1)

    def load_state_dict(self, state):
        value = unpack_rollout_payload(state)
        restored = []
        for buffered in value["buffer"]:
            ref = None
            if value["version"] == 2:
                ref = buffered["group"].manifest
                group = unpack_rollout_payload(buffered["group"])
                if buffered["lease"] is not None:
                    for sample in iter_samples(group):
                        sample._queue_lease = dict(buffered["lease"])
            else:
                group = buffered
            lease = group_lease(group)
            if lease:
                status = ray.get(self.controller.status.remote(lease.task_id))
                if status["state"] == "completed":
                    # An older buffer can precede completion of this task. Its
                    # saved prefix is replay input for the new collection, while
                    # the later global result remains an immutable production fact.
                    receipt = ray.get(self.controller.result.remote(Lease(**status["lease"])))
                    for sample in iter_samples(group):
                        del sample._queue_lease
                        sample._queue_receipt = asdict(receipt)
                        sample._queue_resume_origin = asdict(lease)
                        sample._queue_generation_start = len(sample.tokens)
                elif status["state"] == "leased" and status["lease"] == asdict(lease):
                    self._leases[lease.task_id] = lease
                    if ref is not None:
                        self._leased_inputs[lease.task_id] = ref
                    else:
                        self._leased_inputs.pop(lease.task_id, None)
                else:
                    page = ray.get(self.controller.restore_lease.remote(self.reader_id, lease.task_id))
                    if not page.assignments:
                        continue
                    assignment = page.assignments[0]
                    group = DiskPayloadRef(assignment.task.input_ref, self.args.rollout_data_dir).load()
                    lease = assignment.lease
                    for sample in iter_samples(group):
                        sample._queue_lease = asdict(lease)
                    self._leases[lease.task_id] = lease
                    self._leased_inputs[lease.task_id] = assignment.task.input_ref
            restored.append(group)
        with self._lock:
            self.buffer, self.metadata = restored, dict(value["metadata"])

    def materialize_samples(self, samples, *, release_files=True):
        from slime.utils.tensor_store import TensorRef

        if isinstance(samples, list):
            return [self.materialize_samples(child) for child in samples]
        sample = copy.copy(samples)
        for key, value in vars(sample).items():
            if isinstance(value, TensorRef):
                setattr(sample, key, value.load())
        return sample

    def get_buffer_length(self):
        return len(self.buffer)

    def update_metadata(self, metadata):
        self.metadata.update(metadata)

    def get_metadata(self):
        return self.metadata

    def __len__(self):
        return self.dataset_size

    def save(self, rollout_id):
        raise RuntimeError("Save queue readers through their owning data source")

    def load(self, rollout_id=None):
        raise RuntimeError("Restore queue readers through their owning data source")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._reader.shutdown(wait=True)
        self._save_buffer()
        with self._lock:
            leases = list(self._leases.values())
        ray.get(self.controller.release.remote(leases))
        self._stop.set()
        self._heartbeats.join(timeout=5)


class QueueDataSourceWithBuffer(QueueReader):
    def __init__(self, args):
        self._owns_controller = getattr(args, "_rollout_queue_controller", None) is None
        controller = create_queue_controller(args)
        self.data_config = ray.get(controller.configuration.remote())
        super().__init__(args, controller, "owner", self.data_config["dataset_size"])
        self.consumers, self._restored_consumers = {}, {}

    def reader_config(self, reader_id, *, prefetch_size=128):
        if reader_id == "owner":
            raise ValueError("Reader ID 'owner' is reserved")
        return QueueReaderConfig(self.args, self.controller, reader_id, len(self))

    def register_consumer(self, name, consumer):
        if name in self.consumers:
            raise ValueError(f"Consumer {name!r} already registered")
        if name in self._restored_consumers:
            consumer.load_state_dict(self._restored_consumers.pop(name))
        self.consumers[name] = consumer

    def save(self, rollout_id):
        from straw.reporting import write_report

        paused = []
        try:
            for consumer in self.consumers.values():
                paused.append((consumer, consumer.pause()))
            state = {
                "version": 1,
                "configuration": self.data_config,
                "reader": self.state_dict(),
                "consumers": {
                    **self._restored_consumers,
                    **{key: value.state_dict() for key, value in self.consumers.items()},
                },
            }
            ref = pack_rollout_payload(state, self.args, rollout_id)
            path = Path(self.args.save) / "rollout" / f"queue_state_{rollout_id}.json"
            store, _, lock = rollout_store(self.args)
            with lock:
                from straw.protocol import digest

                storage_owner = f"checkpoint:{path.resolve()}:{digest(asdict(ref.manifest))}"
                store.retain(storage_owner, [ref.manifest])
                store.release_publications([ref.manifest])
            write_report(
                path,
                {
                    "version": 1,
                    "root": ref.root,
                    "manifest": asdict(ref.manifest),
                    "storage_owner": storage_owner,
                },
            )
            if getattr(self.args, "_queue_restored_source_ref", None) is not None:
                ray.get(self.controller.handoff_restored_source.remote(ref.manifest))
        finally:
            for consumer, was_paused in paused:
                if not was_paused:
                    consumer.resume()

    def load(self, rollout_id=None):
        import json

        from straw.protocol import RecordSetRef

        if not self.args.load:
            return
        if self.consumers:
            raise RuntimeError("Restore before starting rollout consumers")
        path = Path(self.args.load) / "rollout" / f"queue_state_{rollout_id}.json"
        if not path.exists():
            return
        value = json.loads(path.read_text())
        state = DiskPayloadRef(RecordSetRef.from_dict(value["manifest"]), value["root"]).load()
        for key in (
            "dataset_size",
            "n_samples_per_prompt",
            "seed",
            "shuffle",
            "run_id",
        ):
            if state["configuration"][key] != self.data_config[key]:
                raise ValueError(f"Data source checkpoint differs in {key}")
        self.load_state_dict(state["reader"])
        self._restored_consumers = state["consumers"]
        self.args._queue_restored_source_ref = RecordSetRef.from_dict(value["manifest"])

    def close(self):
        if self._closed:
            return
        for consumer in self.consumers.values():
            consumer.close()
        super().close()
        if self._owns_controller:
            ray.get(self.controller.close.remote())
            ray.kill(self.controller, no_restart=True)
            self.args._rollout_queue_controller = None
