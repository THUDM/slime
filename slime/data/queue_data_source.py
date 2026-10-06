"""Persistent prompt, continuation and delivery tasks shared by all readers.

One job-owned Ray actor hosts the coordinator and a serialized dataset producer.
Readers load task inputs directly from the shared store; datasets and generated
Samples never travel through the coordinator RPC. Producer writes run outside
the coordinator lock, allowing existing workers to complete during refill.
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from straw.coordinator import Coordinator
from straw.errors import LeaseExpired, StaleAttempt
from straw.protocol import Lease, Record, RecordSetRef, TaskSpec, encode
from straw.reporting import write_report

from slime.data.checkpoint import RestorePlan, SourceRestore
from slime.data.data_source import DataSource, RolloutDataSource
from slime.data.transport import (
    DiskPayloadRef,
    group_lease,
    pack_rollout_payload,
    record_generation_provenance,
    release_rollout_publications,
    resolve_rollout_data_dir,
    rollout_store,
    unpack_rollout_payload,
)
from slime.rollout.base_types import iter_samples
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class RolloutQueueController:
    """Own a job's durable queue and checkpoint branch in one Ray actor.

    The plain class also supports local tests without starting Ray.
    """

    def __init__(self, args, *, producer=None, restore_plan=None, defer_gc=False):
        self.args = args
        self.restore_plan = restore_plan or RestorePlan()
        if (getattr(args, "load_debug_rollout_data", None) or "").endswith(".straw.json"):
            # Replay shares immutable storage, but never the source queue's
            # leases, accepted positions or training cursor.
            self.restore_plan = RestorePlan(queue_id=f"debug:{uuid.uuid4().hex}")
        self._checkpoint_lock = None
        with ExitStack() as cleanup:
            # The controller owns both the queue and its checkpoint branch.
            # Hold the save-directory lock until close() (or actor exit), and
            # publish the branch before any queue writes can become durable.
            if self.restore_plan.root is not None:
                plan = self.restore_plan
                root = Path(plan.root)
                (root / "rollout").mkdir(parents=True, exist_ok=True)
                self._checkpoint_lock = cleanup.enter_context((root / "rollout/session.lock").open("a"))
                try:
                    fcntl.flock(self._checkpoint_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f"Another training job owns save directory {root}") from error
                path = root / "rollout/current.json"
                current = path.read_text() if path.exists() else None
                if current != plan.expected_current:
                    raise RuntimeError("Current checkpoint branch changed during startup; restart to resolve it again")
                resolve_rollout_data_dir(args)
                branch = {**plan.branch, "pool": args.rollout_data_dir}
                write_report(Path(args.save) / "rollout/branch.json", branch)
                tracker = root / "latest_checkpointed_iteration.txt"
                if parent := branch.get("parent"):
                    write_report(tracker, parent["step"])
                write_report(
                    path,
                    {
                        "version": 1,
                        "directory": args.save,
                        "tracker": tracker.read_text() if tracker.exists() else None,
                    },
                )
                logger.warning(
                    "straw checkpoint directory: %s; logical save directory: %s; restoring: %s step %s",
                    args.save,
                    root,
                    args.load,
                    getattr(args, "ckpt_step", None),
                )
            self.store, self.codec, self._writer_lock = rollout_store(args)
            self._gc_error = None
            if not hasattr(Coordinator, "yield_tasks"):
                raise RuntimeError(
                    "straw transport requires yield_tasks and priority scheduling. "
                    "Upgrade on every rollout/training node: pip install --upgrade straw-queue"
                )
            fork = self.restore_plan.mode == "snapshot"
            queue_options = {}
            queue_id = self.restore_plan.queue_id
            if queue_id is not None:
                queue_options = {"queue_id": queue_id, "namespace": True}
            if fork:
                from straw.protocol import digest

                if not getattr(args, "save", None) or not getattr(args, "load", None):
                    raise ValueError("Checkpoint restore requires resolved load and save paths")
                if Path(args.save).resolve() == Path(args.load).resolve():
                    raise ValueError("Checkpoint restore requires an isolated output branch")
                if (Path(args.save) / "rollout" / "fork-active.json").exists():
                    raise ValueError("This queue branch has already started; resolve checkpoint selection again")
                queue_id = queue_id or "fork:" + digest(str(Path(args.save).resolve()))
                queue_options = {"queue_id": queue_id, "namespace": True}
                recovering = (self.store.backend.root / "queues" / digest(queue_id) / "run.json").exists()
            else:
                recovering = self.restore_plan.mode == "resume"
                if recovering and queue_id is not None and self.restore_plan.root:
                    from straw.protocol import digest

                    # Initialization may have stopped after publishing the branch
                    # but before creating its queue. Only this namespace may start empty.
                    recovering = (self.store.backend.root / "queues" / digest(queue_id) / "run.json").exists()
                if recovering and queue_id is None and getattr(args, "load", None):
                    branch = Path(args.load) / "rollout" / "fork-active.json"
                    if branch.exists():
                        queue_options = {"queue_id": json.loads(branch.read_text())["queue_id"], "namespace": True}
            self._queue = Coordinator(
                self.store,
                **queue_options,
                exclusive_owner="job owns this non-restarting actor; prior coordinator and readers must be stopped",
                recover=recovering,
                lease_seconds=getattr(args, "rollout_queue_lease_seconds", 300),
            )
            cleanup.callback(self._queue.close)
            self.producer = producer
            self._initial_cursor = None
            self._producer_lock = threading.Lock()
            self._reader_tokens = {}
            self._retired_readers = set()
            self._active_readers = set()
            self.branch_id = uuid.uuid4().hex
            self._training_token = None
            self._training_lock = threading.RLock()
            self._gc_stop = threading.Event()
            self._gc_thread = None
            self._fork_ready = not fork
            self._fork_active = False
            if fork and self._queue.producer_state("fork-active") is not None:
                raise ValueError("This queue branch has already started; resolve checkpoint selection again")
            if self.restore_plan.mode == "resume":
                # Restart requires the entire prior job to have stopped. Unlike a
                # lease timeout, that supervisor assertion ends old reads.
                readers = self.queue.outstanding_reads()
                for offset in range(0, len(readers), 128):
                    self.queue.release_task_reads(readers[offset : offset + 128])
            # Recovery invalidates leases; unfinished collections from the stopped
            # manager cannot produce a new branch's batch. Accepted facts stay intact.
            for task_id, task in list(self.queue.tasks.items()):
                if task_id.startswith("collection:") and task["state"] == "pending":
                    self.queue.cancel_task(task_id, request_id=f"abandon:{self.branch_id}:{task_id}")
            # Joint restore can rewind the training cursor. Reconcile its durable
            # storage ownership before GC inspects restored live result roots.
            if not fork and not defer_gc:
                self._start_gc()
            # Successful construction transfers ownership to close(). A failed
            # constructor closes the coordinator and releases the save lock here.
            cleanup.pop_all()

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
            if path := self.restore_plan.dataset_cursor:
                import torch

                state = torch.load(path, map_location="cpu", weights_only=False)
                for key in ("sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata"):
                    if key not in state:
                        raise ValueError(f"Dataset checkpoint is missing {key}")
                    setattr(self.producer, key, copy.deepcopy(state[key]))
                if self.producer.dataset is not None and self.args.rollout_shuffle:
                    self.producer.dataset.shuffle(self.producer.epoch_id)
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
            dataset = {
                key: getattr(self.args, key, None)
                for key in (
                    "input_key",
                    "label_key",
                    "metadata_key",
                    "tool_key",
                    "multimodal_keys",
                    "rollout_max_prompt_len",
                    "apply_chat_template",
                    "apply_chat_template_kwargs",
                    "hf_checkpoint",
                )
            }
            if getattr(self.args, "prompt_data", None):
                import hashlib

                from slime.utils.data import _parse_generalized_path

                path, row_slice = _parse_generalized_path(self.args.prompt_data)
                checksum = hashlib.sha256()
                with open(path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        checksum.update(chunk)
                dataset.update(sha256=checksum.hexdigest(), row_slice=str(row_slice))
            return {
                "dataset_size": len(source),
                "n_samples_per_prompt": self.args.n_samples_per_prompt,
                "seed": self.args.rollout_seed,
                "shuffle": self.args.rollout_shuffle,
                "run_id": self.store.run_id,
                "branch_id": self.branch_id,
                "queue_id": self.queue.queue_id,
                "dataset": dataset,
            }

    def identity(self):
        self._check_gc_error()
        return {"run_id": self.store.run_id, "branch_id": self.branch_id}

    def pending_snapshot(self):
        """Capture pending inputs and the producer cursor at one refill boundary."""
        with self._producer_lock:
            self._source()
            producer = self.queue.producer_state("dataset")
            cursor = copy.deepcopy(producer["cursor"] if producer else self._initial_cursor)
            tasks = []
            for spec in self.queue.pending_tasks(task_prefix="prompt:"):
                value = asdict(spec)
                value["input_ref"] = DiskPayloadRef(spec.input_ref, str(self.store.backend.root))
                tasks.append(value)
            inflight = self.queue.metrics()["tasks"].get("leased", 0) > 0
            with self._writer_lock:
                return self.codec.publish(
                    {"tasks": tasks, "producer_cursor": cursor, "inflight": inflight},
                    submission_id=f"pending:{uuid.uuid4().hex}",
                )

    def fork_source(self, source_ref):
        """Import a paused source snapshot into a new queue, sharing payload bytes.

        Task submissions and ready-result acceptance are idempotent so an
        interrupted import can finish before any generation is admitted.
        Receipt positions are local to a queue: ready groups receive new ones.
        """
        from straw.protocol import digest

        from slime.data.transport import RolloutGroupRef

        if self.restore_plan.mode != "snapshot":
            raise ValueError("Source forks require an isolated queue")
        owner = f"fork-source:{self.queue.queue_id}"
        with self._producer_lock, self._training_lock, self._writer_lock:
            previous = self.queue.producer_state("fork-source")
            if previous:
                if previous["source"] != asdict(source_ref):
                    raise ValueError("Fork destination already belongs to a different checkpoint")
                self._fork_ready = True
                self._start_gc()
                return RecordSetRef.from_dict(previous["restored"])
            state = self.codec.load(source_ref)
            reader = state["reader"].load()
            pending = reader["pending"].load()
            if "producer_cursor" not in pending or pending.get("inflight", True):
                raise ValueError("Fork requires a producer cursor and a drained source checkpoint")
            if set(state["consumers"]) - {"fully_async"}:
                raise ValueError("Fork does not support custom consumer state yet")
            # Protect the entire source graph during import, including reader
            # metadata that consumers will only load when they start later.
            self.store.retain(owner, [source_ref])
            self.queue.submit_tasks(
                f"fork-producer:{source_ref.digest}",
                [],
                producer_id="dataset",
                producer_state={"version": 1, "cursor": pending["producer_cursor"]},
            )
            tasks = []
            for value in pending["tasks"]:
                value = dict(value)
                value["input_ref"] = value["input_ref"].manifest
                value["metadata"] = {**value["metadata"], "source_positions": []}
                tasks.append(TaskSpec(**value))
            for start in range(0, len(tasks), 64):
                self.queue.submit_tasks(f"fork-pending:{source_ref.digest}:{start}", tasks[start : start + 64])
            consumers = state["consumers"]
            if "fully_async" in consumers:
                ready = []
                for index, (group, verdict) in enumerate(consumers["fully_async"]["scheduler"]["ready"]):
                    if group is None:
                        ready.append((group, verdict))
                        continue
                    task_id = f"prompt:fork-ready:{source_ref.digest}:{index}"
                    self.queue.submit_tasks(task_id, [TaskSpec(task_id, group.manifest)])
                    status = self.queue.task_status(task_id)
                    if status["state"] == "completed":
                        receipt = self.result(Lease(**status["lease"]))
                    else:
                        page = self.queue.acquire("fork-import", 1, task_ids=[task_id])
                        if not page.assignments:
                            raise RuntimeError(f"Cannot import ready group: {page.status}")
                        lease = page.assignments[0].lease
                        # The native protocol binds result metadata to this new
                        # attempt. A small wrapper shares the old payload graph.
                        result = self.codec.publish(
                            DiskPayloadRef(group.manifest, group.root),
                            submission_id=f"fork-result:{lease.attempt_id}",
                            metadata={"task_id": lease.task_id, "attempt_id": lease.attempt_id},
                        )
                        receipt = self.complete(lease, result)
                    ready.append((RolloutGroupRef(receipt.result_ref, group.root, group.index, receipt, ()), verdict))
                consumers["fully_async"]["scheduler"]["ready"] = ready
            self.reader_metadata("owner", reader["metadata"])
            restored = self.codec.publish(consumers, submission_id=f"fork-consumers:{source_ref.digest}")
            self.store.retain(owner, [source_ref, restored])
            self.store.release_publications([restored])
            self.queue.submit_tasks(
                f"fork-complete:{digest(asdict(source_ref))}",
                [],
                producer_id="fork-source",
                producer_state={"source": asdict(source_ref), "restored": asdict(restored)},
            )
            self._fork_ready = True
        self._start_gc()
        return restored

    def _activate_fork(self):
        if not self._fork_ready:
            raise RuntimeError("Load the fork checkpoint before admitting rollout work")
        if self.restore_plan.mode == "snapshot":
            from straw.reporting import write_report

            with self._producer_lock:
                if not self._fork_active:
                    self.queue.submit_tasks(
                        "fork-active", [], producer_id="fork-active", producer_state={"started": True}
                    )
                    write_report(
                        Path(self.args.save) / "rollout" / "fork-active.json", {"queue_id": self.queue.queue_id}
                    )
                    self._fork_active = True

    def pending_count(self):
        return sum(spec.metadata.get("returned", False) for spec in self.queue.pending_tasks(task_prefix="prompt:"))

    def return_groups(self, updates, deliveries):
        """Return partials atomically; accepted results use separate delivery tasks."""
        from straw.protocol import digest

        with self._producer_lock:
            if updates:
                self.queue.yield_tasks(updates, request_id="return:" + digest(updates))
            for delivery in deliveries:
                ref = RecordSetRef.from_dict(delivery["input_ref"])
                task_id = f"prompt:delivery:{digest(delivery)}"
                if self.queue.task_status(task_id) is not None:
                    continue
                spec = TaskSpec(
                    task_id,
                    ref,
                    metadata=delivery["metadata"],
                    priority=delivery["priority"],
                    scheduling_key=delivery["scheduling_key"],
                    estimated_tokens=ref.tokens,
                )
                self.queue.submit_tasks(task_id, [spec])

    def restore_pending(self, snapshot):
        """Restore saved input versions without mutating accepted result history.

        Restore is called before rollout readers start. Full dataset/model
        rollback is a separate checkpoint protocol; this restores pending work.
        """
        updates, deliveries = [], []
        for saved in self.codec.load(snapshot)["tasks"]:
            ref = saved["input_ref"].manifest
            status = self.queue.task_status(saved["task_id"])
            fields = {
                key: saved.get(key, default)
                for key, default in (
                    ("metadata", {}),
                    ("priority", 0),
                    ("scheduling_key", 0),
                )
            }
            if status and status["state"] in {"pending", "leased"}:
                if status["state"] == "leased":
                    self.release([Lease(**status["lease"])])
                page = self.queue.acquire("checkpoint-restore", 1, task_ids=[saved["task_id"]])
                if not page.assignments:
                    raise RuntimeError(f"Cannot restore pending task: {saved['task_id']} ({page.status})")
                updates.append(
                    {
                        "lease": asdict(page.assignments[0].lease),
                        "input_ref": asdict(ref),
                        **fields,
                    }
                )
            else:
                receipt = self.result(Lease(**status["lease"])) if status and status["state"] == "completed" else None
                metadata = dict(fields["metadata"])
                if receipt:
                    metadata["source_positions"] = sorted(
                        set(metadata.get("source_positions", [])) | {receipt.position}
                    )
                deliveries.append(
                    {
                        "input_ref": asdict(ref),
                        **fields,
                        "metadata": metadata,
                    }
                )
            if len(updates) == 64:
                self.return_groups(updates, [])
                updates = []
        self.return_groups(updates, deliveries)

    def reader_metadata(self, reader_id, metadata=None):
        if metadata is not None:
            with self._writer_lock:
                ref = self.codec.publish(
                    {"metadata": metadata},
                    submission_id=f"reader-metadata:{uuid.uuid4().hex}",
                )
            self.reader_state(reader_id, ref)
        state = self.reader_state(reader_id)
        return self.codec.load(RecordSetRef.from_dict(state["state_ref"])).get("metadata", {}) if state else {}

    def take(self, reader_id, count):
        """Supply queued groups, refilling an empty queue from the dataset.

        The refill path takes original prompt groups from RolloutDataSource,
        writes them to straw, then enqueues their references and the new dataset
        cursor. Existing tasks, including partials, are acquired first. ``count``
        counts prompt groups, each containing n_samples_per_prompt samples.
        """
        self._activate_fork()
        self._active_readers.add(reader_id)
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
            # Start from the last committed cursor: a failed refill may have
            # advanced the in-memory source without successfully enqueueing it.
            state = self.queue.producer_state("dataset")
            for key, value in (state["cursor"] if state else self._initial_cursor).items():
                setattr(source, key, copy.deepcopy(value))
            if source.dataset is not None and self.args.rollout_shuffle:
                source.dataset.shuffle(source.epoch_id)
            start = source.sample_group_index
            # Bounded refill, without unused index reservations held in readers.
            count = min(max(count, 8), 64)
            groups = source.get_samples(count)
            group_ids = [next(iter_samples(group)).group_index for group in groups]
            task_ids = [f"prompt:{group_id}" for group_id in group_ids]
            # Persist the sample groups; queue tasks carry references only.
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
            # Commit tasks and the advanced cursor together, so a failed refill
            # cannot skip dataset inputs on the next attempt.
            self.queue.submit_tasks(
                f"dataset:{start}",
                tasks,
                producer_id="dataset",
                producer_state={"version": 1, "cursor": cursor},
            )
        # Return leases and references; readers load payloads from shared storage.
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
        result = self.queue.fail_task(
            lease,
            request_id=f"reject:{lease.attempt_id}",
            failure={"category": "Filtered", "reason": reason},
            retryable=False,
        )
        positions = self.queue.task_status(lease.task_id)["spec"]["metadata"].get("source_positions", [])
        if positions:
            # A failed delivery cannot replay its earlier accepted versions.
            # Advance their retention only after fencing this task's lease.
            with self._writer_lock:
                ref = self.codec.publish(
                    {"positions": positions, "reason": reason},
                    submission_id=f"reject:{lease.attempt_id}",
                )
            self.record_dispositions(ref)
        return result

    def result(self, lease):
        return self.queue.lookup_submission(f"result:{lease.attempt_id}")

    def results(self, leases):
        return [self.result(lease) if lease else None for lease in leases]

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
            # Fence the reader before recovering its results. Late acquire or
            # completion calls from that reader must not compete with its successor.
            self._retired_readers.add(reader_id)
            accepted = self.queue.retire_worker(reader_id)
            state = self._training_state()
            processed = set(state["processed_positions"])
            superseded = {
                position
                for task in self.queue.tasks.values()
                for position in task["spec"]["metadata"].get("source_positions", [])
            }
            receipts = [
                receipt
                for receipt in accepted
                if receipt["position"] >= state["processed_cursor"]
                and receipt["position"] not in delivered | processed
                and receipt["position"] not in superseded
            ]
            with self._writer_lock:
                return self.codec.publish(receipts, submission_id=f"recover-reader:{uuid.uuid4().hex}")

    def recover_manager_readers(self, generation, excluded):
        """Fence one dead manager's readers while keeping this queue authoritative."""
        # Reader IDs are scoped to a manager incarnation. Fence only that
        # incarnation, and avoid returning samples already retained for replay.
        with self._writer_lock:
            delivered = self.codec.publish(excluded, submission_id=f"recover-manager:{uuid.uuid4().hex}")
        results = []
        for reader_id in sorted(self._active_readers):
            if reader_id.startswith(generation + ":"):
                ref = self.recover_reader_results(reader_id, delivered)
                results.extend(self.codec.load(ref))
        return results

    def recover_pending_rollout(self):
        """Replay accepted groups after a whole-job failure before the first batch.

        Once conversion/training has started, queue progress alone cannot identify
        the model/optimizer version. That recovery requires a joint checkpoint.
        The restarted job must have stopped all prior readers.
        """
        if self.restore_plan.mode != "resume":
            raise RuntimeError("Whole-job replay requires automatic WAL recovery selection")
        with self._producer_lock, self._training_lock:
            state = self._training_state()
            if self.queue.batches or state["processed_cursor"] or state["processed_positions"]:
                raise RuntimeError(
                    "Queue-only recovery is limited to an interrupted first rollout; "
                    "restore matching model/optimizer and rollout checkpoints after batch conversion begins"
                )
            superseded = {
                position
                for task in self.queue.tasks.values()
                for position in task["spec"]["metadata"].get("source_positions", [])
            }
            receipts = [
                receipt
                for receipt in self.queue.commits
                if receipt["task_id"].startswith("prompt:") and receipt["position"] not in superseded
            ]
            with self._writer_lock:
                return self.codec.publish(receipts, submission_id=f"recover-job:{self.branch_id}")

    def begin_collection(self, request_id):
        self._activate_fork()
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

    def _advance(self, state, positions):
        positions = set(state["processed_positions"]) | set(positions)
        # A delivery acknowledges the earlier accepted versions it replaced,
        # whether it reaches training or is rejected by a selection hook.
        for position in list(positions - set(state["processed_positions"])):
            if position < state["processed_cursor"]:
                continue
            receipt = self.queue.read_commits(position, 1).commits[0]
            task = self.queue.task_status(receipt.task_id)
            positions.update(task["spec"]["metadata"].get("source_positions", []))
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
        """Move restore ownership after pending tasks and consumers have been snapshotted.

        The caller retains the complete replacement source before this call,
        with its readers paused. Old checkpoint owners remain independent until
        a successor joint commit allows their normal retirement.
        """
        with self._training_lock:
            if self.restore_plan.mode == "snapshot":
                self.store.retain(f"fork-source:{self.queue.queue_id}", [source_ref])
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
                from slime.data.transport import RolloutGroupRef

                retained, visited = set(), set()

                def visit(value):
                    if isinstance(value, RolloutGroupRef) and value.receipt:
                        retained.add(value.receipt.position)
                    elif isinstance(value, DiskPayloadRef):
                        if value.manifest not in visited:
                            visited.add(value.manifest)
                            visit(self.codec.load(value.manifest))
                    elif isinstance(value, Sample):
                        retained.update(getattr(value, "_queue_source_positions", []))
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
                        retained.update(value.get("source_positions", []))
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
                            "reason": "checkpoint branch excludes outputs absent from its saved source",
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

    def metrics(self):
        return self.queue.metrics()

    def close(self):
        try:
            self._gc_stop.set()
            if self._gc_thread is not None:
                self._gc_thread.join()
            try:
                self._queue.close()
            finally:
                with self._writer_lock:
                    self.store.seal()
            self._check_gc_error()
        finally:
            # Keep the checkpoint branch locked through queue and writer cleanup,
            # including failures. Actor termination also closes this descriptor.
            if self._checkpoint_lock is not None:
                self._checkpoint_lock.close()


def create_queue_controller(args, *, restore_plan=None):
    """Create one job-owned actor; callers retain and explicitly share its handle."""
    resolve_rollout_data_dir(args)
    return (
        ray.remote(RolloutQueueController)
        .options(
            num_cpus=0,
            max_concurrency=4,
            max_restarts=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=False),
        )
        .remote(copy.copy(args), restore_plan=restore_plan)
    )


@dataclass(frozen=True)
class QueueReaderConfig:
    args: object
    controller: object
    reader_id: str
    dataset_size: int
    branch_id: str | None = None

    def open(self, args=None):
        return QueueReader(
            self.args if args is None else args,
            self.controller,
            self.reader_id,
            self.dataset_size,
            branch_id=self.branch_id,
        )


class QueueReader(DataSource):
    """One process's queue client: task acquisition, returns and lease renewal.

    Readers share the coordinator and durable inputs, but track their own active
    leases. Job-wide consumers and checkpoint files belong to QueueDataSource.
    """

    def __init__(self, args, controller, reader_id, dataset_size, *, branch_id=None):
        self.args, self.controller, self.reader_id = args, controller, reader_id
        self.branch_id = branch_id or ray.get(controller.identity.remote())["branch_id"]
        self.dataset_size = dataset_size
        # Only actively executing leases are local. Returned work lives in the
        # durable queue and can be acquired by any reader after this one stops.
        self._leases = {}
        self._closed = False
        self._lock = threading.RLock()
        self._reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="queue-input")
        self._stop = threading.Event()
        self._heartbeat_error = None
        filter_path = getattr(args, "buffer_filter_path", None)
        if filter_path is not None:
            raise ValueError(
                "--buffer-filter-path is not supported by straw; the queue prioritizes ready groups, partials, "
                "then fresh prompts, using older weight versions first within each stage"
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
                            elif outcome != "extended":
                                raise LeaseExpired(f"Reader lost lease for {lease.task_id}: {outcome}")
            except Exception as error:
                self._heartbeat_error = error
                return

    def get_samples(self, num_samples):
        """Acquire groups from durable scheduling state.

        The queue serves ready deliveries, then partials, then untouched prompts.
        Within each stage it applies the stored scheduling key and FIFO order.
        Payloads are read directly from shared storage; RPC carries references.
        """
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        if self._closed:
            raise RuntimeError("Queue reader is closed")
        if self._heartbeat_error:
            raise RuntimeError("Queue lease heartbeat failed") from self._heartbeat_error
        groups = []
        while len(groups) < num_samples:
            if self._closed:
                raise RuntimeError("Queue reader is closed")
            if self._heartbeat_error:
                raise RuntimeError("Queue lease heartbeat failed") from self._heartbeat_error
            page = ray.get(self.controller.take.remote(self.reader_id, min(num_samples - len(groups), 64)))
            if page.status in {"end_of_input", "draining"}:
                break
            if not page.assignments:
                if page.status not in {"backpressured", "empty"}:
                    raise RuntimeError(f"Queue cannot supply rollout inputs: {page.status}")
                time.sleep(0.05)
                continue
            store, codec, _ = rollout_store(self.args)
            with store.read_session() as session:
                for assignment in page.assignments:
                    lease = assignment.lease
                    group = codec.load(assignment.task.input_ref, reader=session)
                    for sample in iter_samples(group):
                        # A delivery task has its own authorization. Prior
                        # receipts only contribute to retention accounting.
                        sample.__dict__.pop("_queue_receipt", None)
                        sample._queue_lease = asdict(lease)
                        sample._queue_generation_start = len(sample.tokens)
                        sample._queue_branch = self.branch_id
                        sample._queue_source_positions = assignment.task.metadata.get("source_positions", [])
                    with self._lock:
                        self._leases[lease.task_id] = lease
                    groups.append(group)
        return groups

    async def get_samples_async(self, num_samples):
        """Run blocking queue/file reads on the reader thread, keeping asyncio responsive."""
        return await asyncio.get_running_loop().run_in_executor(self._reader, self.get_samples, num_samples)

    def add_samples(self, groups):
        """Persist returned groups and make them claimable by any reader.

        A usable partial yields its task with a new input_ref. Incomplete R3/SC
        capture returns the task with its previous input instead. Already accepted
        results get delivery tasks; their original completion remains immutable.
        """
        if len(groups) > 64:
            for offset in range(0, len(groups), 64):
                self.add_samples(groups[offset : offset + 64])
            return
        if not groups:
            return
        groups_to_save, leases_to_retry, retry_mismatches = [], [], []
        for group in groups:
            mismatches = self._continuation_mismatches(group)
            if not mismatches:
                groups_to_save.append(group)
                continue
            lease = group_lease(group)
            if lease is None:
                raise ValueError(f"Incomplete continuation has no durable task input to retry: {mismatches}")
            release_rollout_publications(group, self.args)
            leases_to_retry.append(lease)
            retry_mismatches.extend(mismatches)
        if leases_to_retry:
            ray.get(self.controller.release.remote(leases_to_retry))
            with self._lock:
                for lease in leases_to_retry:
                    if self._leases.get(lease.task_id) == lease:
                        self._leases.pop(lease.task_id)
            logger.warning(
                "Requeued %d incomplete continuations from their last durable inputs; (sample, field, actual, expected): %s",
                len(leases_to_retry),
                retry_mismatches[:8],
            )
        if not groups_to_save:
            return
        leases = [group_lease(group) for group in groups_to_save]
        receipts = ray.get(self.controller.results.remote(leases))
        for group, receipt in zip(groups_to_save, receipts, strict=True):
            record_generation_provenance(group)
            if receipt:
                for sample in iter_samples(group):
                    sample.__dict__.pop("_queue_lease", None)
                    sample._queue_receipt = asdict(receipt)
        store, codec, writer_lock = rollout_store(self.args)
        with writer_lock:
            refs = codec.publish_many(
                groups_to_save,
                submission_ids=[f"continuation:{uuid.uuid4().hex}" for _ in groups_to_save],
            )
        updates, deliveries = [], []
        for group, lease, receipt, ref in zip(groups_to_save, leases, receipts, refs, strict=True):
            samples = list(iter_samples(group))
            ready = all(
                s.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) and s.reward is not None
                for s in samples
            )
            partial = any(s.response_length for s in samples)
            versions = [
                int(str(v))
                for s in samples
                for v in (s.weight_versions or [])
                if str(v).isascii() and str(v).removeprefix("-").isdigit()
            ]
            # current_version - oldest_version orders exactly like oldest_version
            # ascending, without rewriting every task after each weight update.
            scheduling_key = min(versions, default=2**63 - 1)
            scheduling_key = max(-(2**63), min(2**63 - 1, scheduling_key))
            origin = asdict(receipt) if receipt else getattr(samples[0], "_queue_receipt", None)
            source_positions = {p for s in samples for p in getattr(s, "_queue_source_positions", [])}
            if origin:
                source_positions.add(origin["position"])
            fields = {
                "input_ref": asdict(ref),
                "priority": 2 if ready else 1 if partial else 0,
                "scheduling_key": scheduling_key,
                "metadata": {
                    "stage": "ready" if ready else "partial" if partial else "fresh",
                    "returned": True,
                    "source_positions": sorted(source_positions),
                },
            }
            if lease and not receipt:
                updates.append({"lease": asdict(lease), **fields})
            else:
                deliveries.append(fields)
        # Publish before yielding: the WAL transition adopts the new input and
        # ends the old lease in one transaction. A lost reply can be retried.
        ray.get(self.controller.return_groups.remote(updates, deliveries))
        with self._lock:
            for lease in leases:
                if lease and self._leases.get(lease.task_id) == lease:
                    self._leases.pop(lease.task_id)
        # Retain no Sample objects or input references after handing work back.

    def _continuation_mismatches(self, group):
        """Return (sample, field, actual rows, expected rows) for R3/top-k SC captures."""
        mismatches = []
        for sample in iter_samples(group):
            if not sample.response_length:
                continue
            if getattr(self.args, "use_rollout_routing_replay", False):
                rows = sample.get_rollout_routed_experts_length()
                if rows != len(sample.tokens) - 1:
                    mismatches.append((sample.index, "routes", rows, len(sample.tokens) - 1))
            if getattr(self.args, "use_score_centering", False) and getattr(self.args, "rollout_top_p", 1) >= 1:
                for name in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
                    value = getattr(sample, name)
                    rows = len(value) if value is not None else 0
                    if rows != sample.response_length:
                        mismatches.append((sample.index, name, rows, sample.response_length))
        return mismatches

    def state_dict(self, *, include_pending=True):
        state = {"version": 3, "metadata": self.get_metadata()}
        if include_pending:
            ref = ray.get(self.controller.pending_snapshot.remote())
            state["pending"] = DiskPayloadRef(ref, self.args.rollout_data_dir)
        return pack_rollout_payload(state, self.args, -1)

    def load_state_dict(self, state):
        value = unpack_rollout_payload(state)
        if value["version"] != 3:
            raise ValueError(
                "Reader-local buffer checkpoints require migration before using the durable queue scheduler"
            )
        if "pending" in value:
            ray.get(self.controller.restore_pending.remote(value["pending"].manifest))
        self.update_metadata(value["metadata"])

    def materialize_samples(self, samples, *, release_files=True):
        from slime.data.tensor import TensorRef

        if isinstance(samples, list):
            return [self.materialize_samples(child) for child in samples]
        sample = copy.copy(samples)
        for key, value in vars(sample).items():
            if isinstance(value, TensorRef):
                setattr(sample, key, value.load())
        return sample

    def get_buffer_length(self):
        return ray.get(self.controller.pending_count.remote())

    def update_metadata(self, metadata):
        current = self.get_metadata()
        current.update(metadata)
        ray.get(self.controller.reader_metadata.remote(self.reader_id, current))

    def get_metadata(self):
        return ray.get(self.controller.reader_metadata.remote(self.reader_id))

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
        with self._lock:
            leases = list(self._leases.values())
        ray.get(self.controller.release.remote(leases))
        self._stop.set()
        self._heartbeats.join(timeout=5)


class QueueDataSource(QueueReader):
    """Job-level data source: reader creation, consumers and source checkpoints.

    The manager reads through the inherited QueueReader interface; distributed
    workers open separate readers against the same controller.
    """

    def __init__(self, args, *, controller=None, restore_plan=None, reader_generation=""):
        self.restore_plan = restore_plan or RestorePlan()
        self.restored_source = SourceRestore(new_queue=self.restore_plan.mode == "empty")
        self._owns_controller = controller is None
        if controller is None:
            controller = create_queue_controller(args, restore_plan=self.restore_plan)
        self.data_config = ray.get(controller.configuration.remote())
        # Give each manager incarnation distinct reader IDs. Reusing old IDs
        # would let late worker calls interfere with the replacement manager.
        self.reader_generation = reader_generation
        reader_id = f"{reader_generation}:owner" if reader_generation else "owner"
        super().__init__(args, controller, reader_id, self.data_config["dataset_size"])
        self.manager_restored = False
        self.consumers, self._restored_consumers = {}, {}

    def reader_config(self, reader_id):
        if reader_id == "owner":
            raise ValueError("Reader ID 'owner' is reserved")
        if self.reader_generation:
            reader_id = f"{self.reader_generation}:{reader_id}"
        return QueueReaderConfig(self.args, self.controller, reader_id, len(self), self.branch_id)

    def manager_state(self):
        # Pending tasks and the producer cursor remain in the live controller.
        # Restoring a manager must not rewind that shared queue to an old snapshot.
        return {"reader_generation": self.reader_generation, "metadata": self.get_metadata()}

    def restore_manager(self, state, *, excluded):
        # Exclude batches already journaled for trainer replay; recover only
        # accepted results that the dead manager had not handed off durably.
        receipts = ray.get(self.controller.recover_manager_readers.remote(state["reader_generation"], excluded))
        for value in receipts:
            from straw.protocol import CommitReceipt

            receipt = CommitReceipt.from_dict(value)
            group = DiskPayloadRef(receipt.result_ref, self.args.rollout_data_dir).load()
            for sample in iter_samples(group):
                sample._queue_receipt = value
            # Native deliveries preserve generation/reward and work for both
            # synchronous and fully-async consumers after manager replacement.
            self.add_samples([group])
        self.update_metadata(state["metadata"])
        # The async scheduler must not also run whole-job queue recovery, which
        # would deliver the same accepted results a second time.
        self.manager_restored = True

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
                "version": 2,
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
                    "queue_id": self.data_config["queue_id"],
                },
            )
            if self.data_config["queue_id"].startswith("fork:"):
                write_report(path.parent / "fork-active.json", {"queue_id": self.data_config["queue_id"]})
            if self.restored_source.source_ref is not None:
                ray.get(self.controller.handoff_restored_source.remote(ref.manifest))
        finally:
            for consumer, was_paused in paused:
                if not was_paused:
                    consumer.resume()

    def load(self, rollout_id=None):
        import json

        from straw.protocol import RecordSetRef

        if not self.args.load or self.restore_plan.mode == "empty":
            return self.restored_source
        if self.consumers:
            raise RuntimeError("Restore before starting rollout consumers")
        path = Path(self.args.load) / "rollout" / f"queue_state_{rollout_id}.json"
        if not path.exists():
            if self.restore_plan.mode == "snapshot":
                raise FileNotFoundError(path)
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
        if self.restore_plan.mode == "snapshot":
            if state["version"] != 2:
                raise ValueError("Queue fork requires a checkpoint with producer state (source version 2)")
            if state["configuration"].get("dataset") != self.data_config.get("dataset"):
                raise ValueError("Data source checkpoint differs in dataset identity/configuration")
            if Path(value["root"]).resolve() != Path(self.args.rollout_data_dir).resolve():
                raise ValueError("Copy-on-write fork requires the original straw pool")
            ref = ray.get(self.controller.fork_source.remote(RecordSetRef.from_dict(value["manifest"])))
            self._restored_consumers = DiskPayloadRef(ref, self.args.rollout_data_dir).load()
        else:
            if state["configuration"].get("queue_id", self.data_config["queue_id"]) != self.data_config["queue_id"]:
                raise ValueError("Data source checkpoint belongs to a different queue namespace")
            self.load_state_dict(state["reader"])
            self._restored_consumers = state["consumers"]
        self.restored_source = SourceRestore(
            new_queue=self.restore_plan.mode == "snapshot",
            source_ref=RecordSetRef.from_dict(value["manifest"]),
        )
        return self.restored_source

    def close(self):
        if self._closed:
            return
        for consumer in self.consumers.values():
            consumer.close()
        super().close()
        if self._owns_controller:
            ray.get(self.controller.close.remote())
            ray.kill(self.controller, no_restart=True)
