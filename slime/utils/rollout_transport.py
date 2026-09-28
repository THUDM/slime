"""Rollout references over immutable queue segments; no pickle payload protocol."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

if TYPE_CHECKING:
    from straw.protocol import CommitReceipt, RecordSetRef

_writers = {}
_writers_lock = threading.Lock()
_async_limits = {}


def seal_rollout_store(args):
    """Seal local packs after generation/publication work has been drained."""
    if args.rollout_data_transport == "straw":
        store, _, lock = rollout_store(args)
        with lock:
            store.seal()


def resolve_rollout_data_dir(args):
    if args.rollout_data_dir is None:
        if args.save is None:
            raise ValueError("straw rollout transport requires --rollout-data-dir or --save on shared storage")
        args.rollout_data_dir = str(Path(args.save) / "rollout_data")
    args.rollout_data_dir = str(Path(args.rollout_data_dir).expanduser().resolve())


def rollout_store(args):
    """One physical writer incarnation per process/run, shared by producer calls."""
    try:
        from straw.backend import FilesystemBackend
        from straw.store import SharedFilesystemStore
        from straw.tensor import MAX_PUBLICATION_BYTES, MAX_TENSOR_BYTES
    except ModuleNotFoundError as error:
        if error.name != "straw":
            raise
        raise ModuleNotFoundError(
            "straw rollout transport requires straw-queue. "
            "Install it on every rollout/training node: pip install straw-queue",
            name="straw",
        ) from error

    from slime.rollout.queue_codec import CODECS, SampleCodec

    root = str(Path(args.rollout_data_dir).resolve())
    run_id = getattr(args, "rollout_queue_run_id", None) or "rollout"
    profile = getattr(args, "rollout_storage_profile", "local")
    declaration_path = getattr(args, "rollout_storage_declaration", None)
    declaration = json.loads(Path(declaration_path).read_text()) if declaration_path else None
    segment_mib = getattr(args, "rollout_queue_segment_mib", 256)
    online_gc = getattr(args, "rollout_queue_online_gc", False)
    key = (os.getpid(), root, run_id, profile, segment_mib, online_gc, json.dumps(declaration, sort_keys=True))
    with _writers_lock:
        if key not in _writers:
            store = SharedFilesystemStore(
                root,
                run_id,
                online_gc=online_gc,
                codecs=CODECS,
                segment_target_bytes=segment_mib * 1024**2,
                max_record_bytes=MAX_TENSOR_BYTES,
                max_buffer_bytes=MAX_PUBLICATION_BYTES,
                backend=FilesystemBackend(root, profile=profile, declaration=declaration),
            )
            _writers[key] = (store, SampleCodec(store, args=args), threading.RLock())
        return _writers[key]


@dataclass(frozen=True)
class DiskPayloadRef:
    """Bounded protocol ref plus a deployment mount hint, overridable by readers.

    Only ``manifest`` is persisted by SampleCodec. Its paths are relative; root
    is a local mount binding, never a dependency embedded in a durable record.
    """

    manifest: RecordSetRef
    root: str

    def load(self, *, root=None):
        from straw.store import SharedFilesystemStore

        from straw.tensor import MAX_PUBLICATION_BYTES, MAX_TENSOR_BYTES

        from slime.rollout.queue_codec import CODECS, SampleCodec

        store = SharedFilesystemStore(
            root or self.root,
            self.manifest.manifest.segment.run_id,
            codecs=CODECS,
            max_record_bytes=MAX_TENSOR_BYTES,
            max_buffer_bytes=MAX_PUBLICATION_BYTES,
        )
        return SampleCodec(store).load(self.manifest)


@dataclass(frozen=True)
class RolloutGroupRef(DiskPayloadRef):
    index: int
    receipt: CommitReceipt | None = None


@dataclass(frozen=True)
class RawRolloutRef(DiskPayloadRef):
    receipt: CommitReceipt
    metrics: dict | None = None


@dataclass(frozen=True)
class TrainBatchRef(DiskPayloadRef):
    batch_id: str
    rank: int
    plan_digest: str


def group_lease(group):
    from slime.rollout.base_types import iter_samples

    samples = list(iter_samples(group))
    leases = [getattr(sample, "_queue_lease", None) for sample in samples]
    if not any(leases):
        return None
    if not all(value == leases[0] for value in leases):
        raise ValueError("A queue group must preserve one task authorization across all trajectories")
    from straw.protocol import Lease

    return Lease(**leases[0])


def inherit_queue_context(source, output):
    """Carry the input authorization across hooks that construct new trajectories."""
    import copy

    from slime.rollout.base_types import iter_samples

    fields = (
        "_queue_lease",
        "_queue_receipt",
        "_queue_source_positions",
        "_queue_resume_origin",
        "_queue_generation_start",
        "queue_policy_segments",
        "queue_generation_requests",
    )
    for sample in iter_samples(output):
        for name in fields:
            if not hasattr(source, name):
                continue
            value = getattr(source, name)
            if name in {"_queue_lease", "_queue_receipt"} and getattr(sample, name, value) != value:
                raise ValueError("A generation hook returned a different queue task authorization")
            if name in {"_queue_lease", "_queue_receipt", "queue_generation_requests"} or not hasattr(sample, name):
                setattr(sample, name, copy.deepcopy(value))
    return output


def record_generation_provenance(group, args):
    from slime.rollout.base_types import iter_samples

    for sample in iter_samples(group):
        start = getattr(sample, "_queue_generation_start", None)
        branch = getattr(args, "_rollout_queue_branch", None)
        if start is not None and branch is not None and len(sample.tokens) > start:
            segments = list(getattr(sample, "queue_policy_segments", []))
            if not segments and start:
                segments.append({"start": 0, "stop": start, "branch": None, "reported_versions": None})
            segments.append(
                {
                    "start": start,
                    "stop": len(sample.tokens),
                    "branch": branch,
                    "reported_versions": sample.weight_versions or None,
                }
            )
            sample.queue_policy_segments = segments
            sample._queue_generation_start = len(sample.tokens)


def pack_rollout_group(group, args, rollout_id):
    if args.rollout_data_transport != "straw":
        return group
    from slime.rollout.base_types import iter_samples

    first = next(iter_samples(group))
    lease = group_lease(group)
    record_generation_provenance(group, args)
    metadata = {"task_id": lease.task_id, "attempt_id": lease.attempt_id} if lease else {}
    ref = pack_rollout_payload(
        group, args, rollout_id, metadata=metadata, submission_id=f"group:{lease.attempt_id}" if lease else None
    )
    receipt = None
    if lease:
        controller = getattr(args, "_rollout_queue_controller", None)
        if controller is None:
            raise ValueError("Queue group has a lease but no coordinator binding")
        receipt = ray.get(controller.complete.remote(lease, ref.manifest))
    return RolloutGroupRef(receipt.result_ref if receipt else ref.manifest, ref.root, first.index, receipt)


def release_rollout_publications(group, args):
    """Relinquish tensor staging before ending the group's current read lifetime."""
    if args.rollout_data_transport == "straw":
        from straw.tensor import TensorRef, release_tensor_publications
        from slime.rollout.base_types import iter_samples

        tensors = []

        def visit(value):
            if isinstance(value, TensorRef):
                tensors.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        for sample in iter_samples(group):
            visit(vars(sample))
        if tensors:
            store, _, lock = rollout_store(args)
            with lock:
                release_tensor_publications(store, tensors)


def discard_rollout_group(group, args, reason="dynamic filter"):
    lease = group_lease(group)
    release_rollout_publications(group, args)
    # Release staging while the task/accepted-result owner still protects any
    # previously adopted continuation, then acknowledge that reading is done.
    if lease:
        ray.get(args._rollout_queue_controller.reject.remote(lease, reason))
    else:
        from slime.rollout.base_types import iter_samples

        positions = {
            sample._queue_receipt["position"] for sample in iter_samples(group) if hasattr(sample, "_queue_receipt")
        }
        if positions:
            decision = pack_rollout_payload({"positions": sorted(positions), "reason": reason}, args, -1)
            ray.get(args._rollout_queue_controller.record_dispositions.remote(decision.manifest))


def load_rollout_samples(value):
    """Read raw collections without changing group or trajectory nesting."""
    from slime.rollout.base_types import iter_samples

    groups = []
    for reference in unpack_rollout_payload(value):
        group = unpack_rollout_payload(reference)
        if isinstance(reference, RolloutGroupRef) and reference.receipt:
            for sample in iter_samples(group):
                sample.__dict__.pop("_queue_lease", None)
                sample._queue_receipt = asdict(reference.receipt)
        groups.append(group)
    return groups


def pack_rollout_payload(value, args, rollout_id, *, metadata=None, submission_id=None):
    if args.rollout_data_transport != "straw":
        return value
    store, codec, lock = rollout_store(args)
    if isinstance(value, DiskPayloadRef):
        store.validate(value.manifest)
        return value
    with lock:
        ref = codec.publish(
            value, submission_id=submission_id or f"payload:{rollout_id}:{uuid.uuid4().hex}", metadata=metadata
        )
    return DiskPayloadRef(ref, str(store.backend.root))


async def publish_rollout_async(value, args, rollout_id, *, group=False):
    return await run_rollout_io(args, pack_rollout_group if group else pack_rollout_payload, value, args, rollout_id)


async def run_rollout_io(args, function, *values):
    """Bound the executor backlog before submitting encoding/sync work."""
    loop = asyncio.get_running_loop()
    key = (os.getpid(), loop)
    if key not in _async_limits:
        _async_limits[key] = asyncio.Semaphore(getattr(args, "rollout_io_concurrency", 4))
    async with _async_limits[key]:
        pending = asyncio.create_task(asyncio.to_thread(function, *values))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # A cancelled coroutine cannot stop a filesystem commit in its thread.
            # Keep capacity and input ownership until the accepted write finishes.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    pass
            pending.result()
            raise


def unpack_rollout_payload(value):
    while isinstance(value, DiskPayloadRef):
        value = value.load()
    return value


def accept_raw_rollout(output, args, rollout_id):
    """Validate accepted producers, or commit a legacy collection once.

    A small wrapper adds task provenance to existing sealed collections without
    rewriting their Sample or tensor payloads.
    """
    from straw.protocol import Lease

    controller = args._rollout_queue_controller
    if isinstance(output, RawRolloutRef):
        receipt = ray.get(controller.accepted.remote(output.receipt))
        if output.manifest != receipt.result_ref:
            raise ValueError("Raw rollout manifest differs from its accepted receipt")
        return output
    # Legacy custom producers may borrow queue inputs and return Samples without
    # publishing task results themselves. Complete those inputs before accepting
    # the compatibility collection, so closing the reader cannot requeue them.
    from slime.rollout.base_types import iter_samples

    borrowed = {}
    for group in unpack_rollout_payload(output.samples):
        if isinstance(group, RolloutGroupRef) and group.receipt:
            continue
        group = unpack_rollout_payload(group)
        for sample in iter_samples(group):
            if value := getattr(sample, "_queue_lease", None):
                lease = Lease(**value)
                borrowed.setdefault(lease, []).append(sample)
    for lease, samples in borrowed.items():
        if ray.get(controller.result.remote(lease)) is None:
            pack_rollout_group(samples, args, rollout_id)
    lease = ray.get(controller.begin_collection.remote(str(rollout_id)))
    store, codec, lock = rollout_store(args)
    with lock:
        ref = codec.publish(
            output.samples,
            submission_id=f"collection:{lease.attempt_id}",
            metadata={"task_id": lease.task_id, "attempt_id": lease.attempt_id},
        )
    receipt = ray.get(controller.complete.remote(lease, ref))
    return RawRolloutRef(receipt.result_ref, str(store.backend.root), receipt, output.metrics)


def _storage_probe(reference, args):
    return pack_rollout_payload(reference.load(root=args.rollout_data_dir), args, 0)


def check_rollout_storage(args):
    marker = uuid.uuid4().hex
    reference = pack_rollout_payload(marker, args, 0)
    probes = [
        ray.remote(_storage_probe)
        .options(num_cpus=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False))
        .remote(reference, args)
        for node in ray.nodes()
        if node["Alive"] and (node["Resources"].get("CPU", 0) or node["Resources"].get("GPU", 0))
    ]
    try:
        for result in ray.get(probes, timeout=60):
            if result.load(root=args.rollout_data_dir) != marker:
                raise ValueError("Rollout storage probe content differs across nodes")
    except Exception as error:
        raise RuntimeError(
            f"All rollout/training nodes must share the run at --rollout-data-dir={args.rollout_data_dir}"
        ) from error
