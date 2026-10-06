"""Rollout references over immutable queue segments; no pickle payload protocol."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:
    from straw.protocol import CommitReceipt, RecordSetRef


def use_straw(args):
    return getattr(args, "rollout_data_transport", "object-store") == "straw"


_writers = {}
_writers_lock = threading.Lock()
_async_limits = {}


def seal_rollout_store(args):
    """Seal local packs after generation/publication work has been drained."""
    if use_straw(args):
        store, _, lock = rollout_store(args)
        with lock:
            store.seal()


def resolve_rollout_data_dir(args):
    debug_path = getattr(args, "load_debug_rollout_data", None)
    if debug_path and debug_path.endswith(".straw.json"):
        import glob
        from string import Formatter

        from straw.protocol import RecordSetRef

        start = getattr(args, "start_rollout_id", None)
        if start is None:
            # Model loading may determine the starting rollout later. Read only
            # an index to bind the shared pool before creating any actors.
            pattern = "".join(
                glob.escape(text) + ("*" if field else "") for text, field, _, _ in Formatter().parse(debug_path)
            )
            paths = sorted(glob.glob(str(Path(pattern).expanduser())))
            if not paths:
                raise FileNotFoundError(f"No Straw rollout archive matches {debug_path}")
            path = paths[0]
        else:
            path = Path(debug_path.format(rollout_id=start)).expanduser()
        index = json.loads(Path(path).read_text())
        if index.get("format") != "slime.straw-debug" or index.get("version") != 1:
            raise ValueError(f"Unsupported straw rollout archive: {path}")
        args.rollout_data_dir = index["root"]
        args.rollout_queue_run_id = RecordSetRef.from_dict(index["manifest"]).manifest.segment.run_id
    if args.rollout_data_dir is None:
        if args.save is None:
            raise ValueError("Straw rollout transport requires --rollout-data-dir or --save on shared storage")
        args.rollout_data_dir = str(Path(args.save) / "rollout_data")
    args.rollout_data_dir = str(Path(args.rollout_data_dir).expanduser().resolve())


def rollout_store(args):
    """One physical writer incarnation per process/run, shared by producer calls."""
    try:
        from straw.backend import FilesystemBackend
    except ModuleNotFoundError as error:
        if error.name != "straw":
            raise
        raise ModuleNotFoundError("Install straw with: pip install straw-queue", name="straw") from error
    from straw.store import SharedFilesystemStore
    from straw.tensor import MAX_PUBLICATION_BYTES, MAX_TENSOR_BYTES

    from slime.data.codec import CODECS, SampleCodec

    root = str(Path(args.rollout_data_dir).resolve())
    run_id = getattr(args, "rollout_queue_run_id", None) or "rollout"
    profile = getattr(args, "rollout_storage_profile", "local")
    declaration_path = getattr(args, "rollout_storage_declaration", None)
    declaration = json.loads(Path(declaration_path).read_text()) if declaration_path else None
    segment_mib = getattr(args, "rollout_queue_segment_mib", None)
    online_gc = getattr(args, "rollout_queue_online_gc", False)
    key = (os.getpid(), root, run_id, profile, segment_mib, online_gc, json.dumps(declaration, sort_keys=True))
    with _writers_lock:
        if key not in _writers:
            # Omit the override unless requested, so Straw owns its default.
            pack_options = {} if segment_mib is None else {"segment_target_bytes": segment_mib * 1024**2}
            store = SharedFilesystemStore(
                root,
                run_id,
                online_gc=online_gc,
                codecs=CODECS,
                max_record_bytes=MAX_TENSOR_BYTES,
                max_buffer_bytes=MAX_PUBLICATION_BYTES,
                backend=FilesystemBackend(root, profile=profile, declaration=declaration),
                **pack_options,
            )
            _writers[key] = (store, threading.RLock())
        store, lock = _writers[key]
        return store, SampleCodec(store, args=args), lock


@dataclass(frozen=True)
class DiskPayloadRef:
    """Bounded protocol ref plus a deployment mount hint, overridable by readers.

    SampleCodec persists the manifest, projection path and small metadata edits.
    Root is a local mount binding, never a dependency in a durable record.
    """

    manifest: RecordSetRef
    root: str
    # Select a completed group's generation inside its existing worker result.
    # Small CA scheduling/reward edits travel alongside the immutable reference.
    path: tuple = field(default=(), kw_only=True)
    sample_metadata: list[dict] | None = field(default=None, kw_only=True)

    def load(self, *, root=None, selection=None):
        from straw.store import SharedFilesystemStore
        from straw.tensor import MAX_PUBLICATION_BYTES, MAX_TENSOR_BYTES

        from slime.data.codec import CODECS, SampleCodec

        store = SharedFilesystemStore(
            root or self.root,
            self.manifest.manifest.segment.run_id,
            codecs=CODECS,
            max_record_bytes=MAX_TENSOR_BYTES,
            max_buffer_bytes=MAX_PUBLICATION_BYTES,
        )
        value = SampleCodec(store).load(self.manifest, selection=selection)
        for part in self.path:
            value = value[part] if isinstance(part, int) else getattr(value, part)
        if self.sample_metadata is not None:
            import copy
            from slime.utils.types import Sample

            samples = []
            if isinstance(value, Sample):
                value = [value]
            for sample, metadata in zip(value, self.sample_metadata, strict=True):
                sample = copy.copy(sample)
                updates = dict(metadata)
                if "metadata" in updates:
                    sample.metadata = {**(sample.metadata or {}), **updates.pop("metadata")}
                sample.__dict__.update(updates)
                samples.append(sample)
            value = samples
        return value


@dataclass(frozen=True)
class RolloutGroupRef(DiskPayloadRef):
    index: int
    receipt: CommitReceipt | None = None
    # A fork reuses the payload but supplies positions in its own accepted log.
    source_positions: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RawRolloutRef(DiskPayloadRef):
    receipt: CommitReceipt
    metrics: dict | None = None


@dataclass(frozen=True)
class TrainBatchRef(DiskPayloadRef):
    """One rank's view of shared conversion records, plus its trainer schedule.

    ``plan_digest`` identifies this layout, so ranks from different DP/mbs
    schedules cannot accidentally train together even when the data is shared.
    """

    batch_id: str
    rank: int
    plan_digest: str
    selection: dict | None = field(default=None, kw_only=True)
    schedule: dict | None = field(default=None, kw_only=True)

    def load(self, *, root=None):
        value = super().load(root=root, selection=self.selection)
        if self.schedule is not None:
            value.update(self.schedule)
        return value


def group_lease(group):
    from straw.protocol import Lease

    from slime.rollout.base_types import iter_samples

    samples = list(iter_samples(group))
    leases = [getattr(sample, "_queue_lease", None) for sample in samples]
    if not any(leases):
        return None
    if not all(value == leases[0] for value in leases):
        raise ValueError("A queue group must preserve one task authorization across all trajectories")
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
        "_queue_branch",
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


def record_generation_provenance(group):
    from slime.rollout.base_types import iter_samples

    for sample in iter_samples(group):
        start = getattr(sample, "_queue_generation_start", None)
        branch = getattr(sample, "_queue_branch", None)
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


def pack_rollout_group(group, args, rollout_id, *, controller=None):
    if getattr(args, "rollout_data_transport", "object-store") != "straw":
        return group
    from slime.rollout.base_types import iter_samples

    first = next(iter_samples(group))
    lease = group_lease(group)
    record_generation_provenance(group)
    metadata = {"task_id": lease.task_id, "attempt_id": lease.attempt_id} if lease else {}
    ref = pack_rollout_payload(
        group, args, rollout_id, metadata=metadata, submission_id=f"group:{lease.attempt_id}" if lease else None
    )
    receipt = None
    if lease:
        if controller is None:
            raise ValueError("Queue group has a lease but no coordinator binding")
        receipt = ray.get(controller.complete.remote(lease, ref.manifest))
    return RolloutGroupRef(receipt.result_ref if receipt else ref.manifest, ref.root, first.index, receipt)


def release_rollout_publications(group, args):
    """Relinquish tensor staging before ending the group's current read lifetime."""
    if use_straw(args):
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


def discard_rollout_group(group, args, reason="dynamic filter", *, controller=None):
    lease = group_lease(group)
    release_rollout_publications(group, args)
    # Release staging while the task/accepted-result owner still protects any
    # previously adopted continuation, then acknowledge that reading is done.
    if lease:
        ray.get(controller.reject.remote(lease, reason))
    else:
        from slime.rollout.base_types import iter_samples

        positions = {
            sample._queue_receipt["position"] for sample in iter_samples(group) if hasattr(sample, "_queue_receipt")
        }
        if positions:
            decision = pack_rollout_payload({"positions": sorted(positions), "reason": reason}, args, -1)
            ray.get(controller.record_dispositions.remote(decision.manifest))


def load_rollout_samples(value):
    """Read raw collections without changing group or trajectory nesting."""
    from slime.rollout.base_types import iter_samples

    def read_group(reference):
        group = unpack_rollout_payload(reference)
        if isinstance(group, list):
            # CA selects several existing worker results for one prompt group.
            expanded = []
            for child in group:
                if isinstance(child, DiskPayloadRef):
                    expanded.extend(read_group(child))
                else:
                    expanded.append(child)
            group = expanded
        if isinstance(reference, RolloutGroupRef) and reference.receipt:
            for sample in iter_samples(group):
                if reference.source_positions is not None:
                    sample._queue_source_positions = list(reference.source_positions)
                sample.__dict__.pop("_queue_lease", None)
                sample._queue_receipt = asdict(reference.receipt)
        return group

    return [read_group(reference) for reference in unpack_rollout_payload(value)]


def pack_rollout_payload(value, args, rollout_id, *, metadata=None, submission_id=None):
    if not use_straw(args):
        return value
    store, codec, lock = rollout_store(args)
    if isinstance(value, DiskPayloadRef) and value.root is not None:
        store.validate(value.manifest)
        return value
    with lock:
        ref = codec.publish(
            value, submission_id=submission_id or f"payload:{rollout_id}:{uuid.uuid4().hex}", metadata=metadata
        )
    return DiskPayloadRef(ref, str(store.backend.root))


async def publish_rollout_async(value, args, rollout_id, *, group=False, controller=None):
    if group:
        return await run_rollout_io(args, pack_rollout_group, value, args, rollout_id, controller=controller)
    return await run_rollout_io(args, pack_rollout_payload, value, args, rollout_id)


async def run_rollout_io(args, function, *values, **kwargs):
    """Bound the executor backlog before submitting encoding/sync work."""
    loop = asyncio.get_running_loop()
    key = (os.getpid(), loop)
    if key not in _async_limits:
        _async_limits[key] = asyncio.Semaphore(getattr(args, "rollout_io_concurrency", 4))
    async with _async_limits[key]:
        pending = asyncio.create_task(asyncio.to_thread(function, *values, **kwargs))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Repeated cancellation still cannot stop a filesystem commit.
            # Keep capacity and input ownership until the accepted write finishes.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    pass
            pending.result()
            raise


def release_payload_publications(args, payloads):
    """Release transfer staging after readers finish and durable owners adopt it."""
    if not use_straw(args):
        return
    refs = {payload.manifest.digest: payload.manifest for payload in payloads if isinstance(payload, DiskPayloadRef)}
    store, _, lock = rollout_store(args)
    refs = list(refs.values())
    with lock:
        for offset in range(0, len(refs), 128):
            store.release_publications(refs[offset : offset + 128])


def unpack_rollout_payload(value):
    while isinstance(value, DiskPayloadRef):
        value = value.load()
    return value


async def unpack_published_payload(value):
    """Load off the event loop; Straw handles visibility for every read path."""
    if not isinstance(value, DiskPayloadRef):
        return await asyncio.to_thread(unpack_rollout_payload, value)
    from straw.errors import CorruptData

    segment = value.manifest.manifest.segment
    context = f"submitted manifest: root={value.root}, segment={segment.path}, offset={segment.offset}"
    try:
        return await asyncio.to_thread(unpack_rollout_payload, value)
    except CorruptData as error:
        error.add_note(context)
        raise


def accept_raw_rollout(output, args, rollout_id, *, controller):
    """Validate an accepted collection, or persist and accept returned Samples.

    A small wrapper adds task provenance to existing sealed collections without
    rewriting their Sample or tensor payloads.
    """
    from straw.protocol import Lease

    if isinstance(output, RawRolloutRef):
        receipt = ray.get(controller.accepted.remote(output.receipt))
        if output.manifest != receipt.result_ref:
            raise ValueError("Raw rollout manifest differs from its accepted receipt")
        return output
    # Custom producers can return Samples without publishing task results.
    # Complete their borrowed inputs before accepting the collection, so closing
    # the reader cannot requeue work that has already finished.
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
            pack_rollout_group(samples, args, rollout_id, controller=controller)
    lease = ray.get(controller.begin_collection.remote(str(rollout_id)))
    store, codec, lock = rollout_store(args)
    with lock:
        ref = codec.publish(
            output.sample_refs if getattr(output, "sample_refs", None) is not None else output.samples,
            submission_id=f"collection:{lease.attempt_id}",
            metadata={"task_id": lease.task_id, "attempt_id": lease.attempt_id},
        )
    receipt = ray.get(controller.complete.remote(lease, ref))
    if output.on_accepted is not None:
        output.on_accepted()
    return RawRolloutRef(receipt.result_ref, str(store.backend.root), receipt, output.metrics)


def _storage_probe(reference, args):
    return pack_rollout_payload(reference.load(root=args.rollout_data_dir), args, 0)


def check_rollout_storage(args):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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
        results = ray.get(probes, timeout=60)
        for result in results:
            if result.load(root=args.rollout_data_dir) != marker:
                raise ValueError("Rollout storage probe content differs across nodes")
        release_payload_publications(args, [reference, *results])
    except Exception as error:
        raise RuntimeError(
            f"All rollout/training nodes must share the run at --rollout-data-dir={args.rollout_data_dir}"
        ) from error
