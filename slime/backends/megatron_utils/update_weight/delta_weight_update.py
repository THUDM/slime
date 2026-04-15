from __future__ import annotations

import logging
import math
import os
import queue
import threading
from dataclasses import dataclass, field

import torch
from safetensors.torch import save, save_file

try:
    import zstandard
except ImportError:
    zstandard = None


_DELTA_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

logger = logging.getLogger(__name__)


@dataclass
class DeltaCompressionCommitState:
    baseline_updates: list[tuple[str, torch.Tensor]]
    artifact_tensors: list[tuple[str, torch.Tensor]] = field(default_factory=list)


@dataclass
class PreparedChunk:
    is_delta: bool
    tensors: list[tuple[str, torch.Tensor]]
    commit_state: DeltaCompressionCommitState


@dataclass
class MaterializedDeltaTransport:
    tensors: list[tuple[str, torch.Tensor]]
    sparse_metadata: list[dict] | None
    load_format: str
    skipped_zero: int = 0


class DeltaArtifactWriter:
    def __init__(self, artifact_dir: str) -> None:
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.queue: queue.Queue[tuple[int, int, dict[str, torch.Tensor]] | None] = queue.Queue()
        self.compressor = zstandard.ZstdCompressor() if zstandard is not None else None
        self.thread = threading.Thread(target=self.run, name="delta-artifact-writer", daemon=True)
        self.thread.start()

    def enqueue(self, weight_version: int, chunk_idx: int, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        if named_tensors:
            self.queue.put((weight_version, chunk_idx, {name: t.contiguous() for name, t in named_tensors}))

    def run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            weight_version, chunk_idx, tensors = item
            stem = f"weight_version_{weight_version:06d}_chunk_{chunk_idx:06d}.safetensors"
            if self.compressor is None:
                save_file(tensors, os.path.join(self.artifact_dir, stem))
                continue

            payload = save(tensors)
            compressed = self.compressor.compress(payload)
            with open(os.path.join(self.artifact_dir, f"{stem}.zst"), "wb") as f:
                f.write(compressed)


class DeltaCompressionTracker:
    def __init__(self, args) -> None:
        self.delta_dtype = _DELTA_DTYPE_MAP[args.delta_compression_dtype]
        self.full_sync_interval = args.delta_compression_full_sync_interval
        if self.full_sync_interval < 1:
            raise ValueError("--delta-compression-full-sync-interval must be >= 1")
        self.baseline: dict[str, torch.Tensor] = {}  # pinned CPU tensors
        self.committed_syncs = 0
        self.chunk_idx = 0
        self.artifact_writer = (
            DeltaArtifactWriter(args.delta_compression_artifact_dir)
            if args.delta_compression_artifact_dir is not None
            else None
        )
        self._stats = None
        self._baseline_dirty = False  # True when async D2H copies are in-flight
        self._d2h_stream: torch.cuda.Stream | None = None  # lazy-init secondary stream for D2H copies

    def prepare_chunk(self, tensors: list[tuple[str, torch.Tensor]]) -> PreparedChunk:
        if self._should_send_full_chunk():
            return self._prepare_full_chunk(tensors)
        return self._prepare_delta_chunk(tensors)

    def flush_baseline(self) -> None:
        """Synchronize any in-flight baseline D2H copies.

        Called before the next H2D transfer (in _prepare_delta_chunk) or at the
        end of a sync to ensure baselines are consistent before reuse.
        """
        if self._baseline_dirty:
            if self._d2h_stream is not None:
                self._d2h_stream.synchronize()
            else:
                torch.cuda.synchronize()
            self._baseline_dirty = False

    def commit_chunk(self, commit_state: DeltaCompressionCommitState, *, weight_version: int) -> None:
        baseline_updates = commit_state.baseline_updates
        self._ensure_stats()
        # baseline_updates holds (name, current_gpu_tensor) references.
        # Save to pinned CPU async — synchronize is deferred to flush_baseline()
        # which runs before the next H2D transfer or at sync end, avoiding a
        # per-chunk cuda synchronize that serializes the pipeline.
        for name, tensor in baseline_updates:
            if name not in self.baseline:
                self.baseline[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self.baseline[name].copy_(tensor.detach(), non_blocking=True)
        self._baseline_dirty = True

        if self.artifact_writer is not None:
            self.artifact_writer.enqueue(weight_version, self.chunk_idx, commit_state.artifact_tensors)
        self.chunk_idx += 1

    def on_sync_succeeded(self) -> None:
        self.flush_baseline()  # ensure all D2H copies are complete before next sync
        self.committed_syncs += 1
        self._log_stats()
        self._stats = None  # reset for next sync so profiling is per-sync

    def _prepare_full_chunk(self, tensors: list[tuple[str, torch.Tensor]]) -> PreparedChunk:
        self._ensure_stats()
        self._stats["full_chunks"] += 1
        self._stats["full_tensors"] += len(tensors)
        self._stats["full_bytes"] += sum(t.numel() * t.element_size() for _, t in tensors)
        return PreparedChunk(
            is_delta=False,
            tensors=tensors,
            commit_state=DeltaCompressionCommitState(baseline_updates=list(tensors)),
        )

    def _prepare_delta_chunk(self, tensors: list[tuple[str, torch.Tensor]]) -> PreparedChunk:
        import time as _t

        self._ensure_stats()
        # Ensure any in-flight D2H copies have landed before we read baselines.
        _t0 = _t.monotonic()
        self.flush_baseline()
        _t_flush = _t.monotonic() - _t0

        # Batch H2D: load all baselines to GPU with non_blocking, one sync.
        _t0 = _t.monotonic()
        prev_gpu_tensors = []
        for name, tensor in tensors:
            if name not in self.baseline:
                raise KeyError(f"delta baseline missing tensor {name!r}; run a full sync before delta sync resumes")
            prev_gpu_tensors.append(self.baseline[name].to(device=tensor.device, non_blocking=True))
        torch.cuda.synchronize()
        _t_h2d = _t.monotonic() - _t0

        # Lazy-init a secondary CUDA stream for D2H baseline copies.
        # D2H on a separate stream runs concurrently with delta compute on
        # the default stream, keeping the fast 22.6s pipeline intact.
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream()

        _t0 = _t.monotonic()
        delta_tensors = []
        artifact_tensors = []
        input_tensor_count = 0
        sent_tensor_count = 0
        for (name, tensor), prev_gpu in zip(tensors, prev_gpu_tensors, strict=True):
            delta = (tensor - prev_gpu).to(self.delta_dtype)
            del prev_gpu
            input_tensor_count += 1
            # Commit baseline via D2H on secondary stream so it runs
            # concurrently with subsequent delta compute on the default
            # stream. The event ensures correct ordering and keeps the
            # storage alive until the D2H finishes — no GPU refs needed
            # in baseline_updates, so gathered buffers are freed promptly.
            if name not in self.baseline:
                self.baseline[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            event = torch.cuda.current_stream().record_event()
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(event)
                self.baseline[name].copy_(tensor.detach(), non_blocking=True)
            delta_tensors.append((name, delta))
            sent_tensor_count += 1
            if self.artifact_writer is not None:
                artifact_tensors.append((name, delta.cpu()))
        self._baseline_dirty = True
        _t_subtract = _t.monotonic() - _t0

        logger.info(
            "delta_profile: prepare_chunk flush=%.3fs h2d=%.3fs subtract=%.3fs tensors=%s",
            _t_flush,
            _t_h2d,
            _t_subtract,
            input_tensor_count,
        )

        self._stats["delta_chunks"] += 1
        self._stats["delta_input_tensors"] += input_tensor_count
        self._stats["delta_sent_tensors"] += sent_tensor_count
        self._stats["delta_sent_bytes"] += sum(t.numel() * t.element_size() for _, t in delta_tensors)

        return PreparedChunk(
            is_delta=True,
            tensors=delta_tensors,
            commit_state=DeltaCompressionCommitState(
                baseline_updates=[],  # committed inline above, no GPU refs stored
                artifact_tensors=artifact_tensors,
            ),
        )

    def _should_send_full_chunk(self) -> bool:
        return self.committed_syncs == 0 or self.committed_syncs % self.full_sync_interval == 0

    def _ensure_stats(self) -> None:
        if self._stats is None:
            self._stats = {
                "full_chunks": 0,
                "full_tensors": 0,
                "full_bytes": 0,
                "delta_chunks": 0,
                "delta_input_tensors": 0,
                "delta_sent_tensors": 0,
                "delta_sent_bytes": 0,
            }

    def _log_stats(self) -> None:
        if self._stats is None:
            return

        logger.info(
            "delta_weight_update_summary full_chunks=%s full_tensors=%s full_bytes=%s delta_chunks=%s "
            "delta_input_tensors=%s delta_sent_tensors=%s delta_sent_bytes=%s",
            self._stats["full_chunks"],
            self._stats["full_tensors"],
            self._stats["full_bytes"],
            self._stats["delta_chunks"],
            self._stats["delta_input_tensors"],
            self._stats["delta_sent_tensors"],
            self._stats["delta_sent_bytes"],
        )


def get_delta_load_format(transport: str) -> str:
    if transport == "dense":
        return "distributed_delta"
    if transport == "sparse_indices":
        return "distributed_delta_sparse_indices"
    if transport == "sparse_bitmask":
        return "distributed_delta_sparse_bitmask"
    raise ValueError(f"Unsupported delta compression transport: {transport}")


def estimate_delta_transport_byte_size(
    tensors: list[tuple[str, torch.Tensor]],
    transport: str,
) -> int:
    if transport == "dense":
        return sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)

    total_bytes = 0
    nonzero_tensor_count = 0
    # ~200 bytes per sparse metadata entry (name, dtype, shape, offsets)
    _METADATA_BYTES_PER_TENSOR = 200
    for _name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        nnz = int(torch.count_nonzero(flat).item())
        if nnz == 0:
            continue  # zero-delta tensors are skipped in materialization
        nonzero_tensor_count += 1
        value_bytes = nnz * tensor.element_size()
        if transport == "sparse_indices":
            total_bytes += nnz * torch.tensor([], dtype=torch.int32).element_size() + value_bytes
            continue
        if transport == "sparse_bitmask":
            total_bytes += int(math.ceil(flat.numel() / 8)) + value_bytes
            continue
        raise ValueError(f"Unsupported delta compression transport: {transport}")
    total_bytes += nonzero_tensor_count * _METADATA_BYTES_PER_TENSOR
    return total_bytes


def materialize_delta_transport(
    tensors: list[tuple[str, torch.Tensor]],
    transport: str,
) -> MaterializedDeltaTransport:
    if transport == "dense":
        return MaterializedDeltaTransport(
            tensors=list(tensors),
            sparse_metadata=None,
            load_format="distributed_delta",
        )

    if transport == "sparse_indices":
        return _materialize_sparse_indices_transport(tensors)

    if transport == "sparse_bitmask":
        return _materialize_sparse_bitmask_transport(tensors)

    raise ValueError(f"Unsupported delta compression transport: {transport}")


def _materialize_sparse_indices_transport(
    tensors: list[tuple[str, torch.Tensor]],
) -> MaterializedDeltaTransport:
    if not tensors:
        return MaterializedDeltaTransport(
            tensors=[],
            sparse_metadata=None,
            load_format="distributed_delta_sparse_indices",
        )

    all_indices = []
    all_values = []
    sparse_metadata: list[dict] = []
    index_offset = 0
    value_offset = 0
    skipped_zero = 0
    for name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        indices_long = torch.nonzero(flat, as_tuple=False).view(-1)
        nnz = int(indices_long.numel())
        if nnz == 0:
            skipped_zero += 1
            continue
        values = flat[indices_long]
        indices = indices_long.to(dtype=torch.int32)
        sparse_metadata.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "numel": flat.numel(),
                "nnz": nnz,
                "index_start": index_offset,
                "index_end": index_offset + nnz,
                "value_start": value_offset,
                "value_end": value_offset + nnz,
            }
        )
        all_indices.append(indices)
        all_values.append(values)
        index_offset += nnz
        value_offset += nnz
    device = tensors[0][1].device
    value_dtype = tensors[0][1].dtype
    packed_indices = torch.cat(all_indices, dim=0) if all_indices else torch.empty(0, dtype=torch.int32, device=device)
    packed_values = torch.cat(all_values, dim=0) if all_values else torch.empty(0, dtype=value_dtype, device=device)
    return MaterializedDeltaTransport(
        tensors=[
            ("__packed_indices__", packed_indices),
            ("__packed_values__", packed_values),
        ],
        sparse_metadata=sparse_metadata,
        load_format="distributed_delta_sparse_indices",
        skipped_zero=skipped_zero,
    )


def _materialize_sparse_bitmask_transport(
    tensors: list[tuple[str, torch.Tensor]],
) -> MaterializedDeltaTransport:
    if not tensors:
        return MaterializedDeltaTransport(
            tensors=[],
            sparse_metadata=None,
            load_format="distributed_delta_sparse_bitmask",
        )

    all_masks = []
    all_values = []
    sparse_metadata: list[dict] = []
    mask_offset = 0
    value_offset = 0
    skipped_zero = 0
    for name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        mask = flat != 0
        values = flat[mask]
        nnz = int(values.numel())
        if nnz == 0:
            skipped_zero += 1
            continue
        packed_mask = _pack_bitmask(mask)
        mask_numel = int(packed_mask.numel())
        sparse_metadata.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "numel": flat.numel(),
                "nnz": nnz,
                "mask_start": mask_offset,
                "mask_end": mask_offset + mask_numel,
                "value_start": value_offset,
                "value_end": value_offset + nnz,
            }
        )
        all_masks.append(packed_mask)
        all_values.append(values)
        mask_offset += mask_numel
        value_offset += nnz
    device = tensors[0][1].device
    value_dtype = tensors[0][1].dtype
    packed_masks = torch.cat(all_masks, dim=0) if all_masks else torch.empty(0, dtype=torch.uint8, device=device)
    packed_values = torch.cat(all_values, dim=0) if all_values else torch.empty(0, dtype=value_dtype, device=device)
    return MaterializedDeltaTransport(
        tensors=[
            ("__packed_masks__", packed_masks),
            ("__packed_values__", packed_values),
        ],
        sparse_metadata=sparse_metadata,
        load_format="distributed_delta_sparse_bitmask",
        skipped_zero=skipped_zero,
    )


def _pack_bitmask(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=mask.device)
    mask_u8 = mask.to(dtype=torch.uint8)
    pad = (-mask_u8.numel()) % 8
    if pad:
        mask_u8 = torch.cat(
            [mask_u8, torch.zeros(pad, dtype=torch.uint8, device=mask_u8.device)],
            dim=0,
        )
    bits = mask_u8.view(-1, 8)
    weights = (2 ** torch.arange(8, dtype=torch.uint8, device=mask_u8.device)).view(1, 8)
    return torch.sum(bits * weights, dim=1, dtype=torch.uint8)
