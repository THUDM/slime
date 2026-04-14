from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
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

    def prepare_chunk(self, tensors: list[tuple[str, torch.Tensor]]) -> PreparedChunk:
        if self._should_send_full_chunk():
            return self._prepare_full_chunk(tensors)
        return self._prepare_delta_chunk(tensors)

    def commit_chunk(self, commit_state: DeltaCompressionCommitState, *, weight_version: int) -> None:
        baseline_updates = commit_state.baseline_updates
        self._ensure_stats()
        t_commit_start = time.monotonic()
        # baseline_updates holds (name, current_gpu_tensor) references.
        # Save to pinned CPU async, one sync at end — same pattern as TensorBackuper.backup().
        for name, tensor in baseline_updates:
            if name not in self.baseline:
                self.baseline[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self.baseline[name].copy_(tensor.detach(), non_blocking=True)
        torch.cuda.synchronize()
        self._stats["profile_baseline_commit_s"] += time.monotonic() - t_commit_start

        if self.artifact_writer is not None:
            self.artifact_writer.enqueue(weight_version, self.chunk_idx, commit_state.artifact_tensors)
        self.chunk_idx += 1

    def on_sync_succeeded(self) -> None:
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
        self._ensure_stats()
        t_h2d_start = time.monotonic()
        prev_gpu_tensors = []
        for name, tensor in tensors:
            if name not in self.baseline:
                raise KeyError(f"delta baseline missing tensor {name!r}; run a full sync before delta sync resumes")
            prev_gpu_tensors.append(self.baseline[name].to(device=tensor.device, non_blocking=True))
        torch.cuda.synchronize()
        self._stats["profile_baseline_h2d_s"] += time.monotonic() - t_h2d_start

        t_compute_start = time.monotonic()
        delta_tensors = []
        artifact_tensors = []
        baseline_updates = []
        input_tensor_count = 0
        sent_tensor_count = 0
        chunk_elements = 0
        chunk_nonzeros = 0
        for (name, tensor), prev_gpu in zip(tensors, prev_gpu_tensors, strict=True):
            delta = (tensor - prev_gpu).to(self.delta_dtype)
            input_tensor_count += 1
            numel = delta.numel()
            nnz = int(torch.count_nonzero(delta).item())
            chunk_elements += numel
            chunk_nonzeros += nnz

            baseline_updates.append((name, tensor))
            delta_tensors.append((name, delta))
            sent_tensor_count += 1
            if self.artifact_writer is not None:
                artifact_tensors.append((name, delta.cpu()))
        self._stats["profile_delta_compute_s"] += time.monotonic() - t_compute_start
        self._stats["profile_sparsity_total_elements"] += chunk_elements
        self._stats["profile_sparsity_total_nonzeros"] += chunk_nonzeros

        self._stats["delta_chunks"] += 1
        self._stats["delta_input_tensors"] += input_tensor_count
        self._stats["delta_sent_tensors"] += sent_tensor_count
        self._stats["delta_sent_bytes"] += sum(t.numel() * t.element_size() for _, t in delta_tensors)

        return PreparedChunk(
            is_delta=True,
            tensors=delta_tensors,
            commit_state=DeltaCompressionCommitState(
                baseline_updates=baseline_updates,
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
                # -- profiling (temporary) --
                "profile_baseline_h2d_s": 0.0,
                "profile_delta_compute_s": 0.0,
                "profile_baseline_commit_s": 0.0,
                "profile_sparsity_total_elements": 0,
                "profile_sparsity_total_nonzeros": 0,
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
        total_elem = self._stats["profile_sparsity_total_elements"]
        total_nnz = self._stats["profile_sparsity_total_nonzeros"]
        density = total_nnz / total_elem if total_elem > 0 else 0.0
        logger.info(
            "delta_profile: baseline_h2d=%.3fs delta_compute=%.3fs baseline_commit=%.3fs "
            "sparsity_elements=%s sparsity_nonzeros=%s density=%.6f",
            self._stats["profile_baseline_h2d_s"],
            self._stats["profile_delta_compute_s"],
            self._stats["profile_baseline_commit_s"],
            total_elem,
            total_nnz,
            density,
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
    t_start = time.monotonic()
    if transport == "dense":
        total = sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)
        logger.info(
            "delta_profile: estimate_byte_size=%.3fs tensors=%s transport=dense estimated_bytes=%s",
            time.monotonic() - t_start,
            len(tensors),
            total,
        )
        return total

    total_bytes = 0
    for _, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        nnz = int(torch.count_nonzero(flat).item())
        value_bytes = nnz * tensor.element_size()
        if transport == "sparse_indices":
            total_bytes += nnz * torch.tensor([], dtype=torch.int32).element_size() + value_bytes
            continue
        if transport == "sparse_bitmask":
            total_bytes += int(math.ceil(flat.numel() / 8)) + value_bytes
            continue
        raise ValueError(f"Unsupported delta compression transport: {transport}")
    logger.info(
        "delta_profile: estimate_byte_size=%.3fs tensors=%s transport=%s estimated_bytes=%s",
        time.monotonic() - t_start,
        len(tensors),
        transport,
        total_bytes,
    )
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
    for name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        indices_long = torch.nonzero(flat, as_tuple=False).view(-1)
        values = flat[indices_long]
        indices = indices_long.to(dtype=torch.int32)
        nnz = int(indices.numel())
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
    for name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        mask = flat != 0
        packed_mask = _pack_bitmask(mask)
        values = flat[mask]
        nnz = int(values.numel())
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
