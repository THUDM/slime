from __future__ import annotations

import logging
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
        if self.committed_syncs == 0:
            return self._prepare_initial_full_chunk(tensors)
        return self._prepare_delta_chunk(tensors)

    def commit_chunk(self, commit_state: DeltaCompressionCommitState, *, weight_version: int) -> None:
        baseline_updates = commit_state.baseline_updates
        # baseline_updates holds (name, current_gpu_tensor) references.
        # Save to pinned CPU async, one sync at end — same pattern as TensorBackuper.backup().
        for name, tensor in baseline_updates:
            if name not in self.baseline:
                self.baseline[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self.baseline[name].copy_(tensor.detach(), non_blocking=True)
        torch.cuda.synchronize()

        if self.artifact_writer is not None:
            self.artifact_writer.enqueue(weight_version, self.chunk_idx, commit_state.artifact_tensors)
        self.chunk_idx += 1

    def on_sync_succeeded(self) -> None:
        self.committed_syncs += 1
        self._log_stats()

    def _prepare_initial_full_chunk(self, tensors: list[tuple[str, torch.Tensor]]) -> PreparedChunk:
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
        prev_gpu_tensors = []
        for name, tensor in tensors:
            if name not in self.baseline:
                raise KeyError(f"delta baseline missing tensor {name!r}; run a full sync before delta sync resumes")
            prev_gpu_tensors.append(self.baseline[name].to(device=tensor.device, non_blocking=True))
        torch.cuda.synchronize()

        delta_tensors = []
        artifact_tensors = []
        baseline_updates = []
        input_tensor_count = 0
        sent_tensor_count = 0
        exact_zero_tensor_count = 0
        total_elements = 0
        nonzero_elements = 0
        max_abs = 0.0
        for (name, tensor), prev_gpu in zip(tensors, prev_gpu_tensors, strict=True):
            delta = (tensor - prev_gpu).to(self.delta_dtype)
            input_tensor_count += 1
            total_elements += delta.numel()

            baseline_updates.append((name, tensor))

            abs_max = float(delta.abs().max().item()) if delta.numel() > 0 else 0.0
            max_abs = max(max_abs, abs_max)
            nonzero_elements += int(delta.count_nonzero().item()) if delta.numel() > 0 else 0

            if delta.numel() == 0 or abs_max == 0.0:
                exact_zero_tensor_count += 1
                continue

            delta_tensors.append((name, delta))
            sent_tensor_count += 1
            if self.artifact_writer is not None:
                artifact_tensors.append((name, delta.cpu()))

        self._stats["delta_chunks"] += 1
        self._stats["delta_input_tensors"] += input_tensor_count
        self._stats["delta_sent_tensors"] += sent_tensor_count
        self._stats["delta_exact_zero_tensors"] += exact_zero_tensor_count
        self._stats["delta_total_elements"] += total_elements
        self._stats["delta_nonzero_elements"] += nonzero_elements
        self._stats["delta_sent_bytes"] += sum(t.numel() * t.element_size() for _, t in delta_tensors)
        self._stats["delta_max_abs"] = max(self._stats["delta_max_abs"], max_abs)

        return PreparedChunk(
            is_delta=True,
            tensors=delta_tensors,
            commit_state=DeltaCompressionCommitState(
                baseline_updates=baseline_updates,
                artifact_tensors=artifact_tensors,
            ),
        )

    def _ensure_stats(self) -> None:
        if self._stats is None:
            self._stats = {
                "full_chunks": 0,
                "full_tensors": 0,
                "full_bytes": 0,
                "delta_chunks": 0,
                "delta_input_tensors": 0,
                "delta_sent_tensors": 0,
                "delta_exact_zero_tensors": 0,
                "delta_total_elements": 0,
                "delta_nonzero_elements": 0,
                "delta_sent_bytes": 0,
                "delta_max_abs": 0.0,
            }

    def _log_stats(self) -> None:
        if self._stats is None:
            return

        total_elements = self._stats["delta_total_elements"]
        delta_nonzero_ratio = self._stats["delta_nonzero_elements"] / total_elements if total_elements else 0.0
        delta_exact_zero_tensor_ratio = (
            self._stats["delta_exact_zero_tensors"] / self._stats["delta_input_tensors"]
            if self._stats["delta_input_tensors"]
            else 0.0
        )
        logger.info(
            "delta_sync_summary full_chunks=%s full_tensors=%s full_bytes=%s delta_chunks=%s "
            "delta_input_tensors=%s delta_sent_tensors=%s delta_exact_zero_tensors=%s "
            "delta_exact_zero_tensor_ratio=%.4f delta_nonzero_ratio=%.6f delta_sent_bytes=%s delta_max_abs=%.6g",
            self._stats["full_chunks"],
            self._stats["full_tensors"],
            self._stats["full_bytes"],
            self._stats["delta_chunks"],
            self._stats["delta_input_tensors"],
            self._stats["delta_sent_tensors"],
            self._stats["delta_exact_zero_tensors"],
            delta_exact_zero_tensor_ratio,
            delta_nonzero_ratio,
            self._stats["delta_sent_bytes"],
            self._stats["delta_max_abs"],
        )
