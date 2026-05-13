from __future__ import annotations

import os
import threading
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from queue import Queue

import torch
import torch.distributed as dist
from safetensors.torch import save, save_file
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import PartialWeightEncoding, PartialWeightParam, PartialWeightSpec
from .update_weight_from_distributed import UpdateWeightFromDistributed

try:
    import zstandard
except ImportError:
    zstandard = None


_DELTA_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class PartialChunk:
    """
    One encoded chunk awaiting PartialSendBucket.
    dense → tensors=[(name, payload)], params=None.
    sparse_* → tensors=[("__packed_keys__", ...), ("__packed_values__", ...)],
    params holds per-param decoding with chunk-local offsets.
    """

    tensors: list[tuple[str, torch.Tensor]]
    params: list[PartialWeightParam] | None
    byte_size: int


def encode_partial(
    named_tensors: list[tuple[str, torch.Tensor]],
    encoding: PartialWeightEncoding,
    mode: str,
) -> PartialChunk:
    """
    Encode partial-update tensors per wire encoding.

    mode='delta': named_tensors values are deltas with 0 at unchanged positions.
    mode='selective': named_tensors values are new param values with NaN at unchanged.

    The sparse encoders use the corresponding active-position predicate.
    """
    if encoding is PartialWeightEncoding.DENSE:
        size = sum(t.numel() * t.element_size() for _, t in named_tensors)
        return PartialChunk(tensors=list(named_tensors), params=None, byte_size=size)
    if mode == "delta":
        is_active = lambda flat: flat != 0  # noqa: E731
    elif mode == "selective":
        is_active = lambda flat: ~torch.isnan(flat)  # noqa: E731
    else:
        raise ValueError(f"unknown partial-update mode: {mode!r}")
    if encoding is PartialWeightEncoding.SPARSE_INDICES:
        return _encode_sparse(named_tensors, _make_indices_kv(is_active))
    if encoding is PartialWeightEncoding.SPARSE_BITMASK:
        return _encode_sparse(named_tensors, _make_bitmask_kv(is_active))
    raise ValueError(f"unknown partial-update encoding: {encoding!r}")


def _encode_sparse(
    named_tensors: list[tuple[str, torch.Tensor]],
    kv_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> PartialChunk:
    """
    Walk named_tensors, ask kv_fn for (keys, values) per param, pack into a
    single (packed_keys, packed_values) PartialChunk with a per-param manifest.
    Params with zero active positions are skipped.
    """
    keys_chunks: list[torch.Tensor] = []
    values_chunks: list[torch.Tensor] = []
    params: list[PartialWeightParam] = []
    keys_off = values_off = 0
    for name, tensor in named_tensors:
        flat = tensor.contiguous().view(-1)
        keys, values = kv_fn(flat)
        nnz = int(values.numel())
        if nnz == 0:
            continue
        keys_count = int(keys.numel())
        params.append(
            PartialWeightParam(
                name=name,
                dtype=str(tensor.dtype).replace("torch.", ""),
                shape=list(tensor.shape),
                keys_start=keys_off,
                keys_end=keys_off + keys_count,
                values_start=values_off,
                values_end=values_off + nnz,
            )
        )
        keys_chunks.append(keys)
        values_chunks.append(values)
        keys_off += keys_count
        values_off += nnz
    if not params:
        return PartialChunk(tensors=[], params=[], byte_size=0)
    packed_keys = torch.cat(keys_chunks, dim=0)
    packed_values = torch.cat(values_chunks, dim=0)
    size = packed_keys.numel() * packed_keys.element_size() + packed_values.numel() * packed_values.element_size()
    return PartialChunk(
        tensors=[("__packed_keys__", packed_keys), ("__packed_values__", packed_values)],
        params=params,
        byte_size=size,
    )


def _make_indices_kv(is_active):
    def kv(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx_long = is_active(flat).nonzero(as_tuple=False).view(-1)
        return idx_long.to(dtype=torch.int32), flat[idx_long]

    return kv


def _make_bitmask_kv(is_active):
    def kv(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = is_active(flat)
        values = flat[mask]
        if mask.numel() == 0:
            return torch.empty(0, dtype=torch.uint8, device=mask.device), values
        mask_u8 = mask.to(dtype=torch.uint8)
        pad = (-mask_u8.numel()) % 8
        if pad:
            mask_u8 = torch.cat([mask_u8, torch.zeros(pad, dtype=torch.uint8, device=mask_u8.device)])
        bits = mask_u8.view(-1, 8)
        weights = (2 ** torch.arange(8, dtype=torch.uint8, device=mask_u8.device)).view(1, 8)
        return torch.sum(bits * weights, dim=1, dtype=torch.uint8), values

    return kv


@dataclass
class PartialSendBucket:
    """
    Accumulates PartialChunks for one batched broadcast. Sparse params are
    eagerly shifted into merged-buffer coordinates on add().
    """

    tensors: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    params: list[PartialWeightParam] = field(default_factory=list)
    byte_size: int = 0
    keys_total: int = 0
    values_total: int = 0

    @property
    def has_updates(self) -> bool:
        return bool(self.tensors)

    def should_flush_before_add(self, update: PartialChunk, byte_limit: int) -> bool:
        return self.has_updates and self.byte_size + update.byte_size > byte_limit

    def add(self, update: PartialChunk) -> None:
        if update.params is not None:
            for p in update.params:
                self.params.append(
                    replace(
                        p,
                        keys_start=p.keys_start + self.keys_total,
                        keys_end=p.keys_end + self.keys_total,
                        values_start=p.values_start + self.values_total,
                        values_end=p.values_end + self.values_total,
                    )
                )
            self.keys_total += update.tensors[0][1].numel()
            self.values_total += update.tensors[1][1].numel()
        self.tensors.extend(update.tensors)
        self.byte_size += update.byte_size

    def flush_payload(
        self,
    ) -> tuple[list[tuple[str, torch.Tensor]], list[PartialWeightParam] | None]:
        """
        Sparse: concat per-chunk packed tensors into one pair. Dense: tensors as-is.
        """
        if not self.params:
            return list(self.tensors), None
        keys = [t for n, t in self.tensors if n == "__packed_keys__"]
        values = [t for n, t in self.tensors if n == "__packed_values__"]
        merged = [
            ("__packed_keys__", torch.cat(keys, dim=0)),
            ("__packed_values__", torch.cat(values, dim=0)),
        ]
        return merged, list(self.params)

    def clear(self) -> None:
        self.tensors.clear()
        self.params.clear()
        self.byte_size = 0
        self.keys_total = 0
        self.values_total = 0


class PartialSync:
    """
    Owns pinned-CPU snapshot of last broadcast's tensors and the base-vs-partial
    decision. PP-source-rank only. Provides compute_delta (for delta mode) and
    compute_selective (for selective mode); both consume the same snapshot.
    """

    def __init__(self, args: Namespace) -> None:
        self.delta_dtype = _DELTA_DTYPE_MAP[args.update_weight_delta_dtype]
        self.base_sync_interval = args.update_weight_base_sync_interval
        if self.base_sync_interval < 1:
            raise ValueError("--update-weight-base-sync-interval must be >= 1")
        self.snapshot: dict[str, torch.Tensor] = {}
        self.committed_syncs = 0
        self.d2h_stream: torch.cuda.Stream | None = None
        self.snapshot_dirty = False

    def should_send_base(self) -> bool:
        return self.committed_syncs == 0 or self.committed_syncs % self.base_sync_interval == 0

    def compute_delta(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """
        new − snapshot at delta_dtype. Both operands are promoted before the
        subtraction so small-magnitude deltas survive. Caller advances snapshot after.
        """
        self.flush_snapshot()
        prev_gpu = []
        for name, tensor in named_tensors:
            if name not in self.snapshot:
                raise KeyError(f"missing snapshot for {name!r}; need a base sync first")
            prev_gpu.append(self.snapshot[name].to(device=tensor.device, non_blocking=True))
        torch.cuda.synchronize()
        deltas = []
        for (name, tensor), prev in zip(named_tensors, prev_gpu, strict=True):
            deltas.append((name, tensor.to(self.delta_dtype) - prev.to(self.delta_dtype)))
            del prev
        return deltas

    def compute_selective(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """
        For each param, mask = (new != snapshot) in the param dtype. Returns
        new values with NaN at unchanged positions (lossless: no arithmetic).
        Caller advances snapshot after.
        """
        self.flush_snapshot()
        prev_gpu = []
        for name, tensor in named_tensors:
            if name not in self.snapshot:
                raise KeyError(f"missing snapshot for {name!r}; need a base sync first")
            prev_gpu.append(self.snapshot[name].to(device=tensor.device, non_blocking=True))
        torch.cuda.synchronize()
        result = []
        for (name, tensor), prev in zip(named_tensors, prev_gpu, strict=True):
            if not tensor.dtype.is_floating_point:
                raise TypeError(
                    f"selective mode requires float param dtype (has NaN); got {tensor.dtype} for {name!r}"
                )
            mask = tensor != prev
            nan = torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
            result.append((name, torch.where(mask, tensor, nan)))
            del prev
        return result

    def update_snapshot_async(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """
        D2H snapshot copy on a side stream so it overlaps downstream broadcast/encode.
        """
        if self.d2h_stream is None:
            self.d2h_stream = torch.cuda.Stream()
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_event(event)
            for name, tensor in named_tensors:
                if name not in self.snapshot:
                    self.snapshot[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
                self.snapshot[name].copy_(tensor.detach(), non_blocking=True)
        self.snapshot_dirty = True

    def flush_snapshot(self) -> None:
        if self.snapshot_dirty:
            if self.d2h_stream is not None:
                self.d2h_stream.synchronize()
            else:
                torch.cuda.synchronize()
            self.snapshot_dirty = False

    def on_sync_succeeded(self) -> None:
        self.flush_snapshot()
        self.committed_syncs += 1


class PartialArtifactWriter:
    """
    Async background writer for per-chunk partial-update artifacts. Active iff
    ``--update-weight-partial-artifact-dir`` is set. Output is per-chunk safetensors
    (zstd-wrapped if ``zstandard`` is installed).
    """

    def __init__(self, artifact_dir: str) -> None:
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.work_queue: Queue = Queue()
        self.compressor = zstandard.ZstdCompressor() if zstandard is not None else None
        self.thread = threading.Thread(target=self._run, name="partial-artifact-writer", daemon=True)
        self.thread.start()

    def enqueue(
        self,
        weight_version: int,
        chunk_idx: int,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        self.work_queue.put((weight_version, chunk_idx, {name: t.contiguous() for name, t in named_tensors}))

    def _run(self) -> None:
        while True:
            item = self.work_queue.get()
            if item is None:
                return
            weight_version, chunk_idx, tensors = item
            stem = f"weight_version_{weight_version:06d}_chunk_{chunk_idx:06d}.safetensors"
            if self.compressor is None:
                save_file(tensors, os.path.join(self.artifact_dir, stem))
                continue
            payload = save(tensors)
            with open(os.path.join(self.artifact_dir, f"{stem}.zst"), "wb") as f:
                f.write(self.compressor.compress(payload))


class UpdateWeightFromDistributedPartial(UpdateWeightFromDistributed):
    """
    Partial-update variant. Sends a sparse-encoded payload per named tensor and
    has SGLang apply it. Two sub-modes, selected by ``--update-weight-mode``:

    * ``delta``: payload values are ``(current − snapshot)`` cast to delta_dtype;
      receiver applies additively (``param += delta``).
    * ``selective``: payload values are the new param values at changed positions,
      with NaN as the "unchanged" sentinel; receiver overwrites the non-NaN
      positions only.

    Periodic base syncs (full broadcasts) refresh the snapshot. The ``_on_chunk``
    hook on the base class is used to keep the snapshot in lockstep during base
    syncs.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        super().__init__(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.mode = args.update_weight_mode  # "delta" or "selective"
        self.partial_sync = PartialSync(args)
        self.artifact_writer = (
            PartialArtifactWriter(args.update_weight_partial_artifact_dir)
            if args.update_weight_partial_artifact_dir is not None
            else None
        )
        self.artifact_chunk_idx = 0
        self.pending_artifacts: list[list[tuple[str, torch.Tensor]]] = []

    def _send_weights(self, pbar: tqdm | None) -> None:
        if self.partial_sync.should_send_base():
            super()._send_weights(pbar)
        else:
            self._send_partial_weights(pbar)
        # Increment on all ranks so should_send_base() stays in lockstep across
        # the PP group. flush_snapshot() is a no-op on non-PP-src ranks.
        self.partial_sync.on_sync_succeeded()

    def _on_chunk(self, hf_chunk: list[tuple[str, torch.Tensor]]) -> None:
        """
        Base-sync hook: snapshot this chunk so the next partial sync has prev to diff against.
        """
        self.partial_sync.update_snapshot_async(hf_chunk)

    def _send_partial_weights(self, pbar: tqdm | None) -> None:
        """
        non-expert (TP) loop → barrier → expert (EP) loop. Each HF chunk is
        converted to a partial-update payload (delta or selective) and bucketed.
        """
        encoding = PartialWeightEncoding(self.args.update_weight_partial_encoding)
        bucket = PartialSendBucket()
        for hf_chunk in self._iter_non_expert_chunks():
            self._enqueue_partial_chunk(hf_chunk, encoding, bucket, pbar)
        self._flush_partial_bucket(bucket, encoding, pbar)

        dist.barrier(group=get_gloo_group())

        for hf_chunk in self._iter_expert_chunks():
            self._enqueue_partial_chunk(hf_chunk, encoding, bucket, pbar)
        self._flush_partial_bucket(bucket, encoding, pbar)

    def _enqueue_partial_chunk(
        self,
        hf_chunk: list[tuple[str, torch.Tensor]],
        encoding: PartialWeightEncoding,
        bucket: PartialSendBucket,
        pbar: tqdm | None,
    ) -> None:
        """
        compute payload (mode-specific) → snapshot new prev → encode → bucket.add.
        """
        if not hf_chunk:
            return
        if self.mode == "delta":
            payload = self.partial_sync.compute_delta(hf_chunk)
        else:  # "selective"
            payload = self.partial_sync.compute_selective(hf_chunk)
        self.partial_sync.update_snapshot_async(hf_chunk)
        chunk = encode_partial(payload, encoding, self.mode)
        if not chunk.tensors:
            return
        if bucket.should_flush_before_add(chunk, self.args.update_weight_buffer_size):
            self._flush_partial_bucket(bucket, encoding, pbar)
        # Append AFTER the flush check so this chunk's artifact lands in the
        # same flush as its broadcast (and not at all if its encoding skipped).
        if self.artifact_writer is not None:
            self.pending_artifacts.append([(n, t.cpu()) for n, t in payload])
        bucket.add(chunk)

    def _flush_partial_bucket(
        self,
        bucket: PartialSendBucket,
        encoding: PartialWeightEncoding,
        pbar: tqdm | None,
    ) -> None:
        """
        Lock → broadcast (with PartialWeightSpec) → unlock → pbar++. Drains
        pending artifacts to the async writer once the broadcast lands.
        load_format is "delta" or "selective" per self.mode.
        """
        if not bucket.has_updates:
            return
        wire_tensors, params = bucket.flush_payload()
        spec = PartialWeightSpec(encoding=encoding, params=params)
        bucket.clear()
        self._update_bucket_weights_from_distributed(wire_tensors, pbar=pbar, load_format=self.mode, partial=spec)
        if self.artifact_writer is not None:
            for artifact in self.pending_artifacts:
                self.artifact_writer.enqueue(self.weight_version, self.artifact_chunk_idx, artifact)
                self.artifact_chunk_idx += 1
            self.pending_artifacts.clear()
