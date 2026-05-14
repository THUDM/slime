from __future__ import annotations

import itertools
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
    nnz is the active-position count across all params (used for density).
    """

    tensors: list[tuple[str, torch.Tensor]]
    params: list[PartialWeightParam] | None
    byte_size: int
    nnz: int = 0


@dataclass
class PartialPayload:
    """
    Per-param compute output flowing into the encoder. ``payload`` is the full-size
    values; ``mask`` marks active positions, or is ``None`` for the delta fast path
    (encoder uses ``torch.nonzero(payload)`` directly). Selective always sets a mask
    so legitimately-zero new values aren't dropped.
    """

    name: str
    payload: torch.Tensor
    mask: torch.Tensor | None


def encode_partial(
    named_payloads: list[PartialPayload],
    encoding: PartialWeightEncoding,
    mode: str,
) -> PartialChunk:
    """
    Encode partial-update payloads per wire encoding. ``mode`` is needed only
    by the dense encoder, which has to materialize the receiver-side sentinel.
    Sparse paths read the mask directly from each PartialPayload.
    """
    if encoding is PartialWeightEncoding.DENSE:
        return _encode_dense(named_payloads, mode)
    if encoding in (PartialWeightEncoding.SPARSE_INDICES, PartialWeightEncoding.SPARSE_BITMASK):
        return _encode_sparse(named_payloads, encoding)
    raise ValueError(f"unknown partial-update encoding: {encoding!r}")


def _encode_dense(named_payloads: list[PartialPayload], mode: str) -> PartialChunk:
    """
    Dense wire: send a full-size tensor per param with a receiver-side
    sentinel at unchanged positions. Delta's payload already has 0 at
    unchanged (current − snapshot is 0 there); selective re-materializes a
    NaN-marked tensor here — lazy, since dense is the debug-only encoding.
    """
    tensors: list[tuple[str, torch.Tensor]] = []
    nnz = 0
    for pp in named_payloads:
        # Delta path may not provide mask; derive nnz from payload directly.
        nnz += int(pp.mask.sum()) if pp.mask is not None else int((pp.payload != 0).sum())
        if mode == "selective":
            nan = torch.full_like(pp.payload, float("nan"))
            tensors.append((pp.name, torch.where(pp.mask, pp.payload, nan)))
        else:  # "delta"
            tensors.append((pp.name, pp.payload))
    size = sum(t.numel() * t.element_size() for _, t in tensors)
    return PartialChunk(tensors=tensors, params=None, byte_size=size, nnz=nnz)


def _encode_sparse(
    named_payloads: list[PartialPayload],
    encoding: PartialWeightEncoding,
) -> PartialChunk:
    """
    Sparse encoder for SPARSE_INDICES and SPARSE_BITMASK. Boundaries are found once
    via ``_sparse_boundaries``; the per-param loop emits encoding-specific keys
    (in-param int32 indices, or bit-packed mask bytes).
    """
    if not named_payloads:
        return PartialChunk(tensors=[], params=[], byte_size=0)

    big_val, bounds, big_idx, cum = _sparse_boundaries(named_payloads)

    params: list[PartialWeightParam] = []
    keys_pieces: list[torch.Tensor] = []
    vals_pieces: list[torch.Tensor] = []
    keys_off = values_off = 0
    prev_b = 0
    prev_off = 0
    for i, pp in enumerate(named_payloads):
        b = bounds[i]
        nnz = b - prev_b
        if nnz > 0:
            if encoding is PartialWeightEncoding.SPARSE_INDICES:
                # Global → in-param coordinates.
                keys_i = (big_idx[prev_b:b] - prev_off).to(torch.int32)
            else:  # SPARSE_BITMASK
                # Delta has no mask; derive from payload. Selective brings its own.
                flat_mask_i = (
                    pp.mask.contiguous().view(-1) if pp.mask is not None else (pp.payload.contiguous().view(-1) != 0)
                )
                keys_i = _pack_bitmask(flat_mask_i)
            values_i = big_val[prev_b:b]
            keys_count = keys_i.numel()
            params.append(
                PartialWeightParam(
                    name=pp.name,
                    dtype=str(pp.payload.dtype).replace("torch.", ""),
                    shape=list(pp.payload.shape),
                    keys_start=keys_off,
                    keys_end=keys_off + keys_count,
                    values_start=values_off,
                    values_end=values_off + nnz,
                )
            )
            keys_pieces.append(keys_i)
            vals_pieces.append(values_i)
            keys_off += keys_count
            values_off += nnz
        prev_b = b
        prev_off = cum[i]

    if not params:
        return PartialChunk(tensors=[], params=[], byte_size=0)
    packed_keys = torch.cat(keys_pieces, dim=0)
    packed_values = torch.cat(vals_pieces, dim=0)
    size = packed_keys.numel() * packed_keys.element_size() + packed_values.numel() * packed_values.element_size()
    return PartialChunk(
        tensors=[("__packed_keys__", packed_keys), ("__packed_values__", packed_values)],
        params=params,
        byte_size=size,
        nnz=values_off,
    )


def _sparse_boundaries(
    named_payloads: list[PartialPayload],
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[int]]:
    """
    concat → one nonzero → searchsorted → one ``.tolist()``: collapses ~30
    per-param host syncs/chunk to 1. Returns ``(big_val, bounds, big_idx, cum)``
    for the encoder's per-param emission step.
    """
    has_mask = named_payloads[0].mask is not None
    device = named_payloads[0].payload.device
    sizes = [pp.payload.numel() for pp in named_payloads]
    cum = list(itertools.accumulate(sizes))
    cum_t = torch.tensor(cum, dtype=torch.int64, device=device)

    big_payload = torch.cat([pp.payload.contiguous().view(-1) for pp in named_payloads], dim=0)
    if has_mask:
        big_mask = torch.cat([pp.mask.contiguous().view(-1) for pp in named_payloads], dim=0)
        big_idx = big_mask.nonzero(as_tuple=False).view(-1)
    else:
        # Delta: zeros in the payload mean "unchanged" — nonzero(payload) directly.
        big_idx = torch.nonzero(big_payload, as_tuple=False).view(-1)
    big_val = big_payload[big_idx]
    bounds = torch.searchsorted(big_idx, cum_t).tolist()
    return big_val, bounds, big_idx, cum


def _pack_bitmask(flat_mask: torch.Tensor) -> torch.Tensor:
    """Bit-pack a 1-D bool mask into uint8 bytes (little-endian within byte)."""
    if flat_mask.numel() == 0:
        return torch.empty(0, dtype=torch.uint8, device=flat_mask.device)
    mask_u8 = flat_mask.to(dtype=torch.uint8)
    pad = (-mask_u8.numel()) % 8
    if pad:
        mask_u8 = torch.cat([mask_u8, torch.zeros(pad, dtype=torch.uint8, device=mask_u8.device)])
    bits = mask_u8.view(-1, 8)
    weights = (2 ** torch.arange(8, dtype=torch.uint8, device=mask_u8.device)).view(1, 8)
    return torch.sum(bits * weights, dim=1, dtype=torch.uint8)


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
    Sender state: pinned-CPU snapshot of the last broadcast, base-sync cadence,
    and the H2D/D2H streams that pipeline snapshot transfer behind encode+broadcast.
    PP-source-rank only.
    """

    def __init__(self, args: Namespace) -> None:
        self.delta_dtype = _DELTA_DTYPE_MAP[args.update_weight_delta_dtype]
        self.base_sync_interval = args.update_weight_base_sync_interval
        if self.base_sync_interval < 1:
            raise ValueError("--update-weight-base-sync-interval must be >= 1")
        self.snapshot: dict[str, torch.Tensor] = {}
        self.committed_syncs = 0
        self.d2h_stream: torch.cuda.Stream | None = None
        self.h2d_stream: torch.cuda.Stream | None = None
        self.snapshot_dirty = False

    def should_send_base(self) -> bool:
        return self.committed_syncs == 0 or self.committed_syncs % self.base_sync_interval == 0

    def prefetch_snapshot(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[torch.Tensor], torch.cuda.Event]:
        """
        Start an async H2D copy of the snapshot tensors for ``named_tensors`` on
        a side stream. Returns (prev_gpu_list, event) — pass to ``compute_payload``
        as ``prefetched=(prev_gpu_list, event)``; the consumer's stream will wait
        on the event before reading prev_gpu.
        """
        if self.h2d_stream is None:
            self.h2d_stream = torch.cuda.Stream()
        prev_gpu: list[torch.Tensor] = []
        with torch.cuda.stream(self.h2d_stream):
            for name, tensor in named_tensors:
                if name not in self.snapshot:
                    raise KeyError(f"missing snapshot for {name!r}; need a base sync first")
                prev_gpu.append(self.snapshot[name].to(device=tensor.device, non_blocking=True))
            event = self.h2d_stream.record_event()
        return prev_gpu, event

    def compute_payload(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        mode: str,
        prefetched: tuple[list[torch.Tensor], torch.cuda.Event],
    ) -> list[PartialPayload]:
        """
        Build a PartialPayload per param from the prefetched snapshot.

        delta: payload = (new − snapshot) at delta_dtype, mask = None — encoder uses
        ``torch.nonzero(payload)`` directly, skipping the bool intermediate.
        selective: payload = new (reference, no copy), mask = (new != snapshot).
        """
        if mode == "delta":

            def per_param(name, tensor, prev):
                return tensor.to(self.delta_dtype) - prev.to(self.delta_dtype), None

        elif mode == "selective":

            def per_param(name, tensor, prev):
                if not tensor.dtype.is_floating_point:
                    raise TypeError(f"selective mode requires float param dtype; got {tensor.dtype} for {name!r}")
                return tensor, tensor != prev

        else:
            raise ValueError(f"unknown partial-update mode: {mode!r}")

        prev_gpu, event = prefetched
        event.wait()

        result: list[PartialPayload] = []
        for (name, tensor), prev in zip(named_tensors, prev_gpu, strict=True):
            payload, mask = per_param(name, tensor, prev)
            result.append(PartialPayload(name=name, payload=payload, mask=mask))
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
    Partial-update variant selected by ``--update-weight-mode``. ``selective``:
    overwrite changed positions only (NaN = unchanged sentinel). ``delta``:
    apply ``(new − snapshot)`` additively cast to delta_dtype. Periodic base
    syncs refresh the snapshot via ``_on_chunk`` on the parent.
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
        self.mode = args.update_weight_mode  # "selective" or "delta"
        self.partial_sync = PartialSync(args)
        self.artifact_writer = (
            PartialArtifactWriter(args.update_weight_partial_artifact_dir)
            if args.update_weight_partial_artifact_dir is not None
            else None
        )
        self.artifact_chunk_idx = 0
        self.pending_artifacts: list[list[tuple[str, torch.Tensor]]] = []
        self.density_nnz = 0
        self.density_numel = 0
        self.wire_bytes = 0

    def _send_weights(self, pbar: tqdm | None) -> None:
        is_base = self.partial_sync.should_send_base()
        self.density_nnz = 0
        self.density_numel = 0
        self.wire_bytes = 0
        if is_base:
            super()._send_weights(pbar)
        else:
            self._send_partial_weights(pbar)
        # Increment on all ranks so should_send_base() stays in lockstep across
        # the PP group. flush_snapshot() is a no-op on non-PP-src ranks.
        self.partial_sync.on_sync_succeeded()
        self._record_metrics(is_base)

    def _on_chunk(self, hf_chunk: list[tuple[str, torch.Tensor]]) -> None:
        """
        Base-sync hook: snapshot this chunk so the next partial sync has prev to diff against.
        Also count dense wire bytes so base syncs share the wire_bytes metric axis.
        """
        self.partial_sync.update_snapshot_async(hf_chunk)
        self.wire_bytes += sum(t.numel() * t.element_size() for _, t in hf_chunk)

    def _send_partial_weights(self, pbar: tqdm | None) -> None:
        """
        non-expert (TP) loop → barrier → expert (EP) loop, with 1-step H2D
        prefetch lookahead so chunk N+1's snapshot transfer overlaps chunk N's
        compute+encode+broadcast.
        """
        encoding = PartialWeightEncoding(self.args.update_weight_partial_encoding)
        bucket = PartialSendBucket()
        # One flush at the start of the sync ensures previous-sync D2H is
        # complete before we prefetch any snapshot for this sync. Per-chunk
        # prefetches don't need to flush again — they read different param
        # names than the in-flight D2H writes within this sync.
        self.partial_sync.flush_snapshot()

        self._iter_with_pipeline(self._iter_non_expert_chunks(), encoding, bucket, pbar)
        self._flush_partial_bucket(bucket, encoding, pbar)

        dist.barrier(group=get_gloo_group())

        self._iter_with_pipeline(self._iter_expert_chunks(), encoding, bucket, pbar)
        self._flush_partial_bucket(bucket, encoding, pbar)

    def _iter_with_pipeline(
        self,
        chunk_iter,
        encoding: PartialWeightEncoding,
        bucket: PartialSendBucket,
        pbar: tqdm | None,
    ) -> None:
        """
        1-step H2D prefetch lookahead so chunk N+1's snapshot transfer overlaps
        chunk N's compute+encode on the default stream.
        """

        def drain(item):
            if item is None:
                return
            chunk, prefetched = item
            self._enqueue_partial_chunk(chunk, encoding, bucket, pbar, prefetched=prefetched)

        pending = None
        for hf_chunk in chunk_iter:
            if not hf_chunk:
                continue
            ahead = (hf_chunk, self.partial_sync.prefetch_snapshot(hf_chunk))
            drain(pending)
            pending = ahead
        drain(pending)

    def _enqueue_partial_chunk(
        self,
        hf_chunk: list[tuple[str, torch.Tensor]],
        encoding: PartialWeightEncoding,
        bucket: PartialSendBucket,
        pbar: tqdm | None,
        prefetched: tuple[list[torch.Tensor], torch.cuda.Event],
    ) -> None:
        """
        compute payloads (mode-specific) → snapshot new prev → encode → bucket.add.
        ``prefetched`` carries the result of ``PartialSync.prefetch_snapshot``;
        compute_payload waits on its event before reading prev_gpu.
        """
        payloads = self.partial_sync.compute_payload(hf_chunk, self.mode, prefetched=prefetched)
        self.partial_sync.update_snapshot_async(hf_chunk)
        chunk = encode_partial(payloads, encoding, self.mode)
        # numel from input payload so zero-nnz params still count — otherwise
        # the ratio biases toward params that did change.
        self.density_numel += sum(pp.payload.numel() for pp in payloads)
        self.density_nnz += chunk.nnz
        self.wire_bytes += chunk.byte_size
        if not chunk.tensors:
            return
        if bucket.should_flush_before_add(chunk, self.args.update_weight_buffer_size):
            self._flush_partial_bucket(bucket, encoding, pbar)
        # Append AFTER the flush check so this chunk's artifact lands in the
        # same flush as its broadcast (and not at all if its encoding skipped).
        if self.artifact_writer is not None:
            self.pending_artifacts.append([(pp.name, pp.payload.cpu()) for pp in payloads])
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
        load_format is "selective" or "delta" per self.mode.
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

    def _record_metrics(self, is_base: bool) -> None:
        """
        Base sync sends every position → density 1.0 by definition.
        """
        counts = torch.tensor(
            [self.density_nnz, self.density_numel, self.wire_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        nnz, numel, wire_bytes = counts.tolist()
        self.update_weight_metrics["perf/update_weights_is_base_sync"] = 1.0 if is_base else 0.0
        self.update_weight_metrics["perf/update_weights_density"] = 1.0 if is_base else nnz / max(numel, 1)
        self.update_weight_metrics["perf/update_weights_wire_bytes"] = wire_bytes
