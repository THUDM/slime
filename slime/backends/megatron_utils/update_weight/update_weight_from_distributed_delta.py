"""
Delta weight sync.

For each sync, the sender bytewise-diffs the current weights against a
pinned-CPU snapshot of the last broadcast, packs the changed positions
and values, and ships them via one of two transports:

  - "nccl": each bucket flush goes out via NCCL broadcast (low-latency,
    high-bandwidth, intra-datacenter).
  - "disk": each bucket flush is written to a versioned shared-FS directory
    as one safetensors file; one HTTP push per sync wakes the rollout
    engines to read+apply (cross-datacenter, bandwidth-limited).

Both transports share one wire layout (``__positions__`` uint8 byte blob +
``__values__`` param-dtype tensor + per-param decoding manifest) and one
receiver-side decoder. Three encodings differ only in how positions are
packed:

  indices     : int32 absolute positions
  deltas      : uint16 gap-deltas (uint32 fallback per param)
  deltas_zstd : ``deltas`` with the safetensors blob wrapped in zstd L1

The receiver overwrites changed positions with the trainer's exact bytes
(no arithmetic), so the apply is lossless and there is no drift to fight
with periodic re-syncs. The first ``update_weights`` call seeds the
snapshot without contacting the rollout engines — they're assumed to have
loaded the same HF checkpoint at init.
"""

import itertools
import json
import logging
import os
import shutil
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from queue import Queue

import numpy as np
import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from safetensors.torch import save as st_save_bytes
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group

from ..sglang import DeltaEncoding, DeltaParam, DeltaSpec
from .update_weight_from_distributed import UpdateWeightFromDistributed


logger = logging.getLogger(__name__)


# ---------- compute + encode -----------------------------------------------


@dataclass
class ParamDiff:
    """
    One per-param compute output. ``values`` is a reference to the full-shape
    current tensor (no copy); ``mask`` is a same-shape bool marking the
    positions whose bytes differ from the snapshot.
    """

    name: str
    values: torch.Tensor
    mask: torch.Tensor


@dataclass
class EncodedChunk:
    """
    One HF chunk after position+value encoding, before bucket merging.

    ``pos_bytes`` and ``val_tensor`` are the chunk-local concatenations across
    all params; per-param byte/element offsets live on ``params``.
    """

    pos_bytes: bytes
    val_tensor: torch.Tensor
    params: list[DeltaParam]
    nnz: int


def _bytewise_diff_mask(current: torch.Tensor, snapshot: torch.Tensor) -> torch.Tensor:
    """
    Per-element bool mask: True where current and snapshot bytes differ. Dtype-agnostic via view-as-integer.
    """
    es = current.element_size()
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    return current.view(int_dtype) != snapshot.view(int_dtype)


def _sparse_boundaries(
    diffs: list[ParamDiff],
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[int]]:
    """
    One concat → one nonzero → one searchsorted → one ``tolist()``: collapses
    per-param host syncs to one per chunk. Returns ``(big_val, bounds, big_idx, cum)``.
    """
    device = diffs[0].values.device
    sizes = [d.values.numel() for d in diffs]
    cum = list(itertools.accumulate(sizes))
    cum_t = torch.tensor(cum, dtype=torch.int64, device=device)

    big_values = torch.cat([d.values.contiguous().view(-1) for d in diffs], dim=0)
    big_mask = torch.cat([d.mask.contiguous().view(-1) for d in diffs], dim=0)
    big_idx = big_mask.nonzero(as_tuple=False).view(-1)
    big_val = big_values[big_idx]
    bounds = torch.searchsorted(big_idx, cum_t).tolist()
    return big_val, bounds, big_idx, cum


def encode_indices(diffs: list[ParamDiff]) -> EncodedChunk:
    """
    int32 absolute positions, per-param. Position blob is uint8 bytes; pos_width=4 for all params.
    """
    if not diffs:
        return EncodedChunk(pos_bytes=b"", val_tensor=torch.empty(0, dtype=torch.bfloat16), params=[], nnz=0)
    big_val, bounds, big_idx, cum = _sparse_boundaries(diffs)
    pos_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    params: list[DeltaParam] = []
    pos_byte_off = val_off = 0
    prev_b = 0
    prev_param_start = 0
    for i, d in enumerate(diffs):
        b = bounds[i]
        nnz = b - prev_b
        if nnz > 0:
            local_idx = (big_idx[prev_b:b] - prev_param_start).to(torch.int32)
            pos_pieces.append(local_idx)
            val_pieces.append(big_val[prev_b:b])
            params.append(
                DeltaParam(
                    name=d.name,
                    dtype=str(d.values.dtype).replace("torch.", ""),
                    shape=list(d.values.shape),
                    pos_start=pos_byte_off,
                    pos_end=pos_byte_off + nnz * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + nnz,
                )
            )
            pos_byte_off += nnz * 4
            val_off += nnz
        prev_b = b
        prev_param_start = cum[i]
    if not params:
        return EncodedChunk(pos_bytes=b"", val_tensor=torch.empty(0, dtype=torch.bfloat16), params=[], nnz=0)
    positions = torch.cat(pos_pieces, dim=0)
    values = torch.cat(val_pieces, dim=0)
    return EncodedChunk(
        pos_bytes=positions.cpu().numpy().tobytes(),
        val_tensor=values,
        params=params,
        nnz=val_off,
    )


def encode_deltas(diffs: list[ParamDiff]) -> EncodedChunk:
    """
    Gap-encode sorted positions: store ``idx[k] - idx[k-1] - 1`` with idx[-1] := -1
    so the first delta equals the first index. Per-param downcast to uint16 if the max
    gap fits, otherwise uint32. At ~2% Bernoulli density on bf16 weights, max gap ≈ 300
    — uint16 fits; the fallback covers pathological inputs without correctness risk.
    Receiver inverts: ``idx = cumsum(delta + 1) - 1``.
    """
    if not diffs:
        return EncodedChunk(pos_bytes=b"", val_tensor=torch.empty(0, dtype=torch.bfloat16), params=[], nnz=0)
    big_val, bounds, big_idx, cum = _sparse_boundaries(diffs)

    per_param_deltas: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    prelim: list[tuple] = []  # (name, dtype, shape, nnz)
    prev_b = 0
    prev_param_start = 0
    for i, d in enumerate(diffs):
        b = bounds[i]
        nnz = b - prev_b
        if nnz > 0:
            local_idx = big_idx[prev_b:b] - prev_param_start  # int64, sorted
            prev = torch.cat(
                [
                    torch.tensor([-1], dtype=local_idx.dtype, device=local_idx.device),
                    local_idx[:-1],
                ]
            )
            deltas = local_idx - prev - 1
            per_param_deltas.append(deltas)
            val_pieces.append(big_val[prev_b:b])
            prelim.append((d.name, str(d.values.dtype).replace("torch.", ""), list(d.values.shape), nnz))
        prev_b = b
        prev_param_start = cum[i]

    if not prelim:
        return EncodedChunk(pos_bytes=b"", val_tensor=torch.empty(0, dtype=torch.bfloat16), params=[], nnz=0)

    # One CPU sync for per-param width selection.
    max_per_param = torch.stack([d.max() for d in per_param_deltas]).cpu().tolist()
    pos_byte_pieces: list[bytes] = []
    pos_byte_off = val_off = 0
    params: list[DeltaParam] = []
    for (name, dt, sh, nnz), deltas, max_d in zip(prelim, per_param_deltas, max_per_param, strict=True):
        width = 2 if int(max_d) <= 65535 else 4
        np_dtype = np.uint16 if width == 2 else np.uint32
        b_chunk = deltas.cpu().numpy().astype(np_dtype, copy=False).tobytes()
        pos_byte_pieces.append(b_chunk)
        params.append(
            DeltaParam(
                name=name,
                dtype=dt,
                shape=sh,
                pos_start=pos_byte_off,
                pos_end=pos_byte_off + len(b_chunk),
                pos_width=width,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        pos_byte_off += len(b_chunk)
        val_off += nnz

    values = torch.cat(val_pieces, dim=0)
    return EncodedChunk(
        pos_bytes=b"".join(pos_byte_pieces),
        val_tensor=values,
        params=params,
        nnz=val_off,
    )


# ---------- snapshot state -------------------------------------------------


class DeltaState:
    """
    Pinned-CPU snapshot of every HF tensor we've broadcast, plus the H2D/D2H
    side streams that pipeline next-chunk snapshot transfer behind the current
    chunk's compute.
    """

    def __init__(self) -> None:
        self.snapshot: dict[str, torch.Tensor] = {}
        self.d2h_stream: torch.cuda.Stream | None = None
        self.h2d_stream: torch.cuda.Stream | None = None
        self.snapshot_dirty = False

    def prefetch_snapshot(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[torch.Tensor], torch.cuda.Event]:
        """
        Start an async H2D copy of the snapshot tensors for ``named_tensors`` on a side stream.
        """
        if self.h2d_stream is None:
            self.h2d_stream = torch.cuda.Stream()
        prev_gpu: list[torch.Tensor] = []
        with torch.cuda.stream(self.h2d_stream):
            for name, tensor in named_tensors:
                if name not in self.snapshot:
                    raise KeyError(f"missing snapshot for {name!r}; first update_weights call seeds the snapshot")
                prev_gpu.append(self.snapshot[name].to(device=tensor.device, non_blocking=True))
            event = self.h2d_stream.record_event()
        return prev_gpu, event

    def compute_diffs(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        prefetched: tuple[list[torch.Tensor], torch.cuda.Event],
    ) -> list[ParamDiff]:
        prev_gpu, event = prefetched
        event.wait()
        return [
            ParamDiff(name=name, values=current, mask=_bytewise_diff_mask(current, prev))
            for (name, current), prev in zip(named_tensors, prev_gpu, strict=True)
        ]

    def update_snapshot_async(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
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


# ---------- bucket ---------------------------------------------------------


@dataclass
class DeltaBucket:
    """
    Accumulates encoded chunks for one flush. Per-param offsets are rebased
    into the bucket's growing position blob + value tensor on ``add``.
    """

    pos_pieces: list[bytes] = field(default_factory=list)
    val_pieces: list[torch.Tensor] = field(default_factory=list)
    params: list[DeltaParam] = field(default_factory=list)
    pos_total: int = 0
    val_total: int = 0
    byte_size: int = 0

    @property
    def has_updates(self) -> bool:
        return bool(self.pos_pieces)

    def should_flush_before_add(self, chunk: EncodedChunk, byte_limit: int) -> bool:
        chunk_bytes = len(chunk.pos_bytes) + chunk.val_tensor.numel() * chunk.val_tensor.element_size()
        return self.has_updates and self.byte_size + chunk_bytes > byte_limit

    def add(self, chunk: EncodedChunk) -> None:
        for p in chunk.params:
            self.params.append(
                replace(
                    p,
                    pos_start=p.pos_start + self.pos_total,
                    pos_end=p.pos_end + self.pos_total,
                    val_start=p.val_start + self.val_total,
                    val_end=p.val_end + self.val_total,
                )
            )
        self.pos_pieces.append(chunk.pos_bytes)
        self.val_pieces.append(chunk.val_tensor)
        self.pos_total += len(chunk.pos_bytes)
        self.val_total += chunk.val_tensor.numel()
        self.byte_size += len(chunk.pos_bytes) + chunk.val_tensor.numel() * chunk.val_tensor.element_size()

    def merged_positions_cpu(self) -> torch.Tensor:
        """One CPU uint8 tensor with the bucket's positions blob."""
        merged = b"".join(self.pos_pieces)
        if not merged:
            return torch.empty(0, dtype=torch.uint8)
        return torch.from_numpy(np.frombuffer(merged, dtype=np.uint8).copy())

    def merged_values(self) -> torch.Tensor:
        if not self.val_pieces:
            return torch.empty(0, dtype=torch.bfloat16)
        return torch.cat(self.val_pieces, dim=0)

    def clear(self) -> None:
        self.pos_pieces.clear()
        self.val_pieces.clear()
        self.params.clear()
        self.pos_total = 0
        self.val_total = 0
        self.byte_size = 0


# ---------- async safetensors writer (disk transport only) -----------------


class AsyncSafetensorsWriter:
    """
    Background thread that drains a queue of file writes. Producers do GPU→CPU
    on the default stream and enqueue; the writer does the slow disk I/O
    (and optional zstd compress) off the critical path. End-of-sync ``drain()``
    blocks until all enqueued writes have landed.
    """

    def __init__(self, compress_with_zstd: bool, zstd_level: int = 1) -> None:
        self._queue: Queue = Queue()
        self._error: BaseException | None = None
        self._compress_with_zstd = compress_with_zstd
        self._zstd_level = zstd_level
        if compress_with_zstd:
            # Lazy import — non-disk users don't pay the dep.
            import zstandard

            self._zstd = zstandard
        self._lock = threading.Lock()
        self.bytes_pre_compress = 0
        self.bytes_post_compress = 0
        self.write_seconds = 0.0
        self._thread = threading.Thread(target=self._run, name="delta-disk-writer", daemon=True)
        self._thread.start()

    def enqueue(
        self,
        path: str,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
    ) -> None:
        if self._error is not None:
            raise RuntimeError(f"writer thread already failed: {self._error!r}")
        self._queue.put((path, tensors, metadata))

    def drain(self) -> None:
        self._queue.join()
        if self._error is not None:
            raise RuntimeError(f"writer thread failed: {self._error!r}") from self._error

    def reset_counters(self) -> None:
        with self._lock:
            self.bytes_pre_compress = 0
            self.bytes_post_compress = 0
            self.write_seconds = 0.0

    def _run(self) -> None:
        cctx = self._zstd.ZstdCompressor(level=self._zstd_level, threads=-1) if self._compress_with_zstd else None
        while True:
            path, tensors, metadata = self._queue.get()
            try:
                if self._error is None:
                    t0 = time.perf_counter()
                    blob = st_save_bytes(tensors, metadata=metadata)
                    pre = len(blob)
                    if cctx is not None:
                        blob = cctx.compress(blob)
                    post = len(blob)
                    tmp = path + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(blob)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, path)
                    elapsed = time.perf_counter() - t0
                    with self._lock:
                        self.bytes_pre_compress += pre
                        self.bytes_post_compress += post
                        self.write_seconds += elapsed
            except BaseException as e:  # noqa: BLE001
                self._error = e
            finally:
                self._queue.task_done()


# ---------- main class -----------------------------------------------------


class UpdateWeightFromDistributedDelta(UpdateWeightFromDistributed):
    """
    Selective delta sync. ``--update-weight-transport`` picks the per-flush carrier:
    "nccl" broadcasts each bucket; "disk" writes each bucket as a safetensors file under
    ``--update-weight-delta-dir`` and pushes once at end-of-sync.
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
        self.transport = args.update_weight_transport
        self.encoding = DeltaEncoding(args.update_weight_encoding)
        self.delta_state = DeltaState()
        self._snapshot_seeded = False
        # Matches SGLang's ``ModelRunner.current_weight_version`` bootstrap default so
        # the first sync's ``prev_version`` chains correctly with the fresh receiver.
        self._last_synced_version = "default"
        # DELTAS_ZSTD shares the gap encoder; zstd is applied at file-write time.
        self._encode = encode_indices if self.encoding is DeltaEncoding.INDICES else encode_deltas

        self.writer: AsyncSafetensorsWriter | None = None
        self.delta_dir: str | None = None
        self._pre_push_hook: Callable | None = None
        if self.transport == "disk":
            self.delta_dir = args.update_weight_delta_dir
            os.makedirs(self.delta_dir, exist_ok=True)
            self.writer = AsyncSafetensorsWriter(
                compress_with_zstd=(self.encoding == DeltaEncoding.DELTAS_ZSTD),
            )
            if getattr(args, "custom_delta_pre_push_path", None):
                from slime.utils.misc import load_function

                self._pre_push_hook = load_function(args.custom_delta_pre_push_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        NCCL transport: delegate to parent (group creation). Disk transport: just
        record the engines + PP-src flag (no NCCL group needed).
        """
        if self.transport == "nccl":
            super().connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            return
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._group_name = f"slime-pp_{pp_rank}"

    def disconnect_rollout_engines(self) -> None:
        if self.transport == "nccl":
            super().disconnect_rollout_engines()

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        First call: seed the CPU snapshot from current model state, no engine RPCs.
        Subsequent calls: pause → diff/encode → flush → finalize → resume.
        """
        if not self._snapshot_seeded:
            self._seed_snapshot()
            self._snapshot_seeded = True
            return

        t_sync_start = time.perf_counter()
        self.weight_version += 1
        if self.transport == "disk":
            self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
            if self._is_pp_src_rank:
                os.makedirs(self._version_dir, exist_ok=True)

        t_pause_ms = 0.0
        if dist.get_rank() == 0:
            t_pause_start = time.perf_counter()
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            t_pause_ms = (time.perf_counter() - t_pause_start) * 1000
        dist.barrier(group=get_gloo_group())

        self.density_nnz = self.density_numel = self.wire_bytes = self._flush_idx = 0
        if self.writer is not None:
            self.writer.reset_counters()
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        t_encode_start = time.perf_counter()
        self._send_weights(pbar)
        t_encode_ms = (time.perf_counter() - t_encode_start) * 1000

        t_drain_start = time.perf_counter()
        if self.writer is not None:
            self.writer.drain()
        self.delta_state.flush_snapshot()
        t_drain_ms = (time.perf_counter() - t_drain_start) * 1000
        dist.barrier(group=get_gloo_group())

        t_finalize_ms = self._finalize_sync()
        self._last_synced_version = str(self.weight_version)
        self._record_metrics(
            pause_ms=t_pause_ms,
            encode_ms=t_encode_ms,
            drain_ms=t_drain_ms,
            finalize_ms=t_finalize_ms,
            total_ms=(time.perf_counter() - t_sync_start) * 1000,
        )

    def _seed_snapshot(self) -> None:
        """
        Populate the snapshot from current model state (TP/EP gather + HF
        convert on PP-src ranks, D2H pinned copy). Cost is one full pass over
        params — ~50s blocking on 355B at init.
        """
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                if hf_chunk:
                    self.delta_state.update_snapshot_async(hf_chunk)
            dist.barrier(group=get_gloo_group())
        self.delta_state.flush_snapshot()

    def _send_weights(self, pbar: tqdm | None) -> None:
        """
        Non-expert (TP) pass → barrier → expert (EP) pass → barrier, each with
        1-step H2D snapshot prefetch lookahead. The delta bucket is reused across
        both passes so a partially-filled non-expert tail can pack with the first
        expert chunk.
        """
        bucket = DeltaBucket()
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            self._pipeline_pass(chunk_iter, bucket, pbar)
            if bucket.has_updates:
                self._flush_bucket(bucket, pbar)
            dist.barrier(group=get_gloo_group())

    def _pipeline_pass(
        self,
        chunk_iter: Iterator[list[tuple[str, torch.Tensor]]],
        bucket: DeltaBucket,
        pbar: tqdm | None,
    ) -> None:
        """
        1-step H2D snapshot prefetch lookahead: chunk N+1's snapshot transfer
        overlaps chunk N's compute+encode on the default stream.
        """
        pending_chunk: list[tuple[str, torch.Tensor]] | None = None
        pending_prefetch: tuple[list[torch.Tensor], torch.cuda.Event] | None = None
        for hf_chunk in chunk_iter:
            if not hf_chunk:
                continue
            next_prefetch = self.delta_state.prefetch_snapshot(hf_chunk)
            if pending_prefetch is not None:
                self._enqueue_chunk(pending_chunk, pending_prefetch, bucket, pbar)
            pending_chunk, pending_prefetch = hf_chunk, next_prefetch
        if pending_prefetch is not None:
            self._enqueue_chunk(pending_chunk, pending_prefetch, bucket, pbar)

    def _enqueue_chunk(
        self,
        hf_chunk: list[tuple[str, torch.Tensor]],
        prefetched: tuple[list[torch.Tensor], torch.cuda.Event],
        bucket: DeltaBucket,
        pbar: tqdm | None,
    ) -> None:
        """
        compute diffs → snapshot new prev → encode → bucket.add (flushing if full).
        """
        diffs = self.delta_state.compute_diffs(hf_chunk, prefetched=prefetched)
        self.delta_state.update_snapshot_async(hf_chunk)
        chunk = self._encode(diffs)
        self.density_numel += sum(d.values.numel() for d in diffs)
        self.density_nnz += chunk.nnz
        self.wire_bytes += len(chunk.pos_bytes) + chunk.val_tensor.numel() * chunk.val_tensor.element_size()
        if not chunk.params:
            return
        if bucket.should_flush_before_add(chunk, self.args.update_weight_buffer_size):
            self._flush_bucket(bucket, pbar)
        bucket.add(chunk)

    def _flush_bucket(self, bucket: DeltaBucket, pbar: tqdm | None) -> None:
        """
        NCCL: broadcast (__positions__, __values__) with a DeltaSpec.
        Disk: enqueue one safetensors file with the same payload + metadata.
        """
        if not bucket.has_updates:
            return
        positions_cpu = bucket.merged_positions_cpu()
        values_gpu = bucket.merged_values()
        params = list(bucket.params)
        bucket.clear()

        if self.transport == "nccl":
            positions_gpu = positions_cpu.to(values_gpu.device, non_blocking=True)
            spec = DeltaSpec(encoding=self.encoding, params=params)
            self._update_bucket_weights_from_distributed(
                [("__positions__", positions_gpu), ("__values__", values_gpu)],
                pbar=pbar,
                load_format="delta",
                delta=spec,
            )
        else:  # disk
            tensors = {"__positions__": positions_cpu, "__values__": values_gpu.cpu()}
            metadata = {
                "encoding": self.encoding.value,
                "params": json.dumps([asdict(p) for p in params]),
                "prev_version": self._last_synced_version,
                "current_version": str(self.weight_version),
            }
            path = os.path.join(
                self._version_dir,
                f"rank{dist.get_rank():04d}_flush{self._flush_idx:06d}.safetensors",
            )
            self.writer.enqueue(path, tensors, metadata)
            if pbar is not None:
                pbar.update(1)
        self._flush_idx += 1

    def _finalize_sync(self) -> float:
        """
        Per-transport end-of-sync. NCCL: nothing to do (each flush already broadcasted).
        Disk: DONE marker → optional pre-push hook → rank-0 push → cleanup → resume.
        Returns wall-clock ms spent in the finalize path (push + cleanup + resume).
        """
        if self.transport == "nccl":
            if dist.get_rank() == 0:
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            dist.barrier(group=get_gloo_group())
            return 0.0

        t0 = time.perf_counter()
        if dist.get_rank() == 0:
            # "DONE" marker tells the SGLang receiver the directory has settled.
            with open(os.path.join(self._version_dir, "DONE"), "w"):
                pass
        if self._pre_push_hook is not None:
            self._pre_push_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())

        if dist.get_rank() == 0:
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self._version_dir,
                        load_format="delta",
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            if not self.args.update_weight_delta_keep_files:
                shutil.rmtree(self._version_dir, ignore_errors=True)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        return (time.perf_counter() - t0) * 1000

    def _record_metrics(
        self,
        pause_ms: float,
        encode_ms: float,
        drain_ms: float,
        finalize_ms: float,
        total_ms: float,
    ) -> None:
        """
        Allreduce density/byte counters across PP-src ranks; stash on
        ``update_weight_metrics`` for the actor to drain into the next step log.
        """
        pre_bytes = self.writer.bytes_pre_compress if self.writer is not None else 0
        post_bytes = self.writer.bytes_post_compress if self.writer is not None else 0
        counts = torch.tensor(
            [self.density_nnz, self.density_numel, self.wire_bytes, pre_bytes, post_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        nnz, numel, wire_bytes, pre_bytes, post_bytes = counts.tolist()

        density = nnz / max(numel, 1)
        compression_ratio = (pre_bytes / post_bytes) if post_bytes > 0 else 1.0

        m = self.update_weight_metrics
        m["perf/update_weights_density"] = density
        m["perf/update_weights_wire_bytes"] = wire_bytes
        m["perf/update_weights_pause_ms"] = pause_ms
        m["perf/update_weights_encode_ms"] = encode_ms
        m["perf/update_weights_drain_ms"] = drain_ms
        m["perf/update_weights_finalize_ms"] = finalize_ms
        m["perf/update_weights_total_ms"] = total_ms
        m["perf/update_weights_flushes_per_rank"] = float(self._flush_idx)
        if self.transport == "disk":
            m["perf/update_weights_disk_bytes_pre_compress"] = pre_bytes
            m["perf/update_weights_disk_bytes_post_compress"] = post_bytes
            m["perf/update_weights_compression_ratio"] = compression_ratio

        if dist.get_rank() == 0:
            logger.info(
                "[delta sync v=%s] transport=%s enc=%s density=%.3f%% encode=%.2fs drain=%.2fs "
                "finalize=%.2fs total=%.2fs flushes/rank=%d",
                self.weight_version,
                self.transport,
                self.encoding.value,
                100.0 * density,
                encode_ms / 1000,
                drain_ms / 1000,
                finalize_ms / 1000,
                total_ms / 1000,
                self._flush_idx,
            )
