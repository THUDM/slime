"""
Mooncake Hybrid Rollout Transfer - Rewritten per design.

Design:
- Legacy multi-key (default): each tensor in separate key, more stable
- Single-Key SGL (optional): aggregate into one buffer, one key
- Mem pool: reuse get buffers (get_into + memoryview, 1 copy)
- put/get: DataTransferBackend adapters
- Pickle 5 OOB: extract tensor buffers, zero-copy deserialize
"""
import os
import pickle
import queue
import struct
import threading
import time
import uuid
import copyreg
import weakref
import logging
from dataclasses import dataclass

import numpy as np
import torch

from slime.utils.data_transfer import MooncakeStoreConfig

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    torch.float32: 0,
    torch.float64: 1,
    torch.int8: 2,
    torch.uint8: 3,
    torch.int16: 4,
    torch.int32: 6,
    torch.int64: 8,
    torch.bool: 10,
    torch.float16: 11,
    torch.bfloat16: 12,
}

# --- Pickle 5 OOB for PyTorch tensors (zero-copy) ---


def _reconstruct_torch_tensor(buf, dtype, shape):
    if isinstance(buf, torch.Tensor):
        t = buf
    else:
        try:
            m = memoryview(buf)
            arr = np.array(m, copy=False)
            # PyTorch requires writable arrays; copy if read-only to avoid undefined behavior
            if not arr.flags.writeable:
                arr = np.array(arr, copy=True)
            t = torch.from_numpy(arr)
        except TypeError:
            t = torch.from_numpy(buf)
    if t.dtype != dtype:
        t = t.view(dtype)
    return t.reshape(shape)


def _reduce_torch_tensor(t: torch.Tensor):
    if t.device.type != "cpu":
        t = t.cpu()
    t_contig = t.contiguous()
    shape, dtype = t_contig.shape, t_contig.dtype
    if t_contig.element_size() == 2 and dtype in (torch.bfloat16, torch.float16):
        t_contig = t_contig.view(torch.int16)
    return (_reconstruct_torch_tensor, (pickle.PickleBuffer(t_contig.numpy()), dtype, shape))


copyreg.pickle(torch.Tensor, _reduce_torch_tensor)


@dataclass(frozen=True)
class HybridRolloutHandle:
    meta_key: str
    meta_size: int
    tensor_keys: list[str]
    tensor_sizes: list[int]
    padded_sizes: list[int] | None = None  # None = legacy multi-key


def _pack_ragged_1d_int32(list_of_arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = np.array([a.shape[0] for a in list_of_arrays], dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.concatenate(list_of_arrays, axis=0, dtype=np.int32)
    return torch.from_numpy(flat), torch.from_numpy(offsets)


def _pack_ragged_1d_float32(list_of_lists: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = np.array([len(x) for x in list_of_lists], dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.empty(int(offsets[-1]), dtype=np.float32)
    pos = 0
    for xs in list_of_lists:
        n = len(xs)
        flat[pos : pos + n] = np.asarray(xs, dtype=np.float32)
        pos += n
    return torch.from_numpy(flat), torch.from_numpy(offsets)


def _unpack_ragged_1d_int32(flat: torch.Tensor, offsets: torch.Tensor) -> list[np.ndarray]:
    flat_np = flat.cpu().numpy().astype(np.int32, copy=False)
    off = offsets.cpu().numpy().astype(np.int64, copy=False)
    return [flat_np[int(off[i]) : int(off[i + 1])] for i in range(len(off) - 1)]


def _unpack_ragged_1d_float32(flat: torch.Tensor, offsets: torch.Tensor) -> list[list[float]]:
    flat_np = flat.cpu().numpy().astype(np.float32, copy=False)
    off = offsets.cpu().numpy().astype(np.int64, copy=False)
    return [flat_np[int(off[i]) : int(off[i + 1])].tolist() for i in range(len(off) - 1)]


def _unpack_ragged(data: dict) -> dict:
    """Standalone unpack for ragged arrays (used by benchmarks)."""
    if "tokens" in data and isinstance(data["tokens"], dict) and data["tokens"].get("__ragged_type__") == "1d_int32":
        data["tokens"] = _unpack_ragged_1d_int32(data["tokens"]["flat"], data["tokens"]["offsets"])
    if "labels" in data and isinstance(data["labels"], dict) and data["labels"].get("__ragged_type__") == "1d_int32":
        data["labels"] = _unpack_ragged_1d_int32(data["labels"]["flat"], data["labels"]["offsets"])
    if (
        "rollout_log_probs" in data
        and isinstance(data["rollout_log_probs"], dict)
        and data["rollout_log_probs"].get("__ragged_type__") == "1d_float32"
    ):
        data["rollout_log_probs"] = _unpack_ragged_1d_float32(
            data["rollout_log_probs"]["flat"],
            data["rollout_log_probs"]["offsets"],
        )
    return data


# --- Numpy meta format (SLIME_USE_NUMPY_META=1): struct + numpy, no pickle ---
NUMPY_META_MAGIC = b"SLM1"
NUMPY_META_VERSION = 1


def _serialize_rollout_meta_numpy(rollout: dict) -> bytes:
    """Serialize scalar metadata to compact binary. Tensor refs not included."""
    n = len(rollout.get("partition", rollout.get("response_lengths", [])))
    if n == 0:
        n = len(rollout.get("total_lengths", []))
    has_labels = "labels" in rollout and isinstance(rollout["labels"], list)
    has_routed = "rollout_routed_experts" in rollout and isinstance(rollout["rollout_routed_experts"], list)
    num_layers = rollout.get("num_layers", 64)
    moe_topk = rollout.get("moe_router_topk", 2)

    parts = [
        NUMPY_META_MAGIC,
        struct.pack("<HH", NUMPY_META_VERSION, 0),
        struct.pack("<i", n),
        struct.pack("<b", 1 if has_labels else 0),
        struct.pack("<b", 1 if has_routed else 0),
        struct.pack("<ii", num_layers, moe_topk),
        struct.pack("<i", 0),  # reserved
    ]
    partition = np.array(rollout.get("partition", list(range(n))), dtype=np.int32)
    response_lengths = np.array(rollout.get("response_lengths", [0] * n), dtype=np.int32)
    rewards = np.array(rollout.get("rewards", [1.0] * n), dtype=np.float32)
    total_lengths = np.array(rollout.get("total_lengths", [0] * n), dtype=np.int32)
    loss_masks = rollout.get("loss_masks", [])
    loss_lengths = np.array([len(m) for m in loss_masks], dtype=np.int32)
    loss_packed = np.array([x for m in loss_masks for x in m], dtype=np.int32)
    parts.extend([
        partition.tobytes(),
        response_lengths.tobytes(),
        rewards.tobytes(),
        total_lengths.tobytes(),
        loss_lengths.tobytes(),
        loss_packed.tobytes(),
    ])
    raw_reward = rollout.get("raw_reward")
    if raw_reward is not None:
        parts.append(np.array(raw_reward, dtype=np.float32).tobytes())
    return b"".join(parts)


def _deserialize_rollout_meta_numpy(
    meta_bytes: bytes, ready: list[torch.Tensor], has_labels: bool, has_routed: bool, n: int
) -> dict:
    """Deserialize meta_bytes + ready tensors into rollout dict."""
    # Header: magic(4) + ver(2) + reserved(2) + n(4) + has_labels(1) + has_routed(1) + num_layers(4) + moe_topk(4) + reserved(4) = 26
    offset = 26
    partition = np.frombuffer(meta_bytes, dtype=np.int32, count=n, offset=offset)
    offset += n * 4
    response_lengths = np.frombuffer(meta_bytes, dtype=np.int32, count=n, offset=offset)
    offset += n * 4
    rewards = np.frombuffer(meta_bytes, dtype=np.float32, count=n, offset=offset)
    offset += n * 4
    total_lengths = np.frombuffer(meta_bytes, dtype=np.int32, count=n, offset=offset)
    offset += n * 4
    loss_lengths = np.frombuffer(meta_bytes, dtype=np.int32, count=n, offset=offset)
    offset += n * 4
    loss_total = int(loss_lengths.sum())
    loss_packed = np.frombuffer(meta_bytes, dtype=np.int32, count=loss_total, offset=offset)
    offset += loss_total * 4
    loss_masks = []
    pos = 0
    for L in loss_lengths:
        loss_masks.append(loss_packed[pos : pos + int(L)].tolist())
        pos += int(L)

    idx = 0
    tokens_flat = ready[idx]
    tokens_off = ready[idx + 1]
    idx += 2
    log_probs_flat = ready[idx]
    log_probs_off = ready[idx + 1]
    idx += 2
    labels_flat, labels_off = None, None
    if has_labels:
        labels_flat = ready[idx]
        labels_off = ready[idx + 1]
        idx += 2
    routed: list[torch.Tensor] = []
    if has_routed:
        for _ in range(n):
            routed.append(ready[idx])
            idx += 1

    data = {
        "partition": partition.tolist(),
        "response_lengths": response_lengths.tolist(),
        "rewards": rewards.tolist(),
        "loss_masks": loss_masks,
        "total_lengths": total_lengths.tolist(),
        "tokens": _unpack_ragged_1d_int32(tokens_flat, tokens_off),
        "rollout_log_probs": _unpack_ragged_1d_float32(log_probs_flat, log_probs_off),
    }
    if has_labels and labels_flat is not None:
        data["labels"] = _unpack_ragged_1d_int32(labels_flat, labels_off)
    if has_routed and routed:
        data["rollout_routed_experts"] = [r.cpu().numpy() for r in routed]
    return data


def _prepare_rollout_for_numpy(
    rollout: dict, profile_out: dict | None = None
) -> tuple[bytes, list[torch.Tensor]]:
    """Pack ragged arrays, serialize meta with numpy format, return (meta_bytes, tensor_buffers).
    Tensor order: tokens_flat, tokens_off, log_probs_flat, log_probs_off, [labels_flat, labels_off], [routed_i...]
    """
    t0 = time.perf_counter()
    tensors: list[torch.Tensor] = []

    if "tokens" in rollout and isinstance(rollout["tokens"], list):
        flat, off = _pack_ragged_1d_int32(rollout["tokens"])
        tensors.extend([flat, off])
    if "rollout_log_probs" in rollout and isinstance(rollout["rollout_log_probs"], list):
        flat, off = _pack_ragged_1d_float32(rollout["rollout_log_probs"])
        tensors.extend([flat, off])
    if "labels" in rollout and isinstance(rollout["labels"], list):
        flat, off = _pack_ragged_1d_int32(rollout["labels"])
        tensors.extend([flat, off])
    if "rollout_routed_experts" in rollout and isinstance(rollout["rollout_routed_experts"], list):
        for arr in rollout["rollout_routed_experts"]:
            t = torch.from_numpy(np.asarray(arr, dtype=np.int32))
            tensors.append(t)

    meta_bytes = _serialize_rollout_meta_numpy(rollout)
    if profile_out is not None:
        profile_out["pack_ms"] = (time.perf_counter() - t0) * 1000
        profile_out["pickle_ms"] = 0
        profile_out["buffer_convert_ms"] = 0
    return meta_bytes, tensors


def _prepare_rollout_for_pickle(
    rollout: dict, profile_out: dict | None = None
) -> tuple[bytes, list[torch.Tensor]]:
    """Pack ragged arrays, pickle with OOB, return (meta_bytes, tensor_buffers)."""
    t0 = time.perf_counter()
    data = rollout.copy()
    if "tokens" in data and isinstance(data["tokens"], list):
        flat, off = _pack_ragged_1d_int32(data["tokens"])
        data["tokens"] = {"__ragged_type__": "1d_int32", "flat": flat, "offsets": off}
    if "labels" in data and isinstance(data["labels"], list):
        flat, off = _pack_ragged_1d_int32(data["labels"])
        data["labels"] = {"__ragged_type__": "1d_int32", "flat": flat, "offsets": off}
    if "rollout_log_probs" in data and isinstance(data["rollout_log_probs"], list):
        flat, off = _pack_ragged_1d_float32(data["rollout_log_probs"])
        data["rollout_log_probs"] = {"__ragged_type__": "1d_float32", "flat": flat, "offsets": off}
    t_after_pack = time.perf_counter()

    buffers = []

    def _cb(b):
        buffers.append(b)

    meta_bytes = pickle.dumps(data, protocol=5, buffer_callback=_cb)
    t_after_pickle = time.perf_counter()
    tensors = []
    for b in buffers:
        arr = np.array(memoryview(b), copy=False)
        tensors.append(torch.from_numpy(arr))
    if profile_out is not None:
        profile_out["pack_ms"] = (t_after_pack - t0) * 1000
        profile_out["pickle_ms"] = (t_after_pickle - t_after_pack) * 1000
        profile_out["buffer_convert_ms"] = (time.perf_counter() - t_after_pickle) * 1000
    return meta_bytes, tensors


def _pad8(x: int) -> int:
    return (x + 7) // 8 * 8


def _pack_ragged_1d_int32_into(
    buf: np.ndarray, offset: int, list_of_arrays: list[np.ndarray]
) -> tuple[int, int, int]:
    """Pack list of int32 arrays into buf at offset. Returns (flat_bytes, offsets_bytes, total_bytes).
    Flat is padded to 8-byte boundary so offsets (int64) are aligned."""
    lengths = np.array([a.shape[0] for a in list_of_arrays], dtype=np.int64)
    n_flat = int(lengths.sum())
    n_off = len(lengths) + 1
    flat_bytes = n_flat * 4
    flat_padded = _pad8(flat_bytes)
    flat = buf.view(np.uint8)[offset : offset + flat_bytes].view(np.int32)
    off_arr = buf.view(np.uint8)[offset + flat_padded : offset + flat_padded + n_off * 8].view(np.int64)
    np.cumsum(lengths, out=off_arr[1:])
    off_arr[0] = 0
    flat[:] = np.concatenate(list_of_arrays, axis=0, dtype=np.int32)
    return flat_bytes, n_off * 8, flat_padded + n_off * 8


def _pack_ragged_1d_int32_into_split(
    buf: np.ndarray,
    offset_flat: int,
    offset_off: int,
    list_of_arrays: list[np.ndarray],
) -> tuple[int, int]:
    """Pack flat and offsets at separate offsets. Returns (flat_bytes, off_bytes)."""
    lengths = np.array([a.shape[0] for a in list_of_arrays], dtype=np.int64)
    n_flat = int(lengths.sum())
    n_off = len(lengths) + 1
    flat_bytes = n_flat * 4
    off_bytes = n_off * 8
    flat = buf.view(np.uint8)[offset_flat : offset_flat + flat_bytes].view(np.int32)
    off_arr = buf.view(np.uint8)[offset_off : offset_off + off_bytes].view(np.int64)
    np.cumsum(lengths, out=off_arr[1:])
    off_arr[0] = 0
    flat[:] = np.concatenate(list_of_arrays, axis=0, dtype=np.int32)
    return flat_bytes, off_bytes


def _pack_ragged_1d_float32_into_split(
    buf: np.ndarray,
    offset_flat: int,
    offset_off: int,
    list_of_lists: list[list[float]],
) -> tuple[int, int]:
    """Pack flat and offsets at separate offsets. Returns (flat_bytes, off_bytes)."""
    lengths = np.array([len(x) for x in list_of_lists], dtype=np.int64)
    n_flat = int(lengths.sum())
    n_off = len(lengths) + 1
    flat_bytes = n_flat * 4
    off_bytes = n_off * 8
    flat = buf.view(np.uint8)[offset_flat : offset_flat + flat_bytes].view(np.float32)
    off_arr = buf.view(np.uint8)[offset_off : offset_off + off_bytes].view(np.int64)
    np.cumsum(lengths, out=off_arr[1:])
    off_arr[0] = 0
    pos = 0
    for xs in list_of_lists:
        n = len(xs)
        flat[pos : pos + n] = np.asarray(xs, dtype=np.float32)
        pos += n
    return flat_bytes, off_bytes


def _pack_ragged_1d_float32_into(
    buf: np.ndarray, offset: int, list_of_lists: list[list[float]]
) -> tuple[int, int, int]:
    """Pack list of float lists into buf at offset. Returns (flat_bytes, offsets_bytes, total_bytes).
    Flat is padded to 8-byte boundary so offsets (int64) are aligned."""
    lengths = np.array([len(x) for x in list_of_lists], dtype=np.int64)
    n_flat = int(lengths.sum())
    n_off = len(lengths) + 1
    flat_bytes = n_flat * 4
    flat_padded = _pad8(flat_bytes)
    flat = buf.view(np.uint8)[offset : offset + flat_bytes].view(np.float32)
    off_arr = buf.view(np.uint8)[offset + flat_padded : offset + flat_padded + n_off * 8].view(np.int64)
    np.cumsum(lengths, out=off_arr[1:])
    off_arr[0] = 0
    pos = 0
    for xs in list_of_lists:
        n = len(xs)
        flat[pos : pos + n] = np.asarray(xs, dtype=np.float32)
        pos += n
    return flat_bytes, n_off * 8, flat_padded + n_off * 8


def _prepare_rollout_direct_pack(
    rollout: dict,
    store,
    profile_out: dict | None = None,
) -> tuple[bytes, list[torch.Tensor], tuple[int, int] | None, object]:
    """
    Scheme C: Pack directly into a single buffer, no intermediate tensor allocation.
    Returns (meta_bytes, tensor_views, registered_range, buffer_holder).
    registered_range=(ptr, size) if buffer was registered; None if from alloc_from_mem_pool.
    buffer_holder keeps the buffer alive.
    """
    import ctypes

    t0 = time.perf_counter()
    data = rollout.copy()

    def _pad64(x: int) -> int:
        return (x + 63) // 64 * 64

    # Compute sizes (flat padded to 8B for int64 offsets alignment)
    sizes: list[int] = []
    if "tokens" in data and isinstance(data["tokens"], list):
        n_flat = sum(a.shape[0] for a in data["tokens"])
        n_off = len(data["tokens"]) + 1
        sizes.append(_pad64(_pad8(n_flat * 4) + n_off * 8))
    if "labels" in data and isinstance(data["labels"], list):
        n_flat = sum(a.shape[0] for a in data["labels"])
        n_off = len(data["labels"]) + 1
        sizes.append(_pad64(_pad8(n_flat * 4) + n_off * 8))
    if "rollout_log_probs" in data and isinstance(data["rollout_log_probs"], list):
        n_flat = sum(len(x) for x in data["rollout_log_probs"])
        n_off = len(data["rollout_log_probs"]) + 1
        sizes.append(_pad64(_pad8(n_flat * 4) + n_off * 8))

    total = sum(sizes)
    if total == 0:
        meta_bytes, tensors = _prepare_rollout_for_pickle(rollout, profile_out)
        return meta_bytes, tensors, None, None

    # Allocate: try alloc_from_mem_pool first, fallback to torch.empty
    ptr = 0
    buf_holder: object = None
    registered_range: tuple[int, int] | None = None
    if hasattr(store, "alloc_from_mem_pool"):
        ptr = store.alloc_from_mem_pool(total)
    if ptr == 0:
        buf = torch.empty(total, dtype=torch.uint8)
        ptr = buf.data_ptr()
        store.register_buffer(ptr, total)
        buf_holder = buf
    registered_range = (ptr, total)

    if buf_holder is not None:
        buf_np = buf_holder.numpy()
    else:
        buf_np = np.frombuffer((ctypes.c_byte * total).from_address(ptr), dtype=np.uint8, count=total)
    cur = 0

    if "tokens" in data and isinstance(data["tokens"], list):
        flat_b, _, sz = _pack_ragged_1d_int32_into(buf_np, cur, data["tokens"])
        n_flat = flat_b // 4
        flat_padded = _pad8(flat_b)
        flat_np = buf_np.view(np.int32)[cur // 4 : (cur + flat_b) // 4]
        off_np = buf_np.view(np.int64)[(cur + flat_padded) // 8 : (cur + sz) // 8]
        flat = torch.from_numpy(flat_np)
        off = torch.from_numpy(off_np)
        data["tokens"] = {"__ragged_type__": "1d_int32", "flat": flat, "offsets": off}
        cur += _pad64(sz)

    if "labels" in data and isinstance(data["labels"], list):
        flat_b, _, sz = _pack_ragged_1d_int32_into(buf_np, cur, data["labels"])
        flat_padded = _pad8(flat_b)
        flat_np = buf_np.view(np.int32)[cur // 4 : (cur + flat_b) // 4]
        off_np = buf_np.view(np.int64)[(cur + flat_padded) // 8 : (cur + sz) // 8]
        flat = torch.from_numpy(flat_np)
        off = torch.from_numpy(off_np)
        data["labels"] = {"__ragged_type__": "1d_int32", "flat": flat, "offsets": off}
        cur += _pad64(sz)

    if "rollout_log_probs" in data and isinstance(data["rollout_log_probs"], list):
        flat_b, _, sz = _pack_ragged_1d_float32_into(buf_np, cur, data["rollout_log_probs"])
        flat_padded = _pad8(flat_b)
        flat_np = buf_np.view(np.float32)[cur // 4 : (cur + flat_b) // 4]
        off_np = buf_np.view(np.int64)[(cur + flat_padded) // 8 : (cur + sz) // 8]
        flat = torch.from_numpy(flat_np)
        off = torch.from_numpy(off_np)
        data["rollout_log_probs"] = {"__ragged_type__": "1d_float32", "flat": flat, "offsets": off}

    buffers = []

    def _cb(b):
        buffers.append(b)

    meta_bytes = pickle.dumps(data, protocol=5, buffer_callback=_cb)
    tensor_views = []
    for b in buffers:
        arr = np.array(memoryview(b), copy=False)
        tensor_views.append(torch.from_numpy(arr))
    if profile_out is not None:
        profile_out["pack_ms"] = (time.perf_counter() - t0) * 1000
        profile_out["pickle_ms"] = 0
        profile_out["buffer_convert_ms"] = 0
    return meta_bytes, tensor_views, registered_range, buf_holder


class MooncakeHybridRolloutTransfer:
    """
    Mooncake rollout transfer: Legacy multi-key (default) or Single-Key SGL.
    Implements DataTransferBackend: put(data) -> handle, get(handle) -> data.
    """

    def __init__(
        self,
        tensor_min_bytes: int = 1 * 1024 * 1024,
        enable_auto_cleanup: bool = True,
        use_legacy_path: bool | None = None,
        mount_segment_size: int | None = None,
        cleanup_delay_seconds: float = 5.0,
        cleanup_batch_size: int = 100,
        ring_buffer_size: int | None = None,
        ring_buffer_count: int = 3,
    ):
        self.tensor_min_bytes = tensor_min_bytes
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_delay_seconds = cleanup_delay_seconds
        self.cleanup_batch_size = cleanup_batch_size
        if ring_buffer_size is None:
            ring_buffer_size = int(
                os.environ.get("SLIME_RING_BUFFER_SIZE_MB", "2048")
            ) * 1024 * 1024
        self._ring_buffer_size = ring_buffer_size
        self._ring_buffer_count = ring_buffer_count
        if use_legacy_path is None:
            use_legacy_path = os.environ.get("SLIME_USE_LEGACY_TRANSFER", "").lower() in ("1", "true", "yes")
        self._use_legacy = use_legacy_path

        overrides = {"mount_segment_size": mount_segment_size} if mount_segment_size is not None else None
        cfg = MooncakeStoreConfig.load_from_env(overrides=overrides)
        from mooncake.store import MooncakeDistributedStore

        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            cfg.local_hostname,
            cfg.metadata_server,
            cfg.mount_segment_size,
            cfg.local_buffer_size,
            cfg.protocol,
            cfg.device_name or "",
            cfg.master_server_address,
        )
        if ret:
            raise RuntimeError(f"Mooncake setup failed: {ret}")

        # Ring buffer: pre-allocated slots for single-key get (zero cold-start)
        self._ring_slots: list[torch.Tensor] = []
        self._ring_available: list[int] = []
        self._buffer_origin: dict[int, int] = {}  # ptr -> ring_slot_idx
        if self._ring_buffer_size > 0 and self._ring_buffer_count > 0:
            align = 4 * 1024 * 1024
            sz = ((self._ring_buffer_size + align - 1) // align) * align
            for i in range(self._ring_buffer_count):
                buf = torch.empty(sz, dtype=torch.uint8)
                self._store.register_buffer(buf.data_ptr(), sz)
                self._ring_slots.append(buf)
                self._ring_available.append(i)
            logger.info(
                "Ring buffer: %d slots x %d MB (get cold-start eliminated)",
                self._ring_buffer_count,
                sz // (1024 * 1024),
            )

        # Mem pool: overflow / legacy path
        self._get_pool: list[torch.Tensor] = []
        self._pool_max = 4096

        # Per-put buffers (legacy/single-key need registered memory)
        self._put_buffers: dict[int, torch.Tensor] = {}
        self._registered_ptrs: set[int] = set()

        # Single buffer for meta + headers (one put at a time)
        self._meta_cap = 32 * 1024 * 1024
        self._header_cap = 4096 * 40
        self._meta_buf = torch.empty(self._meta_cap, dtype=torch.uint8)
        self._header_buf = torch.empty(self._header_cap, dtype=torch.uint8)
        self._store.register_buffer(self._meta_buf.data_ptr(), self._meta_cap)
        self._store.register_buffer(self._header_buf.data_ptr(), self._header_cap)

        # Async cleanup: delayed deletion after get_rollout (like MooncakeDataTransfer)
        self._pending_deletion = queue.PriorityQueue()
        self._cleanup_thread = None
        self._cleanup_thread_lock = threading.Lock()
        self._cleanup_stop_event = threading.Event()
        if self.enable_auto_cleanup:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, name="MooncakeHybridCleanupThread", daemon=True
        )
        self._cleanup_thread.start()
        logger.info("Mooncake hybrid cleanup thread started")

    def _cleanup_worker(self) -> None:
        """Background worker that deletes keys when their deletion time is reached."""
        while not self._cleanup_stop_event.is_set():
            try:
                keys_to_delete: list[str] = []
                sleep_until: float | None = None
                while len(keys_to_delete) < self.cleanup_batch_size:
                    try:
                        deletion_time, key = self._pending_deletion.get(timeout=0.5)
                    except queue.Empty:
                        break
                    current_time = time.time()
                    if current_time >= deletion_time:
                        keys_to_delete.append(key)
                    else:
                        self._pending_deletion.put((deletion_time, key))
                        sleep_until = deletion_time
                        break
                if keys_to_delete:
                    for key in keys_to_delete:
                        try:
                            result = self._store.remove(key)
                            if result != 0:
                                logger.warning("Failed to delete key %s, error code: %s", key, result)
                        except Exception as e:
                            logger.warning("Exception while deleting key %s: %s", key, e)
                if sleep_until is not None:
                    sleep_time = min(0.5, max(0.1, sleep_until - time.time()))
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    time.sleep(0.5)
            except Exception as e:
                logger.error("Error in cleanup worker: %s", e, exc_info=True)
                time.sleep(1.0)

    def _schedule_handle_deletion(self, handle: HybridRolloutHandle) -> None:
        """Schedule all keys for a handle for deletion after the delay period."""
        deletion_time = time.time() + self.cleanup_delay_seconds
        keys = [handle.meta_key] + list(handle.tensor_keys)
        for key in keys:
            self._pending_deletion.put((deletion_time, key))

    def put(self, data: dict) -> HybridRolloutHandle:
        return self.put_rollout(data)

    def get(self, handle: HybridRolloutHandle, auto_cleanup: bool | None = None) -> dict:
        return self.get_rollout(handle, auto_cleanup=auto_cleanup)

    def _alloc_get_buffer(
        self, size: int, profile: dict | None = None, use_ring: bool = True
    ) -> torch.Tensor:
        t0 = time.perf_counter() if profile is not None else None
        # 1. Ring: single-key path, size fits, slot available
        if (
            use_ring
            and self._ring_slots
            and size <= self._ring_slots[0].numel()
            and self._ring_available
        ):
            idx = self._ring_available.pop()
            buf = self._ring_slots[idx]
            self._buffer_origin[buf.data_ptr()] = idx
            if profile is not None:
                profile["alloc_ms"] = (time.perf_counter() - t0) * 1000
                profile["alloc_from_pool"] = True  # ring counts as "warm"
            return buf
        # 2. Pool: best-fit
        best_i, best_sz = -1, float("inf")
        for i, b in enumerate(self._get_pool):
            if b.numel() >= size and b.numel() < best_sz:
                best_sz, best_i = b.numel(), i
        if best_i >= 0:
            buf = self._get_pool.pop(best_i)
            if profile is not None:
                profile["alloc_ms"] = (time.perf_counter() - t0) * 1000
                profile["alloc_from_pool"] = True
            return buf
        # 3. Dynamic alloc
        align = 4 * 1024 * 1024
        sz = ((size + align - 1) // align) * align
        buf = torch.empty(sz, dtype=torch.uint8)
        self._store.register_buffer(buf.data_ptr(), sz)
        if profile is not None:
            profile["alloc_ms"] = (time.perf_counter() - t0) * 1000
            profile["alloc_from_pool"] = False
        return buf

    def _return_get_buffer(self, buf: torch.Tensor) -> None:
        ptr = buf.data_ptr()
        if ptr in self._buffer_origin:
            idx = self._buffer_origin.pop(ptr)
            self._ring_available.append(idx)
            return
        if len(self._get_pool) < self._pool_max:
            self._get_pool.append(buf)
        else:
            try:
                self._store.unregister_buffer(ptr)
            except Exception:
                pass

    def _get_put_buffer(self, idx: int, size: int) -> torch.Tensor:
        if idx not in self._put_buffers or self._put_buffers[idx].numel() < size:
            if idx in self._put_buffers:
                old_ptr = self._put_buffers[idx].data_ptr()
                self._registered_ptrs.discard(old_ptr)
                try:
                    self._store.unregister_buffer(old_ptr)
                except Exception:
                    pass
            align = 4 * 1024 * 1024
            sz = ((size + align - 1) // align) * align
            buf = torch.empty(sz, dtype=torch.uint8)
            self._store.register_buffer(buf.data_ptr(), sz)
            self._registered_ptrs.add(buf.data_ptr())
            self._put_buffers[idx] = buf
        return self._put_buffers[idx]

    def put_rollout(
        self, rollout: dict, profile_out: dict | None = None
    ) -> HybridRolloutHandle:
        t0 = time.perf_counter()
        use_numpy_meta = os.environ.get("SLIME_USE_NUMPY_META", "").lower() in ("1", "true")
        use_direct_pack = os.environ.get("SLIME_PACK_DIRECT_TO_BUFFER", "").lower() in ("1", "true")
        if use_numpy_meta and not self._use_legacy:
            meta_bytes, tensors = _prepare_rollout_for_numpy(rollout, profile_out)
            registered_range = None
        elif use_direct_pack and not self._use_legacy:
            meta_bytes, tensors, registered_range, _buf_holder = _prepare_rollout_direct_pack(
                rollout, self._store, profile_out
            )
        else:
            meta_bytes, tensors = _prepare_rollout_for_pickle(rollout, profile_out)
            registered_range = None
        t_prepare = (time.perf_counter() - t0) * 1000
        meta_size = len(meta_bytes)
        rid = str(uuid.uuid4())

        use_split_keys = os.environ.get("SLIME_META_TENSOR_SPLIT_KEYS", "").lower() in ("1", "true")
        if self._use_legacy:
            handle = self._put_legacy(
                rid, meta_bytes, meta_size, tensors, profile_out
            )
        elif use_split_keys and not use_numpy_meta:
            handle = self._put_two_key(
                rid, meta_bytes, meta_size, tensors, profile_out
            )
        else:
            handle = self._put_single_key(
                rid, meta_bytes, meta_size, tensors, profile_out,
                registered_range=registered_range,
            )

        if profile_out is not None:
            profile_out["prepare_ms"] = t_prepare
            profile_out["prepare_bytes"] = meta_size
            profile_out["num_tensors"] = len(tensors)
            profile_out["total_bytes"] = meta_size + sum(
                t.numel() * t.element_size() for t in tensors
            )
            profile_out["put_total_ms"] = (time.perf_counter() - t0) * 1000
        return handle

    def _put_legacy(
        self,
        rid: str,
        meta_bytes: bytes,
        meta_size: int,
        tensors: list[torch.Tensor],
        profile_out: dict | None = None,
    ) -> HybridRolloutHandle:
        """Legacy: meta + each tensor in separate key."""
        t0 = time.perf_counter()
        if meta_size > self._meta_cap:
            raise RuntimeError(f"Meta size {meta_size} > capacity {self._meta_cap}")
        self._meta_buf[:meta_size] = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8)

        keys = []
        ptrs_list = []
        sizes_list = []
        tensor_sizes = []

        # Meta key: header(40) + meta
        hdr = struct.pack("iiqqqq", 3, 1, meta_size, -1, -1, -1)
        self._header_buf[:40] = torch.tensor(bytearray(hdr), dtype=torch.uint8)
        keys.append(f"rollout:{rid}:meta")
        ptrs_list.append([self._header_buf.data_ptr(), self._meta_buf.data_ptr()])
        sizes_list.append([40, meta_size])
        tensor_sizes.append(meta_size)

        for i, t in enumerate(tensors):
            if not t.is_contiguous():
                t = t.contiguous()
            ptr = t.data_ptr()
            size = t.numel() * t.element_size()
            if ptr not in self._registered_ptrs:
                buf = self._get_put_buffer(i, size)
                buf[:size].view(t.dtype).reshape(t.shape).copy_(t)
                ptr = buf.data_ptr()
            hdr = struct.pack(
                "iiqqqq",
                DTYPE_MAP.get(t.dtype, 3),
                t.ndim,
                *t.shape,
                *([-1] * (4 - t.ndim)),
            )
            self._header_buf[(i + 1) * 40 : (i + 2) * 40] = torch.tensor(bytearray(hdr), dtype=torch.uint8)
            keys.append(f"rollout:{rid}:{i}")
            ptrs_list.append([self._header_buf.data_ptr() + (i + 1) * 40, ptr])
            sizes_list.append([40, size])
            tensor_sizes.append(size)

        t_before_put = time.perf_counter()
        ret = self._store.batch_put_from_multi_buffers(keys, ptrs_list, sizes_list)
        if profile_out is not None:
            profile_out["buffer_prep_ms"] = (t_before_put - t0) * 1000
            profile_out["batch_put_ms"] = (time.perf_counter() - t_before_put) * 1000
        for r in ret:
            if r != 0:
                raise RuntimeError(f"batch_put_from_multi_buffers failed: {ret}")

        return HybridRolloutHandle(
            meta_key=keys[0],
            meta_size=meta_size,
            tensor_keys=keys[1:],
            tensor_sizes=tensor_sizes[1:],
            padded_sizes=None,
        )

    def _put_two_key(
        self,
        rid: str,
        meta_bytes: bytes,
        meta_size: int,
        tensors: list[torch.Tensor],
        profile_out: dict | None = None,
    ) -> HybridRolloutHandle:
        """Two-key: meta in one key, concatenated tensors in another (pickle OOB only)."""
        t0 = time.perf_counter()
        if meta_size > self._meta_cap:
            raise RuntimeError(f"Meta size {meta_size} > capacity {self._meta_cap}")
        self._meta_buf[:meta_size] = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8)

        meta_key = f"rollout:{rid}:meta"
        tensor_key = f"rollout:{rid}:tensors"

        hdr = struct.pack("iiqqqq", 3, 1, meta_size, -1, -1, -1)
        self._header_buf[:40] = torch.tensor(bytearray(hdr), dtype=torch.uint8)

        ptrs_list = []
        sizes_list = []
        tensor_sizes = []

        meta_ptrs = [self._header_buf.data_ptr(), self._meta_buf.data_ptr()]
        meta_sizes = [40, meta_size]
        ptrs_list.append(meta_ptrs)
        sizes_list.append(meta_sizes)

        tensor_ptrs = []
        tensor_sizes_out = []
        for i, t in enumerate(tensors):
            if not t.is_contiguous():
                t = t.contiguous()
            ptr = t.data_ptr()
            size = t.numel() * t.element_size()
            if ptr not in self._registered_ptrs:
                buf = self._get_put_buffer(i, size)
                buf[:size].view(t.dtype).reshape(t.shape).copy_(t)
                ptr = buf.data_ptr()
            tensor_ptrs.append(ptr)
            tensor_sizes_out.append(size)

        ptrs_list.append(tensor_ptrs)
        sizes_list.append(tensor_sizes_out)

        keys = [meta_key, tensor_key]
        t_before_put = time.perf_counter()
        ret = self._store.batch_put_from_multi_buffers(keys, ptrs_list, sizes_list)
        if profile_out is not None:
            profile_out["buffer_prep_ms"] = (t_before_put - t0) * 1000
            profile_out["batch_put_ms"] = (time.perf_counter() - t_before_put) * 1000
        for r in ret:
            if r != 0:
                raise RuntimeError(f"batch_put_from_multi_buffers failed: {ret}")

        return HybridRolloutHandle(
            meta_key=meta_key,
            meta_size=meta_size,
            tensor_keys=[tensor_key],
            tensor_sizes=tensor_sizes_out,
            padded_sizes=tensor_sizes_out,
        )

    def _put_single_key(
        self,
        rid: str,
        meta_bytes: bytes,
        meta_size: int,
        tensors: list[torch.Tensor],
        profile_out: dict | None = None,
        registered_range: tuple[int, int] | None = None,
    ) -> HybridRolloutHandle:
        """Single-Key SGL: one key, one contiguous SGL."""
        t0 = time.perf_counter()
        n = len(tensors) + 1
        if n > self._header_cap // 40:
            raise RuntimeError(f"Too many tensors: {n}")
        if meta_size > self._meta_cap:
            raise RuntimeError(f"Meta size {meta_size} > capacity {self._meta_cap}")

        t_meta = time.perf_counter()
        self._meta_buf[:meta_size] = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8)
        meta_t = self._meta_buf[:meta_size]
        all_t = [meta_t] + tensors
        t_after_meta = time.perf_counter()

        ptrs, sizes, padded, actual = [], [], [], []
        temp_registered: list[int] = []
        use_register_inplace = os.environ.get("SLIME_REGISTER_PICKLE_BUFFERS", "").lower() in ("1", "true")
        rng_ptr, rng_size = registered_range or (0, 0)
        t_copy_tensors = 0.0
        t_headers = 0.0
        for i, t in enumerate(all_t):
            # Ensure contiguous: Mooncake reads raw bytes from ptr; non-contiguous tensors have gaps
            if not t.is_contiguous():
                t = t.contiguous()
            ptr = t.data_ptr()
            sz = t.numel() * t.element_size()
            in_registered_range = registered_range is not None and ptr >= rng_ptr and ptr + sz <= rng_ptr + rng_size
            _t0 = time.perf_counter()
            if i > 0 and ptr not in self._registered_ptrs and not in_registered_range:
                if use_register_inplace:
                    try:
                        self._store.register_buffer(ptr, sz)
                        self._registered_ptrs.add(ptr)
                        temp_registered.append(ptr)
                    except Exception:
                        use_register_inplace = False
                        buf = self._get_put_buffer(i - 1, sz)
                        buf[:sz].view(t.dtype).reshape(t.shape).copy_(t)
                        ptr = buf.data_ptr()
                else:
                    buf = self._get_put_buffer(i - 1, sz)
                    buf[:sz].view(t.dtype).reshape(t.shape).copy_(t)
                    ptr = buf.data_ptr()
            t_copy_tensors += time.perf_counter() - _t0
            pad = (sz + 63) // 64 * 64
            padded.append(pad)
            actual.append(sz)
            _t1 = time.perf_counter()
            hdr = struct.pack(
                "iiqqqq",
                DTYPE_MAP.get(t.dtype, 3),
                t.ndim,
                *t.shape,
                *([-1] * (4 - t.ndim)),
            )
            self._header_buf[i * 40 : (i + 1) * 40] = torch.tensor(bytearray(hdr), dtype=torch.uint8)
            t_headers += time.perf_counter() - _t1
            ptrs.extend([self._header_buf.data_ptr() + i * 40, ptr])
            sizes.extend([40, pad])

        t_before_put = time.perf_counter()
        key = f"rollout:{rid}"
        if os.environ.get("MC_DEBUG_LOCAL_MEMCPY") in ("1", "true"):
            total_sz = sum(sizes)
            logger.info(
                "[MC_DEBUG] batch_put_from_multi_buffers: key=%s num_slices=%d total_bytes=%d ptrs=%s",
                key, len(ptrs), total_sz,
                [(hex(p), s) for p, s in zip(ptrs, sizes)][:6],
            )
            if len(ptrs) > 6:
                logger.info("[MC_DEBUG]   ... and %d more slices", len(ptrs) - 6)
        ret = self._store.batch_put_from_multi_buffers([key], [ptrs], [sizes])
        if profile_out is not None:
            profile_out["buffer_prep_ms"] = (t_before_put - t0) * 1000
            profile_out["meta_copy_ms"] = (t_after_meta - t_meta) * 1000
            profile_out["copy_tensors_ms"] = t_copy_tensors * 1000
            profile_out["copy_headers_ms"] = t_headers * 1000
            profile_out["batch_put_ms"] = (time.perf_counter() - t_before_put) * 1000
        for r in ret:
            if r != 0:
                raise RuntimeError(f"batch_put_from_multi_buffers failed: {ret}")
        for ptr in temp_registered:
            self._registered_ptrs.discard(ptr)
            try:
                self._store.unregister_buffer(ptr)
            except Exception:
                pass

        return HybridRolloutHandle(
            meta_key=key,
            meta_size=meta_size,
            tensor_keys=[],
            tensor_sizes=actual[1:],
            padded_sizes=padded,
        )

    def get_rollout(
        self,
        handle: HybridRolloutHandle,
        return_packed: bool = False,
        auto_cleanup: bool | None = None,
        profile_out: list | None = None,
    ) -> dict:
        if handle.padded_sizes is not None and len(handle.tensor_keys) == 1:
            data = self._get_two_key(handle, return_packed, profile_out)
        elif handle.padded_sizes is not None:
            data = self._get_single_key(handle, return_packed, profile_out)
        else:
            data = self._get_legacy(handle, return_packed, profile_out)
        should_cleanup = auto_cleanup if auto_cleanup is not None else self.enable_auto_cleanup
        if should_cleanup:
            with self._cleanup_thread_lock:
                if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
                    self._start_cleanup_thread()
                self._schedule_handle_deletion(handle)
        return data

    def _get_two_key(
        self, handle: HybridRolloutHandle, return_packed: bool, profile_out: list | None = None
    ) -> dict:
        """Two-key: meta in one key, concatenated tensors in another (pickle OOB only)."""
        profile = {} if profile_out is not None else None
        meta_size = handle.meta_size
        tensor_sizes = handle.tensor_sizes
        total_meta = 40 + meta_size
        total_tensors = sum(tensor_sizes)

        alloc_t0 = time.perf_counter() if profile is not None else None
        buf_meta = self._alloc_get_buffer(total_meta, None, use_ring=False)
        buf_tensors = self._alloc_get_buffer(total_tensors, None, use_ring=False)
        if profile is not None:
            profile["alloc_ms"] = (time.perf_counter() - alloc_t0) * 1000

        batch_t0 = time.perf_counter() if profile is not None else None
        rets = self._store.batch_get_into(
            [handle.meta_key, handle.tensor_keys[0]],
            [buf_meta.data_ptr(), buf_tensors.data_ptr()],
            [total_meta, total_tensors],
        )
        for r in rets:
            if r < 0:
                raise RuntimeError(f"batch_get_into failed: {rets}")
        if profile is not None:
            profile["batch_get_ms"] = (time.perf_counter() - batch_t0) * 1000

        unpack_t0 = time.perf_counter() if profile is not None else None
        meta_bytes = memoryview(buf_meta.numpy()[40 : 40 + meta_size]).tobytes()
        self._return_get_buffer(buf_meta)
        offset = 0
        ready = []
        for sz in tensor_sizes:
            v = buf_tensors[offset : offset + sz]
            ready.append(v)
            offset += sz

        class _RefCount:
            def __init__(self, n, pool, b):
                self.n, self.pool, self.b = n, pool, b

            def dec(self):
                self.n -= 1
                if self.n == 0:
                    self.pool(self.b)

        if ready:
            rc = _RefCount(len(ready), self._return_get_buffer, buf_tensors)
            for v in ready:
                weakref.finalize(v, rc.dec)
        else:
            self._return_get_buffer(buf_tensors)

        data = pickle.loads(meta_bytes, buffers=[t.numpy() for t in ready])
        data = self._unpack_ragged(data) if not return_packed else data
        if profile is not None:
            profile["unpack_ms"] = (time.perf_counter() - unpack_t0) * 1000
            profile_out.append(profile)
        return data

    def _get_single_key(
        self, handle: HybridRolloutHandle, return_packed: bool, profile_out: list | None = None
    ) -> dict:
        if not handle.padded_sizes or len(handle.padded_sizes) != len(handle.tensor_sizes) + 1:
            raise ValueError(
                f"Invalid handle: padded_sizes len {len(handle.padded_sizes or [])} "
                f"!= tensor_sizes len {len(handle.tensor_sizes)} + 1"
            )
        profile = {} if profile_out is not None else None
        total = sum(40 + p for p in handle.padded_sizes)
        # Add 64KB margin: Mooncake may require slightly more; avoid -600 Buffer too small
        total = total + 65536
        buf = self._alloc_get_buffer(total, profile, use_ring=True)

        max_retries = 30000
        retry_703 = 0
        batch_get_t0 = time.perf_counter() if profile is not None else None
        for _ in range(max_retries):
            rets = self._store.batch_get_into([handle.meta_key], [buf.data_ptr()], [total])
            ok = True
            for r in rets:
                if r < 0:
                    if r == -703:
                        retry_703 += 1
                        time.sleep(0.001)
                        ok = False
                        break
                    raise RuntimeError(f"batch_get_into failed: {rets}")
            if ok:
                break
        else:
            raise RuntimeError(f"NOT_FOUND after {max_retries} retries: {handle.meta_key}")
        if profile is not None:
            profile["batch_get_ms"] = (time.perf_counter() - batch_get_t0) * 1000
            profile["retry_703"] = retry_703

        unpack_t0 = time.perf_counter() if profile is not None else None
        offset = 40
        if offset + handle.meta_size > total:
            raise RuntimeError(f"Meta overrun: offset {offset} + meta_size {handle.meta_size} > total {total}")
        meta_bytes = memoryview(buf.numpy()[offset : offset + handle.meta_size]).tobytes()
        offset += handle.padded_sizes[0]

        ready = []
        for i, sz in enumerate(handle.tensor_sizes):
            if offset + 40 > total:
                raise RuntimeError(f"Header overrun at tensor {i}: offset {offset} + 40 > total {total}")
            h = struct.unpack("iiqqqq", memoryview(buf.numpy()[offset : offset + 40]))
            dtype = next((dt for dt, e in DTYPE_MAP.items() if e == h[0]), torch.uint8)
            ndim = int(h[1])
            if ndim < 0 or ndim > 4:
                raise RuntimeError(f"Invalid ndim {ndim} at tensor {i}")
            shape = tuple(int(h[2 + j]) for j in range(ndim))
            offset += 40
            if offset + sz > total:
                raise RuntimeError(f"Tensor {i} overrun: offset {offset} + sz {sz} > total {total}")
            v = buf[offset : offset + sz].view(dtype).reshape(shape)
            ready.append(v)
            offset += handle.padded_sizes[i + 1]

        class _RefCount:
            def __init__(self, n, pool, b):
                self.n, self.pool, self.b = n, pool, b

            def dec(self):
                self.n -= 1
                if self.n == 0:
                    self.pool(self.b)

        if ready:
            rc = _RefCount(len(ready), self._return_get_buffer, buf)
            for v in ready:
                weakref.finalize(v, rc.dec)
        else:
            self._return_get_buffer(buf)

        if meta_bytes[:4] == NUMPY_META_MAGIC:
            _, _, _, n, has_labels, has_routed = struct.unpack("<4sHHibb", meta_bytes[:14])
            data = _deserialize_rollout_meta_numpy(meta_bytes, ready, bool(has_labels), bool(has_routed), n)
        else:
            data = pickle.loads(meta_bytes, buffers=[t.numpy() for t in ready])
            data = self._unpack_ragged(data) if not return_packed else data
        if profile is not None:
            profile["unpack_ms"] = (time.perf_counter() - unpack_t0) * 1000
            profile_out.append(profile)
        return data

    def _get_legacy(
        self, handle: HybridRolloutHandle, return_packed: bool, profile_out: list | None = None
    ) -> dict:
        profile = {} if profile_out is not None else None
        keys = [handle.meta_key] + handle.tensor_keys
        sizes = [handle.meta_size] + handle.tensor_sizes

        alloc_t0 = time.perf_counter() if profile is not None else None
        bufs = []
        ptrs, szs = [], []
        for s in sizes:
            b = self._alloc_get_buffer(40 + s, None, use_ring=False)
            bufs.append(b)
            ptrs.append(b.data_ptr())
            szs.append(40 + s)
        if profile is not None:
            profile["alloc_ms"] = (time.perf_counter() - alloc_t0) * 1000
            profile["alloc_from_pool"] = False

        batch_t0 = time.perf_counter() if profile is not None else None
        rets = self._store.batch_get_into(keys, ptrs, szs)
        for r in rets:
            if r < 0:
                raise RuntimeError(f"batch_get_into failed: {rets}")
        if profile is not None:
            profile["batch_get_ms"] = (time.perf_counter() - batch_t0) * 1000
            profile["retry_703"] = 0

        unpack_t0 = time.perf_counter() if profile is not None else None
        meta_bytes = memoryview(bufs[0].numpy()[40 : 40 + handle.meta_size]).tobytes()
        self._return_get_buffer(bufs[0])

        ready = []
        for i, b in enumerate(bufs[1:]):
            h = struct.unpack("iiqqqq", memoryview(b.numpy()[:40]))
            dtype = next((dt for dt, e in DTYPE_MAP.items() if e == h[0]), torch.uint8)
            ndim = h[1]
            shape = tuple(h[2 : 2 + ndim])
            v = b[40 : 40 + handle.tensor_sizes[i]].view(dtype).reshape(shape)
            weakref.finalize(v, self._return_get_buffer, b)
            ready.append(v)

        if meta_bytes[:4] == NUMPY_META_MAGIC:
            _, _, _, n, has_labels, has_routed = struct.unpack("<4sHHibb", meta_bytes[:14])
            data = _deserialize_rollout_meta_numpy(meta_bytes, ready, bool(has_labels), bool(has_routed), n)
        else:
            data = pickle.loads(meta_bytes, buffers=[t.numpy() for t in ready])
            data = self._unpack_ragged(data) if not return_packed else data
        if profile is not None:
            profile["unpack_ms"] = (time.perf_counter() - unpack_t0) * 1000
            profile_out.append(profile)
        return data

    def _unpack_ragged(self, data: dict) -> dict:
        return _unpack_ragged(data)

    def cleanup(self, handle: HybridRolloutHandle) -> None:
        """Immediately remove all keys for this handle (bypasses delayed deletion)."""
        self._store.remove(handle.meta_key)
        for k in handle.tensor_keys:
            self._store.remove(k)

    def shutdown(self) -> None:
        """Stop the cleanup thread gracefully."""
        self._cleanup_stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5.0)
            if self._cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not stop within timeout")
            else:
                logger.info("Cleanup thread stopped successfully")
