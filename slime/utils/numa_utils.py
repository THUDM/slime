import ctypes
import logging
import os
import sys
import threading
from itertools import groupby
from uuid import UUID

from slime.utils import accelerator

logger = logging.getLogger(__name__)


def _cpu_ranges(cpus):
    if cpus is None:
        return "unknown"
    ranges = []
    for _, group in groupby(enumerate(sorted(cpus)), lambda pair: pair[1] - pair[0]):
        values = [cpu for _, cpu in group]
        ranges.append(str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}")
    return ",".join(ranges) or "empty"


def _set_cpu_affinity(rank):
    """Narrow the calling thread's CPUs to those preferred by its selected CUDA GPU."""
    device = identity = before = after = None
    initialized = False
    outcome = "skipped"
    stage = "capability check"
    reason = ""
    try:
        if sys.platform != "linux" or not all(
            hasattr(os, name) for name in ("sched_getaffinity", "sched_setaffinity")
        ):
            reason = "Linux thread affinity APIs unavailable"
            return
        if not accelerator.supports("nvml_affinity"):
            reason = "backend does not support NVML affinity"
            return

        stage = "selected device identity"
        device = accelerator.current_device()
        before = after = os.sched_getaffinity(0)
        device_uuid = getattr(accelerator.get_device_properties(device), "uuid", None)
        if not isinstance(device_uuid, (str, bytes)):
            # PyTorch's _CUuuid exposes its 16 bytes as a list of unsigned integers.
            uuid_bytes = getattr(device_uuid, "bytes", None)
            if not isinstance(uuid_bytes, list) or len(uuid_bytes) != 16:
                reason = "selected device has no supported UUID"
                return
            device_uuid = bytes(uuid_bytes)
        if isinstance(device_uuid, bytes):
            device_uuid = f"GPU-{UUID(bytes=device_uuid)}" if len(device_uuid) == 16 else device_uuid.decode("ascii")
        # shortcut: skip MIG affinity until instance-to-parent resolution is validated.
        if device_uuid.startswith("MIG-"):
            identity = device_uuid
            reason = "MIG affinity is unsupported"
            return
        if not device_uuid.startswith("GPU-"):
            reason = "selected device has no supported GPU UUID"
            return
        identity = f"GPU-{UUID(device_uuid[4:])}"

        stage = "NVML import/initialization"
        import pynvml

        pynvml.nvmlInit()
        initialized = True
        stage = "NVML UUID lookup"
        handle = pynvml.nvmlDeviceGetHandleByUUID(identity)
        stage = "NVML MIG check"
        if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
            reason = "MIG affinity is unsupported"
            return

        stage = "GPU CPU mask query"
        # pynvml represents each affinity word with c_ulonglong.
        word_bits = ctypes.sizeof(ctypes.c_ulonglong) * 8
        mask = pynvml.nvmlDeviceGetCpuAffinity(handle, max(before) // word_bits + 1)
        target = {cpu for cpu in before if mask[cpu // word_bits] & (1 << (cpu % word_bits))}
        if not target:
            reason = "GPU CPU mask does not intersect existing affinity"
            return

        stage = "sched_setaffinity"
        os.sched_setaffinity(0, target)
        outcome = "applied"
        stage = "affinity readback"
        after = None
        after = os.sched_getaffinity(0)
        reason = "effective mask read back"
    except Exception as exc:
        reason = f"{stage} failed ({type(exc).__name__}): {str(exc)[:200]}"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                reason += f"; NVML shutdown failed ({type(exc).__name__}): {str(exc)[:200]}"
        logger.info(
            "CPU affinity (calling thread): rank=%s tid=%s cuda=%s uuid=%s before=%s after=%s %s: %s",
            rank,
            threading.get_native_id(),
            device,
            identity,
            _cpu_ranges(before),
            _cpu_ranges(after),
            outcome,
            reason,
        )
