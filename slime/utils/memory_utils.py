import ctypes
import gc
import logging
import resource
import sys
from functools import lru_cache

import psutil
import torch
import torch.distributed as dist

from slime.utils import accelerator

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cuda_stack_api():
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    driver.cuCtxGetLimit.restype = ctypes.c_int
    driver.cuCtxSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
    driver.cuCtxSetLimit.restype = ctypes.c_int
    return driver


def reset_cuda_stack_size() -> None:
    """Release an enlarged CUDA per-thread stack after model offload."""
    if torch.version.cuda is None or torch.version.hip is not None or not torch.cuda.is_initialized():
        return
    torch.cuda.synchronize()
    driver = _cuda_stack_api()
    previous = ctypes.c_size_t()
    error = driver.cuCtxGetLimit(ctypes.byref(previous), 0)  # CU_LIMIT_STACK_SIZE
    if error:
        raise RuntimeError(f"cuCtxGetLimit(CU_LIMIT_STACK_SIZE) failed: CUDA error {error}")
    if previous.value <= 1024:
        return
    error = driver.cuCtxSetLimit(0, 1024)
    if error:
        raise RuntimeError(f"cuCtxSetLimit(CU_LIMIT_STACK_SIZE) failed: CUDA error {error}")
    logger.info("Reset CUDA stack limit after offload: %d -> 1024 bytes", previous.value)


def get_process_host_memory_gib(process: psutil.Process | None = None) -> tuple[float, float]:
    """Return current and peak resident host memory in GiB."""
    process = process or psutil.Process()
    rss_gib = process.memory_info().rss / 1024**3
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    hwm_gib = max_rss / (1024**2 if sys.platform != "darwin" else 1024**3)
    return rss_gib, hwm_gib


def clear_memory(clear_host_memory: bool = False):
    accelerator.synchronize()
    gc.collect()
    accelerator.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = accelerator.current_device()
    free, total = accelerator.mem_get_info(device)
    vm = psutil.virtual_memory()
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(accelerator.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(accelerator.memory_reserved(device)),
        "host_total_GB": _byte_to_gb(vm.total),
        "host_available_GB": _byte_to_gb(vm.available),
        "host_used_GB": _byte_to_gb(vm.used),
        "host_free_GB": _byte_to_gb(vm.free),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
