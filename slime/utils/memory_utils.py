import gc
import logging

import psutil
import torch
import torch.distributed as dist

from slime.utils import accelerator

logger = logging.getLogger(__name__)


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
