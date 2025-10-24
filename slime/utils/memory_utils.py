import gc
import torch
import torch.distributed as dist


def clear_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def available_memory():
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return {
        "gpu": str(torch.cuda.current_device()),
        "total_GB": round(total / (1024**3), 2),
        "free_GB": round(free / (1024**3), 2),
        "used_GB": round((total - free) / (1024**3), 2),
    }


def print_memory(msg):
    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    print(f"[Rank {dist.get_rank()}] Memory-Usage {msg}:", memory_info)
    return memory_info
