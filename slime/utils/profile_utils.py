import time
from pathlib import Path

import torch


# The memory_snapshot_path is not a full path, but we name like this to be compatible with megatron
def attach_oom_dump_memory_history(path_dump):
    print("Attach OOM dump memory history.")

    torch.cuda.memory._record_memory_history(
        max_entries=1000000,
        # record stack information for the trace events
        # trace_alloc_record_context=True,
        stacks="all",
    )

    def oom_observer(device, alloc, device_alloc, device_free):
        print(f"Observe OOM, will dump snapshot to {path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=})")
        torch.cuda.memory._dump_snapshot(path_dump)

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


def dump_snapshot_and_stop(path_dump):
    print(f"Dump memory snapshot to: {path_dump}")
    torch.cuda.memory._dump_snapshot(path_dump)
    torch.cuda.memory._record_memory_history(enabled=None)


def get_memory_snapshot_full_path(args):
    return (
        Path(args.memory_snapshot_dir)
        / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
    )
