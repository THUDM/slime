import time
import torch

from pathlib import Path
from torch.profiler import profile, ProfilerActivity


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

def benchmark_func(
    fn,
    *args,
    n_warmup=10,
    n_repeat=100,
    ref_flops=None,
    ref_bytes=None,
    ref_time=None,
    n_profile=0,
    trace_dir=None,
    name="",
    **kwargs,
):
    func_name = getattr(fn, "__name__", None)
    func_name = name if func_name == "apply" or func_name is None else func_name

    for i in range(n_warmup):
        fn(*args, **kwargs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    ts = time.time()
    for i in range(n_repeat):
        start_events[i].record()
        fn(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    if n_profile > 0:
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
                ProfilerActivity.XPU,
            ]
        ) as prof:
            for i in range(n_profile):
                fn(*args, **kwargs)
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", top_level_events_only=True, row_limit=100
            )
        )
        if trace_dir is not None:
            assert trace_dir.endswith(".json")
            prof.export_chrome_trace(trace_dir)

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times = sorted(times)
    clip = max(1, n_repeat // 100)
    times = sum(times[clip:-clip])

    average_event_time = times * 1000 / (n_repeat - 2 * clip)

    fs = ""
    if ref_flops is not None:
        flops = ref_flops / 1e12 / (average_event_time / 1e6)
        fs = f"FLOPS:{flops:.2f}T"
    bs = ""
    if ref_bytes is not None:
        bs = f"bandwidth:{ref_bytes / average_event_time / 1e3:.1f}G/S"
    ss = ""
    if ref_time is not None:
        ss = f"speedup:{ref_time / average_event_time:.3f}"

    print(f"{func_name:<30} {name} time:{average_event_time:.1f} us {fs} {bs} {ss}")
    return average_event_time
