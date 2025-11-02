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


class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._prof_overall = None
        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._prof_overall = _create_profiler(args, name="train_overall")
            self._prof_overall.start()

    def step(self):
        if self._prof_overall is None:
            return

        self._prof_overall.step()

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    prof = _create_profiler(args, name=name)
    prof.start()
    for item in iterator:
        yield item
        prof.step()


def _create_profiler(args, name):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )
