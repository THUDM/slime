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
        self._prof = None
        if args.use_pytorch_profiler and torch.distributed.get_rank() == 0:
            self._prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=max(args.profile_step_start - 1, 0),
                    warmup=1 if args.profile_step_start > 0 else 0,
                    active=args.profile_step_end - args.profile_step_start,
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
            )
            self._prof.start()

    def before_actor_train_step(self):
        if self._prof is None:
            return

        # We follow Megatron semantics which calls `step` before real training step and `stop` after it
        self._prof.step()

    def after_actor_train_step(self, rollout_id):
        if self._prof is None:
            return

        if rollout_id == self.args.profile_step_end:
            self._prof.stop()
            self._prof = None
