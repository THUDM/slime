import logging
import os
import time
import traceback
from pathlib import Path

import torch

from slime.utils import accelerator
from slime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """Read a boolean env var. Accepts 1/true/yes (case-insensitive) as truthy."""
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no")


def _should_profile_this_rank() -> bool:
    """Rank 0 only by default; per-rank profiler buffers can host-OOM on large MoE.
    Set SLIME_PROFILE_ALL_RANKS=1 to profile every rank."""
    if _env_flag("SLIME_PROFILE_ALL_RANKS"):
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class TrainProfiler:
    """torch.profiler + memory profilers across training phases. Targets (via
    ``--profile-target``, a list): train_overall (full rollout, large traces),
    train_actor (one grad-accum step, small), train_log_probs (one log-probs forward)."""

    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None
        self._torch_profiler_train_actor = None
        self._torch_profiler_train_actor_started = False
        self._torch_profiler_train_log_probs = None
        self._torch_profiler_train_log_probs_started = False

        if _should_profile_this_rank():
            if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
                self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")
            if args.use_pytorch_profiler and ("train_actor" in args.profile_target):
                self._torch_profiler_train_actor = _create_torch_profiler(args, name="train_actor")
            if args.use_pytorch_profiler and ("train_log_probs" in args.profile_target):
                self._torch_profiler_train_log_probs = _create_torch_profiler(args, name="train_log_probs")
            if args.record_memory_history and ("train_overall" in args.profile_target):
                self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
                self._memory_profiler_overall.start()

    def on_init_end(self):
        # train_overall starts at init; per-step profilers start lazily on first tick.
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        """Advance the train_overall profiler once per rollout."""
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def step_train_actor(self):
        """Advance the train_actor profiler one tick per grad-accum step."""
        if self._torch_profiler_train_actor is None:
            return
        if not self._torch_profiler_train_actor_started:
            self._torch_profiler_train_actor.start()
            self._torch_profiler_train_actor_started = True
        self._torch_profiler_train_actor.step()

    def step_train_log_probs(self):
        """Advance the train_log_probs profiler one tick per log-probs step."""
        if self._torch_profiler_train_log_probs is None:
            return
        if not self._torch_profiler_train_log_probs_started:
            self._torch_profiler_train_log_probs.start()
            self._torch_profiler_train_log_probs_started = True
        self._torch_profiler_train_log_probs.step()

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target) and _should_profile_this_rank()):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    # record_shapes/with_flops/with_stack/profile_memory can produce 10+ GB traces
    # and OOM the host on large MoE — all off by default, opt in via env var.
    record_shapes = _env_flag("SLIME_PROFILE_RECORD_SHAPES")
    with_flops = _env_flag("SLIME_PROFILE_WITH_FLOPS")
    with_stack = _env_flag("SLIME_PROFILE_WITH_STACK")
    profile_memory = _env_flag("SLIME_PROFILE_MEMORY")

    activities = [torch.profiler.ProfilerActivity.CPU]
    activity_name = accelerator.device_type().upper()
    if hasattr(torch.profiler.ProfilerActivity, activity_name):
        activities.append(getattr(torch.profiler.ProfilerActivity, activity_name))
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
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
        record_shapes=record_shapes,
        with_flops=with_flops,
        with_stack=with_stack,
        profile_memory=profile_memory,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        self._recording = False

    @staticmethod
    def _memory_module():
        return accelerator.memory_module()

    def start(self):
        logger.info("Attach OOM dump memory history.")
        memory_module = self._memory_module()
        if memory_module is None or not hasattr(memory_module, "_record_memory_history"):
            logger.warning("Accelerator memory history is unavailable; skip torch memory profiler.")
            return
        if not hasattr(memory_module, "_dump_snapshot"):
            logger.warning("Accelerator memory snapshot is unavailable; skip torch memory profiler.")
            return

        memory_module._record_memory_history(
            max_entries=1000000,
            stacks="all",
        )
        self._recording = True

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            memory_module._dump_snapshot(str(self._path_dump))
            print_memory("when oom")

        attach_oom_observer = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
        if attach_oom_observer is not None:
            attach_oom_observer(oom_observer)
        else:
            logger.warning("Accelerator OOM observer is unavailable; memory snapshot on OOM is disabled.")

    def stop(self):
        if not self._recording:
            return
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        memory_module = self._memory_module()
        if memory_module is None or not hasattr(memory_module, "_dump_snapshot"):
            logger.warning("Accelerator memory snapshot is unavailable; skip dump.")
            return
        memory_module._dump_snapshot(str(self._path_dump))
        memory_module._record_memory_history(enabled=None)
        self._recording = False


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
