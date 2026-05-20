import logging
import os
import time
import traceback
from pathlib import Path

import torch

from slime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    """Read a boolean env var. Accepts 1/true/yes (case-insensitive) as truthy."""
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no")


def _should_profile_this_rank() -> bool:
    """Only rank 0 profiles by default. Each rank holds a full torch.profiler
    buffer during the active window (~60 GB on a 26B MoE), so running on all
    16 ranks adds ~1 TB of host RAM pressure and has caused host-OOM kills.
    Set SLIME_PROFILE_ALL_RANKS=1 to opt into per-rank traces when diagnosing
    cross-rank sync or PP-stage imbalance.
    """
    if _env_flag("SLIME_PROFILE_ALL_RANKS"):
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class TrainProfiler:
    """Manages torch.profiler and memory profilers across training phases.

    Three profile targets, typically used one at a time via ``--profile-target``:

    - ``train_overall`` — one active window covers a full rollout
      (ref_log_probs + log_probs + actor_train + update_weights). Useful for
      cumulative stats but produces huge traces (8+ GB gzipped on a 26B MoE).
    - ``train_actor`` — one active window covers a single grad-accum step
      inside ``actor_train``. Trace is ~500× smaller (~15 MB) and actually
      openable in Perfetto/Chrome. Hooked via ``step_train_actor`` at the
      boundary of each step in ``megatron_utils/model.py::train``.
    - ``train_log_probs`` — one active window per log-probs forward. Hooked
      via ``step_train_log_probs``.

    ``--profile-target`` is a list, so multiple targets can be passed at once;
    the code paths check membership independently rather than enforcing
    exclusivity.
    """

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
        # Only the train_overall profiler starts at init; the per-step ones
        # start lazily on their first tick so they don't waste a warmup slot
        # on code that runs before training begins.
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        """Called once per rollout from the actor loop. Advances the
        train_overall profiler's state machine."""
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def step_train_actor(self):
        """Called at each grad-accum step boundary inside ``actor_train``.
        Each call advances the train_actor profiler by one tick."""
        if self._torch_profiler_train_actor is None:
            return
        if not self._torch_profiler_train_actor_started:
            self._torch_profiler_train_actor.start()
            self._torch_profiler_train_actor_started = True
        self._torch_profiler_train_actor.step()

    def step_train_log_probs(self):
        """Called at each log-probs forward-step boundary."""
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
    if not (
        args.use_pytorch_profiler
        and (name in args.profile_target)
        and _should_profile_this_rank()
    ):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    # On large MoE models (26B+), record_shapes / with_flops / with_stack /
    # profile_memory are memory amplifiers that can produce 10+ GB traces and
    # OOM the host. All off by default; flip via env var for tighter capture
    # windows where the extra metadata is worth the cost.
    record_shapes = _env_flag("SLIME_PROFILE_RECORD_SHAPES")
    with_flops = _env_flag("SLIME_PROFILE_WITH_FLOPS")
    with_stack = _env_flag("SLIME_PROFILE_WITH_STACK")
    profile_memory = _env_flag("SLIME_PROFILE_MEMORY")
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
    def start(self):
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            torch.cuda.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        torch.cuda.memory._dump_snapshot(self._path_dump)
        torch.cuda.memory._record_memory_history(enabled=None)


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
