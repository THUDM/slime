import logging
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from slime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    tensorboard_dir = args.tensorboard_dir
    if tensorboard_dir is not None:
        # Use absolute path to ensure it works across Ray actors with different working directories
        tensorboard_dir = str(Path(tensorboard_dir).resolve())
        os.makedirs(tensorboard_dir, exist_ok=True)

    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
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
        # Use absolute path to ensure it works across Ray actors with different working directories
        snapshot_dir = Path(args.memory_snapshot_dir).resolve()
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._path_dump = (
            snapshot_dir
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )
        logger.info(f"Memory snapshot will be saved to: {self._path_dump} (absolute path)")


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
            try:
                logger.info(
                    f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
                )
                traceback.print_stack()
                
                # Ensure the snapshot path is converted to string
                snapshot_path = str(self._path_dump)
                logger.info(f"Attempting to save snapshot to: {snapshot_path}")
                
                torch.cuda.memory._dump_snapshot(snapshot_path)
                
                # Verify file was created
                if Path(snapshot_path).exists():
                    file_size = Path(snapshot_path).stat().st_size
                    logger.info(f"Successfully saved snapshot to {snapshot_path} (size: {file_size} bytes)")
                else:
                    logger.warning(f"Snapshot file was not created at {snapshot_path}")
                
                print_memory("when oom")
            except Exception as e:
                logger.error(f"Error in OOM observer: {e}", exc_info=True)
                traceback.print_exc(file=sys.stderr)
                raise

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        try:
            snapshot_path = str(self._path_dump)
            logger.info(f"Dump memory snapshot to: {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)
            
            # Verify file was created
            if Path(snapshot_path).exists():
                file_size = Path(snapshot_path).stat().st_size
                logger.info(f"Successfully saved snapshot to {snapshot_path} (size: {file_size} bytes)")
            else:
                logger.warning(f"Snapshot file was not created at {snapshot_path}")
            
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as e:
            logger.error(f"Error dumping memory snapshot: {e}", exc_info=True)
            traceback.print_exc(file=sys.stderr)
            raise


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=str(self._path_dump),
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
