import abc
import os
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

from slime.ray.ray_actor import RayActor
from slime.utils.misc import ActorStatus

from tracer import vinit, TracePoint, MemTracePoint


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port, global_rank, task_id):
        os.environ["GLOBAL_RANK"] = str(global_rank)
        vinit()
        tp = TracePoint(f"task-{task_id}-rank-{rank}: init TrainRayActor", "1")
        tp.begin()
        self._world_size = world_size
        self._rank = rank
        self._task_id = task_id
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(start_port=20000+self._task_id*100)

        self.status = ActorStatus.PENDING

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        MemTracePoint.record("create actor")
        tp.end()

    def init(self, args, role, with_ref=False):
        self.args = args
        tp = TracePoint(f"task-{self.args.task_id}: train actor init", "1")
        tp.begin()
        self.role = role
        self.with_ref = with_ref

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        # Use task-specific group name to avoid conflicts between tasks
        group_name = f"train-task-{self._task_id}"
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
            group_name=group_name,
        )

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        # set current device
        args.local_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(f"cuda:{args.local_rank}")
        tp.end()

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        raise NotImplementedError

    @abc.abstractmethod
    def set_data_buffer(self, data_buffer):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, with_data_fetching=True):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, rollout_id):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, iteration, with_optimizer=True):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def update_weights_from_disk(self):
        raise NotImplementedError
