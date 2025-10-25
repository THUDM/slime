import abc
import os
import random
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

from slime.ray.ray_actor import RayActor
from slime.utils.distributed_utils import init_gloo_group
from slime.utils.memory_utils import clear_memory, print_memory


def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port, wandb_run_id):
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args, role, wandb_run_id, with_ref=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        try:
            if torch.version.hip is not None:
                print(f"Detected ROCm/HIP environment, skipping NUMA affinity setup")
                # will find the coresponding API to implement ROCm version as below
            else:
                import pynvml

                pynvml.nvmlInit()

                local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node

                handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                pynvml.nvmlDeviceSetCpuAffinity(handle)

                print(f"Set NUMA affinity for GPU {local_rank}")
                pynvml.nvmlShutdown()

        except ImportError:
            print(f"Warning: pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            print(f"Warning: Failed to set NUMA affinity: {e}")

    def clear_memory(self):
        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, iteration):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def connect_actor_critic(self, critic_group):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
