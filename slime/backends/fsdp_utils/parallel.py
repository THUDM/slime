import logging
from argparse import Namespace

import torch.distributed as dist
from ring_flash_attn import substitute_hf_flash_attn
from torch.distributed.device_mesh import init_device_mesh

from slime.utils.distributed_utils import get_gloo_group

from ..training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


class FSDPParallelState(ParallelState):
    def __init__(self, args: Namespace):
        super().__init__()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.cp_size = args.context_parallel_size
        self.dp_size = world_size // self.cp_size
        self.dp_cp_size = world_size

        self.dp_rank = rank // self.cp_size
        self.cp_rank = rank % self.cp_size
        self.dp_cp_rank = rank
        self.dp_src_rank = self.dp_rank // world_size

        self.tp_size = 1
        self.tp_rank = 0
        self.tp_group = dist.new_group([rank])

        self.mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size // self.cp_size, self.cp_size), mesh_dim_names=("dp", "cp")
        )
        self.dp_mesh = self.mesh["dp"]

        self.dp_group = self.mesh.get_group("dp")
        self.cp_group = self.mesh.get_group("cp")
        self.dp_cp_group = dist.group.WORLD
        self.dp_cp_group_gloo = get_gloo_group()

        logger.info(
            f"[Rank {rank}] Device mesh (2D): world_size={world_size}, "
            f"cp_size={self.cp_size}, dp_size={world_size // self.cp_size}"
        )
        logger.info(f"[Rank {rank}] Mesh shape: {self.mesh.shape}, " f"dp_rank={self.dp_rank}, cp_rank={self.cp_rank}")

        # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
        if self.cp_size > 1:
            substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
            logger.info(f"[Rank {rank}] CP initialized via device mesh")
        else:
            logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")
