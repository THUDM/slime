import logging
from argparse import Namespace

import torch.distributed as dist
from ring_flash_attn import substitute_hf_flash_attn
from torch.distributed.device_mesh import init_device_mesh

from slime.utils.distributed_utils import get_gloo_group

from ..training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def create_fsdp_parallel_state(args: Namespace) -> ParallelState:
    """Create a ParallelState instance for FSDP configuration."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    cp_size = args.context_parallel_size
    dp_rank = rank // cp_size
    cp_rank = rank % cp_size

    mesh = init_device_mesh("cuda", mesh_shape=(world_size // cp_size, cp_size), mesh_dim_names=("dp", "cp"))

    logger.info(
        f"[Rank {rank}] Device mesh (2D): world_size={world_size}, "
        f"cp_size={cp_size}, dp_size={world_size // cp_size}"
    )
    logger.info(f"[Rank {rank}] Mesh shape: {mesh.shape}, " f"dp_rank={dp_rank}, cp_rank={cp_rank}")

    # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
    if cp_size > 1:
        substitute_hf_flash_attn(mesh.get_group("cp"), heads_k_stride=1)
        logger.info(f"[Rank {rank}] CP initialized via device mesh")
    else:
        logger.info(f"[Rank {rank}] Pure DP mode (cp_size=1)")

    parallel_state = ParallelState(
        dp_rank=dp_rank,
        dp_src_rank=dp_rank // world_size,
        dp_size=world_size // cp_size,
        cp_rank=cp_rank,
        cp_size=cp_size,
        dp_cp_rank=rank,
        dp_cp_size=world_size,
        dp_group=mesh.get_group("dp"),
        dp_cp_group=dist.group.WORLD,
        dp_cp_group_gloo=get_gloo_group(),
        cp_group=mesh.get_group("cp"),
        tp_size=1,
        tp_rank=0,
        tp_group=dist.new_group([rank]),
    )

    parallel_state.dp_mesh = mesh["dp"]

    return parallel_state
