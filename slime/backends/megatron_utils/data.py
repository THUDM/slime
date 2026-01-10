import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from ..training_utils.types import ParallelState

logger = logging.getLogger(__name__)


class MegatronParallelState(ParallelState):
    """
    ParallelState for Megatron backend, initialized from mpu module.
    
    This class provides a convenient way to create ParallelState from Megatron's mpu
    without explicitly passing all parameters.
    
    Example:
        from megatron.core import mpu
        parallel_state = MegatronParallelState(with_context_parallel=True)
    """
    
    def __init__(
        self, 
        model: torch.nn.Module | Sequence[torch.nn.Module] | None = None,
        with_context_parallel: bool = True,
    ):
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        microbatch_group_size_per_vp_stage = None

        if vpp_size is None:
            vpp_size = 1
        elif vpp_size > 1:
            assert model is not None
            from megatron.core.utils import get_model_config
            model_to_check = model[0] if isinstance(model, Sequence) else model
            config = get_model_config(model_to_check)
            microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
        
        super().__init__(
            dp_rank=mpu.get_data_parallel_rank(with_context_parallel=with_context_parallel),
            dp_size=mpu.get_data_parallel_world_size(with_context_parallel=with_context_parallel),
            dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=with_context_parallel),
            cp_rank=mpu.get_context_parallel_rank(),
            cp_size=mpu.get_context_parallel_world_size(),
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            is_pp_last_stage=mpu.is_pipeline_last_stage(),
            vpp_size=vpp_size,
            microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
            dp_group=mpu.get_data_parallel_group(with_context_parallel=with_context_parallel),
            dp_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=with_context_parallel),
            cp_group=mpu.get_context_parallel_group(),
            tp_group=mpu.get_tensor_model_parallel_group(),
            dp_mesh=None,
            cp_mesh=None,
        )

def get_packed_seq_params(batch: dict[str, torch.Tensor], args: Namespace) -> PackedSeqParams:
    if args.qkv_format == "thd":
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            max_seqlen_q=batch["max_seqlen"],
            max_seqlen_kv=batch["max_seqlen"],
            qkv_format="thd",
        )
        batch["packed_seq_params"] = packed_seq_params
        return packed_seq_params
    else:
        return None
        
