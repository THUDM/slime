import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config

from ..training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def create_megatron_parallel_state(
    model: torch.nn.Module | Sequence[torch.nn.Module] | None = None,
) -> ParallelState:
    vpp_size_value = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size_value is None:
        vpp_size = 1
        microbatch_group_size_per_vp_stage = None
    elif vpp_size_value > 1:
        assert model is not None
        model_to_check = model[0] if isinstance(model, Sequence) else model
        config = get_model_config(model_to_check)
        vpp_size = vpp_size_value
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    else:
        vpp_size = 1
        microbatch_group_size_per_vp_stage = None

    parallel_state = ParallelState(
        dp_rank=mpu.get_data_parallel_rank(with_context_parallel=False),
        dp_src_rank=mpu.get_data_parallel_src_rank(with_context_parallel=True),
        dp_size=mpu.get_data_parallel_world_size(with_context_parallel=False),
        cp_rank=mpu.get_context_parallel_rank(),
        cp_size=mpu.get_context_parallel_world_size(),
        dp_cp_rank=mpu.get_data_parallel_rank(with_context_parallel=True),
        dp_cp_size=mpu.get_data_parallel_world_size(with_context_parallel=True),
        dp_group=mpu.get_data_parallel_group(with_context_parallel=False),
        dp_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        dp_cp_group_gloo=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        cp_group=mpu.get_context_parallel_group(),
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        tp_group=mpu.get_tensor_model_parallel_group(),
        is_pp_last_stage=mpu.is_pipeline_last_stage(),
        vpp_size=vpp_size,
        microbatch_group_size_per_vp_stage=microbatch_group_size_per_vp_stage,
    )

    return parallel_state


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
