from dataclasses import replace

import pytest
import torch

from slime.backends.megatron_utils.update_weight.expert_routing import (
    _build_expert_transfer_plan,
    _expert_transfer_size,
    _ExpertParam,
)
from slime.utils.types import ParamInfo

NUM_GPUS = 0


def _param(*, expert: int, projection: int, source_rank: int, target_rank: int, size: int) -> _ExpertParam:
    info = ParamInfo(
        name=f"module.module.decoder.layers.3.mlp.experts.linear_fc{projection}.weight{expert}",
        dtype=torch.bfloat16,
        shape=torch.Size([size // 2]),
        attrs={},
        size=size,
        src_rank=source_rank,
    )
    return _ExpertParam(
        info=info,
        layer=3,
        expert=expert,
        target_ranks=(target_rank,),
    )


def test_transfer_plan_splits_same_rank_experts_at_expert_boundaries():
    params = []
    for expert, rank in ((0, 0), (1, 0), (2, 1), (3, 1)):
        params.extend(
            [
                _param(expert=expert, projection=1, source_rank=rank, target_rank=rank, size=60),
                _param(expert=expert, projection=2, source_rank=rank, target_rank=rank, size=60),
            ]
        )

    plan = _build_expert_transfer_plan(params, buffer_size=150)

    assert len(plan) == 1
    assert len(plan[0]) == 2
    transfers = [transfer for batch in plan[0] for transfer in batch]
    assert len(transfers) == 4
    assert all(_expert_transfer_size(transfer) == 120 for transfer in transfers)
    assert all(len({param.expert for param in transfer.params}) == 1 for transfer in transfers)
    assert {(param.expert, param.info.name) for transfer in transfers for param in transfer.params} == {
        (param.expert, param.info.name) for param in params
    }


def test_transfer_plan_rejects_one_expert_larger_than_buffer():
    first = _param(expert=0, projection=1, source_rank=0, target_rank=0, size=80)
    second = replace(
        first,
        info=replace(
            first.info,
            name="module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
            size=80,
        ),
    )

    with pytest.raises(ValueError, match="exceeds update_weight_buffer_size"):
        _build_expert_transfer_plan([first, second], buffer_size=150)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
