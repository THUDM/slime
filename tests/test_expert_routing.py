import sys
from argparse import Namespace
from collections import Counter
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.update_weight import expert_routing
from slime.backends.megatron_utils.update_weight.expert_routing import (
    _build_expert_params,
    _build_expert_transfer_plan,
    _expert_transfer_size,
    _ExpertParam,
)
from slime.utils.types import ParamInfo

NUM_GPUS = 0


@pytest.mark.parametrize("prefix", ["module.module.", "module.module.language_model."])
def test_expert_targets_cover_language_model_wrappers(prefix):
    infos = [
        replace(
            _param(expert=expert, projection=projection, source_rank=0, target_ranks=(0,), size=60).info,
            name=f"{prefix}decoder.layers.3.mlp.experts.linear_fc{projection}.weight{expert}",
        )
        for expert in range(4)
        for projection in (1, 2)
    ]
    params = _build_expert_params(infos, [(0, 2), (1, 3)], num_experts=4)
    assert len(params) == 8
    for param in params:
        assert param.layer == 3
        assert param.target_ranks == ((0, 2) if param.expert < 2 else (1, 3))

    plan = _build_expert_transfer_plan(params, buffer_size=240)
    assert {
        param.info.name for group in plan for batch in group for transfer in batch for param in transfer.params
    } == {info.name for info in infos}


def _param(
    *, expert: int, projection: int, source_rank: int, target_ranks: tuple[int, ...], size: int, layer: int = 3
) -> _ExpertParam:
    info = ParamInfo(
        name=f"module.module.decoder.layers.{layer}.mlp.experts.linear_fc{projection}.weight{expert}",
        dtype=torch.bfloat16,
        shape=torch.Size([size // 2]),
        attrs={},
        size=size,
        src_rank=source_rank,
    )
    return _ExpertParam(
        info=info,
        layer=layer,
        expert=expert,
        target_ranks=target_ranks,
    )


def test_transfer_plan_splits_same_rank_experts_at_expert_boundaries():
    params = []
    for expert, rank in ((0, 0), (1, 0), (2, 1), (3, 1)):
        params.extend(
            [
                _param(expert=expert, projection=1, source_rank=rank, target_ranks=(rank,), size=60),
                _param(expert=expert, projection=2, source_rank=rank, target_ranks=(rank,), size=60),
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
    first = _param(expert=0, projection=1, source_rank=0, target_ranks=(0,), size=80)
    second = replace(
        first,
        info=replace(
            first.info,
            name="module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
            size=80,
        ),
    )

    with pytest.raises(ValueError, match="exceeds the 150-byte staging budget"):
        _build_expert_transfer_plan([first, second], buffer_size=150)


def _payloads(plan):
    return Counter(
        (transfer.source_rank, transfer.target_ranks, tuple(p.info.name for p in transfer.params))
        for group in plan
        for batch in group
        for transfer in batch
    )


def _batch_layers(plan):
    return [tuple(dict.fromkeys(t.params[0].layer for t in batch)) for group in plan for batch in group]


def test_transfer_plan_merges_uneven_stages_deterministically():
    # Ranks 0 and 2 belong to the same physical PP stage but own different layers.
    params = [
        _param(expert=0, projection=projection, source_rank=rank, target_ranks=(0, 1, 2), size=30, layer=layer)
        for layer, rank in enumerate((0, 2, 1, 1, 2))
        for projection in (1, 2)
    ]
    # Count repeated payload entries instead of just comparing membership.
    params.append(params[0])
    stages = (0, 1, 0)
    baseline = _build_expert_transfer_plan(params, buffer_size=150)
    merged = _build_expert_transfer_plan(params, buffer_size=150, pp_stage_by_rank=stages)

    assert _batch_layers(baseline) == [(0,), (1,), (2,), (3,), (4,)]
    assert _batch_layers(merged) == [(0, 2), (1, 3), (4,)]
    assert _payloads(merged) == _payloads(baseline)
    assert merged == _build_expert_transfer_plan(list(reversed(params)), buffer_size=150, pp_stage_by_rank=stages)


@pytest.mark.parametrize(("buffer_size", "expected"), [(120, [(0, 1)]), (119, [(0,), (1,)])])
@pytest.mark.parametrize(
    ("first_targets", "second_targets"),
    [
        ((2,), (0, 1)),  # Rank 0 stages both a send and a receive.
        ((0, 2), (1, 2)),  # Rank 2 receives from both stages.
        ((0, 2), (0, 1)),  # Rank 0 is also its own target but reuses its send buffer.
    ],
)
def test_transfer_plan_limits_per_rank_staging_bytes(buffer_size, expected, first_targets, second_targets):
    params = [
        _param(expert=0, projection=1, source_rank=0, target_ranks=first_targets, size=60, layer=0),
        _param(expert=0, projection=1, source_rank=1, target_ranks=second_targets, size=60, layer=1),
    ]
    plan = _build_expert_transfer_plan(params, buffer_size, pp_stage_by_rank=(0, 1, 0, 1))

    assert _batch_layers(plan) == expected


def test_transfer_plan_repacks_an_oversized_layer_under_the_selected_budget():
    params = [
        _param(expert=0, projection=1, source_rank=0, target_ranks=(0, 1), size=60, layer=0),
        _param(expert=1, projection=1, source_rank=0, target_ranks=(0, 1), size=60, layer=0),
        _param(expert=0, projection=1, source_rank=1, target_ranks=(0, 1), size=60, layer=1),
    ]
    baseline = _build_expert_transfer_plan(params, buffer_size=120)
    merged = _build_expert_transfer_plan(params, buffer_size=60, pp_stage_by_rank=(0, 1))

    assert [len(group) for group in baseline] == [1, 1]
    assert [len(group) for group in merged] == [2, 1]
    assert all(len(batch) == 1 for group in merged for batch in group)
    assert _payloads(merged) == _payloads(baseline)


@pytest.mark.parametrize("stages", [(0, 0, 0), (0, 1, 1)])
def test_transfer_plan_stays_layer_wise_without_single_stage_ownership(stages):
    params = [
        _param(expert=expert, projection=1, source_rank=rank, target_ranks=(0, 1, 2), size=60, layer=layer)
        for layer, expert, rank in ((0, 0, 0), (0, 1, 1), (1, 0, 2))
    ]
    baseline = _build_expert_transfer_plan(params, buffer_size=120)
    assert _build_expert_transfer_plan(params, buffer_size=120, pp_stage_by_rank=stages) == baseline


@pytest.mark.parametrize(("merge_bytes", "expected"), [(0, [(0,), (1,)]), (180, [(0, 1)]), (59, [])])
def test_configure_expert_routing_uses_expert_budget_only(monkeypatch, merge_bytes, expected):
    experts = [
        _param(expert=expert, projection=projection, source_rank=rank, target_ranks=(), size=30, layer=layer).info
        for layer, rank in enumerate((0, 2))
        for expert in (0, 1)
        for projection in (1, 2)
    ]
    dense = replace(experts[0], name="dense")
    names_by_rank = [tuple(info.name for info in experts if info.src_rank == rank) for rank in range(4)]
    stages = (0, 0, 1, 1)
    stage_gathers = []

    def all_gather_object(output, obj, *, group):
        if isinstance(obj, int):
            stage_gathers.append(obj)
            output[:] = stages
        else:
            output[:] = names_by_rank

    def pack_dense(infos, buffer_size):
        assert buffer_size == 120
        return [infos]

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(
        get_expert_tensor_parallel_world_size=lambda: 1,
        get_expert_model_parallel_world_size=lambda: 2,
        get_pipeline_model_parallel_world_size=lambda: 2,
        get_pipeline_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(
        sys.modules,
        f"{expert_routing.__package__}.hf_weight_iterator_direct",
        SimpleNamespace(pack_param_info_buckets=pack_dense),
    )
    monkeypatch.setattr(expert_routing, "get_gloo_group", lambda: None)
    monkeypatch.setattr(expert_routing.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(expert_routing.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(expert_routing.dist, "get_world_size", lambda: 4)

    dense_buckets, plan = expert_routing.configure_expert_routing(
        args=Namespace(
            sglang_ep_size=2, num_experts=2, update_weight_buffer_size=120, expert_sync_merge_bytes=merge_bytes
        ),
        full_param_info_buckets=[[dense, *experts]],
        get_local_weight_names=lambda: names_by_rank[0],
        engine_gpu_counts=(2, 2),
        engine_gpu_offsets=(0, 2),
        engine_parallel_configs=None,
        use_distribute=False,
    )

    assert stage_gathers == ([0] if merge_bytes else [])
    assert dense_buckets == ([[dense]] if expected else None)
    assert _batch_layers(plan) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
