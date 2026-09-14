from types import SimpleNamespace

import pytest
import ray
import torch

from slime.ray import rollout as rollout_module
from slime.ray.rollout import RolloutManager
from slime.utils.misc import RolloutDataRefs
from slime.utils.routing_replay_data import prepare_routing_replay_shard


def _args(*, replay=True, sequence_parallel=True, transport="object-store"):
    return SimpleNamespace(
        global_batch_size=2,
        use_rollout_routing_replay=replay,
        rollout_data_transport=transport,
        num_experts=17,
        data_pad_size_multiplier=4,
        sequence_parallel=sequence_parallel,
        allgather_cp=False,
    )


def _manager(*, replay=True, sequence_parallel=True, transport="object-store"):
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.args = _args(replay=replay, sequence_parallel=sequence_parallel, transport=transport)
    manager.train_parallel_config = {
        "dp_size": 1,
        "cp_size": 2,
        "tp_size": 2,
        "pp_size": 1,
        "world_size": 4,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    return manager


def _data():
    tokens = [torch.arange(7), torch.arange(10)]
    routes = [
        torch.arange((len(token_ids) - 1) * 3 * 2, dtype=torch.int32).reshape(-1, 3, 2) % 17 for token_ids in tokens
    ]
    return {
        "tokens": tokens,
        "rollout_routed_experts": routes,
        "rollout_ids": [0, 1],
    }


def test_split_puts_independent_cp_tp_route_objects(monkeypatch):
    # Deliberately reverse the samples across microbatches: this pins that the
    # actor-local payload follows the dynamic schedule rather than source order.
    monkeypatch.setattr(
        rollout_module,
        "build_dp_schedule",
        lambda *args, **kwargs: ([[0, 1]], [[[1], [0]]], [2], [2]),
    )
    puts = []

    def fake_put(value, **kwargs):
        ref = object()
        puts.append((ref, value, kwargs))
        return ref

    monkeypatch.setattr(ray, "put", fake_put)
    data = _data()
    result = _manager()._split_train_data_by_dp(data)

    assert isinstance(result, RolloutDataRefs)
    assert set(result.routed_experts) == {(0, cp, tp) for cp in range(2) for tp in range(2)}
    assert len({box.inner for box in result.routed_experts.values()}) == 4

    common_payload = next(value for ref, value, _ in puts if ref is result.data[0].inner)
    assert "rollout_routed_experts" not in common_payload

    for (dp_rank, cp_rank, tp_rank), box in result.routed_experts.items():
        assert dp_rank == 0
        actual = next(value for ref, value, _ in puts if ref is box.inner)
        expected = prepare_routing_replay_shard(
            tokens=data["tokens"],
            routed_experts=data["rollout_routed_experts"],
            micro_batch_indices=[[1], [0]],
            cp_rank=cp_rank,
            cp_size=2,
            tp_rank=tp_rank,
            tp_size=2,
            num_experts=17,
            data_pad_size_multiplier=4,
            sequence_parallel=True,
            allgather_cp=False,
        )
        assert len(actual) == 2
        assert all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True))
        for shard in actual:
            assert shard.storage_offset() == 0
            assert shard.untyped_storage().nbytes() == shard.nbytes


@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_split_reuses_route_object_across_tp_without_sequence_parallel(monkeypatch, transport):
    monkeypatch.setattr(
        rollout_module,
        "build_dp_schedule",
        lambda *args, **kwargs: ([[0, 1]], [[[0, 1]]], [1], [2]),
    )
    puts = []

    def fake_put(value, **kwargs):
        ref = object()
        puts.append((ref, value, kwargs))
        return ref

    monkeypatch.setattr(ray, "put", fake_put)
    result = _manager(sequence_parallel=False, transport=transport)._split_train_data_by_dp(_data())

    assert isinstance(result, RolloutDataRefs)
    # One common object plus one route object per CP rank. TP does not add
    # objects because every TP rank consumes identical routes without SP.
    assert len(puts) == 3
    expected_put_kwargs = {"_tensor_transport": "nixl"} if transport == "nixl" else {}
    assert all(kwargs == expected_put_kwargs for _, _, kwargs in puts)
    for cp_rank in range(2):
        tp0_ref = result.routed_experts[(0, cp_rank, 0)].inner
        tp1_ref = result.routed_experts[(0, cp_rank, 1)].inner
        assert tp0_ref is tp1_ref
    assert result.routed_experts[(0, 0, 0)].inner is not result.routed_experts[(0, 1, 0)].inner


def test_prepared_tp_shards_do_not_retain_full_cp_storage():
    # A dim-0 slice is still reported as contiguous by PyTorch even though it
    # retains its parent's complete storage. This is the shape that caused
    # Ray to serialize one full CP tensor for every TP rank.
    routes = [torch.arange(64 * 3 * 2, dtype=torch.int32).reshape(64, 3, 2)]
    tokens = [torch.arange(65)]

    shards = [
        prepare_routing_replay_shard(
            tokens=tokens,
            routed_experts=routes,
            micro_batch_indices=[[0]],
            cp_rank=0,
            cp_size=1,
            tp_rank=tp_rank,
            tp_size=4,
            num_experts=17,
            data_pad_size_multiplier=4,
            sequence_parallel=True,
            allgather_cp=False,
        )[0]
        for tp_rank in range(4)
    ]

    # One terminal row plus alignment padding grows 64 rows to 80 here; the
    # important invariant is that storage grows only with the logical shards,
    # not once per TP rank with their pre-slice parent.
    expected_total_nbytes = 80 * 3 * 2 * torch.tensor([], dtype=torch.int32).element_size()
    assert sum(shard.nbytes for shard in shards) == expected_total_nbytes
    assert sum(shard.untyped_storage().nbytes() for shard in shards) == expected_total_nbytes
    for shard in shards:
        assert shard.is_contiguous()
        assert shard.storage_offset() == 0
        assert shard.untyped_storage().nbytes() == shard.nbytes


def test_replay_requires_routed_experts_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(
        rollout_module,
        "build_dp_schedule",
        lambda *args, **kwargs: ([[0, 1]], [[[0, 1]]], [1], [2]),
    )
    data = _data()
    del data["rollout_routed_experts"]

    with pytest.raises(ValueError, match="rollout_routed_experts is required"):
        _manager(replay=True)._split_train_data_by_dp(data)


def test_non_replay_path_does_not_transfer_unused_routes(monkeypatch):
    monkeypatch.setattr(
        rollout_module,
        "build_dp_schedule",
        lambda *args, **kwargs: ([[0, 1]], [[[0, 1]]], [1], [2]),
    )
    payloads = []

    def fake_put(value, **kwargs):
        payloads.append(value)
        return value

    monkeypatch.setattr(ray, "put", fake_put)
    result = _manager(replay=False)._split_train_data_by_dp(_data())

    assert isinstance(result, list)
    assert len(payloads) == 1
    assert "rollout_routed_experts" not in payloads[0]
