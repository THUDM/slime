import argparse

import pytest
import torch

from tools.benchmark_weight_sync import (
    EngineGroup,
    ExperimentConfig,
    TransferTask,
    _aggregate_experiment,
    _prepare_tensors,
    build_engine_groups,
    build_transfer_tasks,
    parse_byte_credit,
    parse_size,
    percentile,
    plan_transfer_waves,
)


def test_parse_size_supports_binary_decimal_and_exact_bytes():
    assert parse_size("16MiB") == 16 * 1024 * 1024
    assert parse_size("2.5MB") == 2_500_000
    assert parse_size("4096") == 4096
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size("0")
    assert parse_byte_credit("0") == 0


@pytest.mark.parametrize(
    ("world_size", "group_size", "expected"),
    [
        (2, 1, [(1,)]),
        (4, 1, [(1,), (2,), (3,)]),
        (4, 3, [(1, 2, 3)]),
    ],
)
def test_engine_groups_are_derived_from_world_size(world_size, group_size, expected):
    groups = build_engine_groups(world_size, engine_group_size=group_size)

    assert [group.ranks for group in groups] == expected


def test_heterogeneous_engine_groups_cover_every_non_trainer_rank_once():
    groups = build_engine_groups(4, engine_group_sizes=[1, 2])

    assert [group.ranks for group in groups] == [(1,), (2, 3)]


def test_invalid_homogeneous_partition_explains_heterogeneous_option():
    with pytest.raises(ValueError, match="--engine-group-sizes"):
        build_engine_groups(4, engine_group_size=2)


def _tasks(bucket_count=3, engine_count=3, message_bytes=1024):
    groups = [EngineGroup(engine_id=index, ranks=(index + 1,)) for index in range(engine_count)]
    return build_transfer_tasks(bucket_count, groups, message_bytes)


def test_serialized_policy_has_one_transfer_per_wave():
    waves = plan_transfer_waves(
        _tasks(),
        policy="serialized",
        max_inflight_buckets=3,
        max_inflight_bytes=0,
        max_inflight_engine_groups=3,
    )

    assert all(len(wave) == 1 for wave in waves)


def test_all_at_once_keeps_all_engines_for_each_bucket_together():
    waves = plan_transfer_waves(
        _tasks(bucket_count=2, engine_count=3),
        policy="all_at_once",
        max_inflight_buckets=1,
        max_inflight_bytes=0,
        max_inflight_engine_groups=1,
    )

    assert [[task.engine_id for task in wave] for wave in waves] == [[0, 1, 2], [0, 1, 2]]


def test_windowed_policy_respects_bucket_byte_and_engine_credits():
    waves = plan_transfer_waves(
        _tasks(bucket_count=4, engine_count=3, message_bytes=1024),
        policy="windowed",
        max_inflight_buckets=2,
        max_inflight_bytes=2048,
        max_inflight_engine_groups=2,
    )

    for wave in waves:
        assert len({task.bucket_id for task in wave}) <= 2
        assert len({task.engine_id for task in wave}) <= 2
        assert len({task.bucket_id for task in wave}) * 1024 <= 2048


def test_one_bucket_must_fit_byte_credit():
    with pytest.raises(ValueError, match="smaller than one bucket"):
        plan_transfer_waves(
            _tasks(message_bytes=1024),
            policy="windowed",
            max_inflight_buckets=1,
            max_inflight_bytes=512,
            max_inflight_engine_groups=1,
        )


def test_transfer_buffers_are_preallocated_once_per_bucket():
    trainer, receiver, load = _prepare_tensors(
        bucket_count=3,
        message_bytes=16,
        rank=1,
        local_engine_group=EngineGroup(engine_id=0, ranks=(1,)),
        device=torch.device("cpu"),
        simulate_load=True,
    )

    assert trainer == {}
    assert list(receiver) == [0, 1, 2]
    assert list(load) == [0, 1, 2]
    assert all(tensor.numel() == 16 for tensor in [*receiver.values(), *load.values()])
    assert all(receiver[bucket_id].data_ptr() != load[bucket_id].data_ptr() for bucket_id in receiver)


def test_percentile_uses_linear_interpolation():
    assert percentile([], 0.5) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_weight_sync_total_uses_slowest_rank_not_only_trainer():
    config = ExperimentConfig(
        transport="p2p",
        message_bytes=1024,
        max_inflight_buckets=1,
        max_inflight_bytes=0,
        max_inflight_engine_groups=1,
        engine_wave_policy="serialized",
        phase_stride_us=0,
    )
    waves = [[TransferTask(bucket_id=0, engine_id=0, message_bytes=1024, transport="p2p")]]
    gathered = [
        {"rank": 0, "hostname": "node-a", "iteration_ms": [5.0, 6.0], "records": []},
        {"rank": 1, "hostname": "node-a", "iteration_ms": [7.0, 4.0], "records": []},
    ]

    result = _aggregate_experiment(
        config,
        waves,
        gathered,
        engine_groups=[EngineGroup(engine_id=0, ranks=(1,))],
    )

    assert result["weight_sync_total_ms"]["p50"] == 6.5
    assert result["trainer_total_ms"]["p50"] == 5.5
    assert result["max_inflight_engine_groups_observed"] == 1
    assert result["cross_rank_clock_comparable"] is True


def test_cross_host_timestamps_are_not_compared():
    config = ExperimentConfig(
        transport="p2p",
        message_bytes=1024,
        max_inflight_buckets=1,
        max_inflight_bytes=0,
        max_inflight_engine_groups=1,
        engine_wave_policy="serialized",
        phase_stride_us=0,
    )
    waves = [[TransferTask(bucket_id=0, engine_id=0, message_bytes=1024, transport="p2p")]]
    common_record = {
        "iteration": 0,
        "bucket_id": 0,
        "engine_id": 0,
        "transfer_ms": 1.0,
        "synthetic_load_ms": None,
        "control_wait_ms": 0.0,
    }
    gathered = [
        {
            "rank": 0,
            "hostname": "node-a",
            "iteration_ms": [2.0],
            "records": [
                {
                    **common_record,
                    "rank": 0,
                    "role": "trainer",
                    "api_launch_timestamp_ns": 10_000,
                }
            ],
        },
        {
            "rank": 1,
            "hostname": "node-b",
            "iteration_ms": [2.0],
            "records": [
                {
                    **common_record,
                    "rank": 1,
                    "role": "engine",
                    "api_launch_timestamp_ns": 20_000,
                }
            ],
        },
    ]

    result = _aggregate_experiment(
        config,
        waves,
        gathered,
        engine_groups=[EngineGroup(engine_id=0, ranks=(1,))],
    )

    assert result["cross_rank_clock_comparable"] is False
    assert result["rank_start_skew_us"]["p50"] is None
    assert result["engine_finish_skew_us"]["p50"] is None
