from types import SimpleNamespace

import pytest

from slime.utils import data as data_utils

NUM_GPUS = 0


def test_process_rollout_data_partitions_raw_reward(monkeypatch):
    monkeypatch.setattr(data_utils.ray, "get", lambda value: value)
    rollout_data_ref = [
        SimpleNamespace(
            inner={
                "partition": [0, 2],
                "total_lengths": [8, 16, 24],
                "raw_reward": [0.0, 1.0, 0.5],
            }
        ),
        SimpleNamespace(inner={}),
    ]

    rollout_data = data_utils.process_rollout_data(
        SimpleNamespace(log_passrate=False),
        rollout_data_ref,
        dp_rank=0,
        dp_size=2,
    )

    assert rollout_data["total_lengths"] == [8, 24]
    assert rollout_data["raw_reward"] == [0.0, 0.5]
    assert "global_raw_reward" not in rollout_data
    assert "partition" not in rollout_data


def test_process_rollout_data_keeps_global_raw_reward_for_passrate(monkeypatch):
    monkeypatch.setattr(data_utils.ray, "get", lambda value: value)
    global_raw_reward = [0.0, 1.0, 0.0, 1.0]
    rollout_data_ref = [
        SimpleNamespace(
            inner={
                "partition": [1, 3],
                "total_lengths": [8, 16, 24, 32],
                "raw_reward": global_raw_reward,
            }
        ),
        SimpleNamespace(inner={}),
    ]

    rollout_data = data_utils.process_rollout_data(
        SimpleNamespace(log_passrate=True),
        rollout_data_ref,
        dp_rank=0,
        dp_size=2,
    )

    assert rollout_data["raw_reward"] == [1.0, 1.0]
    assert rollout_data["global_raw_reward"] is global_raw_reward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
