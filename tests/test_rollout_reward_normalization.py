from argparse import Namespace

import pytest

from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample


def _make_rollout_manager():
    rollout_manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(rollout_manager_cls)
    manager.custom_reward_post_process_func = None
    manager.args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    return manager


def test_reward_normalization_keeps_uneven_rollout_groups_separate():
    manager = _make_rollout_manager()
    samples = [
        Sample(group_index=0, index=0, reward=1.0),
        Sample(group_index=0, index=1, reward=3.0),
        Sample(group_index=1, index=2, reward=5.0),
    ]

    raw_rewards, rewards = manager._post_process_rewards(samples)

    assert raw_rewards == [1.0, 3.0, 5.0]
    assert rewards == pytest.approx([-0.7071063, 0.7071063, 0.0], abs=1e-6)
