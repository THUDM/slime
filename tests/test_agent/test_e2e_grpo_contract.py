"""The E2E must train on real successes and failures with normalized advantages."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_e2e_helpers import audit_grpo_training  # noqa: E402

NUM_GPUS = 0


def training_records():
    samples, trained = [], []
    # std([1, 0, 0, 1], correction=1) = sqrt(1/3).
    scale = 0.5 / ((1 / 3) ** 0.5 + 1e-6)
    for index, reward in enumerate([1, 0, 0, 1]):
        advantage = scale if reward else -scale
        for _ in range(2):
            samples.append({"rollout_id": index, "group_index": 7, "reward": reward})
            trained.append(
                {
                    "rollout_ids": index,
                    "rewards": advantage,
                    "loss_masks": torch.tensor([1, 0, 1]),
                    "advantages": torch.tensor([advantage, 0, advantage]),
                }
            )
    return samples, trained


def test_checks_actual_positive_and_negative_training_advantages():
    report = audit_grpo_training(*training_records())
    group = report["groups"][0]
    assert group["raw_rewards"] == [1, 0, 0, 1]
    assert group["normalized_advantages"][0] > 0 > group["normalized_advantages"][1]
    assert report["training_token_advantages_verified"]


@pytest.mark.parametrize("corruption", ["raw_rewards", "positive_only", "missing_failed_rollout"])
def test_rejects_training_that_loses_grpo_signal(corruption):
    samples, trained = training_records()
    if corruption == "raw_rewards":
        trained[2]["rewards"] = 0
    elif corruption == "positive_only":
        trained[2]["advantages"].abs_()
    else:
        trained = [sample for sample in trained if sample["rollout_ids"] != 1]
    with pytest.raises(AssertionError):
        audit_grpo_training(samples, trained)


@pytest.mark.parametrize("reward", [0, 1])
def test_uniform_groups_keep_their_real_zero_advantages(reward):
    samples, trained = training_records()
    for sample in samples:
        sample["reward"] = reward
    for sample in trained:
        sample["rewards"] = 0
        sample["advantages"].zero_()
    report = audit_grpo_training(samples, trained)
    assert report["groups"][0]["raw_rewards"] == [reward] * 4
    assert report["groups"][0]["normalized_advantages"] == [0] * 4
    assert not report["has_learning_signal"]


def test_two_tasks_are_normalized_independently_in_one_training_batch():
    samples, trained = training_records()
    second_samples, second_trained = training_records()
    for sample in second_samples:
        sample.update(rollout_id=sample["rollout_id"] + 4, group_index=8, reward=1)
    for sample in second_trained:
        sample["rollout_ids"] += 4
        sample["rewards"] = 0
        sample["advantages"].zero_()
    report = audit_grpo_training(samples + second_samples, trained + second_trained)
    assert [group["raw_rewards"] for group in report["groups"]] == [[1, 0, 0, 1], [1, 1, 1, 1]]
    assert [group["has_learning_signal"] for group in report["groups"]] == [True, False]
    assert report["has_learning_signal"]
    # Centering across all eight trajectories would incorrectly reward task 2.
    second_trained[0]["rewards"] = 0.25
    with pytest.raises(AssertionError):
        audit_grpo_training(samples + second_samples, trained + second_trained)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
