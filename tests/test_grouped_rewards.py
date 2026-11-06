"""GRPO statistics count each agent rollout once, regardless of its forks."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.data.batch_builder import BatchBuilder  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def builder(**overrides):
    args = dict(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    return BatchBuilder(SimpleNamespace(**(args | overrides)))


def segment(group, rollout, reward, mask=(1,)):
    return Sample(
        group_index=group,
        index=rollout,
        rollout_id=rollout,
        reward=reward,
        tokens=[100, *range(len(mask))],
        response_length=len(mask),
        loss_mask=list(mask),
        rollout_log_probs=[-0.5] * len(mask),
    )


@pytest.mark.parametrize("std", [False, True])
def test_unequal_forks_and_shuffled_prompt_groups(std):
    samples = [
        segment(20, 4, 8),
        segment(10, 1, 1),
        segment(20, 3, 10),
        segment(10, 1, 1),
        segment(10, 2, 0),
        segment(20, 4, 8),
        segment(10, 1, 1),
    ]
    raw, normalized = builder(grpo_std_normalization=std)._post_process_rewards(samples)
    assert raw == [8, 1, 10, 1, 0, 8, 1]
    expected = [-1, 0.5, 1, 0.5, -0.5, -1, 0.5]
    if std:
        low = 0.5 / (math.sqrt(0.5) + 1e-6)
        high = 1 / (math.sqrt(2) + 1e-6)
        expected = [-high, low, high, low, -low, -high, low]
    assert normalized == pytest.approx(expected)


def test_segment_count_matching_batch_size_does_not_override_group_ids():
    samples = [segment(9, 1, 1), segment(9, 1, 1), segment(9, 1, 1), segment(9, 2, 0)]
    assert builder()._post_process_rewards(samples)[1] == pytest.approx([0.5, 0.5, 0.5, -0.5])


def test_single_rollout_group_has_finite_zero_advantage():
    samples = [segment(3, 8, 1), segment(3, 8, 1), segment(4, 9, 0)]
    assert builder(grpo_std_normalization=True)._post_process_rewards(samples)[1] == [0, 0, 0]


def test_rollout_reward_or_prompt_group_mismatch_is_rejected():
    for samples in (
        [segment(1, 3, 0), segment(1, 3, 1)],
        [segment(1, 3, 0), segment(2, 3, 0)],
    ):
        with pytest.raises(ValueError, match="rollout"):
            builder()._post_process_rewards(samples)


def test_incomplete_prompt_group_metadata_is_rejected():
    with pytest.raises(ValueError, match="group_index"):
        builder()._post_process_rewards([segment(1, 0, 1), segment(None, 1, 0)])


def test_legacy_flat_batches_without_identifiers_keep_positional_groups():
    samples = [Sample(reward=r) for r in [1, 0, 10, 8]]
    assert builder()._post_process_rewards(samples)[1] == pytest.approx([0.5, -0.5, 1, -1])


def test_disabled_normalization_preserves_outcome_rewards():
    samples = [segment(5, 2, 1), segment(5, 2, 1), segment(5, 3, 0)]
    assert builder(rewards_normalization=False)._post_process_rewards(samples) == ([1, 1, 0], [1, 1, 0])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"])
@pytest.mark.parametrize("std", [False, True])
@pytest.mark.parametrize("reward", [0.3, 0.1 + 0.2, 0.9])
def test_uniform_fractional_rewards_have_exactly_zero_advantage(estimator, std, reward):
    # Every rollout in the group got the same reward, so there is no
    # preference between them and every advantage must be exactly zero.
    # Centering identical float32 rewards used to leave a residue of
    # about an ulp, which the std division amplified into a small
    # nonzero advantage shared by the whole group (about +0.03 for 0.3).
    samples = [segment(1, i, reward) for i in range(16)]
    raw, normalized = builder(advantage_estimator=estimator, grpo_std_normalization=std)._post_process_rewards(samples)
    assert raw == [reward] * 16
    assert normalized == [0.0] * 16


def test_uniform_group_next_to_mixed_group():
    samples = [segment(1, i, 0.3) for i in range(4)] + [
        segment(2, 10, 1),
        segment(2, 11, 0),
        segment(2, 12, 0),
        segment(2, 13, 0),
    ]
    _, normalized = builder()._post_process_rewards(samples)
    assert normalized == pytest.approx([0, 0, 0, 0, 0.75, -0.25, -0.25, -0.25])


@pytest.mark.parametrize("reward", [float("nan"), float("inf")])
def test_uniform_nonfinite_rewards_stay_nonfinite(reward):
    samples = [segment(1, i, reward) for i in range(4)]
    _, normalized = builder(grpo_std_normalization=True)._post_process_rewards(samples)
    assert all(math.isnan(v) for v in normalized)


def test_conversion_aggregates_loss_denominators_over_original_rollout():
    samples = [segment(5, 2, 1, (1, 0, 1)), segment(5, 3, 0), segment(5, 2, 1, (1, 1, 1))]
    # An omitted rollout_id falls back to index, as the Sample contract specifies.
    for sample in samples:
        sample.rollout_id = None
    data = builder().convert(samples)
    assert data["rollout_ids"] == [2, 3, 2]
    assert data["rollout_mask_sums"] == [5, 1, 5]
    assert data["rewards"] == pytest.approx([0.5, -0.5, 0.5])
    assert data["tokens"] == [sample.tokens for sample in samples]
    assert data["rollout_log_probs"] == [sample.rollout_log_probs for sample in samples]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
