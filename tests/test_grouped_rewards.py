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
