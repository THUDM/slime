"""CPU tests for GRPO grouping by ``group_index`` / ``rollout_id``.

The default ``_post_process_rewards`` used to reshape by
``n_samples_per_prompt`` and fall back to one global group when the batch
was uneven. The unit is the rollout: siblings that share ``rollout_id``
must carry the same outcome reward (``get_trajectory`` assigns it in full
to every segment), the baseline counts that rollout once, and
``group_index`` buckets rollouts per prompt.
"""

from argparse import Namespace

import pytest

from slime.utils.ppo_utils import normalize_rewards_by_rollout
from slime.utils.types import Sample

NUM_GPUS = 0


def make_args(**overrides) -> Namespace:
    defaults = dict(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_sample(*, group_index=0, index=0, rollout_id=0, reward=0.0) -> Sample:
    return Sample(group_index=group_index, index=index, rollout_id=rollout_id, reward=reward)


def _normalize(samples, **args_overrides):
    args = make_args(**args_overrides)
    raw = [sample.get_reward_value(args) for sample in samples]
    return raw, normalize_rewards_by_rollout(args, samples, raw)


@pytest.mark.unit
def test_empty_rewards_stay_empty():
    assert normalize_rewards_by_rollout(make_args(), [], []) == []


@pytest.mark.unit
def test_default_one_sample_per_rollout_is_per_prompt_group_norm():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=0, reward=1.0),
        make_sample(group_index=0, index=1, rollout_id=1, reward=3.0),
        make_sample(group_index=1, index=2, rollout_id=2, reward=5.0),
        make_sample(group_index=1, index=3, rollout_id=3, reward=11.0),
    ]
    _, processed = _normalize(samples)
    assert processed == pytest.approx([-1.0, 1.0, -3.0, 3.0])


@pytest.mark.unit
def test_fanned_rollout_counts_once_and_broadcasts_the_advantage():
    # rollout 11 is three siblings sharing the same full reward; the prompt
    # baseline is over two rollouts [0, 1], not four segments.
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
        make_sample(group_index=1, index=2, rollout_id=20, reward=2.0),
        make_sample(group_index=1, index=2, rollout_id=20, reward=2.0),
        make_sample(group_index=1, index=3, rollout_id=21, reward=4.0),
    ]
    raw, processed = _normalize(samples)
    assert raw == [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0]
    assert processed == pytest.approx([-0.5, 0.5, 0.5, 0.5, -1.0, -1.0, 1.0])


@pytest.mark.unit
def test_shared_sibling_reward_is_not_summed():
    # get_trajectory assigns the trajectory reward in full to every segment.
    # Summing sibling rewards would double-count that outcome.
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward=2.0),
        make_sample(group_index=0, index=0, rollout_id=10, reward=2.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=6.0),
    ]
    _, processed = _normalize(samples, n_samples_per_prompt=2, rollout_batch_size=1)
    assert processed == pytest.approx([-2.0, -2.0, 2.0])


@pytest.mark.unit
def test_std_normalization_uses_per_prompt_std_over_rollouts():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
    ]
    _, processed = _normalize(samples, n_samples_per_prompt=2, rollout_batch_size=1, grpo_std_normalization=True)
    assert processed == pytest.approx([-(2**-0.5), 2**-0.5, 2**-0.5], abs=1e-5)


@pytest.mark.unit
def test_single_rollout_prompt_under_std_is_zero_not_nan():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=0, reward=5.0),
        make_sample(group_index=1, index=1, rollout_id=1, reward=1.0),
        make_sample(group_index=1, index=2, rollout_id=2, reward=3.0),
    ]
    _, processed = _normalize(samples, grpo_std_normalization=True)
    inv_sqrt2 = 2**-0.5
    assert processed == pytest.approx([0.0, -inv_sqrt2, inv_sqrt2], abs=1e-4)


@pytest.mark.unit
def test_uneven_prompt_groups_get_true_per_prompt_baselines():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=0, reward=1.0),
        make_sample(group_index=0, index=1, rollout_id=1, reward=3.0),
        make_sample(group_index=0, index=2, rollout_id=2, reward=5.0),
        make_sample(group_index=1, index=3, rollout_id=3, reward=7.0),
        make_sample(group_index=1, index=4, rollout_id=4, reward=13.0),
    ]
    _, processed = _normalize(samples)
    assert processed == pytest.approx([-2.0, 0.0, 2.0, -3.0, 3.0])


@pytest.mark.unit
def test_noncontiguous_group_indices_share_reward_group():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
        make_sample(group_index=1, index=1, rollout_id=20, reward=10.0),
        make_sample(group_index=0, index=2, rollout_id=11, reward=2.0),
        make_sample(group_index=1, index=3, rollout_id=21, reward=14.0),
    ]
    _, processed = _normalize(samples)
    assert processed == pytest.approx([-1.0, -2.0, 1.0, 2.0])


@pytest.mark.unit
def test_missing_group_indices_use_legacy_boundaries():
    samples = [
        make_sample(group_index=None, index=i, rollout_id=i, reward=reward)
        for i, reward in enumerate([0.0, 2.0, 10.0, 14.0])
    ]
    _, processed = _normalize(samples, n_samples_per_prompt=2, rollout_batch_size=2)
    assert processed == pytest.approx([-1.0, 1.0, -2.0, 2.0])


@pytest.mark.unit
def test_missing_group_indices_uneven_count_uses_one_global_group():
    samples = [
        make_sample(group_index=None, index=i, rollout_id=i, reward=reward) for i, reward in enumerate([0.0, 2.0, 4.0])
    ]
    _, processed = _normalize(samples, n_samples_per_prompt=2, rollout_batch_size=2)
    assert processed == pytest.approx([-2.0, 0.0, 2.0])


@pytest.mark.unit
def test_rows_without_rollout_identity_stay_distinct():
    samples = [
        make_sample(group_index=0, index=None, rollout_id=None, reward=0.0),
        make_sample(group_index=0, index=None, rollout_id=None, reward=2.0),
    ]
    _, processed = _normalize(samples)
    assert processed == pytest.approx([-1.0, 1.0])


@pytest.mark.unit
def test_rejects_different_sibling_rewards():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=2.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=6.0),
    ]
    with pytest.raises(
        ValueError,
        match=r"all samples in rollout 11 must share one reward; rows \[1, 2\] have rewards \[2.0, 6.0\]",
    ):
        _normalize(samples, n_samples_per_prompt=2, rollout_batch_size=1)


@pytest.mark.unit
def test_cispo_uses_grpo_std_normalization():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward=0.0),
        make_sample(group_index=0, index=1, rollout_id=11, reward=1.0),
    ]
    _, processed = _normalize(
        samples,
        advantage_estimator="cispo",
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        grpo_std_normalization=True,
    )
    assert processed == pytest.approx([-(2**-0.5), 2**-0.5], abs=1e-5)


@pytest.mark.unit
def test_reinforce_plus_plus_baseline_only_zero_mean_no_std():
    samples = [
        make_sample(group_index=0, index=i, rollout_id=i, reward=reward)
        for i, reward in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    _, processed = _normalize(
        samples,
        advantage_estimator="reinforce_plus_plus_baseline",
        n_samples_per_prompt=4,
        rollout_batch_size=1,
        grpo_std_normalization=True,
    )
    assert processed == pytest.approx([-1.5, -0.5, 0.5, 1.5])


@pytest.mark.unit
def test_shared_reward_uses_selected_reward_key():
    samples = [
        make_sample(group_index=0, index=0, rollout_id=10, reward={"score": 2.0, "detail": "first"}),
        make_sample(group_index=0, index=0, rollout_id=10, reward={"score": 2.0, "detail": "second"}),
        make_sample(group_index=0, index=1, rollout_id=11, reward={"score": 6.0}),
    ]
    raw, processed = _normalize(samples, reward_key="score", n_samples_per_prompt=2, rollout_batch_size=1)
    assert raw == [2.0, 2.0, 6.0]
    assert processed == pytest.approx([-2.0, -2.0, 2.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
