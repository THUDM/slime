"""CPU unit tests for slime.utils.ppo_utils.group_relative_advantages.

The unit is the rollout, not the training sample: segments of one fanned rollout share a
rollout_id and split the rollout reward (reward / len), so the baseline counts each rollout
once; prompt ids bucket rollouts into per-prompt groups even when counts are uneven.
"""

import pytest
import torch

from slime.utils.ppo_utils import group_relative_advantages

NUM_GPUS = 0

inv_sqrt2 = 0.70710678


def _legacy_positional_group_norm(raw_rewards, *, n_samples_per_prompt, rollout_batch_size, std_normalization):
    """The pre-fix _post_process_rewards math, verbatim, as the bit-identity reference."""
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size:
        rewards = rewards.reshape(-1, n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    rewards = rewards - rewards.mean(dim=-1, keepdim=True)
    if std_normalization:
        rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
    return rewards.flatten().tolist()


@pytest.mark.unit
def test_none_rollout_id_fails_loud():
    # Dedup keys on rollout id; a None would merge unrelated samples into one rollout.
    with pytest.raises(ValueError, match="rollout id"):
        group_relative_advantages(
            [1.0, 2.0, 3.0],
            rollout_ids=[0, None, 2],
            prompt_ids=[0, 0, 1],
            n_samples_per_prompt=2,
            rollout_batch_size=1,
            std_normalization=False,
        )


@pytest.mark.unit
def test_default_one_sample_per_rollout_is_plain_per_prompt_group_norm():
    advantages = group_relative_advantages(
        [1.0, 3.0, 5.0, 11.0],
        rollout_ids=[0, 1, 2, 3],
        prompt_ids=[0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    # prompt0 [1,3] -> [-1,+1]; prompt1 [5,11] -> [-3,+3].
    assert advantages == pytest.approx([-1.0, 1.0, -3.0, 3.0])


@pytest.mark.unit
def test_rigid_input_is_bit_identical_to_legacy_positional_reshape():
    # One sample per rollout, prompt-major, every prompt exactly n_samples_per_prompt: the
    # per-prompt buckets are the legacy reshape rows, so the result is bit-for-bit identical.
    raw_rewards = [0.137, -2.9, 3.7, 0.001, 7.25, -0.6, 1.0, 1.0, -5.5, 2.25, 0.0, 9.125]
    n_samples_per_prompt, rollout_batch_size = 4, 3
    advantages = group_relative_advantages(
        raw_rewards,
        rollout_ids=list(range(12)),
        prompt_ids=[i // n_samples_per_prompt for i in range(12)],
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_batch_size=rollout_batch_size,
        std_normalization=True,
    )
    legacy = _legacy_positional_group_norm(
        raw_rewards,
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_batch_size=rollout_batch_size,
        std_normalization=True,
    )
    assert advantages == legacy


@pytest.mark.unit
def test_fanned_rollout_counts_once_and_segments_share_the_advantage():
    # rollout 1 is fanned into two segments splitting its reward 3.0 -> [1.5, 1.5]; the baseline
    # is over 4 rollouts [1, 3, 5, 11], not 5 segments, so the per-prompt means are unchanged.
    advantages = group_relative_advantages(
        [1.0, 1.5, 1.5, 5.0, 11.0],
        rollout_ids=[0, 1, 1, 2, 3],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 1.0, 1.0, -3.0, 3.0])
    assert advantages[1] == advantages[2]
    # A global baseline (mean of [1,3,5,11]=5) would give rollout 0 advantage -4; per-prompt gives -1.
    assert advantages[0] == pytest.approx(-1.0)


@pytest.mark.unit
def test_std_normalization_uses_per_prompt_group_std_over_rollouts():
    # rollout 0 fans into two zero-reward segments; std is over the 4 rollouts [0,2 | 10,14],
    # not the 5 segments. Each prompt is symmetric so it normalizes to +-1/sqrt(2).
    advantages = group_relative_advantages(
        [0.0, 0.0, 2.0, 10.0, 14.0],
        rollout_ids=[0, 0, 1, 2, 3],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=True,
    )
    assert advantages == pytest.approx([-inv_sqrt2, -inv_sqrt2, inv_sqrt2, -inv_sqrt2, inv_sqrt2], abs=1e-4)
    assert advantages[0] == advantages[1]


@pytest.mark.unit
def test_single_rollout_prompt_under_std_is_zero_not_nan():
    # A prompt with one rollout has undefined std; it must stay mean-centered (0), not NaN.
    advantages = group_relative_advantages(
        [5.0, 1.0, 3.0],
        rollout_ids=[0, 1, 2],
        prompt_ids=[0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=True,
    )
    assert advantages == pytest.approx([0.0, -inv_sqrt2, inv_sqrt2], abs=1e-4)


@pytest.mark.unit
def test_uneven_prompt_groups_get_true_per_prompt_baselines():
    # 5 rollouts when 2*2=4 are expected: prompt0 has 3, prompt1 has 2; each centers on its own mean.
    raw_rewards = [1.0, 3.0, 5.0, 7.0, 13.0]
    advantages = group_relative_advantages(
        raw_rewards,
        rollout_ids=[0, 1, 2, 3, 4],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    # prompt0 [1,3,5] -> [-2,0,2]; prompt1 [7,13] -> [-3,3].
    assert advantages == pytest.approx([-2.0, 0.0, 2.0, -3.0, 3.0])
    # The legacy path could only fall back to one global group here, so it must differ.
    global_fallback = _legacy_positional_group_norm(
        raw_rewards, n_samples_per_prompt=2, rollout_batch_size=2, std_normalization=False
    )
    assert advantages != pytest.approx(global_fallback)


@pytest.mark.unit
def test_rollout_id_spanning_two_prompts_raises():
    # One rollout id under two prompt ids has no well-defined baseline group.
    with pytest.raises(ValueError, match="prompt ids"):
        group_relative_advantages(
            [1.0, 2.0],
            rollout_ids=[7, 7],
            prompt_ids=[0, 1],
            n_samples_per_prompt=1,
            rollout_batch_size=2,
            std_normalization=False,
        )


@pytest.mark.unit
def test_metadata_less_uneven_counts_dedup_then_fall_back_to_one_global_group():
    # prompt_ids=None (custom rollout path): rollout 1's reward 2.0 splits into [1.0, 1.0]; after
    # dedup there are 3 rollouts [1, 2, 3] when 4 are expected -> one global group, mean 2.
    advantages = group_relative_advantages(
        [1.0, 1.0, 1.0, 3.0],
        rollout_ids=[0, 1, 1, 2],
        prompt_ids=None,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 0.0, 0.0, 1.0])
    assert advantages[1] == advantages[2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
