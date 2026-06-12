"""GRPO group-relative advantage with the rollout (not the training sample) as the unit.

Pins the reward-attribution contract for fanned agent rollouts: a rollout split into
several segments (compaction / sub-agent / fork branches) that share a ``rollout_id`` counts
once in the baseline, prompt ids (``Sample.group_index``) bucket rollouts into true per-prompt
baseline groups even when counts are uneven, and every segment carries the rollout's advantage.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.ppo_utils import group_relative_advantages


def _legacy_positional_group_norm(raw_rewards, *, n_samples_per_prompt, rollout_batch_size, std_normalization):
    """The pre-fix `_post_process_rewards` math, verbatim, as the bit-identity reference."""
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size:
        rewards = rewards.reshape(-1, n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean
    if std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)
    return rewards.flatten().tolist()


@pytest.mark.unit
def test_none_rollout_id_fails_loud():
    # Highest-stakes guard: dedup is keyed on rollout id, so a silent None would merge
    # every None-id sample into ONE rollout and collapse unrelated samples onto a single
    # advantage. The call site falls back rollout_id -> index, so None means both were
    # unset -- that must raise, not train on garbage.
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
    # prompt0 [1,3] mean 2 -> [-1,+1]; prompt1 [5,11] mean 8 -> [-3,+3].
    assert advantages == pytest.approx([-1.0, 1.0, -3.0, 3.0])


@pytest.mark.unit
def test_rigid_input_is_bit_identical_to_legacy_positional_reshape():
    # On the rigid default path (one sample per rollout, every prompt exactly
    # n_samples_per_prompt rollouts, prompt-major order) the per-prompt buckets are
    # exactly the legacy reshape rows, so the result must be bit-for-bit identical --
    # exact float equality, std normalization included.
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
    # The baseline is over 4 rollouts, NOT 5 samples, so each prompt's mean is unchanged
    # by the fan-out.
    advantages = group_relative_advantages(
        [1.0, 3.0, 3.0, 5.0, 11.0],
        rollout_ids=[0, 1, 1, 2, 3],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-1.0, 1.0, 1.0, -3.0, 3.0])
    assert advantages[1] == advantages[2]
    # A global baseline would subtract mean([1,3,5,11])=5 and give rollout 0 an advantage
    # of -4; the per-prompt baseline gives -1. This asserts per-prompt.
    assert advantages[0] == pytest.approx(-1.0)


@pytest.mark.unit
def test_std_normalization_divides_by_per_prompt_group_std_after_fan_out():
    # rollout 0 is fanned into two segments (rewards [0, 0]); the std must be taken over the
    # 4 *rollouts* [0, 2 | 10, 14], not the 5 segments. Each prompt is symmetric about its
    # mean, so it normalizes to +-1/sqrt(2) (unbiased std) and both fanned segments share it.
    advantages = group_relative_advantages(
        [0.0, 0.0, 2.0, 10.0, 14.0],
        rollout_ids=[0, 0, 1, 2, 3],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=True,
    )
    inv_sqrt2 = 0.70710678
    assert advantages == pytest.approx([-inv_sqrt2, -inv_sqrt2, inv_sqrt2, -inv_sqrt2, inv_sqrt2], abs=1e-4)
    assert advantages[0] == advantages[1]


@pytest.mark.unit
def test_uneven_prompt_groups_get_true_per_prompt_baselines():
    # Dynamic sampling / fan-out leaves 5 rollouts when 2*2=4 are expected: prompt0 has 3
    # rollouts, prompt1 has 2. With prompt ids, each prompt is centered on ITS OWN mean:
    # prompt0 [1,3,5] -> [-2,0,2], prompt1 [7,13] -> [-3,3].
    raw_rewards = [1.0, 3.0, 5.0, 7.0, 13.0]
    advantages = group_relative_advantages(
        raw_rewards,
        rollout_ids=[0, 1, 2, 3, 4],
        prompt_ids=[0, 0, 0, 1, 1],
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        std_normalization=False,
    )
    assert advantages == pytest.approx([-2.0, 0.0, 2.0, -3.0, 3.0])
    # The legacy positional path could only fall back to ONE global group here
    # (mean 5.8, mixing both prompts); the per-prompt baseline must differ from it.
    global_fallback = _legacy_positional_group_norm(
        raw_rewards, n_samples_per_prompt=2, rollout_batch_size=2, std_normalization=False
    )
    assert global_fallback == pytest.approx([-4.8, -2.8, -0.8, 1.2, 7.2])
    assert advantages != pytest.approx(global_fallback)


@pytest.mark.unit
def test_rollout_id_spanning_two_prompts_raises():
    # One rollout id under two prompt ids has no well-defined baseline group; merging
    # them would silently leak reward across prompts.
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
    # Custom rollout paths may not set Sample.group_index (prompt_ids=None). Then the
    # legacy positional behavior is preserved verbatim: rollout 1 is fanned into two
    # segments, so after dedup there are 3 rollouts [1, 2, 3] when 4 are expected ->
    # single global group, mean 2 -> [-1, 0, 1] with rollout 1's segments sharing 0.
    advantages = group_relative_advantages(
        [1.0, 2.0, 2.0, 3.0],
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
