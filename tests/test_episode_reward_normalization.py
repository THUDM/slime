"""CPU regressions for query-local, episode-weighted reward normalization."""

import sys
from types import SimpleNamespace

import pytest
import torch

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.data.batch_builder import BatchBuilder
from slime.utils.ppo_utils import get_grpo_returns
from slime.utils.types import Sample

NUM_GPUS = 0


def _builder(**overrides):
    config = dict(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    config.update(overrides)
    return BatchBuilder(SimpleNamespace(**config))


def _sample(group, rollout, reward):
    return Sample(group_index=group, rollout_id=rollout, reward=reward, tokens=[1, 2], response_length=1)


def _episode(group, rollout, reward, compact):
    """Exercise the real recorder: compaction turns one episode into two rows."""
    manager = TrajectoryManager()
    ids = {"Q": 1, "A1": 11, "O1": 21, "A2": 12, "O2": 22, "A3": 13, "O3": 23, "Z": 99, "A4": 14}

    def message(name):
        role = "assistant" if name.startswith("A") else "tool" if name.startswith("O") else "user"
        return {"role": role, "content": name}

    turns = [
        (["Q"], "A1"),
        (["Q", "A1", "O1"], "A2"),
        (["Q", "A1", "O1", "A2", "O2"], "A3"),
        (["Q", "Z", "A3", "O3"] if compact else ["Q", "A1", "O1", "A2", "O2", "A3", "O3"], "A4"),
    ]
    for prompt, output in turns:
        manager.record_turn(
            "session",
            prompt_messages=[message(name) for name in prompt],
            response_message=message(output),
            turn=TurnRecord(
                prompt_ids=[ids[name] for name in prompt],
                output_ids=[ids[output]],
                output_log_probs=[-0.1],
                finish_reason="stop",
            ),
        )
    samples = manager.get_trajectory(
        "session", base_sample=Sample(index=rollout, group_index=group, rollout_id=rollout), reward=reward
    )
    assert len(samples) == (2 if compact else 1)
    assert sum(sum(sample.loss_mask) for sample in samples) == 4
    return samples


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("use_std", [False, True])
def test_native_compaction_preserves_episode_advantages(compact, use_std):
    episodes = [(0, 0, 1.0, compact), (0, 1, 0.0, False), (1, 2, 0.0, False), (1, 3, 0.0, compact)]
    samples = [sample for spec in episodes for sample in _episode(*spec)]
    converted = _builder(grpo_std_normalization=use_std).convert(samples)
    magnitude = 0.5 / (2**-0.5 + 1e-6) if use_std else 0.5
    expected = {0: magnitude, 1: -magnitude, 2: 0.0, 3: 0.0}
    assert converted["rewards"] == pytest.approx([expected[s.rollout_id] for s in samples])
    assert converted["raw_reward"] == [s.reward for s in samples]
    assert converted["rollout_mask_sums"] == [4] * len(samples)

    # Feed the normalized rewards into the production GRPO return calculation.
    returns = get_grpo_returns(torch.tensor(converted["rewards"]), [torch.zeros(s.response_length) for s in samples])
    for sample, ret in zip(samples, returns, strict=True):
        active = ret[torch.tensor(sample.loss_mask).bool()]
        torch.testing.assert_close(active, torch.full_like(active, expected[sample.rollout_id]))


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"])
@pytest.mark.parametrize("use_std", [False, True])
def test_uneven_groups_count_each_episode_once(estimator, use_std):
    # Interleave groups, include a singleton, and split the successful episode.
    samples = [_sample(9, 4, 1), _sample(2, 8, 7), _sample(9, 5, 0), _sample(9, 4, 1), _sample(3, 6, 0)]
    _, rewards = _builder(advantage_estimator=estimator, grpo_std_normalization=use_std)._post_process_rewards(samples)
    magnitude = 0.5 / (2**-0.5 + 1e-6) if use_std and estimator != "reinforce_plus_plus_baseline" else 0.5
    assert rewards == pytest.approx([magnitude, 0, -magnitude, magnitude, 0])
    reverse_rewards = _builder(advantage_estimator=estimator, grpo_std_normalization=use_std)._post_process_rewards(
        samples[::-1]
    )[1]
    assert reverse_rewards == pytest.approx(rewards[::-1])


def test_explicit_groups_override_nominal_batch_shape():
    # Four rows still equal rollout_batch_size * n_samples_per_prompt. Reshaping
    # would mix the two queries even without taking the old fallback branch.
    samples = [_sample(0, 0, 1), _sample(1, 2, 0), _sample(0, 1, 0), _sample(1, 3, 0)]
    assert _builder()._post_process_rewards(samples)[1] == [0.5, 0, -0.5, 0]


@pytest.mark.parametrize("use_std", [False, True])
def test_legacy_fixed_size_groups_preserve_normalization(use_std):
    samples = [_sample(None, None, reward) for reward in [0.2, 0.8, 2, 4]]
    expected = torch.tensor([0.2, 0.8, 2, 4]).reshape(2, 2)
    expected = expected - expected.mean(dim=-1, keepdim=True)
    if use_std:
        expected = expected / (expected.std(dim=-1, keepdim=True) + 1e-6)
    actual = _builder(grpo_std_normalization=use_std)._post_process_rewards(samples)[1]
    torch.testing.assert_close(torch.tensor(actual), expected.flatten(), rtol=0, atol=0)


@pytest.mark.parametrize("use_std", [False, True])
def test_equal_rewards_and_singleton_are_finite_zero(use_std):
    samples = [_sample(0, 0, 5), _sample(0, 1, 5), _sample(1, 2, 3)]
    assert _builder(grpo_std_normalization=use_std)._post_process_rewards(samples)[1] == [0, 0, 0]


def test_missing_rollout_ids_are_independent_rows():
    samples = [_sample(0, None, 1), _sample(0, None, 0)]
    for sample in samples:
        sample.index = 0  # Do not merge rows by a missing or reused sample index.
    assert _builder()._post_process_rewards(samples)[1] == [0.5, -0.5]


def test_reward_key_is_used_for_episode_consistency():
    samples = [
        _sample(0, 0, {"score": 1, "diagnostic": 10}),
        _sample(0, 0, {"score": 1, "diagnostic": 20}),
        _sample(0, 1, {"score": 0, "diagnostic": 30}),
    ]
    raw, rewards = _builder(reward_key="score")._post_process_rewards(samples)
    assert raw == [1, 1, 0]
    assert rewards == [0.5, 0.5, -0.5]


def test_conflicting_rewards_for_one_episode_are_rejected():
    samples = [_sample(0, 0, 1), _sample(0, 0, 0), _sample(0, 1, 0)]
    with pytest.raises(ValueError, match="reward.*rollout_id"):
        _builder()._post_process_rewards(samples)


@pytest.mark.parametrize(
    "samples",
    [
        [_sample(0, 0, 1), _sample(None, 1, 0)],
        [_sample(None, None, 1), _sample(None, None, 0), _sample(None, None, 0)],
        [_sample(None, 0, 1), _sample(None, 0, 1), _sample(None, 1, 0), _sample(None, 2, 0)],
    ],
)
def test_ambiguous_query_groups_are_rejected(samples):
    with pytest.raises(ValueError, match="group_index"):
        _builder()._post_process_rewards(samples)


def test_custom_hook_takes_precedence():
    builder = _builder()
    samples = [_sample(0, 0, 1), _sample(0, 0, 0)]

    def custom(args, received):
        assert args is builder.args
        assert received is samples
        return [1, 0], [10, 20]

    builder.custom_reward_post_process_func = custom
    assert builder._post_process_rewards(samples) == ([1, 0], [10, 20])


@pytest.mark.parametrize("overrides", [{"rewards_normalization": False}, {"advantage_estimator": "ppo"}])
def test_no_normalization_preserves_raw_rewards(overrides):
    samples = [_sample(0, 0, 1), _sample(None, 0, 0)]
    assert _builder(**overrides)._post_process_rewards(samples) == ([1, 0], [1, 0])


def test_empty_rewards():
    assert _builder()._post_process_rewards([]) == ([], [])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
