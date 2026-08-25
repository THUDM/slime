"""CPU unit tests for ``slime.utils.data.process_rollout_data``.

``RolloutManager._split_train_data_by_dp`` ships two kinds of fields to the
trainer:

  * per-sample fields (``response_lengths``, ``loss_masks``, ...) already
    sliced down to the samples this DP rank owns, and
  * ``raw_reward`` / ``total_lengths``, sent whole because something on the
    training side still needs the full rollout batch.

``process_rollout_data`` is where the second group is reconciled with the
first. These tests pin that contract:

  1. ``total_lengths`` comes out DP-local (already the case).
  2. ``raw_reward`` stays global — ``log_passrate`` reshapes it into
     ``[rollout_batch_size, n_samples_per_prompt]`` groups, which only works
     on the full batch.
  3. ``local_raw_reward`` is the DP-local view, positionally aligned with the
     per-sample fields.

(3) is the regression guard for the ``--log-correct-samples`` crash: that
block zips rewards against ``response_lengths`` / ``total_lengths`` /
``loss_masks`` / ``log_probs`` by position, so feeding it the global
``raw_reward`` walked off the end of this rank's lists with an
``IndexError`` (and, before running off the end, silently attributed one
sample's reward to a different sample).
"""

from __future__ import annotations

import pytest
import ray

from slime.utils.data import process_rollout_data

NUM_GPUS = 0


class _FakeBox:
    """Stand-in for ``slime.ray.utils.Box``: payload lives behind ``.inner``."""

    def __init__(self, inner):
        self.inner = inner


@pytest.fixture
def unwrap_ray_get(monkeypatch):
    """``process_rollout_data`` uses Ray only to deref the per-rank Box.

    Patching ``ray.get`` to the identity keeps these tests single-process
    (no cluster start-up) while still exercising the real function.
    """
    monkeypatch.setattr(ray, "get", lambda ref: ref)


def _split_train_data_by_dp(partitions, raw_reward, response_lengths, total_lengths):
    """Mirror what ``RolloutManager._split_train_data_by_dp`` packages per rank."""
    return [
        _FakeBox(
            {
                "partition": partition,
                "response_lengths": [response_lengths[j] for j in partition],
                "raw_reward": list(raw_reward),
                "total_lengths": list(total_lengths),
            }
        )
        for partition in partitions
    ]


# 8 samples; only the odd-indexed ones are correct. Lengths encode their own
# global index so a mis-pairing is visible in the assertion message.
RAW_REWARD = [0, 1, 0, 1, 0, 1, 0, 1]
RESPONSE_LENGTHS = [100, 101, 102, 103, 104, 105, 106, 107]
TOTAL_LENGTHS = [200, 201, 202, 203, 204, 205, 206, 207]


@pytest.mark.parametrize(
    "partitions",
    [
        pytest.param([[0, 2, 4, 6], [1, 3, 5, 7]], id="dp2-interleaved"),
        pytest.param([[0, 1, 2, 3], [4, 5, 6, 7]], id="dp2-contiguous"),
        pytest.param([[0, 3], [1, 6], [2, 5], [4, 7]], id="dp4-balanced"),
        # Even at dp_size=1 the partition is a permutation: first-fit packing
        # reorders samples by length.
        pytest.param([[3, 0, 7, 1, 5, 2, 6, 4]], id="dp1-permuted"),
    ],
)
def test_local_raw_reward_is_dp_local_and_aligned(unwrap_ray_get, partitions):
    dp_size = len(partitions)
    refs = _split_train_data_by_dp(partitions, RAW_REWARD, RESPONSE_LENGTHS, TOTAL_LENGTHS)

    for dp_rank, partition in enumerate(partitions):
        rollout_data = process_rollout_data(rollout_data_ref=refs, dp_rank=dp_rank, dp_size=dp_size)

        local_raw_reward = rollout_data["local_raw_reward"]
        assert local_raw_reward == [RAW_REWARD[j] for j in partition]
        # Positional alignment with the per-sample fields is the whole point.
        assert len(local_raw_reward) == len(rollout_data["response_lengths"])
        assert len(local_raw_reward) == len(rollout_data["total_lengths"])


@pytest.mark.parametrize(
    "partitions",
    [
        pytest.param([[0, 2, 4, 6], [1, 3, 5, 7]], id="dp2-interleaved"),
        pytest.param([[3, 0, 7, 1, 5, 2, 6, 4]], id="dp1-permuted"),
    ],
)
def test_correct_sample_selection_matches_owned_samples(unwrap_ray_get, partitions):
    """Replay the ``--log-correct-samples`` selection loop.

    Regression for the ``IndexError`` this used to raise on DP > 1.
    """
    dp_size = len(partitions)
    refs = _split_train_data_by_dp(partitions, RAW_REWARD, RESPONSE_LENGTHS, TOTAL_LENGTHS)

    for dp_rank, partition in enumerate(partitions):
        rollout_data = process_rollout_data(rollout_data_ref=refs, dp_rank=dp_rank, dp_size=dp_size)

        response_lengths = rollout_data["response_lengths"]
        total_lengths = rollout_data["total_lengths"]

        correct_response_lengths = []
        correct_total_lengths = []
        for i, raw_reward in enumerate(rollout_data["local_raw_reward"]):
            if raw_reward == 1:
                correct_response_lengths.append(response_lengths[i])
                correct_total_lengths.append(total_lengths[i])

        expected = [j for j in partition if RAW_REWARD[j] == 1]
        assert correct_response_lengths == [RESPONSE_LENGTHS[j] for j in expected]
        assert correct_total_lengths == [TOTAL_LENGTHS[j] for j in expected]


def test_raw_reward_stays_global(unwrap_ray_get):
    """``log_passrate`` needs the whole batch, so the global copy must survive."""
    partitions = [[0, 2, 4, 6], [1, 3, 5, 7]]
    refs = _split_train_data_by_dp(partitions, RAW_REWARD, RESPONSE_LENGTHS, TOTAL_LENGTHS)

    for dp_rank in range(len(partitions)):
        rollout_data = process_rollout_data(rollout_data_ref=refs, dp_rank=dp_rank, dp_size=len(partitions))
        assert rollout_data["raw_reward"] == RAW_REWARD


def test_missing_raw_reward_is_tolerated(unwrap_ray_get):
    """Forward-only passes ship no ``raw_reward``; don't invent one."""
    partition = [1, 0]
    refs = [
        _FakeBox(
            {
                "partition": partition,
                "response_lengths": [101, 100],
                "total_lengths": [200, 201],
            }
        )
    ]

    rollout_data = process_rollout_data(rollout_data_ref=refs, dp_rank=0, dp_size=1)

    assert "local_raw_reward" not in rollout_data
    assert rollout_data["total_lengths"] == [201, 200]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
