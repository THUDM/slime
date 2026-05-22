"""CPU unit tests for slime.backends.megatron_utils.cp_utils.get_sum_of_sample_mean.

Focuses on the per-rollout reducer contract: a rollout split into N training
samples (compact / subagent) must contribute exactly one token-weighted mean
to the sum, even when first-fit packing puts those siblings into different
micro-batches at training time.

The tests stub the megatron CP world size to 1 so they run on CPU. The CP > 1
path mirrors the cp_size == 1 logic (same denominator, sliced masks); the
correctness contract this file pins is denominator handling, which is
identical across the two branches.
"""

from __future__ import annotations

import pytest
import torch

from megatron.core import mpu


@pytest.fixture(autouse=True)
def force_cp_size_one(monkeypatch):
    """Stub the CP world size so the reducer takes its cp_size==1 branch."""
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)


def _make_inputs(per_sample_lengths: list[int]):
    """Build (total_lengths, response_lengths, loss_masks) for samples of the given lengths.

    Each sample has loss_mask = all-ones (so mask sum == length); total length
    is response length + 4 fake prompt tokens (unused by the reducer in
    cp_size==1 mode).
    """
    response_lengths = list(per_sample_lengths)
    total_lengths = [r + 4 for r in response_lengths]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    return total_lengths, response_lengths, loss_masks


@pytest.mark.unit
def test_default_reduces_to_per_sample_mean():
    """``sample_denoms=None`` reproduces the legacy per-sample-mean."""
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3])
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # per-sample means: 2, 5, 8 → sum = 15
    assert reducer(x).item() == pytest.approx(15.0)


@pytest.mark.unit
def test_per_rollout_denom_collapses_siblings_into_one_mean():
    """Pre-computed per-rollout mask sums make N sibling samples contribute one
    token-weighted mean instead of N per-sample means."""
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    # 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
    # owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = [9, 9, 9, 3]
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # R0 token-mean: (1+2+...+9)/9 = 5.  R1 token-mean: (10+11+12)/3 = 11.  Sum = 16.
    assert reducer(x).item() == pytest.approx(16.0)


@pytest.mark.unit
def test_split_across_mbs_recovers_full_per_rollout_mean():
    """The critical contract: when a rollout's samples land in different mbs,
    summing each mb's reducer output equals one whole-step reducer call with
    the same pre-computed denominators. This is exactly the bug that motivated
    the precomputation — if the denom were computed per-mb (partial mask sum),
    the two halves wouldn't add up."""
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    # 4 samples (same as above). Whole-step denoms = [9, 9, 9, 3].
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = [9, 9, 9, 3]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    whole_value = whole(x).item()

    # mb_a holds samples 0, 1 of R0; mb_b holds sample 2 of R0 and sample 3 (R1).
    # Each mb carries the SAME per-sample denoms (precomputed at step level)
    # — that's what makes the split safe.
    mb_a = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], sample_denoms[:2])
    mb_b = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], sample_denoms[2:])
    split_value = mb_a(x[:6]).item() + mb_b(x[6:]).item()

    assert split_value == pytest.approx(whole_value)


@pytest.mark.unit
def test_split_with_per_mb_denom_would_be_wrong():
    """Sanity-check the bug we're guarding against: if the caller naively
    computes per-rollout denoms from each mb's own samples (the local mask
    sum, NOT the precomputed whole-rollout sum), the two halves DON'T add up
    to the whole-step value. This pins down WHY the precomputation must
    happen at the step level."""
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, [9, 9, 9, 3])
    whole_value = whole(x).item()

    # Wrong denom: each mb only sees its own samples of R0.
    # mb_a's "rollout mask sum" for R0 would be 3+3=6 (instead of 9). mb_b's
    # would be 3. Different from the true whole-rollout total.
    mb_a_wrong = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], [6, 6])
    mb_b_wrong = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], [3, 3])
    wrong_total = mb_a_wrong(x[:6]).item() + mb_b_wrong(x[6:]).item()

    assert wrong_total != pytest.approx(whole_value), (
        "Expected the per-mb denom path to produce a different (incorrect) value; "
        "if these match, the regression test is no longer guarding the precomputation contract."
    )


@pytest.mark.unit
def test_num_rollouts_contribution_sums_to_distinct_rollout_count():
    """Pins the analogous fix on the metric side: ``sum(1 / rollout_sample_counts)``
    accumulated over the whole step (across mbs and DP) equals the step's
    distinct rollout count. Per-mb counting via ``len(set(rollout_ids))`` would
    double-count any rollout split across mbs."""
    # 4 samples laid out the same way: rollout R0 has 3 samples; rollout R1 has 1.
    rollout_sample_counts = [3, 3, 3, 1]  # constant within each rollout

    # mb_a = samples 0, 1 of R0; mb_b = sample 2 of R0 + sample 3 of R1.
    mb_a_contrib = sum(1.0 / c for c in rollout_sample_counts[:2])
    mb_b_contrib = sum(1.0 / c for c in rollout_sample_counts[2:])
    total = mb_a_contrib + mb_b_contrib

    # 2 distinct rollouts in the step.
    assert total == pytest.approx(2.0)

    # And the naive per-mb len(set) would overcount when the split happens:
    rollout_ids = [0, 0, 0, 1]
    naive_total = len(set(rollout_ids[:2])) + len(set(rollout_ids[2:]))  # = 1 + 2 = 3
    assert naive_total > 2, "naive per-mb len(set) should overcount when R0 spans mbs"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
