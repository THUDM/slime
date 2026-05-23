"""CPU unit tests for slime.backends.megatron_utils.cp_utils.get_sum_of_sample_mean.

Focuses on the per-rollout reducer contract: a rollout split into N training
samples (compact / subagent) must contribute exactly one token-weighted mean
to the sum, even when first-fit packing puts those siblings into different
micro-batches at training time.

The CPU-only CI image does not ship megatron, so we stub
``megatron.core.mpu`` *before* importing the module under test. The stub
forces ``cp_size == 1``; the CP > 1 branch in ``get_sum_of_sample_mean``
mirrors the cp_size == 1 denominator logic (same per-sample denoms, sliced
masks), so the contracts pinned here apply to both paths.
"""

from __future__ import annotations

import sys
import types

# --- Stub megatron.core.mpu (must run before the cp_utils import below) ---
_fake_mpu = types.ModuleType("megatron.core.mpu")
_fake_mpu.get_context_parallel_world_size = lambda: 1
_fake_mpu.get_context_parallel_rank = lambda: 0
_fake_core = types.ModuleType("megatron.core")
_fake_core.mpu = _fake_mpu
_fake_megatron = types.ModuleType("megatron")
_fake_megatron.core = _fake_core
sys.modules.setdefault("megatron", _fake_megatron)
sys.modules.setdefault("megatron.core", _fake_core)
sys.modules.setdefault("megatron.core.mpu", _fake_mpu)

import pytest  # noqa: E402
import torch  # noqa: E402

from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)


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


def _denoms(*values: int) -> torch.Tensor:
    """Wrap per-sample denoms as the float tensor that the actor side promotes
    them to before calling the reducer."""
    return torch.tensor(values, dtype=torch.float32)


@pytest.mark.unit
def test_default_reduces_to_per_sample_mean():
    """``sample_denoms=None`` reproduces the legacy per-sample-mean."""
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3])
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # per-sample means: 2, 5, 8 → sum = 15
    assert reducer(x).item() == pytest.approx(15.0)


@pytest.mark.unit
def test_per_rollout_denom_collapses_siblings_into_one_mean():
    """Pre-computed per-rollout mask sums make N sibling samples contribute one
    token-weighted mean instead of N per-sample means."""
    # 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
    # owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = _denoms(9, 9, 9, 3)
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
    # 4 samples (same as above). Whole-step denoms = [9, 9, 9, 3].
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = _denoms(9, 9, 9, 3)
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
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, _denoms(9, 9, 9, 3))
    whole_value = whole(x).item()

    # Wrong denom: each mb only sees its own samples of R0.
    # mb_a's "rollout mask sum" for R0 would be 3+3=6 (instead of 9). mb_b's
    # would be 3. Different from the true whole-rollout total.
    mb_a_wrong = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], _denoms(6, 6))
    mb_b_wrong = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], _denoms(3, 3))
    wrong_total = mb_a_wrong(x[:6]).item() + mb_b_wrong(x[6:]).item()

    assert wrong_total != pytest.approx(whole_value), (
        "Expected the per-mb denom path to produce a different (incorrect) value; "
        "if these match, the regression test is no longer guarding the precomputation contract."
    )


# ---------------------------------------------------------------------------
# End-to-end report invariance: same samples must give the same reported
# number regardless of how they're packed into mbs / spread across DP, and
# regardless of whether CP is on. These mirror the actual train_one_step
# reporting math:
#
#   per-rollout-mean path:
#       reported = sum_of_reducer_per_mb / step_global_batch_size
#   per-token-loss path:
#       reported = sum_of_reducer_per_mb / sum_of_per_mb_num_tokens
#
# The reducer is the same callable used at train time (and inside
# log_rollout_data on the rollout side).
# ---------------------------------------------------------------------------


# 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
# owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
# Per-rollout-mean: R0 = 5, R1 = 11, sum = 16, divided by 2 rollouts → 8.
# Per-token-loss:   sum of all x = 78, total clamped mask = 12, → 6.5.
_FIXED_RESPONSE_LENGTHS = [3, 3, 3, 3]
_FIXED_TOTAL_LENGTHS = [r + 4 for r in _FIXED_RESPONSE_LENGTHS]
_FIXED_LOSS_MASKS = [torch.ones(r, dtype=torch.float32) for r in _FIXED_RESPONSE_LENGTHS]
_FIXED_ROLLOUT_DENOMS = [9.0, 9.0, 9.0, 3.0]
_FIXED_X_PER_SAMPLE = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0, 6.0]),
    torch.tensor([7.0, 8.0, 9.0]),
    torch.tensor([10.0, 11.0, 12.0]),
]
_FIXED_STEP_GBS = 2  # 2 distinct rollouts in the step
_EXPECTED_PER_ROLLOUT_MEAN_REPORT = 8.0
_EXPECTED_PER_TOKEN_LOSS_REPORT = 78.0 / 12.0


# Each entry: list of "rank"s, each rank is a list of mbs, each mb is the
# sample-index list packed into that mb. Covers: single mb, evenly split by
# rollout, split inside a rollout (R0 across mbs), uneven distribution, and
# fully singleton mbs per rank.
_PARTITION_CONFIGS = [
    # 1 rank, 1 mb
    [[[0, 1, 2, 3]]],
    # 1 rank, 2 mbs split at rollout boundary
    [[[0, 1, 2], [3]]],
    # 1 rank, 2 mbs splitting R0 across them — the tricky case
    [[[0, 1], [2, 3]]],
    # 2 ranks, 1 mb each
    [[[0, 1]], [[2, 3]]],
    # 2 ranks, R0 split across BOTH ranks (worst case for split-across-mb bug)
    [[[0, 1, 3]], [[2]]],
    # 4 ranks, 1 sample per rank
    [[[0]], [[1]], [[2]], [[3]]],
]


def _simulate_report(partition, *, per_token_loss: bool) -> float:
    """Reproduce train_one_step's reporting math for one partition config."""
    metric_sum = 0.0
    num_tokens_sum = 0
    for rank_mbs in partition:
        for mb_indices in rank_mbs:
            mb_total = [_FIXED_TOTAL_LENGTHS[i] for i in mb_indices]
            mb_resp = [_FIXED_RESPONSE_LENGTHS[i] for i in mb_indices]
            mb_masks = [_FIXED_LOSS_MASKS[i] for i in mb_indices]
            mb_x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in mb_indices])
            if per_token_loss:
                # Per-token-loss: caller uses ``calculate_per_token_loss=True``
                # to get ``sum_of_token`` (no per-sample denom).
                reducer = get_sum_of_sample_mean(mb_total, mb_resp, mb_masks, calculate_per_token_loss=True)
                num_tokens_sum += sum(max(int(m.sum().item()), 1) for m in mb_masks)
            else:
                mb_denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in mb_indices], dtype=torch.float32)
                reducer = get_sum_of_sample_mean(mb_total, mb_resp, mb_masks, mb_denoms)
            metric_sum += reducer(mb_x).item()
    if per_token_loss:
        return metric_sum / num_tokens_sum
    return metric_sum / _FIXED_STEP_GBS


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
def test_per_rollout_mean_report_invariant_to_mb_distribution(partition):
    """Same samples should yield the same per-rollout-mean report regardless of
    how they're spread across DP ranks / micro-batches — this is what lets us
    change parallelism without changing wandb numbers."""
    assert _simulate_report(partition, per_token_loss=False) == pytest.approx(_EXPECTED_PER_ROLLOUT_MEAN_REPORT)


@pytest.mark.unit
@pytest.mark.parametrize("partition", _PARTITION_CONFIGS)
def test_per_token_loss_report_invariant_to_mb_distribution(partition):
    """Same invariant for the per-token-loss reporting path."""
    assert _simulate_report(partition, per_token_loss=True) == pytest.approx(_EXPECTED_PER_TOKEN_LOSS_REPORT)


def _simulate_rollout_report(samples_per_rank):
    """Reproduce log_rollout_data + gather_log_data's averaging math for the
    per-token metric branch.

    Each "rank" applies the reducer once over its full sample subset, then we
    aggregate ``(per_rank_sum, count)`` tuples across DP via
    ``sum_total / count_total`` — the same shape the real ``gather_log_data``
    uses. The fixed contract is ``count_total == total rollouts`` (so the
    report ends up dividing ``sum_DP_full`` by ``step_global_batch_size`` and
    matches the train side); we spread the rollout count evenly across DP
    ranks so this holds regardless of the partition.
    """
    dp_size = len(samples_per_rank)
    rollout_count_per_rank = _FIXED_STEP_GBS / dp_size
    pairs: list[tuple[float, float]] = []
    for indices in samples_per_rank:
        if not indices:
            pairs.append((0.0, rollout_count_per_rank))
            continue
        tl = [_FIXED_TOTAL_LENGTHS[i] for i in indices]
        rl = [_FIXED_RESPONSE_LENGTHS[i] for i in indices]
        masks = [_FIXED_LOSS_MASKS[i] for i in indices]
        denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in indices], dtype=torch.float32)
        x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in indices])
        reducer = get_sum_of_sample_mean(tl, rl, masks, denoms)
        pairs.append((reducer(x).item(), rollout_count_per_rank))
    total_sum = sum(p[0] for p in pairs)
    total_count = sum(p[1] for p in pairs)
    return total_sum / total_count


_DP_PARTITIONS = [
    # 1 rank holds everything
    [[0, 1, 2, 3]],
    # 2 ranks, balanced by rollout
    [[0, 1, 2], [3]],
    # 2 ranks splitting R0 across mb-and-rank
    [[0, 1], [2, 3]],
    # 2 ranks with R0 spread across BOTH (one of R0's samples is on rank 1)
    [[0, 1, 3], [2]],
    # 4 ranks, one sample each (R0's samples spread across 3 ranks)
    [[0], [1], [2], [3]],
]


@pytest.mark.unit
@pytest.mark.parametrize("dp_partition", _DP_PARTITIONS)
def test_rollout_report_matches_train_report_in_single_step(dp_partition):
    """In a 1-step rollout, the rollout-side report (log_rollout_data → gather)
    must equal the train-side report (train_one_step ``value / step_global_batch_size``)
    for the same samples — otherwise wandb numbers between phases drift.

    Both go through the same reducer with the same precomputed denominators;
    the contract this test pins is that the gather count plumbing on the
    rollout side sums to the same denominator the train side uses
    (``step_global_batch_size``), independent of how the rollout's samples
    are spread across DP ranks.
    """
    rollout_report = _simulate_rollout_report(dp_partition)
    assert rollout_report == pytest.approx(_EXPECTED_PER_ROLLOUT_MEAN_REPORT)


@pytest.mark.unit
def test_cp_chunking_preserves_per_rollout_mean_report(monkeypatch):
    """Turning CP on must not change the reported number.

    Real flow: each CP rank only sees its chunk of the response tokens; the
    reducer's CP>1 branch slices ``loss_mask`` to match. Summing each CP
    rank's reducer output across CP ranks reproduces the cp=1 result, which
    is what train_one_step then divides by ``step_global_batch_size``.
    """
    from megatron.core import mpu as _mpu

    # Use lengths that line up cleanly with the CP chunking
    # (chunk_size = ceil(total_length / (2*cp_size))).
    total_lengths = [12, 12]  # 2 samples
    response_lengths = [8, 8]  # 4 prompt + 8 response each
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    sample_denoms = torch.tensor([16.0, 16.0], dtype=torch.float32)  # = sum of both mask totals (one rollout)
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]
    x_concat = torch.cat(x_full)

    # --- cp=1 baseline ---
    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda: 0)
    reducer_cp1 = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    baseline = reducer_cp1(x_concat).item()

    # --- cp=2: sum partial reducer outputs across the two CP ranks ---
    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: 2)
    cp_total = 0.0
    for cp_rank in range(2):
        monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
        # Slice each sample's response-token tensor to the chunks this CP
        # rank owns, mirroring what the forward pass would feed in.
        x_chunks_per_sample = []
        for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
            prompt_length = tl - rl
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
            chunk_0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            chunk_1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            x_chunks_per_sample.append(torch.cat([chunk_0, chunk_1]))
        x_for_rank = torch.cat(x_chunks_per_sample)
        reducer_cp2 = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
        cp_total += reducer_cp2(x_for_rank).item()

    assert cp_total == pytest.approx(baseline)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
