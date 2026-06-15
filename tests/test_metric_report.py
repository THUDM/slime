"""Single-process metric-report invariance tests.

Pins train-side / rollout-side report formulas implemented in
``slime.backends.megatron_utils.cp_utils.reduce_train_step_metrics`` and
``rollout_log_metric_contribution``: the reported number for a given set
of samples must be the same regardless of

  - how samples are distributed across micro-batches / DP ranks
  - whether context parallelism is on or off
  - whether the path is per-rollout-mean or per-token-loss

Single-process variants use a mock dp-with-cp group + a no-op
``dist.all_reduce`` to keep things lightweight; the multi-process
end-to-end variants (real torch.distributed) live in
``test_metric_report_dist.py``.
"""

from __future__ import annotations

# Import the helpers BEFORE the slime imports so the megatron stub lands
# in sys.modules first. pytest's prepend importmode puts this file's
# directory (``tests/``) on sys.path, which is what makes the bare-name
# import work without an ``__init__.py``.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    reduce_train_step_metrics,
    rollout_log_metric_contribution,
)


NUM_GPUS = 0


@pytest.fixture
def mock_dp_with_cp_group(monkeypatch):
    """A sentinel "process group" object plus a no-op ``dist.all_reduce``.

    Lets the train-step report tests exercise the production call shape
    (``dist.all_reduce(values, group=dp_with_cp_group)``) without standing
    up a real torch.distributed runtime. The test itself simulates cross-
    rank summation in pure Python and feeds the already-summed tensor in;
    the no-op all_reduce leaves it untouched.
    """
    import torch.distributed as dist

    monkeypatch.setattr(dist, "all_reduce", lambda tensor, group=None, op=None: None)
    return object()  # opaque sentinel — only used as the ``group`` argument


# ---------------------------------------------------------------------------
# Mirrors the actual train_one_step reporting math:
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
    [[[0, 1, 2, 3]]],  # 1 rank, 1 mb
    [[[0, 1, 2], [3]]],  # 1 rank, 2 mbs split at rollout boundary
    [[[0, 1], [2, 3]]],  # 1 rank, 2 mbs splitting R0 across them — the tricky case
    [[[0, 1]], [[2, 3]]],  # 2 ranks, 1 mb each
    [[[0, 1, 3]], [[2]]],  # 2 ranks, R0 split across BOTH ranks (worst case for split-across-mb bug)
    [[[0]], [[1]], [[2]], [[3]]],  # 4 ranks, 1 sample per rank
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

    Each "rank" applies the reducer once over its full sample subset, then
    ``rollout_log_metric_contribution`` (the same helper data.py uses) emits
    the ``(per_rank_sum, count)`` tuple. We aggregate via
    ``Σsum / Σcount`` — the same shape ``gather_log_data`` uses.
    """
    dp_size = len(samples_per_rank)
    pairs: list[tuple[float, float]] = []
    for indices in samples_per_rank:
        if not indices:
            pairs.append(
                rollout_log_metric_contribution(
                    0.0, cp_size=1, num_rollouts_in_rollout=_FIXED_STEP_GBS, dp_size=dp_size
                )
            )
            continue
        tl = [_FIXED_TOTAL_LENGTHS[i] for i in indices]
        rl = [_FIXED_RESPONSE_LENGTHS[i] for i in indices]
        masks = [_FIXED_LOSS_MASKS[i] for i in indices]
        denoms = torch.tensor([_FIXED_ROLLOUT_DENOMS[i] for i in indices], dtype=torch.float32)
        x = torch.cat([_FIXED_X_PER_SAMPLE[i] for i in indices])
        reducer = get_sum_of_sample_mean(tl, rl, masks, denoms)
        pairs.append(
            rollout_log_metric_contribution(
                reducer(x).item(),
                cp_size=1,
                num_rollouts_in_rollout=_FIXED_STEP_GBS,
                dp_size=dp_size,
            )
        )
    total_sum = sum(p[0] for p in pairs)
    total_count = sum(p[1] for p in pairs)
    return total_sum / total_count


_DP_PARTITIONS = [
    [[0, 1, 2, 3]],  # 1 rank holds everything
    [[0, 1, 2], [3]],  # 2 ranks, balanced by rollout
    [[0, 1], [2, 3]],  # 2 ranks splitting R0 across mb-and-rank
    [[0, 1, 3], [2]],  # 2 ranks with R0 spread across BOTH (one of R0's samples is on rank 1)
    [[0], [1], [2], [3]],  # 4 ranks, one sample each (R0's samples spread across 3 ranks)
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
def test_train_one_step_per_rollout_mean_report_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """End-to-end check of train_one_step's report formula across CP sizes.

    Mirrors the actual reduction order:
      1. Each (DP, CP) rank computes per-mb reducer output.
      2. Per-rank values are summed across mbs locally.
      3. All-reduce sums across DP*CP ranks.
      4. ``reduce_train_step_metrics`` applied (the same helper
         ``train_one_step`` calls, so this test stays honest if the
         implementation changes).

    cp_size = 1 vs cp_size = 2 must give the same reported number —
    otherwise wandb metrics would drift the moment a user enables CP.
    """
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    sample_denoms = torch.tensor([16.0, 16.0], dtype=torch.float32)
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]
    step_global_batch_size = 1  # one rollout in the step

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        # values[0] is the per-token-loss path's num_tokens slot; for
        # per-rollout-mean it's a zero placeholder (loss_function sets 0).
        value_after_allreduce = 0.0
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
            if cp_size == 1:
                x_for_rank = torch.cat(x_full)
            else:
                x_chunks_per_sample = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    x_chunks_per_sample.append(torch.cat([c0, c1]))
                x_for_rank = torch.cat(x_chunks_per_sample)
            reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
            value_after_allreduce += reducer(x_for_rank).item()
        reduced = reduce_train_step_metrics(
            [{"keys": ["metric"], "values": torch.tensor([0.0, value_after_allreduce])}],
            calculate_per_token_loss=False,
            step_global_batch_size=step_global_batch_size,
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["metric"]

    assert simulate(1) == pytest.approx(simulate(2))


@pytest.mark.unit
def test_train_one_step_per_token_loss_report_invariant_to_cp(monkeypatch, mock_dp_with_cp_group):
    """Same end-to-end check for the per-token-loss path: divisor is
    ``values[0] = num_tokens`` (computed in loss.py from FULL loss masks),
    which each CP rank duplicates and all-reduce sums by ``cp_size``. The
    ``cp_factor = cp_size`` multiplier inside ``reduce_train_step_metrics``
    cancels that inflation, so the report stays CP-invariant.
    """
    from megatron.core import mpu as _mpu

    total_lengths = [12, 12]
    response_lengths = [8, 8]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    num_tokens_per_mb = sum(int(m.sum().item()) for m in loss_masks)  # = 16
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]

    def simulate(cp_size: int) -> float:
        monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: cp_size)
        value_after_allreduce = 0.0
        num_tokens_after_allreduce = 0  # each CP rank reports the same num_tokens
        for cp_rank in range(cp_size):
            monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
            if cp_size == 1:
                x_for_rank = torch.cat(x_full)
            else:
                x_chunks_per_sample = []
                for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
                    prompt_length = tl - rl
                    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
                    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                    x_chunks_per_sample.append(torch.cat([c0, c1]))
                x_for_rank = torch.cat(x_chunks_per_sample)
            reducer = get_sum_of_sample_mean(
                total_lengths, response_lengths, loss_masks, calculate_per_token_loss=True
            )
            value_after_allreduce += reducer(x_for_rank).item()
            num_tokens_after_allreduce += num_tokens_per_mb
        reduced = reduce_train_step_metrics(
            [
                {
                    "keys": ["metric"],
                    "values": torch.tensor([num_tokens_after_allreduce, value_after_allreduce], dtype=torch.float32),
                }
            ],
            calculate_per_token_loss=True,
            step_global_batch_size=999,  # unused in per-token-loss path
            cp_size=cp_size,
            dp_with_cp_group=mock_dp_with_cp_group,
        )
        return reduced["metric"]

    assert simulate(1) == pytest.approx(simulate(2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
