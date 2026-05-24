"""CPU-only unit tests for slime.utils.dp_schedule.build_dp_schedule.

These tests exercise the pack-first-distribute-second scheduler without
importing ray / sglang / torch.distributed, so they run under the same
``pytest examples/coding_agent_rl/tests`` flow as the SWE example tests.

PR #1933 originally added these as ``tests/test_dp_schedule.py``; we keep the
file here (next to the SWE tests that are already wired into pytest config)
so they get picked up without touching pyproject testpaths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Make slime importable when this file is executed directly via pytest.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from slime.utils.dp_schedule import build_dp_schedule  # noqa: E402


def _args(use_dynamic_batch_size: bool, balance_data: bool = False, **kw) -> SimpleNamespace:
    base = dict(
        micro_batch_size=2,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=100,
        balance_data=balance_data,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _config(dp_size=2, cp_size=1, vpp_size=1, mb_group=1) -> dict:
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": mb_group,
    }


def _check_invariants(
    partitions: list[list[int]],
    micro_batch_indices: list[list[list[int]]],
    num_microbatches: list[int],
    global_batch_sizes: list[int],
    *,
    num_samples_kept: int,
    dp_size: int,
):
    # Every kept sample placed exactly once.
    flat = [i for part in partitions for i in part]
    assert sorted(flat) == list(range(num_samples_kept)), f"flat={flat}"

    # Every rank runs the same num_microbatches per step (PP sync invariant).
    for r in range(dp_size):
        # micro_batch_indices[r] is flat across steps. Length should equal sum
        # of num_microbatches (each step contributes num_microbatches[s] mbs).
        assert len(micro_batch_indices[r]) == sum(num_microbatches), (
            f"rank {r}: {len(micro_batch_indices[r])} mbs != sum(num_microbatches)={sum(num_microbatches)}"
        )

    # Flattened micro_batch_indices for each rank should tile range(len(partitions[r])).
    for r in range(dp_size):
        flat_locals = [i for mbs in micro_batch_indices[r] for i in mbs]
        assert sorted(flat_locals) == list(range(len(partitions[r]))), (
            f"rank {r} local indices not a permutation of range(N): {flat_locals}"
        )


def test_static_one_step_one_sample_per_rollout():
    # 8 samples / dp=2 / gbs=4 rollouts → 2 steps, 2 samples per step per rank.
    args = _args(use_dynamic_batch_size=False, micro_batch_size=1)
    rollout_indices = list(range(8))
    total_lengths = [10] * 8
    partitions, mb_idx, n_mbs, gbs = build_dp_schedule(
        args, _config(dp_size=2), total_lengths,
        global_batch_size=4, rollout_indices=rollout_indices,
    )
    assert gbs == [4, 4]
    # static mb_size=1, dp=2 → each step has 4 mbs total, 2 per rank
    assert n_mbs == [2, 2]
    _check_invariants(partitions, mb_idx, n_mbs, gbs, num_samples_kept=8, dp_size=2)


def test_dynamic_compact_two_samples_per_rollout():
    # 4 rollouts, each producing 2 training samples (compact / subagent shape)
    # → 8 samples total. global_batch_size=2 rollouts/step → 2 steps × 4
    # samples = 8 samples. dp=2 → at least 4 mbs per step (align_to=dp=2 so
    # need a multiple of 2; first-fit yields 1 or 2 here).
    args = _args(use_dynamic_batch_size=True, max_tokens_per_gpu=50)
    rollout_indices = [0, 0, 1, 1, 2, 2, 3, 3]
    total_lengths = [10] * 8
    partitions, mb_idx, n_mbs, gbs = build_dp_schedule(
        args, _config(dp_size=2), total_lengths,
        global_batch_size=2, rollout_indices=rollout_indices,
    )
    assert gbs == [2, 2]
    # 4 samples × 10 tokens = 40 ≤ 50 budget → first-fit gives 1 mbs;
    # align to dp=2 → split to 2 mbs per step → 1 per rank
    assert n_mbs == [1, 1]
    _check_invariants(partitions, mb_idx, n_mbs, gbs, num_samples_kept=8, dp_size=2)

    # Critical PR-#1933 invariant: samples of the same rollout MUST land in
    # the same step (i.e., not be split across step boundaries).
    # Step boundary check: micro_batch_indices[r][k] holds ABSOLUTE local
    # indices into partitions[r] (built via local_start + range). Step 0's
    # mbs occupy [0, n0) and step 1's occupy [n0, ...).
    for r in range(2):
        step0_locals = mb_idx[r][0]
        step1_locals = mb_idx[r][1] if len(mb_idx[r]) > 1 else []
        for li in step0_locals:
            gi = partitions[r][li]
            assert rollout_indices[gi] in {0, 1}, (
                f"rank {r} step 0 saw sample {gi} with rollout_id "
                f"{rollout_indices[gi]} but expected {{0, 1}}"
            )
        for li in step1_locals:
            gi = partitions[r][li]
            assert rollout_indices[gi] in {2, 3}, (
                f"rank {r} step 1 saw sample {gi} with rollout_id "
                f"{rollout_indices[gi]} but expected {{2, 3}}"
            )


def test_dynamic_trims_trailing_rollouts():
    # 5 rollouts, gbs=2 → 2 full steps, 1 trailing rollout dropped.
    args = _args(use_dynamic_batch_size=True, max_tokens_per_gpu=100)
    rollout_indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    total_lengths = [10] * 10
    partitions, mb_idx, n_mbs, gbs = build_dp_schedule(
        args, _config(dp_size=2), total_lengths,
        global_batch_size=2, rollout_indices=rollout_indices,
    )
    assert gbs == [2, 2]  # 2 steps × 2 rollouts each = 4 rollouts; 1 dropped
    # Only 8 samples kept (4 rollouts × 2 samples).
    _check_invariants(partitions, mb_idx, n_mbs, gbs, num_samples_kept=8, dp_size=2)
    # Confirm rollout 4 was dropped (sample positions 8, 9 not placed).
    flat = sorted(i for part in partitions for i in part)
    assert flat == list(range(8))


def test_fewer_rollouts_than_gbs_asserts():
    args = _args(use_dynamic_batch_size=True, max_tokens_per_gpu=100)
    rollout_indices = [0, 1]  # only 2 rollouts
    with pytest.raises(AssertionError, match="num_rollouts"):
        build_dp_schedule(
            args, _config(dp_size=2), [10, 10],
            global_batch_size=4, rollout_indices=rollout_indices,
        )


def test_step_size_below_dp_size_asserts():
    # 2 rollouts × 1 sample each = 2 samples / step, but dp_size=4 → crash.
    args = _args(use_dynamic_batch_size=True, max_tokens_per_gpu=100)
    rollout_indices = [0, 1]
    with pytest.raises(AssertionError, match="dp_size"):
        build_dp_schedule(
            args, _config(dp_size=4), [10, 10],
            global_batch_size=2, rollout_indices=rollout_indices,
        )


def test_balance_data_round_robin_fallback():
    args = _args(use_dynamic_batch_size=True, max_tokens_per_gpu=200, balance_data=False)
    rollout_indices = list(range(8))
    total_lengths = [50, 10, 50, 10, 50, 10, 50, 10]
    partitions, mb_idx, n_mbs, gbs = build_dp_schedule(
        args, _config(dp_size=2), total_lengths,
        global_batch_size=4, rollout_indices=rollout_indices,
    )
    _check_invariants(partitions, mb_idx, n_mbs, gbs, num_samples_kept=8, dp_size=2)


def test_balance_data_kk_partition():
    args = _args(use_dynamic_batch_size=True, max_tokens_per_gpu=200, balance_data=True)
    rollout_indices = list(range(8))
    total_lengths = [50, 10, 50, 10, 50, 10, 50, 10]
    partitions, mb_idx, n_mbs, gbs = build_dp_schedule(
        args, _config(dp_size=2), total_lengths,
        global_batch_size=4, rollout_indices=rollout_indices,
    )
    _check_invariants(partitions, mb_idx, n_mbs, gbs, num_samples_kept=8, dp_size=2)
