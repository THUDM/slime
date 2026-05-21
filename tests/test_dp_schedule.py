"""CPU unit tests for slime.utils.dp_schedule.build_dp_schedule.

The tests assert the invariants documented at the top of dp_schedule.py against
a range of static / dynamic / VPP / oversize / balance / uneven scenarios.

Trim, dynamic-gbs resolution, custom step splitting, and per-rank rollout_data
packaging all live in ``RolloutManager._split_train_data_by_dp``; these tests
just exercise the schedule itself (``partitions``, ``micro_batch_indices``).
"""

from types import SimpleNamespace

import pytest

from slime.utils.dp_schedule import (
    build_dp_schedule,
    even_step_split,
    near_equal_step_split,
    validate_step_sample_indices,
)


def make_args(
    *,
    micro_batch_size=1,
    use_dynamic_batch_size=False,
    max_tokens_per_gpu=None,
    balance_data=False,
):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=balance_data,
    )


def make_tp(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def assert_invariants(
    partitions,
    micro_batch_indices,
    num_microbatches,
    *,
    dp_size,
    total_lengths,
    max_per_bin=None,
    global_batch_sizes=None,
    expected_step_sizes=None,
):
    """Check the invariants documented at the top of dp_schedule.py.

    No longer asserts equal-size per-rank partitions — Stage 2 allows uneven
    sample counts per rank. It still checks the union/disjointness invariants
    and the (more important) "same num_mbs per rank" PP-sync invariant.
    """
    seen_global: set[int] = set()
    for r in range(dp_size):
        partition = partitions[r]
        mbi = micro_batch_indices[r]

        # Same num_mbs per rank (PP sync); num_microbatches is shared, so each rank's
        # flat mbs count must match sum(num_microbatches).
        assert len(mbi) == sum(num_microbatches), f"rank {r}: mbs count mismatch"

        # Flattened micro_batch_indices == range(len(partition)) (each sample covered
        # exactly once, by exactly one mbs).
        flat = [i for mbs in mbi for i in mbs]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(range(len(total_lengths))), "some samples not assigned to any rank"

    if global_batch_sizes is not None and expected_step_sizes is not None:
        assert global_batch_sizes == list(
            expected_step_sizes
        ), f"global_batch_sizes {global_batch_sizes} != expected {list(expected_step_sizes)}"

    if max_per_bin is None:
        return

    # Every mbs <= max_per_bin tokens, EXCEPT a singleton bin holding an oversized sample.
    for r in range(dp_size):
        partition = partitions[r]
        for mbs in micro_batch_indices[r]:
            bin_total = sum(total_lengths[partition[i]] for i in mbs)
            if bin_total > max_per_bin:
                assert len(mbs) == 1, f"rank {r}: mbs sum {bin_total} > {max_per_bin} but contains {len(mbs)} samples"


def _even(num_samples, gbs):
    """Shorthand: default even split for the equal-divisible case."""
    return even_step_split(num_samples, gbs)


@pytest.mark.unit
def test_static_stride_single_step():
    """Static + strided DP split, single step."""
    total_lengths = [10] * 16
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(args, tp, total_lengths, step_sample_indices=_even(16, 16))

    assert nmb == [2]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        total_lengths=total_lengths,
        global_batch_sizes=gbs_per_step,
        expected_step_sizes=[16],
    )


@pytest.mark.unit
def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps. Each rank still runs same num_mbs per step."""
    total_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]  # 2 steps of 8
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(args, tp, total_lengths, step_sample_indices=_even(16, 8))

    assert nmb == [2, 2]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        total_lengths=total_lengths,
        global_batch_sizes=gbs_per_step,
        expected_step_sizes=[8, 8],
    )


@pytest.mark.unit
def test_dynamic_uniform():
    """Dynamic mbs on uniform-length samples."""
    total_lengths = [5] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(args, tp, total_lengths, step_sample_indices=_even(8, 8))

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        total_lengths=total_lengths,
        max_per_bin=10,
        global_batch_sizes=gbs_per_step,
        expected_step_sizes=[8],
    )


@pytest.mark.unit
def test_dynamic_skewed_lengths():
    """Skewed lengths (the case where K-K used to over-pack a single bin)."""
    total_lengths = [9, 9, 9, 9, 1, 1, 1, 1]
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(args, tp, total_lengths, step_sample_indices=_even(8, 8))

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        total_lengths=total_lengths,
        max_per_bin=10,
        global_batch_sizes=gbs_per_step,
        expected_step_sizes=[8],
    )


@pytest.mark.unit
def test_dynamic_oversized_sample_lands_alone():
    """A single sample exceeding max_per_bin must end up alone in its mbs (with no
    other samples crammed in)."""
    total_lengths = [15, 3, 3, 3, 3, 3, 3, 3]  # 15 > C=10
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(args, tp, total_lengths, step_sample_indices=_even(8, 8))

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        total_lengths=total_lengths,
        max_per_bin=10,
        global_batch_sizes=gbs_per_step,
        expected_step_sizes=[8],
    )
    # Find the rank holding the oversized sample and verify it lives alone in some mbs.
    oversize_idx = total_lengths.index(15)
    found = False
    for r in range(2):
        partition = partitions[r]
        if oversize_idx not in partition:
            continue
        local = partition.index(oversize_idx)
        for mbs in mbi[r]:
            if local in mbs:
                assert mbs == [local], f"oversized sample shares an mbs: {mbs}"
                found = True
    assert found


@pytest.mark.unit
def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    total_lengths = [4] * 32  # 2 steps of 16; per step, ~8 bins of 8 needed at C=8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(args, tp, total_lengths, step_sample_indices=_even(32, 16))

    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        total_lengths=total_lengths,
        max_per_bin=8,
        global_batch_sizes=gbs_per_step,
        expected_step_sizes=[16, 16],
    )


@pytest.mark.unit
def test_dynamic_uneven_sample_count():
    """Pack-first lets DP ranks hold different numbers of samples (here gbs=10,
    dp=4 → ranks must still run the same ``num_mbs`` but sample counts may differ)."""
    # 10 samples of length 5, max-per-bin 10 → first-fit packs into 5 bins of 2 each.
    # 5 is not a multiple of dp_size=4, so the schedule splits the largest bin until
    # K is the next multiple of 4 → K=8, num_mbs_per_rank=2.
    total_lengths = [5] * 10
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, step_sample_indices=[list(range(10))]
    )

    # Per-step gbs = real sample count.
    assert gbs_per_step == [10]
    # Same num_mbs across ranks (PP sync).
    for r in range(4):
        assert len(mbi[r]) == sum(nmb), f"rank {r}: {len(mbi[r])} mbs, want {sum(nmb)}"
    # All 10 samples assigned exactly once across ranks.
    seen: set[int] = set()
    for r in range(4):
        assert seen.isdisjoint(partitions[r])
        seen.update(partitions[r])
    assert seen == set(range(10))
    # Per-rank mbs respect the cap.
    for r in range(4):
        for mbs in mbi[r]:
            assert sum(total_lengths[partitions[r][i]] for i in mbs) <= 10


@pytest.mark.unit
def test_dynamic_uneven_balanced_distribution():
    """``--balance_data`` distributes mbs across DP ranks by KK on mbs token sums."""
    total_lengths = [4, 4, 4, 4, 4, 4, 4, 4, 4]
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, step_sample_indices=[list(range(9))]
    )

    assert gbs_per_step == [9]
    for r in range(2):
        assert len(mbi[r]) == sum(nmb)
    seen: set[int] = set()
    for r in range(2):
        seen.update(partitions[r])
    assert seen == set(range(9))
    for r in range(2):
        for mbs in mbi[r]:
            assert sum(total_lengths[partitions[r][i]] for i in mbs) <= 8


@pytest.mark.unit
def test_dynamic_low_K_padded_by_splitting():
    """Few-but-large samples: first-fit produces fewer bins than dp_size, schedule
    must split the largest bins until K reaches dp_size."""
    total_lengths = [3, 3, 3, 3]
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, step_sample_indices=[list(range(4))]
    )

    assert nmb == [1]
    for r in range(4):
        assert len(partitions[r]) >= 1
    seen: set[int] = set()
    for r in range(4):
        seen.update(partitions[r])
    assert seen == set(range(4))


@pytest.mark.unit
def test_near_equal_step_split_even_and_uneven():
    """``near_equal_step_split`` distributes a remainder one-per-step from the front."""
    assert near_equal_step_split(24, num_steps=3) == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]
    assert near_equal_step_split(23, num_steps=3) == [list(range(0, 8)), list(range(8, 16)), list(range(16, 23))]
    assert near_equal_step_split(25, num_steps=3) == [list(range(0, 9)), list(range(9, 17)), list(range(17, 25))]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
    """A custom splitter producing 7 / 8 / 9 sample steps from 24 samples — the
    schedule still gives every DP rank the same num_mbs per step (PP sync) but
    each step's reported gbs is the real per-step sample count."""
    total_lengths = [4] * 24
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=2)

    step_sample_indices = [
        list(range(0, 7)),  # 7 samples
        list(range(7, 15)),  # 8 samples
        list(range(15, 24)),  # 9 samples
    ]
    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, step_sample_indices=step_sample_indices
    )

    assert gbs_per_step == [7, 8, 9]
    # Each rank runs the same total num_mbs.
    for r in range(2):
        assert len(mbi[r]) == sum(nmb)
    # Every sample placed exactly once.
    seen: set[int] = set()
    for r in range(2):
        seen.update(partitions[r])
    assert seen == set(range(24))


@pytest.mark.unit
def test_validate_step_sample_indices_happy_path():
    """Normal usage: complete cover, no overlap → returns normalised list."""
    out = validate_step_sample_indices([range(0, 3), range(3, 8)], num_samples=8)
    assert out == [[0, 1, 2], [3, 4, 5, 6, 7]]


@pytest.mark.unit
def test_validate_step_sample_indices_rejects_empty():
    with pytest.raises(AssertionError, match="empty list of steps"):
        validate_step_sample_indices([], num_samples=8)


@pytest.mark.unit
def test_validate_step_sample_indices_rejects_overlap():
    with pytest.raises(AssertionError, match="reuses sample indices"):
        validate_step_sample_indices([[0, 1, 2], [2, 3]], num_samples=4)


@pytest.mark.unit
def test_validate_step_sample_indices_rejects_out_of_range():
    with pytest.raises(AssertionError, match="out-of-range"):
        validate_step_sample_indices([[0, 1, 9]], num_samples=4)


@pytest.mark.unit
def test_validate_step_sample_indices_allows_dropping_with_warning(caplog):
    """Dropping samples is allowed but should emit a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="slime.utils.dp_schedule"):
        out = validate_step_sample_indices([[0, 1]], num_samples=5)
    assert out == [[0, 1]]
    assert any("dropped 3 sample" in r.message for r in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
