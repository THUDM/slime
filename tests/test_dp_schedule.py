"""CPU unit tests for slime.utils.dp_schedule.build_dp_rollout_data.

The tests assert the invariants documented at the top of dp_schedule.py against a
range of static / dynamic / VPP / oversize / balance scenarios.

Trim and dynamic-gbs resolution live in the caller (``RolloutManager``), so these
tests just hand ``build_dp_rollout_data`` a ``data`` dict whose ``tokens`` length
is already a multiple of the supplied ``global_batch_size``.
"""

from types import SimpleNamespace

import pytest

from slime.utils.dp_schedule import build_dp_rollout_data, compute_dynamic_global_batch_size


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


def make_data(lengths):
    """``data["tokens"][i]`` is a list of ``lengths[i]`` placeholder token ids."""
    return {"tokens": [list(range(L)) for L in lengths]}


def assert_invariants(rollout_data_list, *, dp_size, total_lengths, max_per_bin=None):
    """Check the invariants documented at the top of dp_schedule.py."""
    # All ranks see the same num_microbatches list (PP-sync requirement).
    nmb = rollout_data_list[0]["num_microbatches"]
    for r in range(1, dp_size):
        assert rollout_data_list[r]["num_microbatches"] == nmb, "num_microbatches diverged across ranks"

    expected_per_rank = len(total_lengths) // dp_size
    seen_global: set[int] = set()
    for r in range(dp_size):
        rd = rollout_data_list[r]
        partition = rd["partition"]

        # Same sample count per rank.
        assert len(partition) == expected_per_rank, f"rank {r}: {len(partition)} samples, want {expected_per_rank}"

        # Each rank has sum(nmb) micro-batches.
        assert len(rd["micro_batch_indices"]) == sum(nmb), f"rank {r}: mbs count mismatch"

        # Flattened micro_batch_indices == range(len(partition)) (each sample covered once
        # by exactly one mbs).
        flat = [i for mbs in rd["micro_batch_indices"] for i in mbs]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(range(len(total_lengths))), "some samples not assigned to any rank"

    if max_per_bin is None:
        return

    # Every mbs <= max_per_bin tokens, EXCEPT a singleton bin holding an oversized sample.
    for r in range(dp_size):
        rd = rollout_data_list[r]
        partition = rd["partition"]
        for mbs in rd["micro_batch_indices"]:
            bin_total = sum(total_lengths[partition[i]] for i in mbs)
            if bin_total > max_per_bin:
                assert len(mbs) == 1, f"rank {r}: mbs sum {bin_total} > {max_per_bin} but contains {len(mbs)} samples"


@pytest.mark.unit
def test_static_stride_single_step():
    """Static + strided DP split, single step."""
    lengths = [10] * 16
    data = make_data(lengths)
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=16)

    assert len(out) == 4
    assert out[0]["num_microbatches"] == [2]
    assert_invariants(out, dp_size=4, total_lengths=lengths)


@pytest.mark.unit
def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps. Each rank must get gbs/dp per step."""
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]  # 2 steps of 8
    data = make_data(lengths)
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=8)

    assert out[0]["num_microbatches"] == [2, 2]
    assert_invariants(out, dp_size=2, total_lengths=lengths)


@pytest.mark.unit
def test_dynamic_uniform():
    """Dynamic mbs on uniform-length samples."""
    lengths = [5] * 8
    data = make_data(lengths)
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=8)

    assert_invariants(out, dp_size=2, total_lengths=lengths, max_per_bin=10)


@pytest.mark.unit
def test_dynamic_skewed_lengths():
    """Skewed lengths (the case where K-K used to over-pack a single bin)."""
    lengths = [9, 9, 9, 9, 1, 1, 1, 1]
    data = make_data(lengths)
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=8)

    assert_invariants(out, dp_size=2, total_lengths=lengths, max_per_bin=10)


@pytest.mark.unit
def test_dynamic_oversized_sample_lands_alone():
    """A single sample exceeding max_per_bin must end up alone in its mbs (with no
    other samples crammed in)."""
    lengths = [15, 3, 3, 3, 3, 3, 3, 3]  # 15 > C=10
    data = make_data(lengths)
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=8)

    assert_invariants(out, dp_size=2, total_lengths=lengths, max_per_bin=10)
    # Find the rank holding the oversized sample and verify it lives alone in some mbs.
    oversize_idx = lengths.index(15)
    found = False
    for r in range(2):
        partition = out[r]["partition"]
        if oversize_idx not in partition:
            continue
        local = partition.index(oversize_idx)
        for mbs in out[r]["micro_batch_indices"]:
            if local in mbs:
                assert mbs == [local], f"oversized sample shares an mbs: {mbs}"
                found = True
    assert found


@pytest.mark.unit
def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    lengths = [4] * 32  # 2 steps of 16; per step, ~8 bins of 8 needed at C=8
    data = make_data(lengths)
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=16)

    nmb = out[0]["num_microbatches"]
    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(out, dp_size=2, total_lengths=lengths, max_per_bin=8)


@pytest.mark.unit
def test_dynamic_global_batch_size_stamped_on_rollout_data():
    """When dynamic_global_batch_size is passed it should appear on every per-rank dict."""
    lengths = [4] * 8
    data = make_data(lengths)
    args = make_args(micro_batch_size=1)
    tp = make_tp(dp_size=4)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=8, dynamic_global_batch_size=8)

    assert out[0]["num_microbatches"] == [2]
    for r in range(4):
        assert out[r]["dynamic_global_batch_size"] == 8


@pytest.mark.unit
def test_compute_dynamic_global_batch_size_floor_and_min():
    """Verify both the floor-to-dp-multiple and the dp_size floor."""
    assert compute_dynamic_global_batch_size(10, 4) == 8
    assert compute_dynamic_global_batch_size(16, 4) == 16
    # Fewer samples than dp_size: clamp to dp_size so we still produce a valid mbs.
    assert compute_dynamic_global_batch_size(2, 4) == 4


@pytest.mark.unit
def test_sample_aligned_fields_are_sliced_by_partition():
    """``rewards`` (sample-aligned) should be sliced per rank; ``raw_reward``
    (passthrough) should appear verbatim."""
    lengths = [3] * 8
    data = make_data(lengths)
    data["rewards"] = list(range(100, 108))  # one per sample
    data["raw_reward"] = "passthrough"  # whole-batch field
    args = make_args(micro_batch_size=1)
    tp = make_tp(dp_size=2)

    out = build_dp_rollout_data(args, tp, data, global_batch_size=8)

    for r in range(2):
        partition = out[r]["partition"]
        assert out[r]["rewards"] == [data["rewards"][j] for j in partition]
        assert out[r]["raw_reward"] == "passthrough"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
