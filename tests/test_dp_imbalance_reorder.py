"""Regression test for forward_only's handling of empty partitions.

Scenario: with `use_dynamic_batch_size=True`, an all_reduce(MAX) across DP
ranks can produce `num_microbatches` larger than the local rank has real
samples for. `_get_capped_partitions` then yields trailing empty partitions
like `[[0,3], [1,2], [], []]`, and `get_batch`'s placeholder guard emits a
1-token fake sample for each empty partition. Those fakes flow through the
forward pass and into `forward_data_store`.

Before the fix, `forward_only` sized `origin_values` to `len(values)` (which
included the fake outputs), and `zip(values, origin_indices, strict=False)`
only populated the real-sample slots — so the tail of the returned list was
`None`. That caused a downstream crash in
`compute_advantages_and_returns` at
    kl = [torch.zeros_like(x, ..., device=x.device) for x in xs]
with `AttributeError: 'NoneType' object has no attribute 'device'` on the
first `None` element.

Related upstream reports: THUDM/slime#1838 (the `torch.cat([])` flavor of
this same DP-imbalance problem, which the `get_batch` placeholder guard
already fixes) and THUDM/slime#1839 (oversized-sample assertion).

This test simulates the exact reorder logic in `forward_only` to pin the
fix in `origin_values = [None] * len(origin_indices)` (as opposed to
`len(values)`).
"""

from __future__ import annotations


def _reorder(values: list, micro_batch_indices: list[list[int]]) -> list:
    """Mirror the reorder in forward_only model.py, post-fix."""
    origin_indices = sum(micro_batch_indices, [])
    origin_values = [None] * len(origin_indices)
    for value, origin_index in zip(values, origin_indices, strict=False):
        origin_values[origin_index] = value
    return origin_values


def test_no_empty_partitions_preserves_order():
    # 4 real samples, 2 microbatches, no empty partitions.
    mb_indices = [[0, 3], [1, 2]]
    values = ["s0", "s3", "s1", "s2"]
    out = _reorder(values, mb_indices)
    assert out == ["s0", "s1", "s2", "s3"]


def test_trailing_empty_partitions_drop_fakes():
    # 4 real samples split across 2 real microbatches + 2 empty ones
    # (produced by _get_capped_partitions when num_mbs exceeds what the
    # local rank needs). Each empty partition contributes one 1-token
    # placeholder via get_batch, which becomes a "fake" value in `values`.
    mb_indices = [[0, 3], [1, 2], [], []]
    values = ["s0", "s3", "s1", "s2", "FAKE0", "FAKE1"]
    out = _reorder(values, mb_indices)
    # Only real samples at origin positions; fakes are dropped.
    assert out == ["s0", "s1", "s2", "s3"]
    assert None not in out


def test_all_partitions_empty_returns_empty_list():
    # Degenerate rank with 0 real samples (empty micro_batch_indices list).
    # Every microbatch is a fake; output is an empty list, not a list of None.
    mb_indices = [[], [], []]
    values = ["FAKE0", "FAKE1", "FAKE2"]
    out = _reorder(values, mb_indices)
    assert out == []


def test_empty_partitions_only_at_tail():
    # _get_capped_partitions uses first-fit bin-packing: samples always go to
    # the lowest-index partition that fits, so any unused capacity appears in
    # trailing partitions — never interleaved. This documents that
    # invariant; the simpler zip-based reorder relies on it.
    # (If KK ever produced mid-schedule empties, len(partition) > 0 would
    # assert in get_seqlen_balanced_partitions, not reach this code path.)
    mb_indices = [[0, 1, 2], [3], [], []]
    values = ["s0", "s1", "s2", "s3", "FAKE0", "FAKE1"]
    out = _reorder(values, mb_indices)
    assert out == ["s0", "s1", "s2", "s3"]
    assert None not in out


def test_fix_vs_pre_fix_behavior_divergence():
    """Sanity check that the OLD reorder (sizing by len(values)) produced
    trailing Nones. Documents the specific bug shape."""
    mb_indices = [[0, 3], [1, 2], [], []]
    values = ["s0", "s3", "s1", "s2", "FAKE0", "FAKE1"]

    # Pre-fix (bug): origin_values sized by len(values) → 2 trailing Nones
    pre_fix = [None] * len(values)
    origin_indices = sum(mb_indices, [])
    for value, origin_index in zip(values, origin_indices, strict=False):
        pre_fix[origin_index] = value
    assert pre_fix[-2:] == [None, None], "pre-fix list should end with Nones"

    # Post-fix: origin_values sized by len(origin_indices) → no trailing Nones
    post_fix = _reorder(values, mb_indices)
    assert None not in post_fix
