"""Unit tests for ``slime.utils.metric_utils.compute_pass_rate``.

Pins the two regimes:

* rigid (legacy fixed-size): ``len(flat_rewards) == num_groups * group_size``.
* ragged (over-sampled): variable per-group sample counts whose total need not
  divide ``group_size`` — the case that crashed the base-slime metric assert
  (e.g. 51 trainable samples, groups of mixed size, not the rigid 8*4=32).
"""

import pytest

from slime.utils.metric_utils import compute_pass_rate


def test_group_size_one_returns_empty():
    assert compute_pass_rate([1, 0, 1], group_size=1) == {}


def test_rigid_layout_pass_at_k():
    # 2 groups x 4 samples; group A has 2 correct, group B has 0 correct.
    flat = [1, 1, 0, 0, 0, 0, 0, 0]
    out = compute_pass_rate(flat, group_size=4, num_groups=2)
    assert set(out) == {"pass@1", "pass@2", "pass@4"}
    # pass@1 = mean correct fraction = (2/4 + 0/4) / 2 = 0.25
    assert out["pass@1"] == pytest.approx(0.25)
    # pass@4 over the whole group: group A always has >=1 correct in a draw of 4
    # (n-c=2 < k=4 -> 1.0), group B never -> mean 0.5.
    assert out["pass@4"] == pytest.approx(0.5)


def test_rigid_layout_matches_legacy_reshape():
    # The group_ids=None path must stay numerically identical to the legacy
    # reshape: full-size groups, every rung eligible, no rung dropped.
    flat = [1, 0, 1, 1, 0, 0, 1, 1]
    out = compute_pass_rate(flat, group_size=4, num_groups=2)
    assert set(out) == {"pass@1", "pass@2", "pass@4"}
    # group A=[1,0,1,1] (3/4 correct), group B=[0,0,1,1] (2/4 correct).
    assert out["pass@1"] == pytest.approx((3 / 4 + 2 / 4) / 2)


def test_rigid_layout_asserts_on_bad_count():
    with pytest.raises(AssertionError):
        compute_pass_rate([1, 0, 1], group_size=4, num_groups=2)


def test_ragged_oversampled_reproduces_crash_and_pins_values():
    # The exact shape that crashed base-slime: 51 samples across ragged groups,
    # total not a multiple of group_size (4). Group sizes: twelve groups
    # summing to 51, every group filled as [1,0,1,0,...].
    group_sizes = [4, 4, 3, 4, 4, 4, 5, 4, 4, 4, 4, 7]
    assert sum(group_sizes) == 51
    flat_rewards = []
    group_ids = []
    for gi, n in enumerate(group_sizes):
        for j in range(n):
            flat_rewards.append(1 if j % 2 == 0 else 0)
            group_ids.append(f"task-{gi}")

    # The rigid path reshapes (num_groups, group_size) and asserts the total
    # divides group_size: 51 != 12*4, so it crashes — this is the bug.
    with pytest.raises(AssertionError):
        compute_pass_rate(flat_rewards, group_size=4, num_groups=12)

    # The ragged path buckets by group id and never asserts. Pin the exact
    # pass@k the fix establishes for this input (not just a 0..1 range).
    out = compute_pass_rate(flat_rewards, group_size=4, group_ids=group_ids)
    assert set(out) == {"pass@1", "pass@2", "pass@4"}
    assert out["pass@1"] == pytest.approx(0.5281746031746032)
    assert out["pass@2"] == pytest.approx(0.8547619047619048)
    # Every group has >= 1 correct in any draw of 4 (n-c < 4 for all), so 1.0.
    assert out["pass@4"] == pytest.approx(1.0)


def test_ragged_per_group_semantics():
    # Two groups: A = [1,1,0] (3 samples, 2 correct), B = [0,0] (2 samples, 0 correct).
    flat = [1, 1, 0, 0, 0]
    gids = ["a", "a", "a", "b", "b"]
    out = compute_pass_rate(flat, group_size=4, group_ids=gids)
    # rungs for group_size=4 -> {1,2,4}; group A has 3 samples, group B has 2.
    # pass@1: mean correct frac = (2/3 + 0/2)/2 = 1/3.
    assert out["pass@1"] == pytest.approx(1 / 3)
    # pass@2: both groups have >=2 samples (eligible).
    #   A: n=3,c=2,k=2 -> n-c=1 < k=2 -> 1.0
    #   B: n=2,c=0,k=2 -> n-c=2 >= k=2 -> 0.0
    #   mean -> 0.5
    assert out["pass@2"] == pytest.approx(0.5)
    # pass@4: only groups with >=4 samples are eligible; neither qualifies -> rung dropped.
    assert "pass@4" not in out


def test_ragged_all_groups_too_small_drops_high_rungs():
    # All groups have a single sample -> only pass@1 survives (pass@2/4 dropped).
    flat = [1, 0, 1]
    gids = ["a", "b", "c"]
    out = compute_pass_rate(flat, group_size=4, group_ids=gids)
    assert "pass@1" in out
    assert "pass@2" not in out
    assert "pass@4" not in out
    assert out["pass@1"] == pytest.approx(2 / 3)


def test_ragged_length_mismatch_asserts():
    with pytest.raises(AssertionError):
        compute_pass_rate([1, 0, 1], group_size=4, group_ids=["a", "b"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
