"""CPU tests for pass@k grouping by ``group_index``.

``compute_pass_rate`` stays rectangular. Call sites that have a
``group_index`` per sample bucket rewards, keep groups whose size is
exactly ``group_size``, and feed that rectangle to the existing helper.
"""

import pytest

from slime.utils.metric_utils import compute_pass_rate, compute_pass_rate_by_group_index

NUM_GPUS = 0


@pytest.mark.unit
def test_group_size_one_returns_empty():
    assert compute_pass_rate([1, 0, 1], group_size=1) == {}
    assert compute_pass_rate_by_group_index([1, 0, 1], [0, 1, 2], group_size=1) == {}


@pytest.mark.unit
def test_rigid_layout_pass_at_k():
    flat = [1, 1, 0, 0, 0, 0, 0, 0]
    out = compute_pass_rate(flat, group_size=4, num_groups=2)
    assert set(out) == {"pass@1", "pass@2", "pass@4"}
    assert out["pass@1"] == pytest.approx(0.25)
    assert out["pass@4"] == pytest.approx(0.5)


@pytest.mark.unit
def test_complete_groups_match_rectangular_compute_pass_rate():
    flat = [1, 0, 1, 1, 0, 0, 1, 1]
    group_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    grouped = compute_pass_rate_by_group_index(flat, group_ids, group_size=4)
    rectangular = compute_pass_rate(flat, group_size=4, num_groups=2)
    assert grouped == rectangular
    assert grouped["pass@1"] == pytest.approx((3 / 4 + 2 / 4) / 2)


@pytest.mark.unit
def test_noncontiguous_group_ids_still_form_complete_groups():
    flat = [1, 0, 0, 1, 1, 0, 1, 1]
    group_ids = [0, 1, 0, 1, 0, 1, 0, 1]
    out = compute_pass_rate_by_group_index(flat, group_ids, group_size=4)
    # group 0 = [1,0,1,1] (3/4), group 1 = [0,1,0,1] (2/4)
    assert out["pass@1"] == pytest.approx((3 / 4 + 2 / 4) / 2)


@pytest.mark.unit
def test_rectangular_compute_pass_rate_still_asserts_on_bad_count():
    with pytest.raises(AssertionError):
        compute_pass_rate([1, 0, 1], group_size=4, num_groups=2)


@pytest.mark.unit
def test_oversampled_batch_excludes_incomplete_groups_instead_of_asserting():
    group_sizes = [4, 4, 3, 4, 4, 4, 5, 4, 4, 4, 4, 7]
    assert sum(group_sizes) == 51
    flat_rewards = []
    group_ids = []
    for gi, n in enumerate(group_sizes):
        for j in range(n):
            flat_rewards.append(1 if j % 2 == 0 else 0)
            group_ids.append(f"task-{gi}")

    with pytest.raises(AssertionError):
        compute_pass_rate(flat_rewards, group_size=4, num_groups=12)

    complete_flat = []
    n_complete = 0
    for gi, n in enumerate(group_sizes):
        if n != 4:
            continue
        n_complete += 1
        start = sum(group_sizes[:gi])
        complete_flat.extend(flat_rewards[start : start + 4])

    out = compute_pass_rate_by_group_index(flat_rewards, group_ids, group_size=4)
    expected = compute_pass_rate(complete_flat, group_size=4, num_groups=n_complete)
    assert out == expected
    assert n_complete == 9


@pytest.mark.unit
def test_excludes_incomplete_groups():
    # Two complete groups of 4 plus a leftover pair: only the complete groups count.
    flat = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    gids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    out = compute_pass_rate_by_group_index(flat, gids, group_size=4)
    expected = compute_pass_rate([1, 1, 1, 1, 0, 0, 0, 0], group_size=4, num_groups=2)
    assert out == expected
    assert out["pass@1"] == pytest.approx(0.5)


@pytest.mark.unit
def test_all_groups_incomplete_returns_empty():
    out = compute_pass_rate_by_group_index([1, 0, 1], ["a", "b", "c"], group_size=4)
    assert out == {}


@pytest.mark.unit
def test_length_mismatch_asserts():
    with pytest.raises(AssertionError):
        compute_pass_rate_by_group_index([1, 0, 1], ["a", "b"], group_size=4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
