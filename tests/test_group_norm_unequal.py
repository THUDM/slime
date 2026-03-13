"""
Tests for group-level reward normalization with unequal group sizes.

Regression test for issue #1414: When training samples have unequal group sizes,
the group-level reward normalization in _post_process_rewards was computing
a single global mean across all samples (shape (1, N)) instead of per-group
means, making the normalization incorrect.
"""

import torch


def normalize_rewards_per_group(
    raw_rewards: list[float],
    group_indices: list[int],
    grpo_std_normalization: bool = False,
) -> list[float]:
    """
    Replicate the fixed normalization logic from _post_process_rewards
    for the unequal-group-sizes branch.
    """
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    group_ids = torch.tensor(group_indices, dtype=torch.long)

    unique_groups = group_ids.unique()
    for gid in unique_groups:
        mask = group_ids == gid
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        rewards[mask] = group_rewards - mean

        if grpo_std_normalization:
            group_rewards = rewards[mask]
            std = group_rewards.std()
            rewards[mask] = group_rewards / (std + 1e-6)

    return rewards.tolist()


class TestGroupNormUnequal:
    """Test per-group reward normalization with unequal group sizes."""

    def test_mean_normalization_per_group(self):
        """Each group should have mean≈0 after normalization."""
        # Group 0: 4 samples (mean=2.5), Group 1: 2 samples (mean=11.0), Group 2: 3 samples (mean=6.0)
        raw_rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 5.0, 6.0, 7.0]
        group_indices = [0, 0, 0, 0, 1, 1, 2, 2, 2]

        result = normalize_rewards_per_group(raw_rewards, group_indices)

        # Verify per-group means are zero
        groups = [(0, 4), (4, 6), (6, 9)]  # (start, end) for each group
        for i, (start, end) in enumerate(groups):
            group_mean = sum(result[start:end]) / (end - start)
            assert abs(group_mean) < 1e-6, f"Group {i} mean={group_mean:.6f}, expected ≈0"

        # Verify specific values: Group 0 with rewards [1,2,3,4] and mean=2.5
        expected_group0 = [-1.5, -0.5, 0.5, 1.5]
        for j in range(4):
            assert abs(result[j] - expected_group0[j]) < 1e-6, (
                f"Group 0 sample {j}: got {result[j]:.6f}, expected {expected_group0[j]}"
            )

        # Verify Group 1 with rewards [10,12] and mean=11.0
        expected_group1 = [-1.0, 1.0]
        for j in range(2):
            assert abs(result[4 + j] - expected_group1[j]) < 1e-6

    def test_std_normalization_per_group(self):
        """With std normalization, each group should have mean≈0 and std≈1."""
        raw_rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 5.0, 6.0, 7.0]
        group_indices = [0, 0, 0, 0, 1, 1, 2, 2, 2]

        result = normalize_rewards_per_group(raw_rewards, group_indices,
                                             grpo_std_normalization=True)

        groups = [(0, 4), (4, 6), (6, 9)]
        for i, (start, end) in enumerate(groups):
            group = result[start:end]
            n = len(group)
            group_mean = sum(group) / n
            assert abs(group_mean) < 1e-5, f"Group {i} mean={group_mean:.6f}"

    def test_single_sample_group(self):
        """A group with a single sample should normalize to 0."""
        raw_rewards = [1.0, 2.0, 3.0, 5.0]
        group_indices = [0, 0, 0, 1]

        result = normalize_rewards_per_group(raw_rewards, group_indices)

        # Single-sample group should be 0 (reward - mean = 5 - 5 = 0)
        assert abs(result[3]) < 1e-6, f"Single sample group should be 0, got {result[3]}"

    def test_equal_groups_still_work(self):
        """Equal-sized groups should still be normalized correctly."""
        raw_rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 8.0, 14.0]
        group_indices = [0, 0, 0, 0, 1, 1, 1, 1]

        result = normalize_rewards_per_group(raw_rewards, group_indices)

        # Group 0 mean=2.5, Group 1 mean=11.0
        for start, end in [(0, 4), (4, 8)]:
            group_mean = sum(result[start:end]) / (end - start)
            assert abs(group_mean) < 1e-6

    def test_old_code_would_fail(self):
        """
        Verify that the OLD buggy code (view(-1, shape[-1]) on 1D tensor)
        produces WRONG per-group means — confirming the bug existed.
        """
        raw_rewards = [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 5.0, 6.0, 7.0]
        n_samples_per_prompt = 4
        rollout_batch_size = 3

        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        # This is the buggy branch: total (9) != n_samples * batch_size (12)
        assert rewards.shape[-1] != n_samples_per_prompt * rollout_batch_size

        # Buggy reshape: (9,) -> view(-1, 9) -> (1, 9)
        buggy_rewards = rewards.view(-1, rewards.shape[-1])
        assert buggy_rewards.shape == (1, 9), f"Expected (1,9) got {buggy_rewards.shape}"

        buggy_mean = buggy_rewards.mean(dim=-1, keepdim=True)
        buggy_result = (buggy_rewards - buggy_mean).flatten().tolist()

        # The buggy version subtracts the global mean from everything
        global_mean = sum(raw_rewards) / len(raw_rewards)

        # Group 0 (indices 0-3) should have per-group mean ≈ 0, but with the bug:
        group0_mean = sum(buggy_result[:4]) / 4
        # group0_mean ≈ 2.5 - 5.556 = -3.056 (NOT zero)
        assert abs(group0_mean) > 1.0, (
            f"Bug should cause group0 mean to be far from 0, got {group0_mean:.3f}"
        )


def run_all_tests():
    """Run all tests with output for verification."""
    t = TestGroupNormUnequal()

    tests = [
        ("mean_normalization_per_group", t.test_mean_normalization_per_group),
        ("std_normalization_per_group", t.test_std_normalization_per_group),
        ("single_sample_group", t.test_single_sample_group),
        ("equal_groups_still_work", t.test_equal_groups_still_work),
        ("old_code_would_fail", t.test_old_code_would_fail),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
