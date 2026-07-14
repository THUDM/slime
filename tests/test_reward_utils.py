import pytest

from slime.rollout.reward_utils import normalize_rewards_by_group

NUM_GPUS = 0


@pytest.mark.unit
def test_normalize_rewards_uses_explicit_uneven_groups():
    rewards = [0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 10.0, 11.0, 12.0, 13.0]
    group_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    normalized = normalize_rewards_by_group(rewards, group_indices, normalize_std=False)

    assert normalized == pytest.approx([-1.5, -0.5, 0.5, 1.5, 0.0, 0.0, 0.0, -1.5, -0.5, 0.5, 1.5])


@pytest.mark.unit
def test_normalize_rewards_preserves_order_and_zeroes_singletons():
    rewards = [-1.0, 4.0, 0.0, 7.0, 1.0, 4.0]
    group_indices = [10, 20, 10, 30, 10, 20]

    normalized = normalize_rewards_by_group(rewards, group_indices, normalize_std=True)

    assert normalized == pytest.approx([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], abs=1e-5)


@pytest.mark.unit
def test_normalize_rewards_requires_group_identity():
    with pytest.raises(ValueError, match="group_index is required.*position 1"):
        normalize_rewards_by_group([1.0, 2.0], [0, None], normalize_std=False)


@pytest.mark.unit
def test_normalize_rewards_supports_legacy_fixed_size_groups():
    normalized = normalize_rewards_by_group(
        [1.0, 3.0, 10.0, 10.0],
        [None, None, None, None],
        normalize_std=False,
        fallback_group_size=2,
    )

    assert normalized == pytest.approx([-1.0, 1.0, 0.0, 0.0])


@pytest.mark.unit
def test_normalize_rewards_rejects_unidentified_uneven_groups():
    with pytest.raises(ValueError, match="group_index is required when reward groups are not uniformly sized"):
        normalize_rewards_by_group(
            [1.0, 2.0, 3.0],
            [None, None, None],
            normalize_std=False,
            fallback_group_size=2,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
