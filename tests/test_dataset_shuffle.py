"""CPU tests for deterministic dataset shuffling without global RNG side effects."""

import random

import pytest

from slime.utils.data import Dataset


NUM_GPUS = 0


def _make_dataset(*, seed=42, size=8):
    dataset = Dataset.__new__(Dataset)
    dataset.origin_samples = list(range(size))
    dataset.samples = dataset.origin_samples
    dataset.epoch_id = -1
    dataset.seed = seed
    return dataset


@pytest.mark.unit
def test_shuffle_preserves_the_seeded_permutation():
    dataset = _make_dataset()
    expected = list(dataset.origin_samples)
    random.Random(dataset.seed).shuffle(expected)

    dataset.shuffle(new_epoch_id=0)

    assert dataset.samples == expected


@pytest.mark.unit
def test_shuffle_does_not_mutate_process_global_rng_state():
    dataset = _make_dataset()
    original_state = random.getstate()
    try:
        random.seed(12345)
        state_before_shuffle = random.getstate()

        dataset.shuffle(new_epoch_id=0)

        assert random.getstate() == state_before_shuffle
    finally:
        random.setstate(original_state)


@pytest.mark.unit
def test_shuffle_is_reproducible_for_each_epoch():
    first = _make_dataset()
    second = _make_dataset()

    first.shuffle(new_epoch_id=3)
    second.shuffle(new_epoch_id=3)

    assert first.samples == second.samples
    assert first.samples != first.origin_samples


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
