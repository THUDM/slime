from types import SimpleNamespace

import pytest

from slime.rollout.data_source import RolloutDataSource
from slime.utils.types import Sample

NUM_GPUS = 0


class _Dataset:
    def __init__(self, prompts):
        self.samples = [Sample(prompt=prompt) for prompt in prompts]
        self.shuffle_epochs = []

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch_id):
        self.shuffle_epochs.append(epoch_id)
        self.samples.reverse()


def _source(prompts, *, sample_offset=0, rollout_shuffle=False):
    source = object.__new__(RolloutDataSource)
    source.args = SimpleNamespace(n_samples_per_prompt=1, rollout_shuffle=rollout_shuffle)
    source.dataset = _Dataset(prompts)
    source.sample_offset = sample_offset
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    return source


def _prompts(groups):
    return [group[0].prompt for group in groups]


def test_get_samples_wraps_across_every_required_epoch():
    source = _source(["a", "b"])

    groups = source.get_samples(7)

    assert _prompts(groups) == ["a", "b", "a", "b", "a", "b", "a"]
    assert source.epoch_id == 3
    assert source.sample_offset == 1
    assert [group[0].index for group in groups] == list(range(7))
    assert [group[0].group_index for group in groups] == list(range(7))


def test_get_samples_wraps_from_nonzero_offset_and_preserves_boundary_position():
    source = _source(["a", "b", "c"], sample_offset=2)

    assert _prompts(source.get_samples(7)) == ["c", "a", "b", "c", "a", "b", "c"]
    assert source.epoch_id == 2
    assert source.sample_offset == 3

    assert _prompts(source.get_samples(1)) == ["a"]
    assert source.epoch_id == 3
    assert source.sample_offset == 1


def test_get_samples_shuffles_at_each_crossed_epoch():
    source = _source(["a", "b"], rollout_shuffle=True)

    assert _prompts(source.get_samples(6)) == ["a", "b", "b", "a", "a", "b"]
    assert source.dataset.shuffle_epochs == [1, 2]
    assert source.epoch_id == 2
    assert source.sample_offset == 2


def test_get_samples_rejects_nonempty_request_from_empty_dataset():
    source = _source([])

    with pytest.raises(RuntimeError, match="empty dataset"):
        source.get_samples(1)


def test_get_samples_rejects_an_invalid_dataset_offset():
    source = _source(["a", "b"], sample_offset=3)

    with pytest.raises(RuntimeError, match="invalid dataset offset"):
        source.get_samples(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
