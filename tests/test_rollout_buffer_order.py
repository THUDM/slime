from types import SimpleNamespace

import pytest

from slime.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer, pop_first, pop_oldest
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.unit
def test_buffer_resumes_oldest_group_first_and_preserves_ties():
    groups = [
        [Sample(index=0), Sample(index=1)],
        [Sample(index=2, weight_versions=["10"]), Sample(index=3, weight_versions=["12"])],
        [Sample(index=4, weight_versions=["12"]), Sample(index=5, weight_versions=["2", "11"])],
        [Sample(index=6, weight_versions=["2"]), Sample(index=7)],
        [Sample(index=8, weight_versions=["default"]), Sample(index=9)],
    ]
    buffer = groups.copy()

    assert pop_oldest(None, None, buffer, 2) == [groups[2], groups[3]]
    assert buffer == [groups[1], groups[0], groups[4]]
    assert pop_oldest(None, None, buffer, 10) == [groups[1], groups[0], groups[4]]
    assert buffer == []


@pytest.mark.unit
@pytest.mark.parametrize("sort_by_staleness", [None, False, True])
def test_buffer_sort_is_opt_in_and_explicit_filter_takes_precedence(monkeypatch, sort_by_staleness):
    monkeypatch.setattr(RolloutDataSource, "__init__", lambda self, args: setattr(self, "args", args))
    fresh = [Sample(index=0)]
    old = [Sample(index=1, weight_versions=["2"])]
    args = SimpleNamespace(buffer_filter_path=None, n_samples_per_prompt=1)
    if sort_by_staleness is not None:
        args.buffer_sort_by_staleness = sort_by_staleness

    source = RolloutDataSourceWithBuffer(args)
    source.add_samples([fresh, old])
    expected = [old, fresh] if sort_by_staleness else [fresh, old]
    assert source.get_samples(1) == [expected[0]]
    assert source.get_samples(1) == [expected[1]]

    args.buffer_filter_path = "slime.rollout.data_source.pop_first"
    source = RolloutDataSourceWithBuffer(args)
    assert source.buffer_filter is pop_first


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
