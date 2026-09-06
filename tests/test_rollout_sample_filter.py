from argparse import Namespace
from pathlib import Path

import pytest

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput, call_rollout_fn
from slime.utils.types import Sample

NUM_GPUS = 0


def _group(start: int) -> list[Sample]:
    return [
        Sample(index=start, prompt=f"p-{start}", response="a", tokens=[1], response_length=1, reward=1.0),
        Sample(index=start + 1, prompt=f"p-{start + 1}", response="b", tokens=[2], response_length=1, reward=0.0),
    ]


def custom_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        sample = _group(0)[0]
        return RolloutFnEvalOutput(
            data={"eval": {"rewards": [sample.reward], "truncated": [False], "samples": [sample]}}
        )
    return RolloutFnTrainOutput(samples=[_group(0), _group(2)])


def drop_last_in_group(args, groups):
    args.filter_calls.append([len(group) for group in groups])
    for group in groups:
        group[-1].remove_sample = True


@pytest.mark.unit
def test_call_rollout_fn_applies_sample_filter_to_custom_rollout_exactly_once(monkeypatch):
    monkeypatch.setattr("slime.rollout.base_types.load_function", lambda path: drop_last_in_group)
    args = Namespace(rollout_sample_filter_path="tests.drop_last_in_group", filter_calls=[])

    output = call_rollout_fn(custom_rollout, args, 1, None, evaluation=False)

    assert args.filter_calls == [[2, 2]]
    assert [sample.remove_sample for group in output.samples for sample in group] == [False, True, False, True]


@pytest.mark.unit
def test_call_rollout_fn_skips_sample_filter_on_eval(monkeypatch):
    monkeypatch.setattr("slime.rollout.base_types.load_function", lambda path: drop_last_in_group)
    args = Namespace(rollout_sample_filter_path="tests.drop_last_in_group", filter_calls=[])

    call_rollout_fn(custom_rollout, args, 1, None, evaluation=True)

    assert args.filter_calls == []


@pytest.mark.unit
def test_call_rollout_fn_skips_sample_filter_when_unset():
    args = Namespace(rollout_sample_filter_path=None, filter_calls=[])

    output = call_rollout_fn(custom_rollout, args, 1, None, evaluation=False)

    assert args.filter_calls == []
    assert [sample.remove_sample for group in output.samples for sample in group] == [False, False, False, False]


@pytest.mark.unit
def test_sglang_async_rollout_does_not_apply_sample_filter():
    source = (Path(__file__).resolve().parents[1] / "slime/rollout/sglang_rollout.py").read_text()
    assert "filter_func = load_function(args.rollout_sample_filter_path)" not in source
    assert "if args.rollout_sample_filter_path is not None:" not in source
