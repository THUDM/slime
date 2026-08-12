"""CPU unit tests for ``slime.utils.eval_config.build_eval_dataset_configs``.

The documented contract (examples/eval_multi_task/README.md) is that
``eval.defaults`` "defines inference parameters shared by every dataset entry.
Override them inside an individual dataset block if needed." — i.e. resolution
is dataset entry > defaults > args, for every ``EvalDatasetConfig`` field.

Historically only the fields listed in the two spec tables flowed through
``defaults``; everything else (``rm_type``, ``repetition_penalty``,
``app_service``, ...) was silently dropped, and a typo'd key in ``defaults``
was silently accepted while the same typo in a dataset entry raised. These
tests pin the full contract, including the ``stop`` / ``stop_token_ids`` /
``min_new_tokens`` fields that lost their resolution in the #1005 refactor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime.utils.eval_config import build_eval_dataset_configs


NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        rollout_temperature=0.8,
        rollout_top_p=1.0,
        rollout_stop=["</train_stop>"],
        rollout_stop_token_ids=[7],
        eval_min_new_tokens=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_non_spec_defaults_reach_every_dataset():
    datasets = build_eval_dataset_configs(
        _args(),
        [{"name": "aime", "path": "/d/aime.jsonl"}, {"name": "gpqa", "path": "/d/gpqa.jsonl"}],
        defaults={"rm_type": "deepscaler", "repetition_penalty": 1.05, "eval_task_timeout": 120},
    )

    for dataset in datasets:
        assert dataset.rm_type == "deepscaler"
        assert dataset.repetition_penalty == 1.05
        assert dataset.eval_task_timeout == 120


@pytest.mark.unit
def test_dataset_entry_overrides_default():
    datasets = build_eval_dataset_configs(
        _args(),
        [
            {"name": "aime", "path": "/d/aime.jsonl", "rm_type": "math", "temperature": 0.2},
            {"name": "gpqa", "path": "/d/gpqa.jsonl"},
        ],
        defaults={"rm_type": "deepscaler", "temperature": 0.7},
    )

    assert datasets[0].rm_type == "math"
    assert datasets[0].temperature == 0.2
    assert datasets[1].rm_type == "deepscaler"
    assert datasets[1].temperature == 0.7


@pytest.mark.unit
def test_stop_fields_resolve_dataset_then_default_then_args():
    datasets = build_eval_dataset_configs(
        _args(eval_min_new_tokens=4),
        [
            {"name": "a", "path": "/d/a.jsonl", "stop": ["</answer>"], "min_new_tokens": 8},
            {"name": "b", "path": "/d/b.jsonl"},
            {"name": "c", "path": "/d/c.jsonl"},
        ],
        defaults={"stop_token_ids": [11, 12]},
    )

    # dataset entry wins
    assert datasets[0].stop == ["</answer>"]
    assert datasets[0].min_new_tokens == 8
    # eval.defaults fills in
    assert datasets[1].stop_token_ids == [11, 12]
    # args are the last fallback
    assert datasets[1].stop == ["</train_stop>"]
    assert datasets[2].stop_token_ids == [11, 12]
    assert datasets[2].min_new_tokens == 4


@pytest.mark.unit
def test_unknown_default_key_raises():
    with pytest.raises(ValueError, match="temperture"):
        build_eval_dataset_configs(
            _args(),
            [{"name": "aime", "path": "/d/aime.jsonl"}],
            defaults={"temperture": 0.7},
        )


@pytest.mark.unit
def test_spec_fields_still_fall_back_to_args():
    datasets = build_eval_dataset_configs(
        _args(rollout_temperature=0.9),
        [{"name": "aime", "path": "/d/aime.jsonl"}],
        defaults={},
    )

    assert datasets[0].temperature == 0.9
