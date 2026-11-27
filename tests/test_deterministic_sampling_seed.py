"""CPU checks for request seeds, independent of a model or SGLang server."""

import ast
import asyncio
import copy
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.rollout.sampling_seed import training_sample_seed

NUM_GPUS = 0


def _group_function(recorded):
    # Execute the production coroutine with only its network/state dependencies
    # replaced. Avoid importing SGLang, tokenizers, or starting a Ray runtime.
    source = Path(__file__).parents[1] / "slime/rollout/sglang_rollout.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate_and_rm_group")
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )

    async def generate(args, sample, sampling_params, evaluation=False):
        await asyncio.sleep(0)
        recorded[sample.index] = sampling_params.copy()
        return sample

    namespace = {
        "asyncio": asyncio,
        "uuid": uuid,
        "logger": logging.getLogger(__name__),
        "training_sample_seed": training_sample_seed,
        "GenerateState": lambda args: SimpleNamespace(
            aborted=False, group_sampling_seeds=[args.rollout_seed + i for i in range(8)]
        ),
        "generate_and_rm": generate,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["generate_and_rm_group"]


def _generate(mode, indices, deterministic=True):
    recorded = {}
    generate_group = _group_function(recorded)
    args = SimpleNamespace(rollout_seed=42, sglang_enable_deterministic_inference=deterministic, group_rm=False)
    if mode is not None:
        args.deterministic_sampling_seed_mode = mode
    params = {"temperature": 1.0}

    async def run():
        for start in indices:
            samples = [SimpleNamespace(index=i, session_id=None) for i in range(start, start + 8)]
            await generate_group(args, samples, params)

    asyncio.run(run())
    assert params == {"temperature": 1.0}
    return recorded


@pytest.mark.parametrize("mode", [None, "group"])
def test_legacy_group_mode_is_preserved(mode):
    rows = _generate(mode, [0, 8, 256])
    for start in (0, 8, 256):
        assert [rows[i]["sampling_seed"] for i in range(start, start + 8)] == list(range(42, 50))


def test_sample_mode_uses_unique_streams_across_prompts_and_updates():
    rows = _generate("sample", range(0, 40 * 256, 8))
    assert len(rows) == 10240
    assert len({p["sampling_seed"] for p in rows.values()}) == len(rows)
    assert all(p["sampling_seed"] == 42 + index for index, p in rows.items())


def test_seed_is_stable_on_retry_reordering_and_restored_index():
    original = _generate("sample", [0, 8, 256])
    assert _generate("sample", [256, 8, 0]) == original
    saved_index = copy.deepcopy(256)
    assert _generate("sample", [saved_index]) == {i: p for i, p in original.items() if i >= saved_index}
    assert _generate("sample", [8, 8]) == {i: p for i, p in original.items() if 8 <= i < 16}


def test_nondeterministic_sampling_is_unchanged():
    assert all("sampling_seed" not in p for p in _generate("sample", [0, 8], deterministic=False).values())


@pytest.mark.parametrize("index", [None, -1, "0", True, 2**31 - 42])
def test_invalid_sample_indices_fail_without_silent_seed_reuse(index):
    with pytest.raises(ValueError):
        training_sample_seed(42, index)


def test_last_supported_seed_does_not_wrap():
    assert training_sample_seed(42, 2**31 - 43) == 2**31 - 1


def _load_dataset_state(state, rollout_id, checkpoint):
    source = Path(__file__).parents[1] / "slime/data/data_source.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RolloutDataSource")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load")
    namespace = {
        "Path": Path,
        "logger": logging.getLogger(__name__),
        "torch": SimpleNamespace(load=lambda path, weights_only=False: checkpoint),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    namespace["load"](state, rollout_id)


@pytest.mark.parametrize("mode", ["group", "sample"])
def test_missing_rollout_state_fails_closed_only_in_sample_mode(tmp_path, mode):
    state = SimpleNamespace(args=SimpleNamespace(
        load=str(tmp_path), rollout_global_dataset=True, deterministic_sampling_seed_mode=mode
    ))
    if mode == "sample":
        with pytest.raises(FileNotFoundError, match="saved rollout dataset state"):
            _load_dataset_state(state, 19, {})
    else:
        _load_dataset_state(state, 19, {})


def test_restored_dataset_index_determines_next_request_seed(tmp_path):
    path = tmp_path / "rollout/global_dataset_state_dict_19.pt"
    path.parent.mkdir()
    path.touch()
    state = SimpleNamespace(
        args=SimpleNamespace(load=str(tmp_path), rollout_global_dataset=True,
                             deterministic_sampling_seed_mode="sample", rollout_shuffle=False),
        metadata={}, dataset=None,
    )
    state.load_state_dict = lambda state_dict: [
        setattr(state, name, value) for name, value in state_dict.items()
    ]
    with pytest.raises(ValueError, match="sample_index"):
        _load_dataset_state(state, 19, {})
    _load_dataset_state(state, 19, {"sample_index": 5120, "sample_group_index": 640})
    assert state.sample_index == 5120
    assert _generate("sample", [state.sample_index])[5120]["sampling_seed"] == 5162
