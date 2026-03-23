from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass

import pytest

try:
    from ._shared import contract_env_name, get_contract_path, install_paths, install_stubs, run_contract_test_for_file
except ImportError:
    try:
        from plugin_contracts._shared import (
            contract_env_name,
            get_contract_path,
            install_paths,
            install_stubs,
            run_contract_test_for_file,
        )
    except ImportError:
        from _shared import (
            contract_env_name,
            get_contract_path,
            install_paths,
            install_stubs,
            run_contract_test_for_file,
        )

install_paths()
install_stubs(with_sglang_router=True, with_transformers=True)

NUM_GPUS = 0

from slime.rollout.base_types import RolloutFnEvalOutput, call_rollout_fn
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.rollout.filter_hub.base_types import DynamicFilterOutput, call_dynamic_filter
from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout
from slime.utils.misc import load_function
from slime.utils.types import Sample


def run_contract_test_file() -> None:
    def _extra(parsed):
        if parsed.group_rm:
            os.environ[contract_env_name("GROUP_RM")] = "1"

    run_contract_test_for_file(
        __file__,
        path_args=[
            "eval-function-path",
            "custom-rm-path",
            "dynamic-sampling-filter-path",
            "buffer-filter-path",
            "data-source-path",
            "rollout-sample-filter-path",
            "rollout-all-samples-process-path",
        ],
        extra_args=[("--group-rm", {"action": "store_true", "default": False})],
        extra_setup=_extra,
    )


def make_sample(index: int, reward: float = 1.0) -> Sample:
    return Sample(
        index=index,
        prompt=f"prompt-{index}",
        response=f"response-{index}",
        label=f"label-{index}",
        tokens=[100 + index, 200 + index],
        response_length=2,
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={},
    )


def make_args(**overrides):
    class Args:
        rollout_global_dataset = False
        buffer_filter_path = None
        n_samples_per_prompt = 2
        custom_rm_path = None
        group_rm = False
        rm_type = None
        reward_key = None

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class ReferenceDataSource:
    def __init__(self, args):
        self.args = args
        self._groups = [[Sample(index=0), Sample(index=1)], [Sample(index=2), Sample(index=3)]]

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        selected = self._groups[:num_samples]
        self._groups = self._groups[num_samples:]
        return selected

    def add_samples(self, samples: list[list[Sample]]):
        self._groups.extend(samples)

    def save(self, rollout_id):
        self.last_saved_rollout_id = rollout_id

    def load(self, rollout_id=None):
        self.last_loaded_rollout_id = rollout_id

    def __len__(self) -> int:
        return len(self._groups)


def reference_dynamic_filter(args, samples: list[Sample], **kwargs):
    keep = not any(sample.metadata.get("drop") for sample in samples)
    return DynamicFilterOutput(keep=keep, reason=None if keep else "drop-flag")


def reference_buffer_filter(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    selected = list(reversed(buffer[-num_samples:]))
    del buffer[-num_samples:]
    return selected


def reference_rollout_sample_filter(args, groups: list[list[Sample]]) -> None:
    for group in groups:
        if group:
            group[-1].remove_sample = True


def reference_rollout_all_samples_process(args, all_groups: list[list[Sample]], data_source) -> None:
    args.processed_group_count = len(all_groups)


async def reference_single_rm(args, sample: Sample, **kwargs):
    return float(sample.index or 0) + 0.1


async def reference_batched_rm(args, samples: list[Sample], **kwargs):
    return [float(sample.index or 0) + 0.2 for sample in samples]


def valid_eval_function(args, rollout_id, data_source, evaluation=False):
    assert evaluation is True
    sample = make_sample(rollout_id, reward=0.5)
    return RolloutFnEvalOutput(
        data={"eval_contract": {"rewards": [sample.reward], "truncated": [False], "samples": [sample]}},
        metrics={"source": "contract"},
    )


class ContractEvalDataSource:
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        return [[Sample(index=index, prompt=f"prompt-{index}")] for index in range(num_samples)]


@dataclass(frozen=True)
class SyncCase:
    name: str
    env_key: str
    default_path: str
    default_check: object
    path_check: object


def check_eval_function_default() -> None:
    default_sig = inspect.signature(default_generate_rollout)
    assert tuple(default_sig.parameters) == ("args", "rollout_id", "data_source", "evaluation")
    assert default_sig.parameters["evaluation"].default is False


def check_eval_function_path(path: str) -> None:
    fn = load_function(path)
    default_sig = inspect.signature(default_generate_rollout)
    candidate_sig = inspect.signature(fn)
    assert tuple(candidate_sig.parameters) == tuple(default_sig.parameters)
    if path != "slime.rollout.sglang_rollout.generate_rollout":
        output = call_rollout_fn(fn, None, 5, ContractEvalDataSource(), evaluation=True)
        assert isinstance(output, RolloutFnEvalOutput)
        assert output.data


def check_dynamic_filter_default() -> None:
    fn = load_function("slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std")
    assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "samples")
    output = call_dynamic_filter(fn, make_args(), [make_sample(0, reward=1.0), make_sample(1, reward=2.0)])
    assert isinstance(output, DynamicFilterOutput)


def check_dynamic_filter_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "samples")
    output = call_dynamic_filter(fn, make_args(), [make_sample(0, reward=1.0), make_sample(1, reward=2.0)])
    assert isinstance(output, DynamicFilterOutput)


def check_buffer_filter_default() -> None:
    fn = load_function("slime.rollout.data_source.pop_first")
    assert tuple(inspect.signature(fn).parameters)[:4] == ("args", "rollout_id", "buffer", "num_samples")


def check_buffer_filter_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:4] == ("args", "rollout_id", "buffer", "num_samples")
    data_source = RolloutDataSourceWithBuffer(make_args())
    data_source.add_samples([[Sample(index=0), Sample(index=1)], [Sample(index=2), Sample(index=3)]])
    assert isinstance(fn(make_args(buffer_filter_path=path), None, data_source.buffer, 1), list)


def check_data_source_default() -> None:
    cls = load_function("slime.rollout.data_source.RolloutDataSourceWithBuffer")
    assert tuple(inspect.signature(cls.__init__).parameters)[:2] == ("self", "args")


def check_data_source_path(path: str) -> None:
    cls = load_function(path)
    assert tuple(inspect.signature(cls.__init__).parameters)[:2] == ("self", "args")
    groups = cls(make_args()).get_samples(1)
    assert isinstance(groups, list)


def check_rollout_sample_filter_default() -> None:
    assert tuple(inspect.signature(reference_rollout_sample_filter).parameters)[:2] == ("args", "groups")


def check_rollout_sample_filter_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "groups")
    groups = [[Sample(index=0), Sample(index=1)], [Sample(index=2), Sample(index=3)]]
    fn(object(), groups)
    assert any(sample.remove_sample for group in groups for sample in group)


def check_rollout_all_samples_process_default() -> None:
    assert tuple(inspect.signature(reference_rollout_all_samples_process).parameters)[:3] == (
        "args",
        "all_groups",
        "data_source",
    )


def check_rollout_all_samples_process_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:3] == ("args", "all_groups", "data_source")
    args = type("Args", (), {})()
    fn(args, [[Sample(index=0)]], object())
    assert hasattr(args, "processed_group_count")


SYNC_CASES = [
    SyncCase(
        "eval_function",
        "EVAL_FUNCTION_PATH",
        "slime.rollout.sglang_rollout.generate_rollout",
        check_eval_function_default,
        check_eval_function_path,
    ),
    SyncCase(
        "dynamic_filter",
        "DYNAMIC_SAMPLING_FILTER_PATH",
        "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        check_dynamic_filter_default,
        check_dynamic_filter_path,
    ),
    SyncCase(
        "buffer_filter",
        "BUFFER_FILTER_PATH",
        "slime.rollout.data_source.pop_first",
        check_buffer_filter_default,
        check_buffer_filter_path,
    ),
    SyncCase(
        "data_source",
        "DATA_SOURCE_PATH",
        "plugin_contracts.test_plugin_path_loading_contracts.ReferenceDataSource",
        check_data_source_default,
        check_data_source_path,
    ),
    SyncCase(
        "rollout_sample_filter",
        "ROLLOUT_SAMPLE_FILTER_PATH",
        "plugin_contracts.test_plugin_path_loading_contracts.reference_rollout_sample_filter",
        check_rollout_sample_filter_default,
        check_rollout_sample_filter_path,
    ),
    SyncCase(
        "rollout_all_samples_process",
        "ROLLOUT_ALL_SAMPLES_PROCESS_PATH",
        "plugin_contracts.test_plugin_path_loading_contracts.reference_rollout_all_samples_process",
        check_rollout_all_samples_process_default,
        check_rollout_all_samples_process_path,
    ),
]


@pytest.mark.parametrize("case", SYNC_CASES, ids=[case.name for case in SYNC_CASES])
def test_path_loading_default_behavior_is_stable(case: SyncCase):
    case.default_check()


@pytest.mark.parametrize("case", SYNC_CASES, ids=[case.name for case in SYNC_CASES])
def test_path_loading_path_aligns_with_expected_format(case: SyncCase):
    case.path_check(get_contract_path(case.env_key, case.default_path))


def test_custom_rm_default_behavior_is_stable():
    reward = asyncio.run(async_rm(make_args(rm_type="random"), make_sample(4)))
    rewards = asyncio.run(
        batched_async_rm(make_args(group_rm=True, rm_type="random"), [make_sample(1), make_sample(2)])
    )
    assert isinstance(reward, (int, float))
    assert isinstance(rewards, list) and len(rewards) == 2


def test_custom_rm_path_aligns_with_expected_format():
    path = get_contract_path("CUSTOM_RM_PATH")
    if get_contract_path("GROUP_RM") == "1":
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "samples")
        rewards = asyncio.run(
            batched_async_rm(
                make_args(
                    group_rm=True,
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm",
                ),
                [make_sample(0), make_sample(1)],
            )
        )
        assert isinstance(rewards, list) and len(rewards) == 2
    else:
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "sample")
        reward = asyncio.run(
            async_rm(
                make_args(
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm"
                ),
                make_sample(3),
            )
        )
        assert isinstance(reward, (int, float))


if __name__ == "__main__":
    run_contract_test_file()
