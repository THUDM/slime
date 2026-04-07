from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

try:
    from ._shared import get_contract_path, install_paths, install_stubs, run_contract_test_for_file
except ImportError:
    try:
        from plugin_contracts._shared import (
            get_contract_path,
            install_paths,
            install_stubs,
            run_contract_test_for_file,
        )
    except ImportError:
        from _shared import get_contract_path, install_paths, install_stubs, run_contract_test_for_file

install_paths()
install_stubs()

NUM_GPUS = 0

from slime.utils.misc import load_function
from slime.utils.types import Sample


def run_contract_test_file() -> None:
    run_contract_test_for_file(
        __file__,
        path_args=[
            "custom-rollout-log-function-path",
            "custom-eval-rollout-log-function-path",
            "custom-reward-post-process-path",
            "custom-convert-samples-to-train-data-path",
            "rollout-data-postprocess-path",
        ],
    )


def reference_custom_rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    args.logged_rollout_id = rollout_id
    return True


def reference_custom_eval_rollout_log(rollout_id, args, data, extra_metrics) -> bool:
    args.logged_eval_rollout_id = rollout_id
    return True


def reference_reward_post_process(args, samples):
    raw_rewards = [sample.reward for sample in samples]
    rewards = [reward + 1.0 for reward in raw_rewards]
    return raw_rewards, rewards


def reference_convert_samples_to_train_data(args, samples):
    return {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": [sample.reward for sample in samples],
        "raw_reward": [sample.reward for sample in samples],
        "truncated": [0 for _ in samples],
        "sample_indices": [sample.index for sample in samples],
        "loss_masks": [sample.loss_mask for sample in samples],
    }


def reference_rollout_data_postprocess(args, rollout_id, rollout_data) -> None:
    args.rollout_data_postprocess_called = True
    args.rollout_data_postprocess_rollout_id = rollout_id
    args.rollout_data_postprocess_data = rollout_data


def make_sample(index: int, reward: float = 1.0) -> Sample:
    return Sample(
        index=index,
        reward=reward,
        tokens=[index, index + 1],
        response_length=2,
        status=Sample.Status.COMPLETED,
        loss_mask=[1, 1],
    )


@dataclass(frozen=True)
class HookCase:
    name: str
    env_key: str
    default_path: str
    source_path: str
    runtime_marker: str
    expected_params: tuple[str, ...]
    invoke: object


def invoke_custom_rollout_log(fn):
    args = type("Args", (), {})()
    assert isinstance(fn(3, args, [Sample(index=0)], {"reward": 1.0}, 0.5), bool)
    assert args.logged_rollout_id == 3


def invoke_custom_eval_rollout_log(fn):
    args = type("Args", (), {})()
    sample = Sample(index=0, reward=1.0)
    assert isinstance(
        fn(4, args, {"eval_set": {"rewards": [1.0], "truncated": [False], "samples": [sample]}}, {"acc": 1.0}), bool
    )
    assert args.logged_eval_rollout_id == 4


def invoke_reward_post_process(fn):
    raw_rewards, rewards = fn(type("Args", (), {})(), [make_sample(0, 0.5), make_sample(1, 1.5)])
    assert len(raw_rewards) == len(rewards) == 2


def invoke_convert_samples_to_train_data(fn):
    train_data = fn(type("Args", (), {})(), [make_sample(0, 0.5), make_sample(1, 1.5)])
    assert {"tokens", "response_lengths", "rewards", "raw_reward", "truncated", "sample_indices", "loss_masks"} <= set(
        train_data
    )


def invoke_rollout_data_postprocess(fn):
    args = type("Args", (), {})()
    rollout_data = {"sample_indices": [0], "rewards": [1.0]}
    assert fn(args, 7, rollout_data) is None
    assert args.rollout_data_postprocess_called is True
    assert args.rollout_data_postprocess_rollout_id == 7
    assert args.rollout_data_postprocess_data == rollout_data


HOOK_CASES = [
    HookCase(
        "custom_rollout_log",
        "CUSTOM_ROLLOUT_LOG_FUNCTION_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_custom_rollout_log",
        "slime/ray/rollout.py",
        "custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time)",
        ("rollout_id", "args", "samples", "rollout_extra_metrics", "rollout_time"),
        invoke_custom_rollout_log,
    ),
    HookCase(
        "custom_eval_rollout_log",
        "CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_custom_eval_rollout_log",
        "slime/ray/rollout.py",
        "custom_log_func(rollout_id, args, data, extra_metrics)",
        ("rollout_id", "args", "data", "extra_metrics"),
        invoke_custom_eval_rollout_log,
    ),
    HookCase(
        "custom_reward_post_process",
        "CUSTOM_REWARD_POST_PROCESS_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_reward_post_process",
        "slime/ray/rollout.py",
        "self.custom_reward_post_process_func(self.args, samples)",
        ("args", "samples"),
        invoke_reward_post_process,
    ),
    HookCase(
        "custom_convert_samples_to_train_data",
        "CUSTOM_CONVERT_SAMPLES_TO_TRAIN_DATA_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_convert_samples_to_train_data",
        "slime/ray/rollout.py",
        "self.custom_convert_samples_to_train_data_func(self.args, samples)",
        ("args", "samples"),
        invoke_convert_samples_to_train_data,
    ),
    HookCase(
        "rollout_data_postprocess",
        "ROLLOUT_DATA_POSTPROCESS_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_rollout_data_postprocess",
        "slime/backends/megatron_utils/actor.py",
        "self.rollout_data_postprocess(self.args, rollout_id, rollout_data)",
        ("args", "rollout_id", "rollout_data"),
        invoke_rollout_data_postprocess,
    ),
]


@pytest.mark.parametrize("case", HOOK_CASES, ids=[case.name for case in HOOK_CASES])
def test_runtime_hook_callsite_is_stable(case: HookCase):
    assert case.runtime_marker in Path(case.source_path).read_text()


@pytest.mark.parametrize("case", HOOK_CASES, ids=[case.name for case in HOOK_CASES])
def test_runtime_hook_path_aligns_with_expected_format(case: HookCase):
    fn = load_function(get_contract_path(case.env_key, case.default_path))
    assert tuple(inspect.signature(fn).parameters) == case.expected_params
    case.invoke(fn)


if __name__ == "__main__":
    run_contract_test_file()
