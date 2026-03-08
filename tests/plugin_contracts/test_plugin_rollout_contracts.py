from __future__ import annotations

import inspect

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
install_stubs(with_sglang_router=True, with_transformers=True)

NUM_GPUS = 0
DEFAULT_ROLLOUT_FUNCTION_PATH = "slime.rollout.sglang_rollout.generate_rollout"
REFERENCE_ROLLOUT_FUNCTION_PATH = "plugin_contracts.test_plugin_rollout_contracts.valid_rollout_function"

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput, call_rollout_fn
from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout
from slime.utils.misc import load_function
from slime.utils.types import Sample


def run_contract_test_file() -> None:
    run_contract_test_for_file(__file__, path_args=["rollout-function-path"])


def make_sample(index: int, reward: float = 1.0) -> Sample:
    tokens = [1000 + index, 2000 + index]
    return Sample(
        index=index,
        prompt=f"prompt-{index}",
        response=f"response-{index}",
        tokens=tokens,
        response_length=len(tokens),
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={},
    )


class ContractDataSource:
    def __init__(self, n_samples_per_prompt: int = 2):
        self.n_samples_per_prompt = n_samples_per_prompt
        self.next_index = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        for _ in range(num_samples):
            group = [make_sample(self.next_index + i, reward=0.0) for i in range(self.n_samples_per_prompt)]
            self.next_index += self.n_samples_per_prompt
            groups.append(group)
        return groups


def valid_rollout_function(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        sample = make_sample(0, reward=0.75)
        return RolloutFnEvalOutput(
            data={"contract_eval": {"rewards": [sample.reward], "truncated": [False], "samples": [sample]}},
            metrics={"source": "contract"},
        )

    groups = data_source.get_samples(2)
    for group_index, group in enumerate(groups):
        for sample_index, sample in enumerate(group):
            sample.tokens = [group_index, sample_index, rollout_id]
            sample.response = f"group-{group_index}-sample-{sample_index}"
            sample.response_length = len(sample.tokens)
            sample.reward = float(group_index + sample_index)
            sample.status = Sample.Status.COMPLETED
    return RolloutFnTrainOutput(samples=groups, metrics={"source": "contract"})


def invalid_rollout_function(args, rollout_id, data_source, evaluation=False):
    sample = make_sample(0)
    sample.reward = None
    return RolloutFnTrainOutput(samples=[[sample]])


def legacy_rollout_function(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        sample = make_sample(1, reward=0.5)
        return {"legacy_eval": {"rewards": [sample.reward], "truncated": [False], "samples": [sample]}}
    return [[make_sample(1)]]


def assert_sample_contract(sample: Sample) -> None:
    assert isinstance(sample, Sample)
    assert isinstance(sample.tokens, list)
    assert all(isinstance(token, int) for token in sample.tokens)
    assert isinstance(sample.response, str)
    assert isinstance(sample.response_length, int)
    assert sample.reward is not None
    assert isinstance(sample.status, Sample.Status)


def assert_train_rollout_contract(output: RolloutFnTrainOutput, n_samples_per_prompt: int) -> None:
    assert isinstance(output, RolloutFnTrainOutput)
    assert output.samples
    for group in output.samples:
        assert len(group) == n_samples_per_prompt
        for sample in group:
            assert_sample_contract(sample)


def assert_eval_rollout_contract(output: RolloutFnEvalOutput) -> None:
    assert isinstance(output, RolloutFnEvalOutput)
    assert output.data
    for dataset_data in output.data.values():
        assert set(dataset_data) >= {"rewards", "truncated", "samples"}
        assert len(dataset_data["rewards"]) == len(dataset_data["truncated"]) == len(dataset_data["samples"])
        for sample in dataset_data["samples"]:
            assert_sample_contract(sample)


def assert_rollout_function_signature_matches_default(fn) -> None:
    default_sig = inspect.signature(default_generate_rollout)
    candidate_sig = inspect.signature(fn)

    assert tuple(candidate_sig.parameters) == tuple(default_sig.parameters)

    for name, default_param in default_sig.parameters.items():
        candidate_param = candidate_sig.parameters[name]
        assert candidate_param.kind == default_param.kind
        assert candidate_param.default == default_param.default


def assert_rollout_function_matches_default_contract(fn) -> None:
    assert_rollout_function_signature_matches_default(fn)

    data_source = ContractDataSource()
    train_output = call_rollout_fn(fn, None, 2, data_source, evaluation=False)
    eval_output = call_rollout_fn(fn, None, 2, data_source, evaluation=True)

    assert_train_rollout_contract(train_output, n_samples_per_prompt=2)
    assert_eval_rollout_contract(eval_output)


def test_rollout_function_path_contract_supports_user_override():
    rollout_path = get_contract_path("ROLLOUT_FUNCTION_PATH", DEFAULT_ROLLOUT_FUNCTION_PATH)
    rollout_fn = load_function(rollout_path)

    assert_rollout_function_signature_matches_default(rollout_fn)

    if rollout_path != DEFAULT_ROLLOUT_FUNCTION_PATH:
        assert_rollout_function_matches_default_contract(rollout_fn)


def test_load_function_can_load_local_rollout_plugin():
    func = load_function(REFERENCE_ROLLOUT_FUNCTION_PATH)
    assert func.__name__ == valid_rollout_function.__name__
    assert func.__module__ == "plugin_contracts.test_plugin_rollout_contracts"


def test_default_rollout_signature_is_stable():
    default_sig = inspect.signature(default_generate_rollout)
    assert tuple(default_sig.parameters) == ("args", "rollout_id", "data_source", "evaluation")
    assert default_sig.parameters["evaluation"].default is False


def test_default_rollout_compat_wrapper_stability():
    data_source = ContractDataSource(n_samples_per_prompt=1)
    train_output = call_rollout_fn(legacy_rollout_function, None, 1, data_source, evaluation=False)
    eval_output = call_rollout_fn(legacy_rollout_function, None, 1, data_source, evaluation=True)
    assert_train_rollout_contract(train_output, n_samples_per_prompt=1)
    assert_eval_rollout_contract(eval_output)


def test_local_rollout_plugin_aligns_with_default_input_output_format():
    assert_rollout_function_matches_default_contract(valid_rollout_function)


def test_misaligned_rollout_plugin_is_rejected():
    with pytest.raises(AssertionError):
        assert_rollout_function_matches_default_contract(invalid_rollout_function)


if __name__ == "__main__":
    run_contract_test_file()
