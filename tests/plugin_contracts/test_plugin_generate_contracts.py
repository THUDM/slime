from __future__ import annotations

import asyncio
import inspect
import types
from contextlib import contextmanager

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
REFERENCE_CUSTOM_GENERATE_PATH = "plugin_contracts.test_plugin_generate_contracts.custom_generate"
REFERENCE_CUSTOM_GENERATE_WITH_EVAL_PATH = (
    "plugin_contracts.test_plugin_generate_contracts.custom_generate_with_evaluation"
)

from slime.rollout.sglang_rollout import generate_and_rm
from slime.utils.misc import load_function
from slime.utils.types import Sample


def run_contract_test_file() -> None:
    run_contract_test_for_file(__file__, path_args=["custom-generate-function-path"])


def make_args(**overrides):
    class Args:
        partial_rollout = False
        mask_offpolicy_in_partial_rollout = False
        group_rm = False
        custom_generate_function_path = None
        sglang_enable_deterministic_inference = False
        rollout_seed = 7
        n_samples_per_prompt = 2

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeGenerateState:
    def __init__(self, args) -> None:
        self.args = args
        self.semaphore = types.SimpleNamespace(__aenter__=None)
        self.pendings = set()
        self.remaining_batch_size = 0
        self.aborted = False
        self.group_sampling_seeds = [args.rollout_seed + i for i in range(args.n_samples_per_prompt)]

    @contextmanager
    def dp_rank_context(self):
        yield 0


async def custom_generate(args, sample: Sample, sampling_params: dict):
    sample.tokens = [11, 12, 13]
    sample.response = "generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.25
    sample.status = Sample.Status.COMPLETED
    return sample


async def custom_generate_with_evaluation(args, sample: Sample, sampling_params: dict, evaluation: bool = False):
    sample.tokens = [21, 22]
    sample.response = "eval-generated" if evaluation else "train-generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.5 if evaluation else 0.75
    sample.status = Sample.Status.COMPLETED
    sample.metadata["evaluation"] = evaluation
    return sample


def assert_sample_contract(sample: Sample) -> None:
    assert isinstance(sample, Sample)
    assert isinstance(sample.tokens, list)
    assert isinstance(sample.response, str)
    assert isinstance(sample.response_length, int)
    assert sample.reward is not None


def assert_custom_generate_signature_matches_expected(fn) -> None:
    params = tuple(inspect.signature(fn).parameters)
    assert params[:3] == ("args", "sample", "sampling_params")


class _DummySemaphore:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PatchedGenerateState(FakeGenerateState):
    def __init__(self, args):
        super().__init__(args)
        self.semaphore = _DummySemaphore()


@pytest.fixture
def patch_generate_state(monkeypatch):
    """Patch GenerateState with a test-safe variant; returns the sglang_rollout module."""
    from slime.rollout import sglang_rollout

    monkeypatch.setattr(sglang_rollout, "GenerateState", _PatchedGenerateState)
    return sglang_rollout


def test_generate_and_rm_default_generate_branch_is_stable(patch_generate_state, monkeypatch):
    sglang_rollout = patch_generate_state

    async def official_default_generate(args, sample: Sample, sampling_params: dict):
        sample.tokens = [31, 32]
        sample.response = "default-generate"
        sample.response_length = 2
        sample.reward = 1.0
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "generate", official_default_generate)

    result = asyncio.run(
        generate_and_rm(
            make_args(custom_generate_function_path=None),
            Sample(index=0, prompt="prompt"),
            sampling_params={"temperature": 0.3},
            evaluation=False,
        )
    )
    assert_sample_contract(result)
    assert result.response == "default-generate"


def test_generate_and_rm_prefers_per_sample_generate_function(patch_generate_state):
    args = make_args(custom_generate_function_path=REFERENCE_CUSTOM_GENERATE_PATH)
    sample = Sample(index=0, prompt="prompt", generate_function_path=REFERENCE_CUSTOM_GENERATE_WITH_EVAL_PATH)
    result = asyncio.run(generate_and_rm(args, sample, sampling_params={"temperature": 0.3}, evaluation=True))
    assert_sample_contract(result)
    assert result.metadata["evaluation"] is True


def test_custom_generate_function_path_supports_user_override(patch_generate_state):
    custom_generate_path = get_contract_path(
        "CUSTOM_GENERATE_FUNCTION_PATH",
        REFERENCE_CUSTOM_GENERATE_PATH,
    )
    assert_custom_generate_signature_matches_expected(load_function(custom_generate_path))
    result = asyncio.run(
        generate_and_rm(
            make_args(custom_generate_function_path=custom_generate_path),
            Sample(index=0, prompt="prompt"),
            sampling_params={"temperature": 0.3},
            evaluation=False,
        )
    )
    assert_sample_contract(result)


def test_generate_and_rm_group_rm_accepts_list_result_from_custom_generate(patch_generate_state, monkeypatch):
    sglang_rollout = patch_generate_state

    async def custom_generate_list(args, sample: Sample, sampling_params: dict):
        sample.status = Sample.Status.COMPLETED
        sibling = Sample(index=1, prompt="prompt-1", status=Sample.Status.COMPLETED)
        return [sample, sibling]

    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: custom_generate_list)

    result = asyncio.run(
        generate_and_rm(
            make_args(custom_generate_function_path="plugin_contracts.fake_generate", group_rm=True),
            Sample(index=0, prompt="prompt-0"),
            sampling_params={"temperature": 0.3},
            evaluation=False,
        )
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(sample, Sample) for sample in result)


if __name__ == "__main__":
    run_contract_test_file()
