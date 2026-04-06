from __future__ import annotations

import asyncio
import importlib.util
from enum import Enum
from pathlib import Path

import pytest


def _load_shared_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "multidomain_shared.py"
    spec = importlib.util.spec_from_file_location("multidomain_shared_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multidomain_shared = _load_shared_module()


def test_default_train_datasets_for_group_tool_call_are_call_level_only():
    assert multidomain_shared.default_train_datasets_for_group("tool_call") == (
        "apibench",
        "xlam_function_calling_60k",
        "agent",
    )


def test_default_train_datasets_for_group_rejects_removed_tool_trajectory_group():
    with pytest.raises(ValueError, match="Unsupported train group 'tool_trajectory'"):
        multidomain_shared.default_train_datasets_for_group("tool_trajectory")


def test_group_signature_for_train_datasets_rejects_removed_train_datasets():
    with pytest.raises(ValueError, match="Unsupported dataset 'toolbench_v1'"):
        multidomain_shared.group_signature_for_train_datasets(["toolbench_v1"])


def test_group_signature_for_train_datasets_returns_mixed_label():
    assert multidomain_shared.group_signature_for_train_datasets(["apibench", "jsonschemabench"]) == "mixed-tool-call+structured"


def test_compute_generic_reward_rejects_bfcl_in_generic_router():
    with pytest.raises(RuntimeError, match="BFCL official evaluation"):
        multidomain_shared.compute_generic_reward(
            {
                "dataset_name": "bfcl_v3",
                "reward_type": "bfcl_official",
            },
            "Question",
            "Answer",
        )


def test_compute_generic_reward_requires_official_ifeval_backend(monkeypatch):
    monkeypatch.setattr(multidomain_shared, "_IFEVAL_BACKEND", None)

    def _missing_backend(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(multidomain_shared.importlib, "import_module", _missing_backend)

    with pytest.raises(RuntimeError, match="Official IFEval backend is required"):
        multidomain_shared.compute_generic_reward(
            {
                "dataset_name": "ifeval",
                "reward_type": "instruction_following_strict",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["alpha"]}],
                "record_id": "ifeval-1",
            },
            "Mention alpha.",
            "alpha",
        )


def test_load_ifeval_backend_accepts_google_research_evaluation_lib(monkeypatch):
    multidomain_shared._IFEVAL_BACKEND = None

    class _EvalLib:
        class InputExample:
            pass

        @staticmethod
        def test_instruction_following_strict(inp, prompt_to_response):
            return None

        @staticmethod
        def test_instruction_following_loose(inp, prompt_to_response):
            return None

    class _Registry:
        INSTRUCTION_DICT = {"x": object()}

    def _import_module(name: str):
        if name == "instruction_following_eval.evaluation_lib":
            return _EvalLib
        if name == "instruction_following_eval.instructions_registry":
            return _Registry
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(multidomain_shared.importlib, "import_module", _import_module)

    backend = multidomain_shared._load_ifeval_backend()
    assert backend["kind"] == "google"
    assert backend["InputExample"] is _EvalLib.InputExample
    assert isinstance(backend["registry"], dict)
    assert list(backend["registry"]) == ["x"]


@pytest.mark.parametrize(
    "reward_type",
    [
        "function_call_single",
        "api_call_text",
    ],
)
def test_compute_generic_reward_accepts_family_specific_train_tool_rewards(reward_type):
    metadata = {
        "dataset_name": "tool_train",
        "ground_truth": [{"name": "weather", "arguments": {"city": "Paris"}}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }
    response = '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'
    if reward_type == "api_call_text":
        metadata = {
            "dataset_name": "apibench_huggingface",
            "raw_api_call": "AutoModel.from_pretrained('bert-base-uncased')",
        }
        response = "AutoModel.from_pretrained('bert-base-uncased')"

    score = multidomain_shared.compute_generic_reward(
        {
            **metadata,
            "reward_type": reward_type,
        },
        "Check weather",
        response,
    )
    legacy_score = multidomain_shared.compute_generic_reward(
        {
            **metadata,
            "reward_type": "tool_call_soft",
        },
        "Check weather",
        response,
    )

    if reward_type == "api_call_text":
        assert score == 1.0
        assert legacy_score == 0.0
    else:
        assert score == legacy_score
        assert score > 0.0


def test_compute_generic_reward_rejects_removed_tool_trajectory_reward_type():
    metadata = {
        "dataset_name": "toolbench_v1",
        "reward_type": "tool_trajectory",
        "assistant_reference": "Assistant: Final answer.",
    }

    assert multidomain_shared.compute_generic_reward(metadata, "Look up nike.", "Assistant: Final answer.") == 0.0


def test_reward_func_uses_top_level_sample_tools_for_tool_rewards(monkeypatch):
    expected_calls = [
        {
            "name": "weather",
            "arguments": {"city": "Paris"},
            "function": {"name": "weather", "arguments": {"city": "Paris"}},
        }
    ]
    tools = [{"type": "function", "function": {"name": "weather", "description": "", "parameters": {}}}]
    seen: dict[str, object] = {}

    def _parse_predicted_calls(response, *, tools, parser_type):
        seen["tools"] = tools
        seen["parser_type"] = parser_type
        return expected_calls

    monkeypatch.setattr(multidomain_shared, "_parse_predicted_calls", _parse_predicted_calls)

    class _Status(Enum):
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    class _Sample:
        Status = _Status

        def __init__(self):
            self.status = _Status.COMPLETED
            self.prompt = [{"role": "user", "content": "check weather"}]
            self.response = "<tool_call>weather({\"city\": \"Paris\"})</tool_call>"
            self.metadata = {
                "dataset_name": "tool_train",
                "reward_type": "function_call_single",
                "ground_truth": expected_calls,
                "parser_type": "qwen3",
            }
            self.tools = tools

    score = asyncio.run(multidomain_shared.reward_func(None, _Sample()))

    assert score > 0.0
    assert seen == {"tools": tools, "parser_type": "qwen3"}


def test_compute_generic_reward_accepts_raw_final_contract_tool_pool_rows():
    metadata = {
        "dataset_name": "agent_function_calling_open_dataset",
        "reward_type": "function_call_single",
        "ground_truth": [
            {
                "name": "maps_text_search",
                "arguments": {"keywords": "Italian Restaurants", "city": "New York"},
                "function": {"name": "maps_text_search", "arguments": {"keywords": "Italian Restaurants", "city": "New York"}},
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "maps_text_search",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    response = '<tool_call>{"name":"maps_text_search","arguments":{"keywords":"Italian Restaurants","city":"New York"}}</tool_call>'

    assert multidomain_shared.compute_generic_reward(metadata, "Find restaurants", response) > 0.0
