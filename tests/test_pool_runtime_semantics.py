from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runtime_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "pool_runtime_semantics.py"
    spec = importlib.util.spec_from_file_location("pool_runtime_semantics_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pool_runtime_semantics = _load_runtime_module()


def test_infer_prompt_family_prefers_explicit_metadata_value():
    family = pool_runtime_semantics.infer_prompt_family(
        {
            "prompt": [{"role": "user", "content": "Load a model"}],
            "metadata": {
                "dataset_name": "apibench_huggingface",
                "domain": "tool",
                "reward_type": "api_call_text",
                "prompt_family": "api_call_codegen",
            },
        }
    )

    assert family == "api_call_codegen"


def test_materialize_runtime_pool_row_injects_api_call_codegen_prompt():
    row = {
        "dataset_name": "apibench_huggingface",
        "domain": "tool",
        "record_id": "api-1",
        "supervision_family": "function_call_single",
        "prompt": [{"role": "user", "content": "Load the model"}],
        "tools": [],
        "metadata": {
            "dataset_name": "apibench_huggingface",
            "domain": "tool",
            "record_id": "api-1",
            "reward_type": "api_call_text",
            "prompt_family": "api_call_codegen",
            "raw_api_call": "AutoModel.from_pretrained('bert-base-uncased')",
            "ground_truth": [
                {
                    "name": "AutoModel.from_pretrained",
                    "arguments": {"_args": ["bert-base-uncased"]},
                    "function": {"name": "AutoModel.from_pretrained", "arguments": {"_args": ["bert-base-uncased"]}},
                }
            ],
        },
    }

    materialized = pool_runtime_semantics.materialize_runtime_pool_row(row)

    assert materialized is not None
    assert materialized["prompt"][0]["role"] == "system"
    assert "python api call" in materialized["prompt"][0]["content"].lower()
    assert materialized["prompt"][1:] == [{"role": "user", "content": "Load the model"}]


def test_materialize_runtime_pool_row_injects_agent_next_action_prompt():
    row = {
        "dataset_name": "agent_function_calling_open_dataset",
        "domain": "tool",
        "record_id": "agent-1",
        "supervision_family": "function_call_single",
        "prompt": [{"role": "user", "content": "Find Italian restaurants"}],
        "tools": [{"type": "function", "function": {"name": "maps_text_search", "parameters": {"type": "object"}}}],
        "metadata": {
            "dataset_name": "agent_function_calling_open_dataset",
            "domain": "tool",
            "record_id": "agent-1",
            "reward_type": "function_call_single",
            "prompt_family": "next_action_tool_call",
            "ground_truth": [
                {
                    "name": "maps_text_search",
                    "arguments": {"keywords": "Italian Restaurants", "city": "New York"},
                    "function": {
                        "name": "maps_text_search",
                        "arguments": {"keywords": "Italian Restaurants", "city": "New York"},
                    },
                }
            ],
        },
    }

    materialized = pool_runtime_semantics.materialize_runtime_pool_row(row)

    assert materialized is not None
    assert materialized["prompt"][0]["role"] == "system"
    system_text = materialized["prompt"][0]["content"].lower()
    assert "next tool call" in system_text or "next action" in system_text
    assert materialized["prompt"][1:] == [{"role": "user", "content": "Find Italian restaurants"}]
