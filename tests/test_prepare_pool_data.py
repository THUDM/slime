from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_prepare_pool_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "prepare_pool_data.py"
    spec = importlib.util.spec_from_file_location("prepare_pool_data_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_pool_data = _load_prepare_pool_module()


def test_convert_xlam_row_for_pool_preserves_prior_tool_context_and_call_metadata():
    row = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": "first call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "weather",
                "tool_call_id": "call_1",
                "content": "sunny",
            },
            {
                "role": "assistant",
                "content": "second call",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "calendar", "arguments": {"day": "Monday"}},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "desc",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
                "x-extra": "keep-me",
            },
            {
                "type": "function",
                "function": {
                    "name": "calendar",
                    "description": "desc2",
                    "parameters": {"type": "object", "properties": {"day": {"type": "string"}}},
                },
            },
        ],
        "extra": {"id": "row-1"},
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "xlam_function_calling_60k")

    assert len(samples) == 2
    second = samples[1]
    prompt = second["prompt"]
    assert prompt[2]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "name": "weather",
            "arguments": {"city": "Paris"},
            "function": {"name": "weather", "arguments": {"city": "Paris"}},
        }
    ]
    assert prompt[3]["name"] == "weather"
    assert prompt[3]["tool_call_id"] == "call_1"
    assert second["metadata"]["ground_truth"] == [
        {
            "id": "call_2",
            "type": "function",
            "name": "calendar",
            "arguments": {"day": "Monday"},
            "function": {"name": "calendar", "arguments": {"day": "Monday"}},
        }
    ]
    assert second["tools"][0]["x-extra"] == "keep-me"
    assert "tools" not in second["metadata"]
    assert second["metadata"]["source_fields"] == {"extra": {"id": "row-1"}}


def test_normalize_prompt_message_for_pool_keeps_multimodal_content_blocks():
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": "image-1.png"},
            {"type": "text", "text": "Describe the image."},
        ],
    }

    normalized = prepare_pool_data._normalize_prompt_message_for_pool(message)

    assert normalized == message


def test_resolve_avalanche_root_falls_back_to_project_root_when_legacy_mount_is_missing(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir()

    resolved = prepare_pool_data._resolve_avalanche_root(
        project_root=project_root,
        legacy_root=tmp_path / "missing-legacy-root",
        env={},
    )

    assert resolved == project_root


def test_resolve_avalanche_root_prefers_ancestor_with_open_data(tmp_path: Path):
    avalanche_root = tmp_path / "avalanche"
    nested_project = avalanche_root / "jy_workspace"
    (avalanche_root / "data" / "open_data").mkdir(parents=True)
    nested_project.mkdir()

    resolved = prepare_pool_data._resolve_avalanche_root(
        project_root=nested_project,
        legacy_root=tmp_path / "missing-legacy-root",
        env={},
    )

    assert resolved == avalanche_root


def test_pool_output_paths_prefers_nested_existing_targets(tmp_path: Path, monkeypatch):
    avalanche_root = tmp_path / "avalanche"
    root_level = avalanche_root / "data" / "pool" / "tool" / "sample.jsonl"
    nested = avalanche_root / "data" / "pool" / "tool" / "train" / "sample.jsonl"
    root_level.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    root_level.write_text("", encoding="utf-8")
    nested.write_text("", encoding="utf-8")
    monkeypatch.setattr(prepare_pool_data, "AVALANCHE_ROOT", avalanche_root)

    output_paths = prepare_pool_data._pool_output_paths("tool", "sample.jsonl")

    assert output_paths == [nested]


def test_convert_agent_row_for_pool_preserves_source_fields_and_record_fields():
    row = {
        "trace_id": "trace-1",
        "model": "gpt-test",
        "session_metadata": {"region": "cq"},
        "function_calls": [
            {
                "messages": [
                    {"role": "user", "content": "where is times square"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "tool_use",
                                "function": {
                                    "name": "maps_text_search",
                                    "arguments": '{"city":"New York"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "maps_text_search",
                        "content": "[]",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "tool_use",
                                "function": {"name": "maps_geo", "arguments": {"address": "Times Square"}},
                            }
                        ],
                    },
                ],
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "tool_use",
                        "function": {"name": "maps_geo", "arguments": {"address": "Times Square"}},
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "maps_geo",
                            "description": "desc",
                            "parameters": {"type": "object", "properties": {"address": {"type": "string"}}},
                        },
                    }
                ],
                "tool_choice": "auto",
            }
        ],
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "agent_function_calling_open_dataset")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["metadata"]["source_fields"] == {
        "model": "gpt-test",
        "session_metadata": {"region": "cq"},
    }
    assert sample["metadata"]["source_record_fields"] == {"tool_choice": "auto"}
    assert sample["prompt"][1]["tool_calls"][0]["id"] == "call_1"
    assert sample["prompt"][2]["tool_call_id"] == "call_1"


def test_convert_ifbench_row_for_pool_preserves_extra_source_fields():
    row = {
        "key": "ifbench-1",
        "prompt": "Do task",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["alpha"]}],
        "split": "train",
        "source_name": "ifbench-open",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "ifbench_test")

    assert len(samples) == 1
    assert samples[0]["prompt"] == [{"role": "user", "content": "Do task"}]
    metadata = samples[0]["metadata"]
    assert metadata["prompt_text"] == "Do task"
    assert metadata["source_fields"] == {
        "source_name": "ifbench-open",
        "split": "train",
    }


def test_convert_gpqa_invalid_row_is_skipped():
    row = {
        "Question": "Which option is correct?",
        "Correct Answer": "",
        "Incorrect Answer 1": "A",
        "Incorrect Answer 2": "B",
        "Incorrect Answer 3": "C",
        "Record ID": "gpqa-invalid",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "gpqa")

    assert samples == []


def test_convert_xlam_row_without_tool_calls_is_skipped():
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "plain answer"},
        ],
        "tools": [],
        "extra": {"id": "row-2"},
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "xlam_function_calling_60k")

    assert samples == []


def test_process_ifrl_writes_message_prompts_without_prebaked_system(tmp_path: Path):
    src = tmp_path / "ifrl_source.jsonl"
    src.write_text(
        json.dumps(
            {
                "id": 1,
                "prompt": "Follow the instruction",
                "dataset": "ifrl-dataset",
                "agent_ref": {"name": "agent"},
                "instruction_id_list": ["rule-1"],
                "kwargs": [{"alpha": 1}],
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    original_root = prepare_pool_data.AVALANCHE_ROOT
    try:
        prepare_pool_data.AVALANCHE_ROOT = tmp_path
        prepare_pool_data.process_ifrl(str(src))
    finally:
        prepare_pool_data.AVALANCHE_ROOT = original_root

    out_path = tmp_path / "data" / "pool" / "ifrl" / "ifrl_ifrl_source.jsonl"
    row = json.loads(out_path.read_text(encoding="utf-8").strip())

    assert row["prompt"] == [{"role": "user", "content": "Follow the instruction"}]
    assert row["metadata"]["domain"] == "ifrl"
    assert row["metadata"]["dataset_name"] == "ifrl-dataset"
    assert row["metadata"]["instruction_id_list"] == ["rule-1"]
    assert row["metadata"]["source_fields"] == {"split": "train"}


def test_convert_scienceqa_row_for_pool_preserves_extra_source_fields():
    row = {
        "question": "What is H2O?",
        "choices": ["Water", "Oxygen", "Hydrogen"],
        "answer": 0,
        "hint": "It is common on Earth.",
        "lecture": "H2O is the chemical formula for water.",
        "topic": "chemistry",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "scienceqa")

    assert len(samples) == 1
    metadata = samples[0]["metadata"]
    assert metadata["source_fields"] == {
        "hint": "It is common on Earth.",
        "lecture": "H2O is the chemical formula for water.",
        "topic": "chemistry",
    }
