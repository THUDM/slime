from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_prepare_pool_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "common" / "prepare_pool_data.py"
    spec = importlib.util.spec_from_file_location("prepare_pool_data_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_pool_data = _load_prepare_pool_module()


def test_convert_xlam_row_for_pool_preserves_raw_fields_without_native_wrapper():
    row = {
        "messages": [
            {"role": "user", "content": "Where can I find beta giveaways?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "live_giveaways_by_type", "arguments": '{"type":"beta"}'},
                    }
                ],
            },
        ],
        "tools": '[{"type":"function","function":{"name":"live_giveaways_by_type","parameters":{"type":"object"}}}]',
        "extra": {"id": 7},
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "xlam_function_calling_60k")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "xlam_function_calling_60k"
    assert sample["domain"] == "tool"
    assert sample["record_id"] == "7"
    assert sample["supervision_family"] == "function_call_single"
    assert sample["label"] == ""
    assert sample["prompt"] == [{"role": "user", "content": "Where can I find beta giveaways?"}]
    assert sample["messages"] == row["messages"]
    assert isinstance(sample["tools"], list)
    assert sample["extra"] == {"id": 7}
    assert "native" not in sample
    assert sample["metadata"] == {
        "dataset_name": "xlam_function_calling_60k",
        "domain": "tool",
        "record_id": "7",
        "reward_type": "function_call_single",
        "prompt_family": "function_call_single",
        "ground_truth": [
            {
                "name": "live_giveaways_by_type",
                "arguments": {"type": "beta"},
                "function": {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            }
        ],
        "supervision_family": "function_call_single",
        "source_fields": {"extra": {"id": 7}},
    }


def test_convert_apibench_row_for_pool_preserves_api_bench_shape():
    row = {
        "code": "###Instruction: Load the model.\n###Output: <<<api_call>>>: AutoModel.from_pretrained(\"bert-base-uncased\")",
        "api_call": 'AutoModel.from_pretrained("bert-base-uncased")',
        "provider": "Hugging Face Transformers",
        "api_data": {"api_name": "bert-base-uncased", "framework": "transformers"},
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "apibench_huggingface")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "apibench_huggingface"
    assert sample["supervision_family"] == "function_call_single"
    assert sample["label"] == ""
    assert sample["prompt"] == [{"role": "user", "content": row["code"]}]
    assert sample["tools"] == []
    assert sample["code"] == row["code"]
    assert sample["api_call"] == row["api_call"]
    assert sample["provider"] == row["provider"]
    assert sample["api_data"] == row["api_data"]
    assert "native" not in sample
    assert sample["metadata"]["reward_type"] == "api_call_text"
    assert sample["metadata"]["prompt_family"] == "api_call_codegen"
    assert sample["metadata"]["raw_api_call"] == row["api_call"]


def test_convert_row_for_pool_rejects_removed_apigen_train_dataset():
    row = {
        "system": "You are a helpful airline agent.",
        "tools": [{"type": "function", "function": {"name": "get_reservation_details"}}],
        "conversations": [],
    }

    with pytest.raises(ValueError, match="Unsupported dataset for pool conversion: apigen_mt_5k"):
        prepare_pool_data.convert_row_for_pool(row, "apigen_mt_5k")


def test_convert_agent_row_for_pool_preserves_trace_item_fields():
    row = {
        "trace_id": "trace-1",
        "model": "qwen3-plus",
        "session_id": "TEMP_SESSION_123",
        "function_calls": [
            {
                "messages": [
                    {"role": "user", "content": "Italian restaurants in New York"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "tool_use",
                                "function": {
                                    "name": "maps_text_search",
                                    "arguments": '{"keywords":"Italian Restaurants","city":"New York"}',
                                },
                            }
                        ],
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "maps_text_search"}}],
            }
        ],
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "agent_function_calling_open_dataset")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "agent_function_calling_open_dataset"
    assert sample["supervision_family"] == "function_call_single"
    assert sample["label"] == ""
    assert sample["trace_id"] == "trace-1"
    assert sample["model"] == "qwen3-plus"
    assert sample["session_id"] == "TEMP_SESSION_123"
    assert sample["prompt"] == [{"role": "user", "content": "Italian restaurants in New York"}]
    assert sample["messages"] == row["function_calls"][0]["messages"]
    assert sample["tools"] == row["function_calls"][0]["tools"]
    assert "native" not in sample
    assert sample["metadata"]["source_fields"] == {
        "model": "qwen3-plus",
        "session_id": "TEMP_SESSION_123",
    }
    assert sample["metadata"]["reward_type"] == "function_call_single"
    assert sample["metadata"]["prompt_family"] == "next_action_tool_call"
    assert sample["metadata"]["ground_truth"] == [
        {
            "name": "maps_text_search",
            "arguments": {"keywords": "Italian Restaurants", "city": "New York"},
            "function": {"name": "maps_text_search", "arguments": {"keywords": "Italian Restaurants", "city": "New York"}},
        }
    ]


def test_convert_agent_row_for_pool_drops_items_without_tool_calls():
    row = {
        "trace_id": "trace-2",
        "model": "qwen3-plus",
        "session_id": "TEMP_SESSION_456",
        "function_calls": [
            {
                "messages": [
                    {"role": "user", "content": "Just answer directly"},
                    {"role": "assistant", "content": "<p>No More Tool Calls Needed</p>"},
                ],
                "tools": [{"type": "function", "function": {"name": "maps_text_search"}}],
            }
        ],
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "agent_function_calling_open_dataset")

    assert samples == []


def test_convert_row_for_pool_rejects_removed_toolbench_train_dataset():
    row = {
        "id": "step-7",
        "conversations": {
            "from": ["system", "user", "assistant", "function", "assistant"],
            "value": [
                "You are AutoGPT.",
                "Find nike's Instagram info.",
                "Thought: search it.\nAction: userinfo_for_instagram_cheapest\nAction Input: {\"username\": \"nike\"}",
                '{"username":"nike","followers":1}',
                "Final answer.",
            ],
        },
    }

    with pytest.raises(ValueError, match="Unsupported dataset for pool conversion: toolbench_v1"):
        prepare_pool_data.convert_row_for_pool(row, "toolbench_v1")


def test_convert_row_for_pool_rejects_removed_toolbench_benchmark_dataset():
    row = {
        "query_id": 7,
        "query": "Find a tutorial",
        "api_list": [{"name": "Search"}],
        "relevant_apis": [["Simple YouTube Search", "Search"], ["Calendar", "CreateEvent"]],
    }

    with pytest.raises(ValueError, match="Unsupported dataset for pool conversion: toolbench_v1_benchmark"):
        prepare_pool_data.convert_row_for_pool(row, "toolbench_v1_benchmark")


def test_convert_bfcl_row_for_pool_promotes_official_fields_to_top_level():
    row = {
        "id": "irrelevance_0",
        "multi_turn": False,
        "functions": '[{"name":"determine_body_mass_index"}]',
        "tools": '[{"type":"function","function":{"name":"determine_body_mass_index","parameters":{"type":"object"}}}]',
        "missed_functions": "{}",
        "initial_config": "{}",
        "involved_classes": [],
        "turns": '[[{"role":"system","content":"sys"},{"role":"user","content":"Question"}]]',
        "language": "Python",
        "test_category": "irrelevance",
        "subset": "irrelevance",
        "ground_truth": "{}",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "bfcl_v3")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "bfcl_v3"
    assert sample["record_id"] == "irrelevance_0"
    assert sample["id"] == "irrelevance_0"
    assert sample["test_category"] == "irrelevance"
    assert sample["subset"] == "irrelevance"
    assert sample["turns"][0][1]["content"] == "Question"
    assert sample["prompt"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "Question"}]
    assert sample["tools"][0]["function"]["name"] == "determine_body_mass_index"
    assert "native" not in sample
    assert "label" not in sample


def test_write_jsonl_rows_uses_atomic_replace(tmp_path: Path):
    dest = tmp_path / "pool" / "tool" / "train" / "xlam.jsonl"
    rows = [{"dataset_name": "xlam_function_calling_60k", "record_id": "1"}]

    prepare_pool_data.write_jsonl_rows(dest, rows)

    assert json.loads(dest.read_text(encoding="utf-8").strip()) == rows[0]


def test_convert_gpqa_row_for_pool_uses_label_for_correct_answer():
    row = {
        "Record ID": "rec-1",
        "Question": "What is the answer?",
        "Correct Answer": "alpha",
        "Incorrect Answer 1": "beta",
        "Incorrect Answer 2": "gamma",
        "Incorrect Answer 3": "delta",
        "High-level domain": "biology",
        "Subdomain": "genetics",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "gpqa")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "gpqa"
    assert sample["domain"] == "stem"
    assert sample["record_id"] == "rec-1"
    assert sample["label"] in {"A", "B", "C", "D"}
    assert sample["question"] == "What is the answer?"
    assert len(sample["options"]) == 4
    assert "native" not in sample
    assert "answer" not in sample
    assert "tools" not in sample
    assert sample["metadata"]["reward_type"] == "stem_mcqa"
    assert sample["metadata"]["answer"] == sample["label"]


def test_convert_ifeval_row_for_pool_promotes_instruction_fields_to_top_level():
    row = {
        "key": 1000,
        "prompt": "Do the task",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["alpha"]}],
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "ifeval")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "ifeval"
    assert sample["domain"] == "structured"
    assert sample["record_id"] == "1000"
    assert sample["prompt"] == [{"role": "user", "content": "Do the task"}]
    assert sample["prompt_text"] == "Do the task"
    assert sample["instruction_id_list"] == ["keywords:existence"]
    assert sample["kwargs"] == [{"keywords": ["alpha"]}]
    assert "native" not in sample
    assert "label" not in sample
    assert "tools" not in sample
    assert sample["metadata"]["reward_type"] == "instruction_following_strict"


def test_convert_jsonschemabench_row_for_pool_promotes_schema_to_top_level():
    row = {
        "unique_id": "calculate_area_88ee549f",
        "json_schema": '{"type":"object","properties":{"radius":{"type":"number"}}}',
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "jsonschemabench")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "jsonschemabench"
    assert sample["domain"] == "structured"
    assert sample["record_id"] == "calculate_area_88ee549f"
    assert sample["schema"] == {"type": "object", "properties": {"radius": {"type": "number"}}}
    assert sample["prompt"] == [{"role": "user", "content": row["json_schema"]}]
    assert "native" not in sample
    assert "label" not in sample
    assert "tools" not in sample
    assert sample["metadata"]["reward_type"] == "structured_json_schema"


def test_convert_nemotron_structured_outputs_row_for_pool_preserves_prompt_and_schema():
    row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": "Extract the fields from the document.",
                }
            ]
        },
        "schema_str": '{"type":"object","properties":{"name":{"type":"string"}}}',
        "schema_type": "json",
        "schema_fields_count": "1",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "nemotron_structured_outputs")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["dataset_name"] == "nemotron_structured_outputs"
    assert sample["domain"] == "structured"
    assert sample["prompt"] == row["responses_create_params"]["input"]
    assert sample["schema"] == {"type": "object", "properties": {"name": {"type": "string"}}}
    assert sample["responses_create_params"] == row["responses_create_params"]
    assert "native" not in sample
    assert "label" not in sample
    assert "tools" not in sample
    assert sample["metadata"]["reward_type"] == "structured_json_schema"


def test_convert_ai2_arc_row_for_pool_promotes_choices_dict_to_mcqa_shape():
    row = {
        "id": "Mercury_SC_415702",
        "question": "Which skin surface will produce the most heat?",
        "choices": {
            "text": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "ai2_arc")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["domain"] == "stem"
    assert sample["question"] == row["question"]
    assert sample["options"] == row["choices"]["text"]
    assert sample["label"] == "A"
    assert "answer" not in sample and "tools" not in sample and "native" not in sample
    assert sample["metadata"]["answer"] == "A"


def test_convert_mmlu_row_for_pool_uses_letter_answer_from_index():
    row = {
        "question": "Find all c in Z_3 such that ...",
        "subject": "abstract_algebra",
        "choices": ["0", "1", "2", "3"],
        "answer": 1,
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "mmlu")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["options"] == row["choices"]
    assert sample["label"] == "B"
    assert "answer" not in sample
    assert sample["metadata"]["reward_type"] == "stem_mcqa"
    assert sample["metadata"]["answer"] == "B"


def test_convert_sciq_row_for_pool_builds_mcqa_options_from_correct_and_distractors():
    row = {
        "question": "What type of organism is used in yogurt?",
        "correct_answer": "mesophilic organisms",
        "distractor1": "protozoa",
        "distractor2": "gymnosperms",
        "distractor3": "viruses",
        "support": "Mesophiles grow best in moderate temperature.",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "sciq")

    assert len(samples) == 1
    sample = samples[0]
    assert len(sample["options"]) == 4
    assert sample["label"] in {"A", "B", "C", "D"}
    assert "answer" not in sample
    assert sample["metadata"]["source_fields"]["support"] == row["support"]
    assert sample["metadata"]["answer"] == sample["label"]


def test_convert_scienceqa_row_for_pool_serializes_image_bytes():
    row = {
        "question": "What is shown?",
        "choices": ["A cat", "A dog"],
        "answer": 1,
        "image": {"bytes": b"png-bytes", "path": "sample.png"},
        "hint": "Look closely",
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "scienceqa")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["label"] == "B"
    assert sample["metadata"]["source_fields"]["image"]["path"] == "sample.png"
    assert isinstance(sample["metadata"]["source_fields"]["image"]["bytes"], str)
    assert "native" not in sample and "answer" not in sample and "tools" not in sample
    assert sample["metadata"]["answer"] == "B"


def test_convert_agieval_row_for_pool_preserves_jsonl_mcqa_shape():
    row = {
        "question": "Find the distance between P and Q.",
        "options": ["(A)1.75km", "(B)2.75km", "(C)3.75km", "(D)4.75km", "(E)5.75km"],
        "label": "C",
        "explanation": "Reasoning",
        "other": {"solution": "Detailed solution"},
    }

    samples = prepare_pool_data.convert_row_for_pool(row, "agieval")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["options"] == row["options"]
    assert sample["label"] == "C"
    assert "answer" not in sample
    assert sample["metadata"]["source_fields"]["explanation"] == "Reasoning"
    assert sample["metadata"]["answer"] == "C"


def test_iter_source_rows_supports_multiple_paths_for_single_dataset(tmp_path: Path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    left.write_text(json.dumps({"id": "l"}) + "\n", encoding="utf-8")
    right.write_text(json.dumps({"id": "r"}) + "\n", encoding="utf-8")

    rows = list(prepare_pool_data._iter_source_rows([left, right], "jsonl"))

    assert rows == [{"id": "l"}, {"id": "r"}]


def test_iter_source_rows_supports_jsonl_directory_reader(tmp_path: Path):
    source_dir = tmp_path / "agieval"
    source_dir.mkdir()
    (source_dir / "b.jsonl").write_text(json.dumps({"label": "B"}) + "\n", encoding="utf-8")
    (source_dir / "a.jsonl").write_text(json.dumps({"label": "A"}) + "\n", encoding="utf-8")

    rows = list(prepare_pool_data._iter_source_rows(source_dir, "jsonl_dir"))

    assert rows == [{"label": "A"}, {"label": "B"}]


def test_build_dataset_maps_aggregated_ai2_arc_key_to_ai2_arc_converter(tmp_path: Path, monkeypatch):
    source_left = tmp_path / "arc_left.jsonl"
    source_right = tmp_path / "arc_right.jsonl"
    output = tmp_path / "ai2_arc.jsonl"
    source_left.write_text(
        json.dumps({"id": "1", "question": "q1", "choices": {"text": ["x", "y"]}, "answerKey": "A"}) + "\n",
        encoding="utf-8",
    )
    source_right.write_text(
        json.dumps({"id": "2", "question": "q2", "choices": {"text": ["x", "y"]}, "answerKey": "B"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setitem(
        prepare_pool_data.DATASET_SPECS,
        "ai2_arc_train",
        {"source": [source_left, source_right], "output": output, "reader": "jsonl"},
    )

    count = prepare_pool_data.build_dataset("ai2_arc_train")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert [row["dataset_name"] for row in rows] == ["ai2_arc", "ai2_arc"]


def test_jsonschemabench_dataset_specs_cover_default_and_named_configs():
    specs = prepare_pool_data.DATASET_SPECS

    assert "jsonschemabench_val" in specs
    assert specs["jsonschemabench_val"]["output"].name == "jsonschemabench_val-00000-of-00001.jsonl"

    assert "jsonschemabench_github_easy_train" in specs
    assert "jsonschemabench_github_easy_val" in specs
    assert "jsonschemabench_github_easy_test" in specs
    assert specs["jsonschemabench_github_easy_train"]["output"].name == "jsonschemabench_Github_easy_train-00000-of-00001.jsonl"


def test_mmlu_auxiliary_train_is_not_in_current_pool_specs():
    assert "mmlu_auxiliary_train" not in prepare_pool_data.DATASET_SPECS
