from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


def _load_prepare_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "multidomain_v2" / "prepare_multidomain_v2_data.py"
    spec = importlib.util.spec_from_file_location("prepare_multidomain_v2_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_multidomain_v2 = _load_prepare_module()


def _load_data_module():
    original_ray = sys.modules.get("ray")
    repo_root = str(Path(__file__).resolve().parents[1])
    had_repo_root = repo_root in sys.path
    if not had_repo_root:
        sys.path.insert(0, repo_root)
    sys.modules["ray"] = types.SimpleNamespace(get=lambda value: value)
    try:
        module = importlib.import_module("slime.utils.data")
    finally:
        if not had_repo_root:
            sys.path.remove(repo_root)
        if original_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = original_ray
    return module


slime_utils_data = _load_data_module()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_discover_sources_returns_sorted_jsonl_files(tmp_path: Path):
    pool_root = tmp_path / "pool"
    _write_jsonl(pool_root / "structured" / "train" / "zeta.jsonl", [{"prompt": [], "label": "3", "metadata": {}}])
    _write_jsonl(pool_root / "tool" / "train" / "alpha.jsonl", [{"prompt": [], "label": "1", "metadata": {}}])
    _write_jsonl(pool_root / "tool" / "train" / "legacy_tool_data_train-00000-of-00004.jsonl", [{"prompt": [], "label": "2", "metadata": {}}])
    _write_jsonl(pool_root / "tool" / "eval" / "ignored.jsonl", [{"prompt": [], "label": "4", "metadata": {}}])
    _write_jsonl(pool_root / "math" / "ignored.jsonl", [{"question": "q-ignored", "label": "0"}])
    (pool_root / "code").mkdir(parents=True, exist_ok=True)
    (pool_root / "code" / "ignore.txt").write_text("ignored\n", encoding="utf-8")

    discovered = prepare_multidomain_v2.discover_sources(pool_root)

    assert [path.relative_to(pool_root).as_posix() for path in discovered] == [
        "structured/train/zeta.jsonl",
        "tool/train/alpha.jsonl",
    ]


def test_discover_sources_excludes_hidden_and_appledouble_jsonl_files(tmp_path: Path):
    pool_root = tmp_path / "pool"
    _write_jsonl(pool_root / "tool" / "train" / "alpha.jsonl", [{"prompt": [], "label": "1", "metadata": {}}])
    _write_jsonl(pool_root / "tool" / "train" / ".hidden.jsonl", [{"prompt": [], "label": "2", "metadata": {}}])
    _write_jsonl(pool_root / "tool" / "train" / "nested" / "._alpha.jsonl", [{"prompt": [], "label": "3", "metadata": {}}])

    discovered = prepare_multidomain_v2.discover_sources(pool_root)

    assert [path.relative_to(pool_root).as_posix() for path in discovered] == [
        "tool/train/alpha.jsonl",
    ]


def test_write_dataset_preserves_normalized_rows_without_rewriting_fields(tmp_path: Path):
    pool_root = tmp_path / "pool"
    source_a = pool_root / "tool" / "a.jsonl"
    source_b = pool_root / "structured" / "b.jsonl"
    _write_jsonl(
        source_a,
        [
            {
                "prompt": [{"role": "user", "content": "prompt-a"}],
                "label": "label-a",
                "metadata": {"reward_type": "tool_call"},
            }
        ],
    )
    _write_jsonl(
        source_b,
        [
            {
                "prompt": [{"role": "user", "content": "prompt-b"}],
                "label": "label-b",
                "metadata": {"reward_type": "structured_json_schema"},
            }
        ],
    )

    dest = tmp_path / "multidomain_v2_train.jsonl"
    written = prepare_multidomain_v2.write_dataset(
        [source_a, source_b],
        dest,
        skip_samples=1,
        max_samples=1,
    )

    rows = _read_jsonl(dest)

    assert written == 1
    assert len(rows) == 1
    assert rows[0]["prompt"][0]["role"] == "system"
    assert (
        "structured output assistant" in rows[0]["prompt"][0]["content"].lower()
        or "information extraction assistant" in rows[0]["prompt"][0]["content"].lower()
    )
    assert rows[0]["prompt"][1:] == [{"role": "user", "content": "prompt-b"}]
    assert rows[0]["label"] == "label-b"
    assert rows[0]["metadata"] == {"reward_type": "structured_json_schema"}
    assert rows[0]["tools"] == []


def test_write_dataset_aligns_missing_tools_to_v1_shape(tmp_path: Path):
    pool_root = tmp_path / "pool"
    source = pool_root / "stem" / "stem.jsonl"
    _write_jsonl(
        source,
        [
            {
                "prompt": [{"role": "user", "content": "prompt-stem"}],
                "label": "A",
                "metadata": {"reward_type": "stem_mcqa"},
            }
        ],
    )

    dest = tmp_path / "multidomain_v2_train.jsonl"
    written = prepare_multidomain_v2.write_dataset([source], dest)
    rows = _read_jsonl(dest)

    assert written == 1
    assert len(rows) == 1
    assert rows[0]["prompt"][0]["role"] == "system"
    assert "stem" in rows[0]["prompt"][0]["content"].lower() or "science" in rows[0]["prompt"][0]["content"].lower()
    assert rows[0]["prompt"][1:] == [{"role": "user", "content": "prompt-stem"}]
    assert rows[0]["label"] == "A"
    assert rows[0]["metadata"] == {"reward_type": "stem_mcqa", "answer": "A"}
    assert rows[0]["tools"] == []


def test_resolve_named_datasets_expands_requested_pool_sources(tmp_path: Path):
    pool_root = tmp_path / "pool"
    expected_relpaths = [
        "tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202510.jsonl",
        "tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202601.jsonl",
        "structured/train/jsonschemabench_train-00000-of-00001.jsonl",
        "structured/train/nemotron_structured_outputs_structured_outputs_251027_nano_v3_sdg_json_train.jsonl",
    ]
    for relpath in expected_relpaths:
        _write_jsonl(
            pool_root / relpath,
            [{"prompt": [{"role": "user", "content": relpath}], "label": "", "metadata": {}}],
        )

    resolved = prepare_multidomain_v2.resolve_named_datasets(
        pool_root,
        ["agent", "jsonschemabench", "nemotron_structured_outputs"],
    )

    assert [path.relative_to(pool_root).as_posix() for path in resolved] == expected_relpaths


def test_resolve_nemotron_knowledge_mcqa_expands_all_train_shards(tmp_path: Path):
    pool_root = tmp_path / "pool"
    expected_relpaths = [
        "stem/train/nemotron_knowledge_mcqa_data_train-00000-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00001-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00002-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00003-of-00004.jsonl",
    ]
    for relpath in expected_relpaths:
        _write_jsonl(
            pool_root / relpath,
            [{"prompt": [{"role": "user", "content": relpath}], "label": "", "metadata": {}}],
        )

    resolved = prepare_multidomain_v2.resolve_named_datasets(pool_root, ["nemotron_knowledge_mcqa"])

    assert [path.relative_to(pool_root).as_posix() for path in resolved] == expected_relpaths


def test_read_file_accepts_manifest_of_pool_sources(tmp_path: Path):
    source_a = tmp_path / "tool" / "a.jsonl"
    source_b = tmp_path / "structured" / "b.jsonl"
    _write_jsonl(source_a, [{"prompt": [{"role": "user", "content": "a"}], "label": "A", "metadata": {}}])
    _write_jsonl(source_b, [{"prompt": [{"role": "user", "content": "b"}], "label": "B", "metadata": {}}])

    manifest = tmp_path / "train_pool_sources.list"
    manifest.write_text(f"{source_a}\n{source_b}\n", encoding="utf-8")

    rows = list(slime_utils_data.read_file(str(manifest)))

    assert rows == [
        {"prompt": [{"role": "user", "content": "a"}], "label": "A", "metadata": {}},
        {"prompt": [{"role": "user", "content": "b"}], "label": "B", "metadata": {}},
    ]


def test_align_row_to_v1_shape_uses_instruction_following_prompts_for_ifbench():
    row = {
        "prompt": [{"role": "user", "content": "Follow the rule"}],
        "label": "",
        "metadata": {
            "domain": "structured",
            "dataset_name": "ifbench_test",
            "reward_type": "instruction_following_soft",
        },
        "tools": [],
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["prompt"][0]["role"] == "system"
    assert "structured output assistant" not in aligned["prompt"][0]["content"].lower()
    assert aligned["prompt"][1:] == [{"role": "user", "content": "Follow the rule"}]


def test_align_row_to_v1_shape_materializes_ifbench_eval_metadata_from_top_level_fields():
    row = {
        "dataset_name": "ifbench_test",
        "domain": "structured",
        "record_id": "ifbench-1",
        "prompt": [{"role": "user", "content": "Follow the rule"}],
        "prompt_text": "Follow the rule",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["alpha"], "unused": None}],
        "metadata": {
            "dataset_name": "ifbench_test",
            "domain": "structured",
            "record_id": "ifbench-1",
            "reward_type": "instruction_following_soft",
        },
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "instruction_following_soft"
    assert aligned["metadata"]["instruction_id_list"] == ["keywords:existence"]
    assert aligned["metadata"]["kwargs"] == [{"keywords": ["alpha"], "unused": None}]
    assert aligned["metadata"]["prompt_text"] == "Follow the rule"
    assert aligned["prompt"][0]["role"] == "system"


def test_align_row_to_v1_shape_materializes_clean_stem_row_without_native():
    row = {
        "dataset_name": "gpqa",
        "domain": "stem",
        "record_id": "rec-1",
        "prompt": [{"role": "user", "content": "Question: ..."}],
        "label": "C",
        "question": "Question: ...",
        "options": ["A1", "B1", "C1", "D1"],
        "metadata": {
            "dataset_name": "gpqa",
            "domain": "stem",
            "record_id": "rec-1",
            "reward_type": "stem_mcqa",
        },
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "stem_mcqa"
    assert aligned["metadata"]["answer"] == "C"
    assert aligned["label"] == "C"
    assert aligned["tools"] == []
    assert aligned["prompt"][0]["role"] == "system"


def test_align_row_to_v1_shape_materializes_clean_structured_row_without_native():
    row = {
        "dataset_name": "jsonschemabench",
        "domain": "structured",
        "record_id": "schema-1",
        "prompt": [{"role": "user", "content": "{\"type\":\"object\"}"}],
        "schema": {"type": "object"},
        "metadata": {
            "dataset_name": "jsonschemabench",
            "domain": "structured",
            "record_id": "schema-1",
            "reward_type": "structured_json_schema",
        },
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "structured_json_schema"
    assert aligned["metadata"]["schema"] == {"type": "object"}
    assert aligned["label"] == ""
    assert aligned["tools"] == []
    assert aligned["prompt"][0]["role"] == "system"


def test_align_row_to_v1_shape_materializes_clean_ifeval_row_without_native():
    row = {
        "dataset_name": "ifeval",
        "domain": "structured",
        "record_id": "ifeval-1",
        "prompt": [{"role": "user", "content": "Do the task"}],
        "prompt_text": "Do the task",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["alpha"]}],
        "metadata": {
            "dataset_name": "ifeval",
            "domain": "structured",
            "record_id": "ifeval-1",
            "reward_type": "instruction_following_strict",
        },
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "instruction_following_strict"
    assert aligned["metadata"]["prompt_text"] == "Do the task"
    assert aligned["metadata"]["instruction_id_list"] == ["keywords:existence"]
    assert aligned["metadata"]["kwargs"] == [{"keywords": ["alpha"]}]
    assert aligned["label"] == ""
    assert aligned["tools"] == []
    assert aligned["prompt"][0]["role"] == "system"


def test_align_row_to_v1_shape_materializes_bfcl_eval_metadata_from_top_level_fields():
    row = {
        "dataset_name": "bfcl_v3",
        "domain": "tool",
        "record_id": "irrelevance_0",
        "id": "irrelevance_0",
        "ground_truth": {},
        "subset": "eval",
        "test_category": "irrelevance",
        "language": "python",
        "turns": [[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question"},
        ]],
        "prompt": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question"},
        ],
        "metadata": {
            "dataset_name": "bfcl_v3",
            "domain": "tool",
            "record_id": "irrelevance_0",
        },
        "tools": [{"type": "function", "function": {"name": "weather", "description": "", "parameters": {}}}],
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "bfcl_official"
    assert aligned["metadata"]["official_eval_name"] == "bfcl"
    assert aligned["metadata"]["test_category"] == "irrelevance"
    assert aligned["tools"] == row["tools"]


def test_align_row_to_v1_shape_materializes_train_function_call_family_from_top_level_fields():
    row = {
        "dataset_name": "xlam_function_calling_60k",
        "domain": "tool",
        "record_id": "row-1",
        "supervision_family": "function_call_single",
        "messages": [
            {"role": "user", "content": "check weather", "tool_calls": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                        "type": "function",
                    }
                ],
            },
        ],
        "source_fields": {"extra": {"id": "row-1"}},
        "metadata": {
            "dataset_name": "xlam_function_calling_60k",
            "domain": "tool",
            "record_id": "row-1",
        },
        "tools": [{"type": "function", "function": {"name": "weather", "description": "", "parameters": {}}}],
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "function_call_single"
    assert aligned["metadata"]["supervision_family"] == "function_call_single"
    assert aligned["metadata"]["ground_truth"] == [
        {
            "name": "weather",
            "arguments": {"city": "Paris"},
            "function": {"name": "weather", "arguments": {"city": "Paris"}},
        }
    ]
    assert aligned["metadata"]["source_fields"] == {"extra": {"id": "row-1"}}
    assert aligned["tools"] == row["tools"]
    assert aligned["prompt"][1:] == [{"role": "user", "content": "check weather"}]


def test_align_row_to_v1_shape_is_near_noop_for_final_contract_tool_train_row():
    row = {
        "dataset_name": "xlam_function_calling_60k",
        "domain": "tool",
        "record_id": "row-1",
        "supervision_family": "function_call_single",
        "prompt": [{"role": "user", "content": "check weather"}],
        "messages": [
            {"role": "user", "content": "check weather"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                        "type": "function",
                    }
                ],
            },
        ],
        "metadata": {
            "dataset_name": "xlam_function_calling_60k",
            "domain": "tool",
            "record_id": "row-1",
            "supervision_family": "function_call_single",
            "reward_type": "function_call_single",
            "ground_truth": [
                {
                    "name": "weather",
                    "arguments": {"city": "Paris"},
                    "function": {"name": "weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        "tools": [{"type": "function", "function": {"name": "weather", "description": "", "parameters": {}}}],
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["metadata"]["reward_type"] == "function_call_single"
    assert aligned["metadata"]["ground_truth"] == row["metadata"]["ground_truth"]
    assert aligned["tools"] == row["tools"]
    assert aligned["prompt"][1:] == row["prompt"]


def test_align_row_to_v1_shape_keeps_tools_top_level_only():
    row = {
        "prompt": [{"role": "user", "content": "Use the tool"}],
        "label": "",
        "metadata": {
            "domain": "tool",
            "dataset_name": "toolbench_v1",
            "reward_type": "tool_call_soft",
        },
        "tools": [{"type": "function", "function": {"name": "weather", "description": "", "parameters": {}}}],
    }

    aligned = prepare_multidomain_v2.align_row_to_v1_normalized_shape(row)

    assert aligned["tools"] == row["tools"]
    assert "tools" not in aligned["metadata"]


def test_v2_run_script_is_not_a_v1_wrapper_anymore():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "multidomain_v2"
        / "run_qwen3_30b_a3b_multidomain_v2_3node.sh"
    )
    script_text = script_path.read_text(encoding="utf-8")

    assert "../multidomain_v1/run_qwen3_30b_a3b_multidomain_v1_3node.sh" not in script_text
    assert "TRAIN_DATA_BASENAME" not in script_text


def test_v2_run_script_keeps_bfcl_eval_on_soft_reward():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "multidomain_v2"
        / "run_qwen3_30b_a3b_multidomain_v2_3node.sh"
    )
    script_text = script_path.read_text(encoding="utf-8")

    assert "EVAL_PROMPT_DATA_ARGS+=(bfcl_v3_eval" not in script_text
    assert "EVAL_PROMPT_DATA_ARGS+=(bfcl_multi_turn_eval" not in script_text


def test_v2_run_script_no_longer_routes_tool_trajectory_to_custom_generate():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "multidomain_v2"
        / "run_qwen3_30b_a3b_multidomain_v2_3node.sh"
    )
    script_text = script_path.read_text(encoding="utf-8")

    assert 'GENERATE_MODULE_PATH=' not in script_text
    assert 'tool_trajectory_generate.generate' not in script_text
    assert 'tool_trajectory_reward.reward_func' not in script_text


def test_v2_run_script_no_longer_references_deleted_v1_prepare_helper():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "multidomain_v2"
        / "run_qwen3_30b_a3b_multidomain_v2_3node.sh"
    )
    script_text = script_path.read_text(encoding="utf-8")

    assert "../multidomain_v1/prepare_multidomain_v1_data.py" not in script_text


def test_v2_run_script_no_longer_references_toolbench_eval():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "multidomain_v2"
        / "run_qwen3_30b_a3b_multidomain_v2_3node.sh"
    )
    script_text = script_path.read_text(encoding="utf-8")

    assert "toolbench_v1_benchmark" not in script_text
    assert "toolbench_benchmark_eval" not in script_text
    assert "EVAL_TOOL_TOOLBENCH_BENCHMARK" not in script_text


def test_v1_run_script_has_been_removed_from_examples():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "multidomain_v1"
        / "run_qwen3_30b_a3b_multidomain_v1_3node.sh"
    )

    assert not script_path.exists()
