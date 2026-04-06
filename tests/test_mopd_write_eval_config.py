from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import yaml

def _load_write_eval_config_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "MOPD" / "write_eval_config.py"
    spec = importlib.util.spec_from_file_location("write_eval_config_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


write_eval_config = _load_write_eval_config_module()


def test_infer_domain_from_pool_rel():
    assert write_eval_config._infer_domain_from_pool_rel("tool/eval/bfcl_v3.jsonl") == "tool"
    assert write_eval_config._infer_domain_from_pool_rel("structured/eval/ifeval.jsonl") == "structured"


def test_preprocess_eval_jsonl_uses_instruction_following_prompt_for_ifbench(tmp_path: Path):
    src = tmp_path / "ifbench.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "Do the task"}],
                "label": "",
                "metadata": {
                    "domain": "structured",
                    "dataset_name": "ifbench_test",
                    "reward_type": "instruction_following_soft",
                },
                "tools": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    count = write_eval_config._preprocess_eval_jsonl(src, dst, "structured")

    row = json.loads(dst.read_text(encoding="utf-8").strip())
    assert count == 1
    assert row["prompt"][0]["role"] == "system"
    assert "structured output assistant" not in row["prompt"][0]["content"].lower()
    assert row["prompt"][1:] == [{"role": "user", "content": "Do the task"}]


def test_preprocess_eval_jsonl_keeps_structured_prompt_for_json_schema(tmp_path: Path):
    src = tmp_path / "jsonschema.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "Return valid JSON"}],
                "label": "",
                "metadata": {
                    "domain": "structured",
                    "dataset_name": "jsonschemabench",
                    "reward_type": "structured_json_schema",
                },
                "tools": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_eval_config._preprocess_eval_jsonl(src, dst, "structured")

    row = json.loads(dst.read_text(encoding="utf-8").strip())
    assert row["prompt"][0]["role"] == "system"
    assert (
        "structured output assistant" in row["prompt"][0]["content"].lower()
        or "information extraction assistant" in row["prompt"][0]["content"].lower()
    )


def test_preprocess_eval_jsonl_materializes_top_level_ifbench_payload(tmp_path: Path):
    src = tmp_path / "ifbench.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        json.dumps(
            {
                "dataset_name": "ifbench_test",
                "domain": "structured",
                "record_id": "ifbench-1",
                "prompt": [{"role": "user", "content": "Do the task"}],
                "prompt_text": "Do the task",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["alpha"]}],
                "metadata": {
                    "dataset_name": "ifbench_test",
                    "domain": "structured",
                    "record_id": "ifbench-1",
                    "reward_type": "instruction_following_soft",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_eval_config._preprocess_eval_jsonl(src, dst, "structured")

    row = json.loads(dst.read_text(encoding="utf-8").strip())
    assert row["metadata"]["reward_type"] == "instruction_following_soft"
    assert row["metadata"]["instruction_id_list"] == ["keywords:existence"]


def test_build_math_eval_row_converts_question_to_runtime_prompt():
    row = write_eval_config._build_math_eval_row(
        {
            "id": 7,
            "question": "What is 1+1?",
            "label": "2",
            "metadata": {"dataset_name": "aime24", "domain": "math"},
        }
    )

    assert row is not None
    assert row["id"] == 7
    assert row["label"] == "2"
    assert row["metadata"]["domain"] == "math"
    assert row["metadata"]["dataset_key"] == "aime24"
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1] == {"role": "user", "content": "What is 1+1?"}


def test_build_code_eval_row_converts_question_to_runtime_prompt():
    row = write_eval_config._build_code_eval_row(
        {
            "id": "code-1",
            "question": "Write a function that returns 42.",
            "prompt_mode": "function_test",
            "label": "",
            "metadata": {"dataset_name": "humanevalplus"},
        }
    )

    assert row is not None
    assert row["id"] == "code-1"
    assert row["metadata"]["domain"] == "code"
    assert row["metadata"]["dataset_key"] == "humanevalplus"
    assert row["prompt"][0]["role"] == "system"
    assert "Python 3" in row["prompt"][0]["content"]
    assert row["prompt"][1]["role"] == "user"
    assert "Write a function that returns 42." in row["prompt"][1]["content"]


def test_main_ignores_official_only_datasets_when_writing_generic_eval_config(
    tmp_path: Path,
    monkeypatch,
):
    pool_root = tmp_path / "pool"
    ifeval_src = pool_root / "structured" / "eval" / "ifeval_ifeval_input_data.jsonl"
    bfcl_src = pool_root / "tool" / "eval" / "bfcl_v3_train-00000-of-00001.jsonl"
    ifeval_src.parent.mkdir(parents=True, exist_ok=True)
    bfcl_src.parent.mkdir(parents=True, exist_ok=True)

    ifeval_src.write_text(
        json.dumps(
            {
                "dataset_name": "ifeval",
                "domain": "structured",
                "record_id": "ifeval-1",
                "prompt": [{"role": "user", "content": "Follow the instruction"}],
                "prompt_text": "Follow the instruction",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["alpha"]}],
                "metadata": {"dataset_name": "ifeval", "domain": "structured", "record_id": "ifeval-1"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    bfcl_src.write_text(
        json.dumps(
            {
                "dataset_name": "bfcl_v3",
                "domain": "tool",
                "record_id": "bfcl-1",
                "id": "bfcl-1",
                "test_category": "simple",
                "ground_truth": [],
                "prompt": [{"role": "user", "content": "Use tools if needed"}],
                "metadata": {"dataset_name": "bfcl_v3", "domain": "tool", "record_id": "bfcl-1"},
                "tools": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(write_eval_config, "EVAL_DATASETS", [("ifeval", "structured/eval/ifeval_ifeval_input_data.jsonl", 1)])
    output = tmp_path / "config" / "eval.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_eval_config.py",
            "--pool-root",
            str(pool_root),
            "--output",
            str(output),
        ],
    )

    write_eval_config.main()

    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert [dataset["name"] for dataset in config["eval"]["datasets"]] == ["ifeval"]
    assert not output.with_name(output.stem + ".official_benchmarks.json").exists()


def test_main_writes_official_manifest_when_requested(tmp_path: Path, monkeypatch):
    pool_root = tmp_path / "pool"
    ifeval_src = pool_root / "structured" / "eval" / "ifeval_ifeval_input_data.jsonl"
    bfcl_src = pool_root / "tool" / "eval" / "bfcl_v3_train-00000-of-00001.jsonl"
    bfcl_multi_turn_src = pool_root / "tool" / "eval" / "bfcl_v3_multi_turn_base_train-00000-of-00001.jsonl"
    ifeval_src.parent.mkdir(parents=True, exist_ok=True)
    bfcl_src.parent.mkdir(parents=True, exist_ok=True)
    bfcl_multi_turn_src.parent.mkdir(parents=True, exist_ok=True)

    ifeval_src.write_text(
        json.dumps(
            {
                "dataset_name": "ifeval",
                "domain": "structured",
                "record_id": "ifeval-1",
                "prompt": [{"role": "user", "content": "Follow the instruction"}],
                "prompt_text": "Follow the instruction",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["alpha"]}],
                "metadata": {"dataset_name": "ifeval", "domain": "structured", "record_id": "ifeval-1"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    bfcl_src.write_text(
        json.dumps(
            {
                "dataset_name": "bfcl_v3",
                "domain": "tool",
                "record_id": "bfcl-1",
                "id": "bfcl-1",
                "test_category": "simple",
                "ground_truth": [],
                "prompt": [{"role": "user", "content": "Use tools if needed"}],
                "metadata": {"dataset_name": "bfcl_v3", "domain": "tool", "record_id": "bfcl-1"},
                "tools": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    bfcl_multi_turn_src.write_text(
        json.dumps(
            {
                "dataset_name": "bfcl_v3_multi_turn_base",
                "domain": "tool",
                "record_id": "bfcl-multi-1",
                "id": "bfcl-multi-1",
                "test_category": "multi_turn_base",
                "ground_truth": [{"name": "search", "arguments": {}}],
                "prompt": [{"role": "user", "content": "Search then summarize"}],
                "metadata": {"dataset_name": "bfcl_v3_multi_turn_base", "domain": "tool", "record_id": "bfcl-multi-1"},
                "tools": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(write_eval_config, "EVAL_DATASETS", [("ifeval", "structured/eval/ifeval_ifeval_input_data.jsonl", 1)])
    output = tmp_path / "config" / "eval.yaml"
    manifest = tmp_path / "config" / "official.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_eval_config.py",
            "--pool-root",
            str(pool_root),
            "--output",
            str(output),
            "--official-manifest-output",
            str(manifest),
        ],
    )

    write_eval_config.main()

    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert [dataset["name"] for dataset in config["eval"]["datasets"]] == ["ifeval"]
    assert [dataset["name"] for dataset in payload["datasets"]] == ["bfcl_v3", "bfcl_v3_multi_turn_base"]
    for dataset in payload["datasets"]:
        path = Path(dataset["path"])
        assert path.exists()
        row = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
        assert row["metadata"]["reward_type"] == "bfcl_official"
