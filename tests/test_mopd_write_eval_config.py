from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_write_eval_config_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "MOPD" / "write_eval_config.py"
    spec = importlib.util.spec_from_file_location("write_eval_config_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


write_eval_config = _load_write_eval_config_module()


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
