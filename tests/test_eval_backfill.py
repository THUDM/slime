from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_eval_backfill_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "multidomain_v2" / "eval_backfill.py"
    spec = importlib.util.spec_from_file_location("eval_backfill_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_backfill = _load_eval_backfill_module()


def test_load_eval_data_rewrites_stale_bfcl_strict_reward_type(tmp_path: Path):
    path = tmp_path / "bfcl_eval.jsonl"
    rows = [
        {
            "prompt": [{"role": "user", "content": "call a tool"}],
            "label": "",
            "tools": [],
            "metadata": {
                "dataset_name": "bfcl_v3",
                "reward_type": "tool_call_strict",
                "ground_truth": [{"name": "foo", "arguments": {"x": [1]}}],
            },
        },
        {
            "prompt": [{"role": "user", "content": "other"}],
            "label": "",
            "tools": [],
            "metadata": {
                "dataset_name": "ifeval",
                "reward_type": "instruction_following_strict",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = eval_backfill.load_eval_data(str(path))

    assert loaded[0]["metadata"]["reward_type"] == "tool_call_soft"
    assert loaded[1]["metadata"]["reward_type"] == "instruction_following_strict"


def test_merge_eval_history_rows_keeps_latest_values_per_eval_step():
    rows = [
        {"_step": 103, "eval/step": 19, "eval/bfcl_v3_eval": 0.0, "eval/mmlu_pro_eval": 0.76},
        {"_step": 206, "eval/step": 39, "eval/bfcl_v3_eval": 0.0004, "eval/mmlu_pro_eval": 0.77},
        {"_step": 1595, "eval/step": 19, "eval/bfcl_v3_eval": 0.4148, "eval/mmlu_pro_eval": 0.7676},
        {"_step": 1678, "eval/step": 39, "eval/bfcl_v3_eval": 0.4037},
        {"_step": 1768, "eval/step": 59, "eval/bfcl_v3_eval": 0.4227, "eval/ifeval_eval": 0.3438},
    ]

    merged = eval_backfill.merge_eval_history_rows(rows)

    assert merged == [
        {"eval/step": 19, "eval/bfcl_v3_eval": 0.4148, "eval/mmlu_pro_eval": 0.7676},
        {"eval/step": 39, "eval/bfcl_v3_eval": 0.4037, "eval/mmlu_pro_eval": 0.77},
        {"eval/step": 59, "eval/bfcl_v3_eval": 0.4227, "eval/ifeval_eval": 0.3438},
    ]
