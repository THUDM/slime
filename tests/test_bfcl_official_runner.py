from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "bfcl_official_runner.py"
    spec = importlib.util.spec_from_file_location("bfcl_official_runner_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bfcl_official_runner = _load_module()


def test_build_bfcl_result_entries_uses_top_level_id_for_single_turn():
    rows = [
        {
            "dataset_name": "bfcl_v3",
            "id": "irrelevance_0",
            "test_category": "irrelevance",
            "prompt": [{"role": "user", "content": "Question"}],
        }
    ]
    entries = bfcl_official_runner.build_bfcl_result_entries(
        rows,
        ["I will not call a tool."],
        backend={"load_dataset_entry": lambda *args, **kwargs: []},
    )
    assert entries == [{"id": "irrelevance_0", "result": "I will not call a tool."}]


def test_build_bfcl_result_entries_assigns_multi_turn_ids_from_official_prompt_order():
    rows = [
        {
            "dataset_name": "bfcl_v3_multi_turn_base",
            "ground_truth": ["call_a", "call_b"],
            "prompt": [{"role": "user", "content": "Task A"}],
        },
        {
            "dataset_name": "bfcl_v3_multi_turn_base",
            "ground_truth": ["call_c"],
            "prompt": [{"role": "user", "content": "Task B"}],
        },
    ]
    backend = {
        "load_dataset_entry": lambda *args, **kwargs: [
            {"id": "multi_turn_base_0"},
            {"id": "multi_turn_base_1"},
        ]
    }
    entries = bfcl_official_runner.build_bfcl_result_entries(
        rows,
        [["step-1", "step-2"], ["step-3"]],
        backend=backend,
    )
    assert entries == [
        {"id": "multi_turn_base_0", "result": [["step-1"], ["step-2"]]},
        {"id": "multi_turn_base_1", "result": [["step-3"]]},
    ]


def test_run_bfcl_official_eval_writes_entries_and_reads_score_headers(tmp_path: Path):
    written: list[tuple[list[dict], Path]] = []
    invoked: list[tuple[list[str], list[str], Path, Path, bool]] = []

    class _FakeHandler:
        def write(self, entries, result_dir, update_mode=False):
            written.append((entries, result_dir))

    def _fake_runner(model_names, test_categories, result_dir, score_dir, allow_missing=False):
        invoked.append((model_names, test_categories, result_dir, score_dir, allow_missing))
        score_path = score_dir / model_names[0] / "non_live"
        score_path.mkdir(parents=True, exist_ok=True)
        (score_path / "BFCL_v4_irrelevance_score.json").write_text(
            json.dumps({"accuracy": 1.0, "correct_count": 1, "total_count": 1}) + "\n",
            encoding="utf-8",
        )

    backend = {
        "VERSION_PREFIX": "BFCL_v4",
        "MODEL_CONFIG_MAPPING": {
            "qwen3-30b-a3b-instruct-2507": type(
                    "_Cfg",
                    (),
                    {
                        "model_name": "qwen3-30b-a3b-instruct-2507",
                        "model_handler": staticmethod(lambda **kwargs: _FakeHandler()),
                        "is_fc_model": False,
                    },
                )()
            },
        "load_dataset_entry": lambda *args, **kwargs: [],
        "runner": _fake_runner,
    }

    original_loader = bfcl_official_runner._load_bfcl_backend
    bfcl_official_runner._load_bfcl_backend = lambda: backend
    try:
        summary = bfcl_official_runner.run_bfcl_official_eval(
            [
                {
                    "dataset_name": "bfcl_v3",
                    "id": "irrelevance_0",
                    "test_category": "irrelevance",
                    "prompt": [{"role": "user", "content": "Question"}],
                }
            ],
            ["No tool call."],
            result_dir=tmp_path / "result",
            score_dir=tmp_path / "score",
        )
    finally:
        bfcl_official_runner._load_bfcl_backend = original_loader

    assert written == [
        ([{"id": "irrelevance_0", "result": "No tool call."}], tmp_path / "result"),
    ]
    assert invoked == [
        (
            ["qwen3-30b-a3b-instruct-2507"],
            ["irrelevance"],
            tmp_path / "result",
            tmp_path / "score",
            True,
        )
    ]
    assert summary["overall_accuracy"] == 1.0
    assert summary["categories"]["irrelevance"]["accuracy"] == 1.0
