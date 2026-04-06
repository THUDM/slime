from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "eval_backfill.py"
    spec = importlib.util.spec_from_file_location("eval_backfill_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_backfill = _load_module()


def test_main_routes_bfcl_eval_to_official_runner(monkeypatch, tmp_path: Path):
    called = {"generate_batch": 0, "run_bfcl": 0, "log": None}

    args = SimpleNamespace(
        migrate_full_run=False,
        migrate_eval_history=False,
        sglang_url="http://localhost:30000",
        eval_data=["bfcl_v3_eval:/tmp/bfcl.jsonl"],
        rollout_id=17,
        wandb_run_id="run-id",
        wandb_project="proj",
        wandb_entity="",
        wandb_host="",
        wandb_key="",
        wandb_group="",
        target_wandb_run_id="",
        target_wandb_run_name="",
        runtime_data_dir=str(tmp_path / "runtime"),
        dry_run=False,
        model_path="/tmp/model",
        max_context_len=32768,
        max_tokens=256,
        batch_size=8,
        bfcl_model_name="qwen3-30b-a3b-instruct-2507",
        reward_module="multidomain_shared.reward_func",
    )

    monkeypatch.setattr(eval_backfill, "parse_args", lambda: args)
    monkeypatch.setattr(eval_backfill, "load_reward_func", lambda _module: object())
    monkeypatch.setattr(eval_backfill, "wait_for_sglang", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_backfill, "get_tokenizer", lambda _path: SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt"))
    monkeypatch.setattr(
        eval_backfill,
        "load_eval_data",
        lambda _path: [
            {
                "dataset_name": "bfcl_v3",
                "id": "irrelevance_0",
                "test_category": "irrelevance",
                "prompt": [{"role": "user", "content": "Question"}],
                "tools": [],
                "metadata": {
                    "dataset_name": "bfcl_v3",
                    "domain": "tool",
                    "reward_type": "bfcl_official",
                },
            }
        ],
    )
    monkeypatch.setattr(eval_backfill, "filter_long_prompts", lambda tokenizer, samples, prompts, max_len: (samples, prompts))

    def _generate_batch(**kwargs):
        called["generate_batch"] += 1
        return ["No tool call."]

    monkeypatch.setattr(eval_backfill, "generate_batch", _generate_batch)

    def _unexpected_compute_rewards(*_args, **_kwargs):
        raise AssertionError("generic reward path should not run for BFCL official eval")

    monkeypatch.setattr(eval_backfill, "compute_rewards", _unexpected_compute_rewards)

    def _run_bfcl(eval_samples, outputs, **kwargs):
        called["run_bfcl"] += 1
        assert outputs == ["No tool call."]
        assert kwargs["model_name"] == "qwen3-30b-a3b-instruct-2507"
        return {"overall_accuracy": 1.0, "categories": {"irrelevance": {"accuracy": 1.0, "total_count": 1}}}

    monkeypatch.setattr(
        eval_backfill,
        "_load_bfcl_runner",
        lambda: {
            "DEFAULT_BFCL_MODEL_NAME": "qwen3-30b-a3b-instruct-2507",
            "generate_bfcl_multi_turn_outputs": lambda **kwargs: [],
            "run_bfcl_official_eval": _run_bfcl,
            "summary_to_metrics": lambda eval_name, summary: {f"eval/{eval_name}": summary["overall_accuracy"]},
        },
    )
    monkeypatch.setattr(eval_backfill, "log_to_wandb", lambda **kwargs: called.__setitem__("log", kwargs["metrics"]))

    eval_backfill.main()

    assert called["generate_batch"] == 1
    assert called["run_bfcl"] == 1
    assert called["log"] == {"eval/bfcl_v3_eval": 1.0}
