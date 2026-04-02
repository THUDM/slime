from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.MOPD import log_mopd_rollout


def _args(trace_dir: Path):
    return SimpleNamespace(
        use_wandb=False,
        wandb_always_use_train_step=False,
        rollout_batch_size=32,
        n_samples_per_prompt=16,
        global_batch_size=512,
        rollout_num_gpus=16,
        multidomain_v1_trace_dir=str(trace_dir),
        multidomain_v1_trace_max_samples=None,
    )


def _sample(*, reward, domain="math", dataset_name="deepmath"):
    return SimpleNamespace(
        prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
        response="answer",
        label="label",
        reward=reward,
        response_length=12,
        effective_response_length=12,
        status=SimpleNamespace(value="completed"),
        metadata={"domain": domain, "dataset_name": dataset_name, "record_id": "r1"},
    )


def test_mopd_rollout_log_handles_dict_reward_and_bypasses_core_logging(tmp_path: Path):
    args = _args(tmp_path)
    samples = [
        _sample(
            reward={
                "meta_info": {"input_token_logprobs": [[0.0, 1, None]]},
                "completion_tokens": 0,
                "cached_tokens": 0,
            }
        )
    ]

    handled = log_mopd_rollout.log_rollout_data(7, args, samples, {}, 2.0)

    assert handled is True
    trace_path = tmp_path / "rollout_0000007.jsonl"
    row = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert row["reward_type"] == "dict"
    assert row["reward_scalar"] is None
    assert row["reward_has_meta_info"] is True


def test_mopd_rollout_log_preserves_scalar_reward_in_trace(tmp_path: Path):
    args = _args(tmp_path)
    samples = [_sample(reward=0.75, domain="tool", dataset_name="toolbench")]

    log_mopd_rollout.log_rollout_data(3, args, samples, {}, 1.5)

    trace_path = tmp_path / "rollout_0000003.jsonl"
    row = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert row["reward_type"] == "float"
    assert row["reward_scalar"] == 0.75
    assert row["domain"] == "tool"
