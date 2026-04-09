"""MOPD eval reward router."""
from __future__ import annotations

import sys
from pathlib import Path

from slime.rollout.rm_hub.math_verify import compute_math_verify_reward
from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout

_SCRIPT_DIR = Path(__file__).resolve().parent
_SLIME_ROOT = _SCRIPT_DIR.parents[1]
if str(_SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLIME_ROOT))

from examples.common.multidomain_shared import reward_func as _shared_reward  # noqa: E402
from slime.rollout.rm_hub.code_verifier import compute_score as _compute_code_score


def _infer_domain(sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("domain") or "").strip().lower()


async def reward_func(args, sample, **kwargs):
    domain = _infer_domain(sample)
    if domain == "math":
        if sample.status == sample.Status.TRUNCATED:
            return 0.0
        return compute_math_verify_reward(sample.response or "", sample.label)
    if domain == "code":
        if sample.status in {sample.Status.TRUNCATED, sample.Status.ABORTED, sample.Status.FAILED}:
            return 0.0
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        try:
            score, _ = _compute_code_score(sample.response or "", metadata, continuous=True, max_partial_cases=10)
            return float(score)
        except Exception:
            return 0.0
    return await _shared_reward(args, sample, **kwargs)


def generate_eval_rollout(args, rollout_id, data_source, evaluation: bool = False):
    original_rm = args.custom_rm_path
    original_pp = args.custom_reward_post_process_path

    try:
        args.custom_rm_path = "examples.MOPD.reward_mopd_eval_router.reward_func"
        args.custom_reward_post_process_path = None
        return default_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        args.custom_rm_path = original_rm
        args.custom_reward_post_process_path = original_pp
