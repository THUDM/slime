"""MOPD eval reward router."""
from __future__ import annotations

import sys
from pathlib import Path

from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from reward_code_execution import reward_func as _code_reward  # noqa: E402
from reward_deepmath_mathverify import reward_func as _math_reward  # noqa: E402
from multidomain_shared import reward_func as _shared_reward  # noqa: E402


def _infer_domain(sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("domain") or "").strip().lower()


async def reward_func(args, sample, **kwargs):
    domain = _infer_domain(sample)
    if domain == "math":
        return await _math_reward(args, sample, **kwargs)
    if domain == "code":
        return await _code_reward(args, sample, **kwargs)
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
