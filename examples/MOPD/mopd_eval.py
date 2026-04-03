"""MOPD eval wrapper — swaps reward function to task-specific verifiers during eval."""
from __future__ import annotations

from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout


def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    original_rm = args.custom_rm_path
    original_pp = args.custom_reward_post_process_path

    try:
        args.custom_rm_path = "examples.MOPD.reward_mopd_eval_router.reward_func"
        args.custom_reward_post_process_path = None
        return default_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        args.custom_rm_path = original_rm
        args.custom_reward_post_process_path = original_pp
