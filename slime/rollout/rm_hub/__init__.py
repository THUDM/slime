import asyncio
import random

import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


def _validate_dict_reward(args, reward):
    """Validate that reward_key is set when RM returns a dict."""
    if isinstance(reward, dict):
        reward_key = getattr(args, "reward_key", None)
        eval_reward_key = getattr(args, "eval_reward_key", None)
        if not reward_key and not eval_reward_key:
            available_keys = list(reward.keys())
            raise ValueError(
                f"RM returned a dict with keys {available_keys}, but neither 'reward_key' nor 'eval_reward_key' is set. "
                f"Please specify --reward_key or --eval_reward_key (e.g., --reward_key score) to extract the reward value."
            )


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        reward = await rm_function(args, sample, **kwargs)
        _validate_dict_reward(args, reward)
        return reward

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    reward = None
    if rm_type == "remote_rm":
        reward = await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        reward = get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        reward = compute_score_dapo(response, label)
    elif rm_type == "math":
        reward = 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        reward = f1_score(response, label)[0]
    elif rm_type == "gpqa":
        reward = compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        reward = compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        reward = random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")

    _validate_dict_reward(args, reward)
    return reward


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float | dict]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        rewards = await rm_function(args, samples, **kwargs)
        # Validate for custom RM (async_rm handles validation for built-in RMs)
        if rewards:
            _validate_dict_reward(args, rewards[0])
        return rewards

    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    # Note: validation is already done in async_rm for each sample
    return rewards
