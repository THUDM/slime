import asyncio
from typing import Union

import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


async def remote_rm(prompt: str, response: str, label: str = None, **kwargs):
    assert "remote_url" in kwargs, "remote_url is required for remote RM"
    remote_url = kwargs["remote_url"]
    payload = {
        "prompt": prompt,
        "response": response,
        "label": label,
    }
    session_kwargs = {}
    if "proxy" in kwargs:
        session_kwargs["proxy"] = kwargs["proxy"]
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(remote_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    rm_type = args.rm_type
    prompt = sample.prompt
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response)
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(prompt, response, label, **kwargs)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    else:
        raise NotImplementedError(f"Rule-based RM for {type} is not implemented.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[Union[int, float]]:
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)

    rm_type = args.rm_type
    prompts = [sample.prompt for sample in samples]
    responses = [sample.response for sample in samples]
    labels = [sample.label for sample in samples]
    if labels is None:
        labels = [None] * len(prompts)
    tasks = [
        async_rm(rm_type, prompt, response, label, **kwargs)
        for prompt, response, label in zip(prompts, responses, labels)
    ]
    rewards = await asyncio.gather(*tasks)
    return rewards
