from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
import aiohttp
import torch


async def reward_func(args, sample, **kwargs):
    payload = {
        "text": sample.prompt + sample.response,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            'skip_special_tokens': False,

        },
        "return_logprob": True,
        "logprob_start_len": 0
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()




def post_process_rewards(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    teacher_log_probs = [torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32) for reward in rewards]
    teacher_token_ids = [torch.tensor([item[1] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.int32) for reward in rewards]

    for sample, t_log_probs, t_token_ids in zip(samples, teacher_log_probs, teacher_token_ids):
        sample.teacher_log_probs = t_log_probs
        sample.teacher_token_ids = t_token_ids

    return teacher_log_probs, teacher_token_ids