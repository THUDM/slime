"""Minimal reward plumbing for top-k Megatron OPD examples.

Top-k OPD gets its training signal from teacher/student top-k and tail
distributions computed during Megatron training. These helpers keep the rollout
reward path runnable when no task reward is needed.
"""

from __future__ import annotations

import torch

from slime.utils.types import Sample


async def zero_reward_func(args, sample_or_samples: Sample | list[Sample], **kwargs):
    if isinstance(sample_or_samples, list):
        return [0.0 for _ in sample_or_samples]
    return {"score": 0.0}


def zero_reward_post_process(args, samples: list[Sample]):
    rewards = [0.0 for _ in samples]
    return rewards, rewards


def placeholder_advantage_function(args, rollout_data):
    values_sources = (
        rollout_data.get("log_probs"),
        rollout_data.get("rollout_log_probs"),
        rollout_data.get("values"),
    )
    device = next((values[0].device for values in values_sources if values), torch.device("cpu"))

    advantages = [
        torch.zeros(response_length, dtype=torch.float32, device=device)
        for response_length in rollout_data["response_lengths"]
    ]
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [advantage.clone() for advantage in advantages]
