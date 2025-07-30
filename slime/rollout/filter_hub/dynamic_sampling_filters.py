import torch
from slime.utils.types import Sample


__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
