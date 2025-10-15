import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
