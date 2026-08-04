import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_reward_nonzero_std_with_fallback"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonzero_std_with_fallback(args, samples: list[Sample], **kwargs):
    """Prefer non-zero-std groups without triggering another sampling round."""

    output = check_reward_nonzero_std(args, samples, **kwargs)
    output.keep_when_insufficient = True
    return output
