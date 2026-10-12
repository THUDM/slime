"""Example-specific hooks; native slime still computes GRPO/OPD and PPO."""


async def zero_reward(args, sample, **kwargs):
    """Pure OPD has no task reward; teacher scores enter the native advantage."""
    return [0.0] * len(sample) if isinstance(sample, list) else 0.0
