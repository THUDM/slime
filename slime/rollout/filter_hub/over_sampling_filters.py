import torch
from slime.utils.types import Sample


__all__ = ["sort_by_reward_std"]


def sort_by_reward_std(args, samples: list[list[Sample]], **kwargs) -> list[list[Sample]]:
    samples_with_std = []
    for group in samples:
        rewards = [item.reward for item in group]
        std = torch.tensor(rewards, dtype=torch.float).std()
        samples_with_std.append((group, std))
    # python sort is stable, so the order of samples with the same std is preserved
    samples_with_std.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in samples_with_std]
