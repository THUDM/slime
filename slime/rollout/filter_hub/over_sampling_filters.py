import torch
from slime.utils.types import Sample


__all__ = ["sort_by_reward_std"]


def sort_by_reward_std(args, samples: list[Sample], **kwargs):
    args.n_samples_per_prompt
    samples_with_std = []
    for i in range(0, len(samples), args.n_samples_per_prompt):
        batch = samples[i : i + args.n_samples_per_prompt]
        rewards = [item.reward for item in batch]
        std = torch.tensor(rewards, dtype=torch.float).std()
        for sample in batch:
            samples_with_std.append((sample, std))
    # python sort is stable, so the order of samples with the same std is preserved
    samples_with_std.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in samples_with_std]
