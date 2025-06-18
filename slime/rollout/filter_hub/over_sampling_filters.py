import torch
from slime.utils.types import Samples


__all__ = ["sort_by_reward_std"]


def sort_by_reward_std(args, samples: list[Samples], **kwargs):
    args.n_samples_per_prompt
    samples_with_std = []
    for i in range(0, len(samples), args.n_samples_per_prompt):
        batch = samples[i : i + args.n_samples_per_prompt]
        rewards = [item[3] for item in batch]
        std = torch.tensor(rewards, dtype=torch.float).std()
        for j in range(args.n_samples_per_prompt):
            samples_with_std.append(batch[i + j], torch.tensor(rewards, std))
    samples_with_std.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in samples_with_std]
