from collections import defaultdict

import torch


def normalize_rewards_by_group(
    rewards: list[float],
    group_indices: list[int | None],
    *,
    normalize_std: bool,
    fallback_group_size: int | None = None,
) -> list[float]:
    """Normalize rewards within the sample group that produced each response.

    ``fallback_group_size`` preserves fixed-size custom rollouts that predate
    ``Sample.group_index``. Uneven groups must provide explicit identities.
    """
    if len(rewards) != len(group_indices):
        raise ValueError(f"rewards and group_indices must have the same length, got {len(rewards)} and {len(group_indices)}")

    if group_indices and all(group_index is None for group_index in group_indices):
        if fallback_group_size is None or fallback_group_size <= 0 or len(group_indices) % fallback_group_size != 0:
            raise ValueError("group_index is required when reward groups are not uniformly sized")
        group_indices = [position // fallback_group_size for position in range(len(group_indices))]

    positions_by_group: dict[int, list[int]] = defaultdict(list)
    for position, group_index in enumerate(group_indices):
        if group_index is None:
            raise ValueError(f"group_index is required for reward normalization, but sample at position {position} has none")
        positions_by_group[group_index].append(position)

    reward_tensor = torch.tensor(rewards, dtype=torch.float)
    normalized_rewards = torch.empty_like(reward_tensor)
    for positions in positions_by_group.values():
        group_rewards = reward_tensor[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if normalize_std and len(positions) > 1:
            group_rewards = group_rewards / (group_rewards.std() + 1e-6)
        normalized_rewards[positions] = group_rewards

    return normalized_rewards.tolist()
