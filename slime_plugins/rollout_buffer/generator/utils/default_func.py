"""
Default functions for the Rollout Buffer System

This module contains the default implementations of data processing functions
that are used when generators don't provide their own custom implementations.
"""

import copy
from typing import Any, Dict, List, Tuple


def is_valid_reward(reward: float) -> bool:
    """
    Check if a reward value is valid.

    Args:
        reward: The reward value to check

    Returns:
        bool: True if reward is valid (between 0 and 1), False otherwise
    """
    return 1 >= reward >= 0


def default_normalize_group_data(group: Tuple[str, List[Dict[str, Any]]], epsilon=1e-8, algo="grpo"):
    """
    Default normalize rewards in a group using z-score normalization.
    If all rewards are 0 -> skip normalization.
    If std is very small -> set normalized reward to 0.

    Args:
        group: (instance_id, [sample_dicts])
        epsilon: Numerical stability parameter
        algo: Algorithm type, only "grpo" is supported for now

    Returns:
        (instance_id, normalized_data) tuple
    """
    assert algo == "grpo", "Only 'grpo' is supported for now."

    instance_id = group[0]
    data = group[1]
    rewards = [item["reward"] for item in data]
    valid_rewards = [r for r in rewards if is_valid_reward(r)]

    if set(valid_rewards) == {0}:
        print(f"[Info] All rewards zero in group {instance_id}, skipping normalization.")
        normalized_rewards = rewards
    else:
        mean_reward = sum(valid_rewards) / len(valid_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in valid_rewards) / len(valid_rewards)) ** 0.5

        if std_reward < epsilon:
            print(f"[Info] Zero variance in group {instance_id} (non-zero constant rewards), setting all to 0.")
            normalized_rewards = [0.0 if is_valid_reward(r) else r for r in rewards]
        else:
            normalized_rewards = [
                (r - mean_reward) / (std_reward + epsilon) if is_valid_reward(r) else r for r in rewards
            ]

    for i, item in enumerate(data):
        item["reward"] = normalized_rewards[i]
        item["raw_reward"] = rewards[i]

    return (instance_id, data)


def default_pad_group_data(batch, group_size):
    """
    Default padding strategy for group data.
    Input batch: (instance_id, [data_1, ..., data_n])
    We multiply the normalized reward by group_size / valid_size to keep the reward range

    Args:
        batch: (instance_id, data) tuple
        group_size: Target group size

    Returns:
        (instance_id, padded_data) tuple
    """
    instance_id = batch[0]
    data = batch[1]

    # to ensure the padding is equal to dummy padding
    for item in data:
        item["reward"] = item["reward"] * group_size / len(data)

    pad_count = group_size - len(data)

    assert pad_count <= len(data), "pad_count should be less than or equal to the length of data"

    if pad_count > 0:
        print(f"padding {pad_count} items")
        data = data + copy.deepcopy(data[:pad_count])
        for i in range(pad_count):
            data[i]["reward"] /= 2
            data[-(i + 1)]["reward"] /= 2

    return (instance_id, data)


def default_is_valid_group(group_data, min_valid_group_size, task_type):
    """
    Default implementation for checking if a group is valid and finished.

    Logic:
    - finished groups are a superset of valid groups
    - all valid groups are finished
    - some finished groups may not be valid (discarded due to quality issues)

    Args:
        group_data: Tuple of (instance_id, items)
        min_valid_group_size: Minimum required group size
        task_type: Task type for task-specific validation

    Returns:
        tuple: (is_valid, is_finished)
    """
    instance_id, items = group_data

    group_size = len(items)
    reward_list = [item["reward"] for item in items]

    # A group is finished if it has reached the minimum size
    is_finished = group_size >= min_valid_group_size

    has_reward_diversity = len(set(reward_list)) > 1

    is_valid = is_finished and has_reward_diversity

    return is_valid, is_finished


def default_get_group_data_meta_info(
    temp_data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Default implementation for getting meta information about the temporary data
    collected between get_batch calls.
    """
    if not temp_data:
        return {
            "total_samples": 0,
            "num_groups": 0,
            "avg_group_size": 0,
            "avg_reward": 0,
            "reward_std": 0,
            "reward_min": 0,
            "reward_max": 0,
        }

    meta_info = {"total_samples": 0, "num_groups": len(temp_data)}

    all_rewards = []
    all_raw_rewards = []
    # Calculate per-group statistics
    for instance_id, samples in temp_data.items():
        group_size = len(samples)
        group_rewards = [s["reward"] for s in samples]  # Calculate group reward standard deviation
        meta_info["total_samples"] += group_size
        all_rewards.extend(group_rewards)
    # Calculate global statistics
    meta_info["avg_group_size"] = meta_info["total_samples"] / meta_info["num_groups"]

    if all_rewards:
        meta_info["avg_reward"] = sum(all_rewards) / len(all_rewards)
        meta_info["reward_min"] = min(all_rewards)
        meta_info["reward_max"] = max(all_rewards)
        # Calculate global reward standard deviation
        squared_diff_sum = sum((r - meta_info["avg_reward"]) ** 2 for r in all_rewards)
        meta_info["reward_std"] = (squared_diff_sum / (len(all_rewards) - 1)) ** 0.5 if len(all_rewards) > 1 else 0
    else:
        meta_info["avg_reward"] = 0
        meta_info["reward_std"] = 0
        meta_info["reward_min"] = 0
        meta_info["reward_max"] = 0
    return meta_info


def default_filter_item(item: dict, task_type: str) -> bool:
    """
    Default function to filter individual items before normalization.
    Returns True if the item is valid, False otherwise.

    Args:
        item (dict): Single data item to validate
        task_type (str): Type of task for task-specific validation

    Returns:
        bool: True if item is valid, False otherwise
    """
    # Basic validation that all items should have
    required_fields = {"instance_id", "reward", "messages"}
    if not all(field in item for field in required_fields):
        return False

    # Validate reward is a number
    if not isinstance(item["reward"], (int, float)):
        return False
    if item["reward"] < 0 or item["reward"] > 1:
        return False

    # Validate messages is a list
    if not isinstance(item["messages"], list):
        return False

    # Validate messages is a list of dicts
    return True
