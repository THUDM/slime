"""
GRPO Generator for HTTP Rollout Buffer

This generator supports on-policy and off-policy GRPO training using
the HTTP buffer server architecture.

Task Type: grpo
Compatible with: Standard GRPO/PPO training pipeline

Date: 2025-12-10
"""

TASK_TYPE = "grpo"


def is_valid_group(group_data, min_valid_group_size, task_type):
    """
    Check if a group of samples is valid for GRPO.

    Args:
        group_data: Tuple of (instance_id, samples)
        min_valid_group_size: Minimum number of samples (n_samples_per_prompt)
        task_type: Task type (should be "grpo")

    Returns:
        True if group is valid
    """
    instance_id, samples = group_data

    # For GRPO, we need exactly min_valid_group_size samples per group
    return len(samples) == min_valid_group_size


def transform_group(group_data, task_type):
    """
    Transform a group of GRPO samples.

    Currently just returns the samples as-is, but can be customized for:
    - Filtering low-quality samples
    - Reward normalization
    - Advantage computation

    Args:
        group_data: Tuple of (instance_id, samples)
        task_type: Task type (should be "grpo")

    Returns:
        Transformed group data
    """
    instance_id, samples = group_data

    # For GRPO, we typically want to keep all samples in a group
    # (they represent different responses to the same prompt)
    return (instance_id, samples)


def get_group_data_meta_info(temp_data):
    """
    Get metadata about buffered GRPO data.

    Args:
        temp_data: Dict mapping instance_id to list of samples

    Returns:
        Dictionary with statistics
    """
    if not temp_data:
        return {
            "total_samples": 0,
            "num_groups": 0,
            "avg_group_size": 0,
            "avg_reward": 0,
            "min_policy_version": None,
            "max_policy_version": None,
        }

    meta_info = {
        "total_samples": 0,
        "num_groups": len(temp_data)
    }

    all_rewards = []
    all_policy_versions = []

    for instance_id, samples in temp_data.items():
        group_size = len(samples)
        meta_info["total_samples"] += group_size

        for sample in samples:
            # Collect rewards
            if "reward" in sample:
                all_rewards.append(sample["reward"])

            # Collect policy versions (for off-policy tracking)
            if "policy_version" in sample:
                all_policy_versions.append(sample["policy_version"])

    # Calculate statistics
    meta_info["avg_group_size"] = meta_info["total_samples"] / meta_info["num_groups"] if meta_info["num_groups"] > 0 else 0

    if all_rewards:
        meta_info["avg_reward"] = sum(all_rewards) / len(all_rewards)
        meta_info["min_reward"] = min(all_rewards)
        meta_info["max_reward"] = max(all_rewards)
    else:
        meta_info["avg_reward"] = 0
        meta_info["min_reward"] = 0
        meta_info["max_reward"] = 0

    if all_policy_versions:
        meta_info["min_policy_version"] = min(all_policy_versions)
        meta_info["max_policy_version"] = max(all_policy_versions)
        meta_info["avg_policy_version"] = sum(all_policy_versions) / len(all_policy_versions)
    else:
        meta_info["min_policy_version"] = None
        meta_info["max_policy_version"] = None

    return meta_info


def run_rollout(data: dict):
    """
    Run rollout generation for GRPO.

    This is a placeholder for async generation. In the standard GRPO pipeline,
    rollout generation happens in the RolloutManager, not in the buffer server.

    The buffer server is used only for storage and retrieval.

    Args:
        data: Configuration dictionary containing:
            - task_type: "grpo"
            - num_repeat_per_sample: n_samples_per_prompt
            - ... other GRPO config

    Notes:
        For GRPO, the actual generation happens via:
        1. RolloutManager.generate() calls the rollout function
        2. Results are sent to HTTP buffer via add_samples()
        3. Buffer stores them grouped by instance_id
        4. Training fetches from buffer via get_samples()
    """
    print(f"[GRPO Generator] Rollout configuration received:")
    print(f"  Task type: {data.get('task_type', 'grpo')}")
    print(f"  Samples per prompt: {data.get('num_repeat_per_sample', 'N/A')}")

    print("[GRPO Generator] Note: For standard GRPO, generation happens in RolloutManager.")
    print("[GRPO Generator] The buffer server is used only for storage/retrieval.")
    print("[GRPO Generator] To enable async generation, implement custom generation logic here.")
