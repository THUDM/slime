"""
Example reward cap scheduler functions for dynamic reward capping.

These functions compute reward caps for individual data sources based on the current rollout step,
enabling curriculum learning strategies where filtering strictness changes during training.

Each function takes a rollout_step and returns a single float cap for one data source.
Multiple sources can use different functions, constants, or lambdas independently.
"""

import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Per-Source Reward Cap Scheduler Functions
# ============================================================================
# Each function returns a single float cap for one data source
# Use these in config like:
#   data_source_reward_caps:
#     - "examples.curriculum_learning.reward_cap_scheduler.increasing_cap"
#     - "examples.curriculum_learning.reward_cap_scheduler.decreasing_cap"


def increasing_cap(rollout_step: int) -> float:
    """
    Increasing cap: starts strict (0.1) and becomes lenient (0.9) over 1000 steps.
    Use for datasets where you want to gradually accept more samples as training progresses.

    Curriculum: Start by only accepting very difficult samples, gradually accept easier ones.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float cap for this data source
    """
    max_steps = 1000
    progress = min(rollout_step / max_steps, 1.0)
    return 0.1 + 0.8 * progress  # 0.1 -> 0.9


def decreasing_cap(rollout_step: int) -> float:
    """
    Decreasing cap: starts lenient (0.9) and becomes strict (0.1) over 1000 steps.
    Use for datasets where you want to gradually filter out easier samples.

    Curriculum: Start by accepting many samples, gradually become more selective.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float cap for this data source
    """
    max_steps = 1000
    progress = min(rollout_step / max_steps, 1.0)
    return 0.9 - 0.8 * progress  # 0.9 -> 0.1


def exponential_increase(rollout_step: int) -> float:
    """
    Exponentially increasing cap with smooth growth.
    Use for datasets where you want smooth, gradual relaxation of filtering.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float cap for this data source
    """
    decay_rate = 0.003
    return 1.0 - (0.9 * math.exp(-decay_rate * rollout_step) + 0.1)  # 0.0 -> ~0.9


def exponential_decrease(rollout_step: int) -> float:
    """
    Exponentially decreasing cap with smooth decay.
    Use for datasets where you want smooth, gradual tightening of filtering.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float cap for this data source
    """
    decay_rate = 0.003
    return 0.9 * math.exp(-decay_rate * rollout_step) + 0.1  # 0.9 -> ~0.1


def step_function_cap(rollout_step: int) -> float:
    """
    Step function: abrupt changes at specific thresholds.
    Useful for staged curriculum learning.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float cap for this data source
    """
    if rollout_step < 200:
        return 0.2  # Stage 1: Very strict
    elif rollout_step < 500:
        return 0.5  # Stage 2: Moderate
    else:
        return 0.8  # Stage 3: Lenient


def warmup_then_constant(rollout_step: int) -> float:
    """
    Linear warmup for first 500 steps, then constant.
    Useful for stabilizing early training with strict filtering,
    then maintaining consistent quality bar.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float cap for this data source
    """
    warmup_steps = 500
    target_cap = 0.6

    if rollout_step < warmup_steps:
        # Linear increase from 0.1 to target_cap
        return 0.1 + (target_cap - 0.1) * (rollout_step / warmup_steps)
    else:
        return target_cap
