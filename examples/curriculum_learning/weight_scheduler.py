import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Per-Source Weight Functions
# ============================================================================
# Each function returns a single float for one data source
# Use these in config like:
#   data_source_weights:
#     - "examples.curriculum_learning.weight_scheduler.decreasing_weight"
#     - "examples.curriculum_learning.weight_scheduler.increasing_weight"


def decreasing_weight(rollout_step: int) -> float:
    """
    Decreasing weight: starts high (0.9) and decreases to low (0.1) over 200 steps.
    Use for easier data sources that should be de-emphasized as training progresses.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    max_steps = 200
    progress = min(rollout_step / max_steps, 1.0)
    return 0.9 - 0.8 * progress  # 0.9 -> 0.1


def increasing_weight(rollout_step: int) -> float:
    """
    Increasing weight: starts low (0.1) and increases to high (0.9) over 200 steps.
    Use for harder data sources that should be emphasized as training progresses.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    max_steps = 200
    progress = min(rollout_step / max_steps, 1.0)
    return 0.1 + 0.8 * progress  # 0.1 -> 0.9


def exponential_decay(rollout_step: int) -> float:
    """
    Exponentially decreasing weight with smooth decay over 200 steps.
    Use for data sources that should be phased out gradually.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    decay_rate = 0.015  # Calibrated for 200 steps
    return 0.9 * math.exp(-decay_rate * rollout_step) + 0.1


def exponential_warmup(rollout_step: int) -> float:
    """
    Exponentially increasing weight with smooth growth over 200 steps.
    Use for data sources that should be phased in gradually.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    decay_rate = 0.015  # Calibrated for 200 steps
    weight_decaying = 0.9 * math.exp(-decay_rate * rollout_step) + 0.1
    return 1.0 - weight_decaying  # Inverse of exponential_decay


def level_increasing(rollout_step: int) -> float:
    """
    Level-based increasing weight with discrete steps.
    Weight increases in 4 levels over 200 steps: 0.5 -> 0.6 -> 0.7 -> 0.8

    Useful for staged curriculum where you want discrete milestones
    for gradually emphasizing a data source more.

    Stages:
    - Steps 0-50: 0.5 (low weight)
    - Steps 50-100: 0.6
    - Steps 100-150: 0.7
    - Steps 150-200: 0.8 (high weight)

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    if rollout_step < 50:
        return 0.5
    elif rollout_step < 100:
        return 0.6
    elif rollout_step < 150:
        return 0.7
    else:
        return 0.8


def level_decreasing(rollout_step: int) -> float:
    """
    Level-based decreasing weight with discrete steps.
    Weight decreases in 4 levels over 200 steps: 0.8 -> 0.7 -> 0.6 -> 0.5

    Useful for staged curriculum where you want discrete milestones
    for gradually de-emphasizing a data source.

    Stages:
    - Steps 0-50: 0.8 (high weight)
    - Steps 50-100: 0.7
    - Steps 100-150: 0.6
    - Steps 150-200: 0.5 (low weight)

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    if rollout_step < 50:
        return 0.8
    elif rollout_step < 100:
        return 0.7
    elif rollout_step < 150:
        return 0.6
    else:
        return 0.5
