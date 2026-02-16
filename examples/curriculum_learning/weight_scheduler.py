import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Per-Source Weight Functions
# ============================================================================
# Each function returns a single float for one data source
# Use these in config like:
#   data_source_weights:
#     - "examples.curriculum_learning.weight_scheduler.easy_to_hard"
#     - "examples.curriculum_learning.weight_scheduler.hard_to_easy"


def easy_to_hard(rollout_step: int) -> float:
    """
    Decreasing weight: starts high (0.9) and decreases to low (0.1) over 1000 steps.
    Use for easier data sources that should be de-emphasized as training progresses.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    max_steps = 100
    progress = min(rollout_step / max_steps, 1.0)
    return 0.9 - 0.8 * progress  # 0.9 -> 0.1


def hard_to_easy(rollout_step: int) -> float:
    """
    Increasing weight: starts low (0.1) and increases to high (0.9) over 1000 steps.
    Use for harder data sources that should be emphasized as training progresses.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    max_steps = 100
    progress = min(rollout_step / max_steps, 1.0)
    return 0.1 + 0.8 * progress  # 0.1 -> 0.9


def exponential_decay(rollout_step: int) -> float:
    """
    Exponentially decreasing weight with smooth decay.
    Use for data sources that should be phased out gradually.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    decay_rate = 0.003
    return 0.9 * math.exp(-decay_rate * rollout_step) + 0.1


def exponential_warmup(rollout_step: int) -> float:
    """
    Exponentially increasing weight with smooth growth.
    Use for data sources that should be phased in gradually.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    decay_rate = 0.003
    weight_decaying = 0.9 * math.exp(-decay_rate * rollout_step) + 0.1
    return 1.0 - weight_decaying  # Inverse of exponential_decay
