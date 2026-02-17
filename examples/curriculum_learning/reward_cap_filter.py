"""
Dynamic sampling filter for per-data-source reward capping.

This filter checks if a sample group's average reward exceeds the configured cap for its data source.
Groups with average rewards above the cap are dropped during dynamic sampling.

Supports both static and dynamic (step-wise) reward caps:
- Static: constant values (e.g., 0.3, 0.5)
- Dynamic: lambda functions or function paths that take rollout_step

Example usage in config:
    prompt_data_source_names:
      - dapo_math_17k
      - verinstruct
    data_source_reward_caps:
      - 0.3  # Constant cap
      - "lambda step: 0.1 + 0.5 * min(step / 1000, 1.0)"  # Dynamic cap
      # OR
      - "examples.curriculum_learning.reward_cap_scheduler.increasing_cap"
    dynamic_sampling_filter_path: examples.curriculum_learning.reward_cap_filter.check_reward_cap_per_source
"""

import logging
from collections.abc import Callable

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

__all__ = ["check_reward_cap_per_source"]

# Module-level cache for parsed cap functions
_CAP_FUNCTIONS_CACHE = {}


def _parse_reward_caps(reward_caps_config: list, source_names: list) -> list[Callable[[int], float]]:
    """
    Parse reward cap configuration into a list of callable functions.

    Each element in reward_caps_config can be:
    1. Float/int constant: 0.5 -> returns constant cap
    2. Lambda string: "lambda step: 0.5 + step/1000" -> returns dynamic cap
    3. Function path: "examples.curriculum_learning.reward_cap_scheduler.increasing_cap" -> returns dynamic cap

    Returns a list of functions, each taking rollout_step and returning a single float cap.
    """
    # Create cache key from config
    cache_key = str(reward_caps_config)
    if cache_key in _CAP_FUNCTIONS_CACHE:
        return _CAP_FUNCTIONS_CACHE[cache_key]

    cap_functions = []
    for i, config in enumerate(reward_caps_config):
        cap_fn = _parse_single_cap(config, i, source_names)
        cap_functions.append(cap_fn)

    _CAP_FUNCTIONS_CACHE[cache_key] = cap_functions
    logger.info(f"Parsed {len(cap_functions)} reward cap functions for data sources")
    return cap_functions


def _parse_single_cap(config: float | int | str | None, index: int, source_names: list) -> Callable[[int], float]:
    """
    Parse a single reward cap configuration into a callable function.

    Args:
        config: Can be a number (constant), lambda string, function path, or None
        index: Index of the data source (for logging)
        source_names: List of source names (for logging)

    Returns:
        A function that takes rollout_step and returns a float cap (or None)
    """
    source_name = source_names[index] if index < len(source_names) else f"source_{index}"

    # Handle None (no cap)
    if config is None:
        logger.info(f"Data source {index} ({source_name}): no reward cap (None)")
        return lambda step: None

    # Handle numeric constant
    if isinstance(config, (float, int)):
        logger.info(f"Data source {index} ({source_name}): constant reward cap = {config}")
        return lambda step: float(config)

    # Handle string (lambda or function path)
    elif isinstance(config, str):
        # Check if it's a lambda function
        if config.strip().startswith("lambda"):
            try:
                cap_fn = eval(config)
                logger.info(f"Data source {index} ({source_name}): lambda reward cap = {config}")
                return cap_fn
            except Exception as e:
                logger.error(f"Failed to parse lambda function for source {index}: '{config}': {e}")
                raise ValueError(f"Invalid lambda function for data source {index}: {config}") from e
        else:
            # Assume it's a function path
            try:
                cap_fn = load_function(config)
                logger.info(f"Data source {index} ({source_name}): loaded reward cap function from {config}")
                return cap_fn
            except Exception as e:
                logger.error(f"Failed to load reward cap function for source {index} from '{config}': {e}")
                raise ValueError(f"Invalid function path for data source {index}: {config}") from e
    else:
        raise ValueError(f"Reward cap config for source {index} must be a number, string, or None, got {type(config)}")


def check_reward_cap_per_source(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """
    Filter sample groups based on per-data-source average reward caps.

    Computes the average reward within the group and drops the group if:
    1. All rewards are 0.0 (no learning signal), OR
    2. Average reward exceeds the configured cap for this data source

    Works with both binary rewards (0/1) and continuous rewards (e.g., 0.0 to 1.0).
    Supports both static caps (constants) and dynamic caps (functions of rollout_step).

    Args:
        args: Arguments containing:
            - prompt_data_source_names: List of data source names
            - data_source_reward_caps: List of reward caps (constants, lambdas, or function paths)
        samples: List of samples in this group (all from same data source)
        **kwargs: Additional arguments

    Returns:
        DynamicFilterOutput indicating whether to keep or drop this group
    """
    # Get configuration
    if not hasattr(args, "prompt_data_source_names") or not hasattr(args, "data_source_reward_caps"):
        # No configuration, pass through
        return DynamicFilterOutput(keep=True)

    source_names = args.prompt_data_source_names
    reward_caps_config = args.data_source_reward_caps

    if len(source_names) != len(reward_caps_config):
        logger.warning(
            f"Mismatch between source_names ({len(source_names)}) and reward_caps ({len(reward_caps_config)}). "
            "Disabling reward filtering."
        )
        return DynamicFilterOutput(keep=True)

    # Parse reward caps into callable functions
    cap_functions = _parse_reward_caps(reward_caps_config, source_names)

    # Build source -> cap_function mapping
    cap_fn_by_source = dict(zip(source_names, cap_functions, strict=True))

    # Get data source from first sample's metadata
    if not samples or not samples[0].metadata or "data_source" not in samples[0].metadata:
        # No data source info, pass through
        return DynamicFilterOutput(keep=True)

    data_source = samples[0].metadata["data_source"]

    # Assert all samples in group have the same data source
    assert all(
        sample.metadata and sample.metadata.get("data_source") == data_source for sample in samples
    ), "All samples in group must have the same data_source, got mixed sources in group"

    # Get cap function for this data source
    cap_fn = cap_fn_by_source.get(data_source)
    if cap_fn is None:
        # No cap configured for this source, pass through
        return DynamicFilterOutput(keep=True)

    # Get rollout step from sample metadata (set by MultipleWeightedRolloutDataSourceWithBuffer)
    rollout_step = samples[0].metadata.get("data_source_rollout_step", 0)

    # Evaluate cap function with current rollout step
    cap = cap_fn(rollout_step)
    if cap is None:
        # Cap function returned None (no filtering for this source)
        return DynamicFilterOutput(keep=True)

    # Compute average reward for this group using pre-computed rewards
    rewards = [sample.reward for sample in samples]

    # Drop if all rewards are 0.0 (no learning signal)
    if all(r == 0.0 for r in rewards):
        return DynamicFilterOutput(keep=False, reason=f"all_zero_{data_source}")

    avg_reward = sum(rewards) / len(samples) if len(samples) > 0 else 0.0

    # Drop if average reward exceeds cap
    if avg_reward > cap:
        return DynamicFilterOutput(keep=False, reason=f"reward_cap_{data_source}")

    return DynamicFilterOutput(keep=True)
