import logging
from collections import defaultdict


from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, compute_statistics

logger = logging.getLogger(__name__)


def custom_rollout_log_function(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """
    Custom logging function that logs metrics separately for each data source.

    Logs both:
    1. Overall metrics (existing behavior)
    2. Per-data-source metrics for response_len and raw_rewards
    """
    # Group samples by data source
    samples_by_source = defaultdict(list)
    for sample in samples:
        if sample.metadata and "data_source" in sample.metadata:
            source_name = sample.metadata["data_source"]
        else:
            source_name = "unknown"
        samples_by_source[source_name].append(sample)

    # Compute overall metrics (existing behavior)
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from slime.utils.metric_utils import dict_add_prefix

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")

    # Compute per-source metrics
    for source_name, source_samples in samples_by_source.items():
        if not source_samples:
            continue

        # Response length statistics
        response_lengths = [sample.effective_response_length for sample in source_samples]
        response_stats = compute_statistics(response_lengths)
        for stat_name, stat_value in response_stats.items():
            log_dict[f"rollout/{source_name}/response_len/{stat_name}"] = stat_value

        # Raw reward statistics
        raw_rewards = [sample.get_reward_value(args) for sample in source_samples]
        reward_stats = compute_statistics(raw_rewards)
        for stat_name, stat_value in reward_stats.items():
            log_dict[f"rollout/{source_name}/raw_reward/{stat_name}"] = stat_value

        # Sample count for this source
        num_final_samples = len(source_samples)
        log_dict[f"rollout/{source_name}/sample_count"] = num_final_samples

        # Dynamic filter statistics: compute ratios of filtered groups
        # Only compute if dynamic filtering is enabled and we have rollout_extra_metrics
        if (
            hasattr(args, "dynamic_sampling_filter_path")
            and args.dynamic_sampling_filter_path is not None
            and rollout_extra_metrics
        ):

            # Count groups filtered due to all zero rewards
            num_filtered_zero = 0
            # Count groups filtered due to reward cap
            num_filtered_cap = 0

            for metric_key, metric_value in rollout_extra_metrics.items():
                # Check for all zero drops: rollout/dynamic_filter/drop_all_zero_{source}
                if metric_key == f"rollout/dynamic_filter/drop_all_zero_{source_name}":
                    num_filtered_zero = metric_value
                # Check for reward cap drops: rollout/dynamic_filter/drop_reward_cap_{source}_*
                elif metric_key.startswith(f"rollout/dynamic_filter/drop_reward_cap_{source_name}"):
                    num_filtered_cap += metric_value

            # Compute ratios: filtered / final_samples
            if num_final_samples > 0:
                if num_filtered_zero > 0:
                    log_dict[f"rollout/{source_name}/filter_ratio_all_zero"] = num_filtered_zero / num_final_samples
                if num_filtered_cap > 0:
                    log_dict[f"rollout/{source_name}/filter_ratio_reward_cap"] = num_filtered_cap / num_final_samples

    # Log data source weights (curriculum learning curves)
    # Read actual weights from sample metadata (more accurate than recomputing)
    if hasattr(args, "prompt_data_source_names"):
        weight_by_source = {}
        rollout_step_by_source = {}

        # Extract weights from sample metadata
        for sample in samples:
            if sample.metadata and "data_source" in sample.metadata:
                source_name = sample.metadata["data_source"]
                if "data_source_weight" in sample.metadata:
                    weight_by_source[source_name] = sample.metadata["data_source_weight"]
                if "data_source_rollout_step" in sample.metadata:
                    rollout_step_by_source[source_name] = sample.metadata["data_source_rollout_step"]

        # Log weights for each source
        for source_name in args.prompt_data_source_names:
            if source_name in weight_by_source:
                log_dict[f"rollout/{source_name}/weight"] = weight_by_source[source_name]
            if source_name in rollout_step_by_source:
                log_dict[f"rollout/{source_name}/rollout_step"] = rollout_step_by_source[source_name]

    # Log data source reward caps (curriculum learning curves for dynamic caps)
    if (
        hasattr(args, "prompt_data_source_names")
        and hasattr(args, "data_source_reward_caps")
        and args.data_source_reward_caps is not None
    ):

        # Import parsing function from reward_cap_filter
        try:
            from examples.curriculum_learning.reward_cap_filter import _parse_reward_caps

            source_names = args.prompt_data_source_names
            reward_caps_config = args.data_source_reward_caps

            # Parse reward caps into callable functions
            cap_functions = _parse_reward_caps(reward_caps_config, source_names)

            # Evaluate and log cap for each source at its current rollout step
            for source_name, cap_fn in zip(source_names, cap_functions, strict=True):
                # Get rollout step for this source (from rollout_step_by_source if available)
                rollout_step = (
                    rollout_step_by_source.get(source_name, 0) if "rollout_step_by_source" in locals() else 0
                )

                # Evaluate cap function
                cap_value = cap_fn(rollout_step)

                # Log cap value (None means no cap for this source)
                if cap_value is not None:
                    log_dict[f"rollout/{source_name}/reward_cap"] = cap_value
        except ImportError as e:
            # If import fails, silently skip logging caps
            logger.debug(f"Could not import reward cap parsing function: {e}")
        except Exception as e:
            # Log any other errors but don't fail
            logger.warning(f"Error logging reward caps: {e}")

    logger.info(f"rollout {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # Return True to indicate we handled logging (prevents default logging)
    return True
