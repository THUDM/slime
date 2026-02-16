import logging
from collections import defaultdict

import numpy as np

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
        if sample.metadata and 'data_source' in sample.metadata:
            source_name = sample.metadata['data_source']
        else:
            source_name = 'unknown'
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
        log_dict[f"rollout/{source_name}/sample_count"] = len(source_samples)

    logger.info(f"rollout {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # Return True to indicate we handled logging (prevents default logging)
    return True
