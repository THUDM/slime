from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any


logger = logging.getLogger(__name__)


class _LoggingUtils:
    @staticmethod
    def log(args, metrics: dict[str, Any], step_key: str) -> None:
        if not getattr(args, "use_wandb", False):
            return
        try:
            import wandb
        except Exception:
            logger.exception("Failed to import wandb for mixed-domain logging.")
            return
        wandb.log(metrics)


logging_utils = _LoggingUtils()


def _compute_rollout_step(args, rollout_id: int) -> int:
    if getattr(args, "wandb_always_use_train_step", False):
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id


def _compute_statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    else:
        median = sorted_values[mid]
    return {
        "mean": sum(sorted_values) / count,
        "median": median,
        "max": max(sorted_values),
        "min": min(sorted_values),
    }


def _sample_reward(sample: Any, args) -> float:
    if hasattr(sample, "get_reward_value"):
        return float(sample.get_reward_value(args))
    reward = getattr(sample, "reward", 0.0)
    if isinstance(reward, dict):
        reward_key = getattr(args, "reward_key", None)
        if reward_key:
            return float(reward.get(reward_key, 0.0))
        for value in reward.values():
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0
    return float(reward or 0.0)


def _sample_response_length(sample: Any) -> int:
    return int(getattr(sample, "effective_response_length", getattr(sample, "response_length", 0)) or 0)


def _sample_status(sample: Any) -> str:
    status = getattr(sample, "status", "")
    return getattr(status, "value", status) or ""


def _metadata_value(sample: Any, key: str, default: str) -> str:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return default
    value = metadata.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _group_samples(samples: list[Any], key: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[_metadata_value(sample, key, "unknown")].append(sample)
    return dict(grouped)


def _build_group_metrics(
    *,
    samples: list[Any],
    args,
    prefix: str,
    score_name: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = len(samples)
    if total == 0:
        return metrics

    rewards = [_sample_reward(sample, args) for sample in samples]
    lengths = [_sample_response_length(sample) for sample in samples]
    truncated = [_sample_status(sample) == "truncated" for sample in samples]

    metrics[f"{prefix}/count"] = total
    metrics[f"{prefix}/fraction"] = total / total
    metrics[f"{prefix}/{score_name}"] = sum(rewards) / total
    metrics[f"{prefix}/truncated_ratio"] = sum(1 for item in truncated if item) / total
    for name, value in _compute_statistics(lengths).items():
        metrics[f"{prefix}/response_len/{name}"] = value
    return metrics


def _build_split_metrics(
    *,
    samples: list[Any],
    args,
    prefix: str,
    score_name: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = len(samples)
    if total == 0:
        return metrics

    for source_name, grouped in _group_samples(samples, "dataset_name").items():
        rewards = [_sample_reward(sample, args) for sample in grouped]
        lengths = [_sample_response_length(sample) for sample in grouped]
        truncated = [_sample_status(sample) == "truncated" for sample in grouped]
        metrics[f"{prefix}_by_source/{source_name}/count"] = len(grouped)
        metrics[f"{prefix}_by_source/{source_name}/fraction"] = len(grouped) / total
        metrics[f"{prefix}_by_source/{source_name}/{score_name}"] = sum(rewards) / len(grouped)
        metrics[f"{prefix}_by_source/{source_name}/truncated_ratio"] = sum(1 for item in truncated if item) / len(grouped)
        for name, value in _compute_statistics(lengths).items():
            metrics[f"{prefix}_by_source/{source_name}/response_len/{name}"] = value

    for domain_name, grouped in _group_samples(samples, "domain").items():
        rewards = [_sample_reward(sample, args) for sample in grouped]
        lengths = [_sample_response_length(sample) for sample in grouped]
        truncated = [_sample_status(sample) == "truncated" for sample in grouped]
        metrics[f"{prefix}_by_domain/{domain_name}/count"] = len(grouped)
        metrics[f"{prefix}_by_domain/{domain_name}/fraction"] = len(grouped) / total
        metrics[f"{prefix}_by_domain/{domain_name}/{score_name}"] = sum(rewards) / len(grouped)
        metrics[f"{prefix}_by_domain/{domain_name}/truncated_ratio"] = sum(1 for item in truncated if item) / len(grouped)
        for name, value in _compute_statistics(lengths).items():
            metrics[f"{prefix}_by_domain/{domain_name}/response_len/{name}"] = value

    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    metrics = _build_split_metrics(
        samples=list(samples),
        args=args,
        prefix="rollout",
        score_name="reward_mean",
    )
    if metrics:
        metrics["rollout/step"] = _compute_rollout_step(args, rollout_id)
        logging_utils.log(args, metrics, step_key="rollout/step")
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    samples: list[Any] = []
    for split in data.values():
        split_samples = split.get("samples") or []
        samples.extend(split_samples)

    metrics = _build_split_metrics(
        samples=samples,
        args=args,
        prefix="eval",
        score_name="score",
    )
    if metrics:
        metrics["eval/step"] = _compute_rollout_step(args, rollout_id)
        logging_utils.log(args, metrics, step_key="eval/step")
    return False
