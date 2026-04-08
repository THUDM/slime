"""Shared rollout and eval logging."""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
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
            logger.exception("Failed to import wandb for rollout logging.")
            return
        wandb.log(metrics)


logging_utils = _LoggingUtils()


def _compute_rollout_step(args, rollout_id: int) -> int:
    if getattr(args, "wandb_always_use_train_step", False):
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id


def _extract_reward_scalar(value: Any, args) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        reward_key = getattr(args, "reward_key", None)
        if reward_key:
            reward_value = value.get(reward_key)
            if isinstance(reward_value, (bool, int, float)):
                return float(reward_value)

        for key in ("reward", "score", "scalar_reward", "value"):
            reward_value = value.get(key)
            if isinstance(reward_value, (bool, int, float)):
                return float(reward_value)

        # Pure OPD teacher responses carry logprob meta info plus telemetry such as
        # token counts. Those are not task rewards and should not be logged as such.
        if isinstance(value.get("meta_info"), dict):
            return 0.0

        return 0.0
    return float(value or 0.0)


def _sample_reward(sample: Any, args) -> float:
    if hasattr(sample, "get_reward_value"):
        return _extract_reward_scalar(sample.get_reward_value(args), args)
    reward = getattr(sample, "reward", 0.0)
    return _extract_reward_scalar(reward, args)


def _sample_response_length(sample: Any) -> int:
    return int(getattr(sample, "effective_response_length", getattr(sample, "response_length", 0)) or 0)


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _sample_status(sample: Any) -> str:
    status = getattr(sample, "status", "")
    return getattr(status, "value", status) or ""


def _metadata_value(sample: Any, key: str, default: str) -> str:
    value = _sample_metadata(sample).get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _group_samples(samples: list[Any], key: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[_metadata_value(sample, key, "unknown")].append(sample)
    return dict(grouped)


def _compute_statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(float(v) for v in values)
    count = len(sorted_values)
    mid = count // 2
    median = (sorted_values[mid - 1] + sorted_values[mid]) / 2.0 if count % 2 == 0 else sorted_values[mid]
    return {
        "mean": sum(sorted_values) / count,
        "median": median,
        "max": max(sorted_values),
        "min": min(sorted_values),
    }


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

    metrics[f"{prefix}/count"] = total
    lengths = [_sample_response_length(s) for s in samples]
    truncated = [_sample_status(s) == "truncated" for s in samples]
    metrics[f"{prefix}/truncated_ratio"] = sum(1 for t in truncated if t) / total
    for name, value in _compute_statistics(lengths).items():
        metrics[f"{prefix}/response_len/{name}"] = value

    for source_name, grouped in _group_samples(samples, "dataset_name").items():
        rewards = [_sample_reward(s, args) for s in grouped]
        glengths = [_sample_response_length(s) for s in grouped]
        gtrunc = [_sample_status(s) == "truncated" for s in grouped]
        n = len(grouped)
        metrics[f"{prefix}_by_source/{source_name}/count"] = n
        metrics[f"{prefix}_by_source/{source_name}/fraction"] = n / total
        metrics[f"{prefix}_by_source/{source_name}/{score_name}"] = sum(rewards) / n
        metrics[f"{prefix}_by_source/{source_name}/truncated_ratio"] = sum(1 for t in gtrunc if t) / n
        for name, value in _compute_statistics(glengths).items():
            metrics[f"{prefix}_by_source/{source_name}/response_len/{name}"] = value

    for domain_name, grouped in _group_samples(samples, "domain").items():
        rewards = [_sample_reward(s, args) for s in grouped]
        glengths = [_sample_response_length(s) for s in grouped]
        gtrunc = [_sample_status(s) == "truncated" for s in grouped]
        n = len(grouped)
        metrics[f"{prefix}_by_domain/{domain_name}/count"] = n
        metrics[f"{prefix}_by_domain/{domain_name}/fraction"] = n / total
        metrics[f"{prefix}_by_domain/{domain_name}/{score_name}"] = sum(rewards) / n
        metrics[f"{prefix}_by_domain/{domain_name}/truncated_ratio"] = sum(1 for t in gtrunc if t) / n
        for name, value in _compute_statistics(glengths).items():
            metrics[f"{prefix}_by_domain/{domain_name}/response_len/{name}"] = value

    return metrics


def _build_perf_metrics(*, samples: list[Any], args, rollout_time: float) -> dict[str, float]:
    metrics: dict[str, float] = {"perf/rollout_time": rollout_time}
    response_lengths = [_sample_response_length(s) for s in samples]
    if not response_lengths or rollout_time <= 0:
        return metrics
    metrics["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time
    rollout_num_gpus = getattr(args, "rollout_num_gpus", None)
    if rollout_num_gpus:
        metrics["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / rollout_num_gpus
    return metrics


def _trace_dir(args) -> str:
    value = getattr(args, "multidomain_v1_trace_dir", "") or os.getenv("MULTIDOMAIN_V1_TRACE_DIR", "")
    return str(value).strip()


def _trace_max_samples(args) -> int | None:
    value = getattr(args, "multidomain_v1_trace_max_samples", None)
    if value in (None, ""):
        value = os.getenv("MULTIDOMAIN_V1_TRACE_MAX_SAMPLES", "")
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _reward_summary(sample: Any) -> dict[str, Any]:
    reward = getattr(sample, "reward", None)
    summary: dict[str, Any] = {"reward_type": type(reward).__name__ if reward is not None else "none", "reward_scalar": None}
    if isinstance(reward, bool):
        summary["reward_scalar"] = float(reward)
    elif isinstance(reward, (int, float)):
        summary["reward_scalar"] = float(reward)
    elif isinstance(reward, dict):
        summary["reward_keys"] = sorted(str(k) for k in reward.keys())[:16]
        summary["reward_has_meta_info"] = isinstance(reward.get("meta_info"), dict)
    return summary


def _write_rollout_trace(*, rollout_id: int, args, samples: list[Any]) -> None:
    trace_path = _trace_dir(args)
    if not trace_path:
        return
    max_samples = _trace_max_samples(args)
    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        if max_samples is not None and i >= max_samples:
            break
        metadata = _sample_metadata(sample)
        rows.append({
            "rollout_id": rollout_id,
            "sample_index": i,
            "dataset_name": metadata.get("dataset_name"),
            "domain": metadata.get("domain"),
            "record_id": metadata.get("record_id"),
            "response_length": _sample_response_length(sample),
            "status": _sample_status(sample),
            "prompt": getattr(sample, "prompt", None),
            "response": getattr(sample, "response", None),
            "label": getattr(sample, "label", None),
            "metadata": metadata,
            **_reward_summary(sample),
        })
    if not rows:
        return
    out = Path(trace_path)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"rollout_{rollout_id:07d}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    rollout_samples = list(samples)
    _write_rollout_trace(rollout_id=rollout_id, args=args, samples=rollout_samples)

    metrics: dict[str, float] = {}
    metrics.update(rollout_extra_metrics or {})
    metrics.update(_build_split_metrics(samples=rollout_samples, args=args, prefix="rollout", score_name="reward_mean"))
    metrics.update(_build_perf_metrics(samples=rollout_samples, args=args, rollout_time=rollout_time))
    metrics["rollout/step"] = _compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="rollout/step")
    return True


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    samples: list[Any] = []
    for split in data.values():
        samples.extend(split.get("samples") or [])

    metrics: dict[str, float] = {}
    metrics.update(extra_metrics or {})
    metrics.update(_build_split_metrics(samples=samples, args=args, prefix="eval", score_name="score"))
    metrics["eval/step"] = _compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="eval/step")
    return False
