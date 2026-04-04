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
            logger.exception("Failed to import wandb for MOPD rollout logging.")
            return
        wandb.log(metrics)


logging_utils = _LoggingUtils()


def _compute_rollout_step(args, rollout_id: int) -> int:
    if getattr(args, "wandb_always_use_train_step", False):
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id


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


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _sample_status(sample: Any) -> str:
    status = getattr(sample, "status", "")
    return getattr(status, "value", status) or ""


def _sample_response_length(sample: Any) -> int:
    return int(getattr(sample, "effective_response_length", getattr(sample, "response_length", 0)) or 0)


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


def _reward_summary(sample: Any) -> dict[str, Any]:
    reward = getattr(sample, "reward", None)
    summary: dict[str, Any] = {
        "reward_type": type(reward).__name__ if reward is not None else "none",
        "reward_scalar": None,
    }
    if isinstance(reward, bool):
        summary["reward_scalar"] = float(reward)
    elif isinstance(reward, (int, float)):
        summary["reward_scalar"] = float(reward)
    elif isinstance(reward, dict):
        summary["reward_keys"] = sorted(str(key) for key in reward.keys())[:16]
        summary["reward_has_meta_info"] = isinstance(reward.get("meta_info"), dict)
    return summary


def _trace_rows(*, rollout_id: int, samples: list[Any], args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_samples = _trace_max_samples(args)
    for sample_index, sample in enumerate(samples):
        if max_samples is not None and sample_index >= max_samples:
            break
        metadata = _sample_metadata(sample)
        rows.append(
            {
                "rollout_id": rollout_id,
                "sample_index": sample_index,
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
            }
        )
    return rows


def _write_rollout_trace(*, rollout_id: int, args, samples: list[Any]) -> None:
    trace_dir = _trace_dir(args)
    if not trace_dir:
        return

    rows = _trace_rows(rollout_id=rollout_id, samples=samples, args=args)
    if not rows:
        return

    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    output_path = trace_path / f"rollout_{rollout_id:07d}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")


def _build_split_metrics(*, samples: list[Any], prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total = len(samples)
    if total == 0:
        return metrics

    lengths = [_sample_response_length(sample) for sample in samples]
    truncated = [_sample_status(sample) == "truncated" for sample in samples]

    metrics[f"{prefix}/count"] = total
    metrics[f"{prefix}/truncated_ratio"] = sum(1 for item in truncated if item) / total
    for name, value in _compute_statistics(lengths).items():
        metrics[f"{prefix}/response_len/{name}"] = value

    for source_name, grouped in _group_samples(samples, "dataset_name").items():
        source_lengths = [_sample_response_length(sample) for sample in grouped]
        source_truncated = [_sample_status(sample) == "truncated" for sample in grouped]
        metrics[f"{prefix}_by_source/{source_name}/count"] = len(grouped)
        metrics[f"{prefix}_by_source/{source_name}/fraction"] = len(grouped) / total
        metrics[f"{prefix}_by_source/{source_name}/truncated_ratio"] = (
            sum(1 for item in source_truncated if item) / len(grouped)
        )
        for name, value in _compute_statistics(source_lengths).items():
            metrics[f"{prefix}_by_source/{source_name}/response_len/{name}"] = value

    for domain_name, grouped in _group_samples(samples, "domain").items():
        domain_lengths = [_sample_response_length(sample) for sample in grouped]
        domain_truncated = [_sample_status(sample) == "truncated" for sample in grouped]
        metrics[f"{prefix}_by_domain/{domain_name}/count"] = len(grouped)
        metrics[f"{prefix}_by_domain/{domain_name}/fraction"] = len(grouped) / total
        metrics[f"{prefix}_by_domain/{domain_name}/truncated_ratio"] = (
            sum(1 for item in domain_truncated if item) / len(grouped)
        )
        for name, value in _compute_statistics(domain_lengths).items():
            metrics[f"{prefix}_by_domain/{domain_name}/response_len/{name}"] = value

    return metrics


def _build_perf_metrics(*, samples: list[Any], args, rollout_time: float) -> dict[str, float]:
    metrics: dict[str, float] = {"perf/rollout_time": rollout_time}
    response_lengths = [_sample_response_length(sample) for sample in samples]
    if not response_lengths or rollout_time <= 0:
        return metrics

    metrics["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time
    rollout_num_gpus = getattr(args, "rollout_num_gpus", None)
    if rollout_num_gpus:
        metrics["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / rollout_num_gpus
    return metrics


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    rollout_samples = list(samples)
    _write_rollout_trace(rollout_id=rollout_id, args=args, samples=rollout_samples)

    metrics = {}
    metrics.update(rollout_extra_metrics or {})
    metrics.update(_build_split_metrics(samples=rollout_samples, prefix="rollout"))
    metrics.update(_build_perf_metrics(samples=rollout_samples, args=args, rollout_time=rollout_time))
    metrics["rollout/step"] = _compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="rollout/step")
    return True


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    samples: list[Any] = []
    for split in data.values():
        split_samples = split.get("samples") or []
        samples.extend(split_samples)

    metrics = {}
    metrics.update(extra_metrics or {})
    metrics.update(_build_split_metrics(samples=samples, prefix="eval"))
    metrics["eval/step"] = _compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="eval/step")
    # Keep default eval logging enabled so per-dataset metrics such as
    # eval/aime24 and eval/livecodebench still appear in W&B.
    return False
