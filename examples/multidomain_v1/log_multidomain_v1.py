from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V0_SCRIPT = ROOT / "multidomain_v0" / "log_mixed_domain.py"


def _load_v0_module():
    spec = importlib.util.spec_from_file_location("log_mixed_domain_v0", V0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v0 = _load_v0_module()
logging_utils = _v0.logging_utils


def _compute_rollout_step(args, rollout_id: int) -> int:
    return _v0._compute_rollout_step(args, rollout_id)


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


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _sample_status(sample: Any) -> str:
    status = getattr(sample, "status", "")
    return getattr(status, "value", status) or ""


def _sample_prompt(sample: Any) -> Any:
    return getattr(sample, "prompt", None)


def _sample_response(sample: Any) -> Any:
    return getattr(sample, "response", None)


def _sample_label(sample: Any) -> Any:
    return getattr(sample, "label", None)


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
                "reward": _v0._sample_reward(sample, args),
                "response_length": _v0._sample_response_length(sample),
                "status": _sample_status(sample),
                "prompt": _sample_prompt(sample),
                "response": _sample_response(sample),
                "label": _sample_label(sample),
                "metadata": metadata,
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


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    rollout_samples = list(samples)
    _write_rollout_trace(rollout_id=rollout_id, args=args, samples=rollout_samples)

    metrics = _v0._build_split_metrics(
        samples=rollout_samples,
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

    metrics = _v0._build_split_metrics(
        samples=samples,
        args=args,
        prefix="eval",
        score_name="score",
    )
    if metrics:
        metrics["eval/step"] = _compute_rollout_step(args, rollout_id)
        logging_utils.log(args, metrics, step_key="eval/step")
    return False
