from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from examples.eval.eval_delegate import EvalDelegateClient, _rebuild_delegate_config
from omegaconf import OmegaConf

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import generate_rollout as base_generate_rollout

logger = logging.getLogger(__name__)

_DELEGATE_CACHE: dict[str, tuple[float | None, EvalDelegateClient | None]] = {}


def generate_rollout(
    args, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert evaluation, "Delegate rollout is only supported for evaluation"
    flattened_metrics: dict[str, float] = {}

    client = _get_delegate_client(args)
    if client is not None:
        metrics, raw_response = client.evaluate(args, rollout_id)
        flattened_metrics = _log_delegate_metrics(args, rollout_id, metrics, raw_response)

    result = base_generate_rollout(args, rollout_id, data_buffer, evaluation=evaluation)
    result.metrics = flattened_metrics
    return result


def _get_delegate_client(args) -> EvalDelegateClient | None:
    config_path = getattr(args, "eval_config", None)
    if not config_path:
        return None

    config_path = str(Path(config_path).expanduser())
    cache_entry = _DELEGATE_CACHE.get(config_path)
    mtime = _safe_mtime(config_path)
    if cache_entry and cache_entry[0] == mtime:
        return cache_entry[1]

    client = _build_delegate_client(args, config_path)
    _DELEGATE_CACHE[config_path] = (mtime, client)
    return client


def _build_delegate_client(args, config_path: str) -> EvalDelegateClient | None:
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger.warning("--eval-config must contain a mapping at the root.")
        return None

    eval_cfg = cfg_dict.get("eval", cfg_dict)
    if not isinstance(eval_cfg, dict):
        logger.warning("--eval-config must define an `eval` mapping or be a mapping itself.")
        return None

    defaults = dict(eval_cfg.get("defaults") or {})
    delegate_entries = list(eval_cfg.get("delegate") or [])
    env_configs = _rebuild_delegate_config(args, delegate_entries, defaults)
    if not env_configs:
        logger.info("No delegate environments configured under `eval.delegate`; skipping external eval.")
        return None

    return EvalDelegateClient.maybe_create(args, env_configs=env_configs)


def _safe_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _log_delegate_metrics(args, rollout_id: int, metrics: dict | None, raw_response: dict | None) -> dict:
    flattened = _flatten_metrics(metrics)
    if raw_response is not None:
        logger.info("External eval raw response for rollout %s: %s", rollout_id, raw_response)
    logger.info("eval %s (external): %s", rollout_id, flattened)
    return flattened


def _flatten_metrics(metric_source: dict | None) -> dict:
    flattened_metrics: dict[str, float] = {}
    if not isinstance(metric_source, dict):
        return flattened_metrics
    for key, value in _iter_metric_items(metric_source):
        if not isinstance(value, (int, float)):
            continue
        metric_key = key if key.startswith("eval/") else f"eval/{key}"
        flattened_metrics[metric_key] = float(value)
    return flattened_metrics


def _iter_metric_items(metrics: dict, prefix: str = ""):
    for key, value in metrics.items():
        metric_key = f"{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            yield from _iter_metric_items(value, metric_key)
        else:
            yield metric_key, value
