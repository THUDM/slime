"""MOPD rollout logging wrappers around the shared log_rollout module."""
from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import log_rollout as _shared_log_rollout  # noqa: E402

logging_utils = _shared_log_rollout.logging_utils


def _call_with_logging_utils(func, *args, **kwargs):
    original = _shared_log_rollout.logging_utils
    _shared_log_rollout.logging_utils = logging_utils
    try:
        return func(*args, **kwargs)
    finally:
        _shared_log_rollout.logging_utils = original


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    return _call_with_logging_utils(
        _shared_log_rollout.log_rollout_data,
        rollout_id,
        args,
        samples,
        rollout_extra_metrics,
        rollout_time,
    )


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    return _call_with_logging_utils(
        _shared_log_rollout.log_eval_rollout_data,
        rollout_id,
        args,
        data,
        extra_metrics,
    )
