"""Custom reward post-processing for teacher-routed MOPD.

Mirrors ``slime.rollout.on_policy_distillation.post_process_rewards``: extracts
each sample's routed-teacher token log-probabilities, trims them to the response
span, stores them in the existing ``sample.teacher_log_probs`` field, and returns
zero scalar rewards for pure distillation.

Because the stored value is per-sample and slime's OPD loss already computes
``reverse_kl = student_logp - teacher_logp`` per sample, routing requires no
training-side change and no custom advantage function.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from slime_mopd.config import MopdConfig, MopdConfigError, load_mopd_config

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_logprob_dtype(args: Any, config: MopdConfig | None = None) -> torch.dtype:
    """Teacher log-prob dtype: explicit config, else the model params dtype."""

    if config is not None and config.logprob_dtype != "auto":
        return _DTYPE_MAP[config.logprob_dtype]
    params_dtype = getattr(args, "params_dtype", None)
    if isinstance(params_dtype, torch.dtype):
        return params_dtype
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    return torch.float32


def extract_teacher_log_probs(raw_reward: Any, response_length: int, dtype: torch.dtype) -> torch.Tensor:
    """Pull token log-probs from an SGLang response and align them to the response."""

    if not isinstance(raw_reward, dict):
        raise TypeError(
            "mopd post-process expects the raw SGLang response dict as sample.reward; "
            f"got {type(raw_reward).__name__}. Ensure --reward-key is not set."
        )
    meta_info = raw_reward.get("meta_info") or {}
    entries = meta_info.get("input_token_logprobs") or []
    if not entries:
        raise ValueError("mopd: teacher response has no meta_info.input_token_logprobs")

    values = [entry[0] for entry in entries[1:]]
    if response_length <= 0:
        return torch.empty(0, dtype=dtype)
    if len(values) < response_length:
        raise ValueError(
            f"mopd: teacher returned {len(values)} token log-probs for a response of {response_length} tokens"
        )
    return torch.tensor(values[-response_length:], dtype=dtype)


def mopd_post_process(args: Any, samples: list[Any], **_: Any) -> tuple[list[float], list[float]]:
    """slime ``--custom-reward-post-process-path`` hook."""

    try:
        config = load_mopd_config(args)
    except MopdConfigError:
        config = None
    dtype = resolve_logprob_dtype(args, config)

    raw_rewards = []
    teacher_log_probs = []
    for sample in samples:
        raw_reward = getattr(sample, "reward", None)
        raw_rewards.append(raw_reward)
        teacher_log_probs.append(extract_teacher_log_probs(raw_reward, int(sample.response_length), dtype))

    for sample, log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = log_probs

    if logger.isEnabledFor(logging.INFO):
        counts: dict[str, int] = {}
        for sample in samples:
            teacher_id = getattr(sample, "mopd_teacher_id", None)
            if teacher_id is None:
                metadata = getattr(sample, "metadata", None) or {}
                teacher_id = metadata.get("teacher") if isinstance(metadata, dict) else None
            key = str(teacher_id) if teacher_id is not None else "unresolved"
            counts[key] = counts.get(key, 0) + 1
        logger.info("mopd: post-processed %d samples; teachers=%s", len(samples), dict(sorted(counts.items())))

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
