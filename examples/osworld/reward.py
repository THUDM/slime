"""Hybrid Reward System for OSWorld GUI Tasks.

Rule-based signals: action parsing, execution, a11y grounding.
Efficiency bonus: fewer steps = higher score.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)
DEBUG_PRINT_LIMIT = int(os.environ.get("OSWORLD_REWARD_DEBUG_LIMIT", "5"))
_debug_print_count = 0


def _maybe_print_reward_debug(
    task_reward: float,
    partial_score: float,
    components: dict[str, float],
    step_signals: list[dict],
) -> None:
    global _debug_print_count
    if _debug_print_count >= DEBUG_PRINT_LIMIT:
        return
    _debug_print_count += 1
    parsed = sum(1 for s in step_signals if s.get("action_parsed", False))
    hashes = [s.get("screen_hash") for s in step_signals if s.get("screen_hash")]
    unique_hashes = len(set(hashes)) if hashes else 0
    hash_changes = sum(1 for s in step_signals if s.get("screen_hash_changed", 0.0))
    print(
        f"[reward_debug] steps={len(step_signals)} parsed={parsed} task_reward={task_reward:.3f} "
        f"partial_score={partial_score:.3f} screen_hash_unique={unique_hashes} "
        f"screen_hash_changes={hash_changes:.1f} components={components}",
        flush=True,
    )


@dataclass
class OSWorldPartialWeights:
    """Weight configuration for hybrid reward components."""

    action_parse_valid: float = 0.10
    action_executed: float = 0.15
    a11y_grounding: float = 0.30
    tool_calling: float = 0.25
    efficiency: float = 0.05
    screen_change: float = 0.05
    a11y_change: float = 0.05
    terminal_change: float = 0.05


def compute_rule_based_signals(step_signals: list[dict]) -> dict[str, float]:
    """Compute verifiable reward signals from step-level data."""
    if not step_signals:
        return {
            "action_parse_valid": 0.0,
            "action_executed": 0.0,
            "a11y_grounding": 0.0,
        }

    parse_valid = sum(1 for s in step_signals if s.get("action_parsed", False)) / len(step_signals)
    executed = sum(1 for s in step_signals if s.get("action_executed", False)) / len(step_signals)
    grounded = sum(1 for s in step_signals if s.get("a11y_grounded", False)) / len(step_signals)

    return {
        "action_parse_valid": parse_valid,
        "action_executed": executed,
        "a11y_grounding": grounded,
    }


def compute_process_signals(step_signals: list[dict]) -> dict[str, float]:
    """Compute process-level signals from step deltas."""
    if not step_signals:
        return {
            "screen_diff": 0.0,
            "a11y_delta": 0.0,
            "terminal_changed": 0.0,
        }

    n = len(step_signals)
    screen_diff = sum(s.get("screen_diff", 0.0) for s in step_signals) / n
    a11y_delta = sum(s.get("a11y_delta", 0.0) for s in step_signals) / n
    terminal_changed = sum(s.get("terminal_changed", 0.0) for s in step_signals) / n

    return {
        "screen_diff": screen_diff,
        "a11y_delta": a11y_delta,
        "terminal_changed": terminal_changed,
    }


def compute_variance_penalties(step_signals: list[dict]) -> tuple[float, float]:
    """Compute penalties that create reward variance between samples."""
    n = max(1, len(step_signals))
    if n < 2:
        return 0.0, 0.0

    repeats = 0
    invalids = 0
    for i, s in enumerate(step_signals):
        if s.get("action_type", "unknown") == "unknown":
            invalids += 1
        if i > 0:
            prev = step_signals[i - 1]
            if s.get("action_type") == prev.get("action_type") and s.get("coordinate") == prev.get("coordinate"):
                repeats += 1

    # Increased repetition penalty - was primary failure mode (clicking same spot)
    rep_penalty = min(0.30, (repeats / n) * 0.40)
    inv_penalty = min(0.15, (invalids / n) * 0.30)
    return rep_penalty, inv_penalty


def compute_fallback_penalty(step_signals: list[dict]) -> float:
    """Penalize non-compliant fallback parsing to enforce tool_call format."""
    if not step_signals:
        return 0.0
    n = len(step_signals)
    fallback_count = sum(1 for s in step_signals if s.get("fallback_used", False))
    return min(0.5, (fallback_count / n) * 0.5)


def compute_wait_penalty(step_signals: list[dict]) -> float:
    """Penalize repeated WAIT actions to push the model toward higher-impact steps."""
    if not step_signals:
        return 0.0
    n = len(step_signals)
    wait_count = sum(1 for s in step_signals if str(s.get("action_type", "")).lower() == "wait")
    return min(0.3, (wait_count / n) * 0.3)


def compute_efficiency_signal(num_steps: int, max_steps: int = 8) -> float:
    """Compute efficiency score (0-1, higher = fewer steps)."""
    if num_steps <= 0:
        return 1.0
    return max(0.0, (max_steps - num_steps) / max_steps)


def compute_tool_calling_signal(
    step_signals: list[dict],
    expected_complexity: str = "medium",
) -> float:
    """Compute tool-calling score to prevent single-action gaming."""
    if not step_signals:
        return 0.0

    action_types = [s.get("action_type", "unknown") for s in step_signals]
    num_actions = len(action_types)

    if num_actions == 0:
        return 0.0

    # Action diversity score
    unique_types = set(action_types)
    diversity_score = min(1.0, len(unique_types) / 3.0)

    # Appropriate complexity score
    complexity_ranges = {
        "low": (2, 5),
        "medium": (5, 12),
        "high": (10, 25),
    }
    min_expected, max_expected = complexity_ranges.get(expected_complexity, (5, 12))

    if min_expected <= num_actions <= max_expected:
        complexity_score = 1.0
    elif num_actions < min_expected:
        complexity_score = num_actions / min_expected
    else:
        complexity_score = max(0.5, max_expected / num_actions)

    # Modality coverage
    modality_types = {"click", "type", "key", "scroll", "drag", "wait"}
    used_modalities = set()
    for action in action_types:
        action_lower = action.lower()
        for modality in modality_types:
            if modality in action_lower:
                used_modalities.add(modality)
                break
    modality_score = min(1.0, len(used_modalities) / 2.0)

    return 0.4 * diversity_score + 0.4 * complexity_score + 0.2 * modality_score


def compute_hybrid_partial_score(
    step_signals: list[dict],
    num_steps: int,
    weights: OSWorldPartialWeights | None = None,
    max_steps: int = 15,
    task_complexity: str = "medium",
    a11y_mode: str = "full",
) -> tuple[float, dict[str, float]]:
    """Compute weighted partial score from all signal components."""
    weights = weights or OSWorldPartialWeights()

    rule_signals = compute_rule_based_signals(step_signals)
    efficiency = compute_efficiency_signal(num_steps, max_steps)

    process_signals = compute_process_signals(step_signals)

    if a11y_mode != "full":
        process_signals["a11y_delta"] = 0.0
        rule_signals["a11y_grounding"] = 0.0
        weights.a11y_grounding = 0.0
        weights.a11y_change = 0.0

    tool_calling = compute_tool_calling_signal(step_signals, task_complexity)

    components = {
        "action_parse_valid": rule_signals["action_parse_valid"],
        "action_executed": rule_signals["action_executed"],
        "a11y_grounding": rule_signals["a11y_grounding"],
        "tool_calling": tool_calling,
        "efficiency": efficiency,
        "screen_diff": process_signals["screen_diff"],
        "a11y_delta": process_signals["a11y_delta"],
        "terminal_changed": process_signals["terminal_changed"],
    }

    partial_score = (
        weights.action_parse_valid * components["action_parse_valid"]
        + weights.action_executed * components["action_executed"]
        + weights.a11y_grounding * components["a11y_grounding"]
        + weights.tool_calling * components["tool_calling"]
        + weights.efficiency * components["efficiency"]
        + weights.screen_change * components["screen_diff"]
        + weights.a11y_change * components["a11y_delta"]
        + weights.terminal_change * components["terminal_changed"]
    )

    return partial_score, components


def compute_shaped_reward(
    task_reward: float,
    partial_score: float,
    step_signals: list[dict] = None,
    alpha: float | None = None,
) -> float:
    """Compute shaped reward with variance penalties."""
    if alpha is None:
        alpha = float(os.environ.get("OSWORLD_REWARD_ALPHA", "0.3"))

    reward = task_reward + alpha * partial_score
    if step_signals:
        rep_pen, inv_pen = compute_variance_penalties(step_signals)
        fallback_pen = compute_fallback_penalty(step_signals)
        wait_pen = compute_wait_penalty(step_signals)
        reward -= rep_pen + inv_pen + fallback_pen + wait_pen

    # Clamp to [0, 1] - penalties can push below 0, alpha scaling can push above 1
    return max(0.0, min(1.0, reward))


def _merge_reward_config(
    reward_config: dict | None,
) -> tuple[float | None, OSWorldPartialWeights | None, str | None]:
    if not reward_config:
        return None, None, None

    alpha = reward_config.get("alpha")
    weights_cfg = reward_config.get("weights", {}) or {}
    weights = OSWorldPartialWeights()
    for key, value in weights_cfg.items():
        if hasattr(weights, key):
            setattr(weights, key, float(value))
    task_complexity = reward_config.get("task_complexity")
    return alpha, weights, task_complexity


def _attach_reward_breakdown(sample: Sample, breakdown: dict) -> None:
    """Attach reward breakdown to train metadata for training-side logging."""
    train_meta = sample.train_metadata or {}
    train_meta["osworld_reward_breakdown"] = breakdown
    sample.train_metadata = train_meta


def compute_reward_from_metadata(sample: Sample, reward_config: dict | None = None) -> float:
    """Compute shaped reward from sample metadata."""
    metadata = sample.metadata or {}

    # Replay samples have pre-computed rewards - don't overwrite
    if metadata.get("is_replay"):
        return sample.reward or metadata.get("replay_reward", 1.0)

    osworld_meta = metadata.get("osworld", {})
    if reward_config is None:
        reward_config = osworld_meta.get("reward_config")
    alpha, weights, task_complexity = _merge_reward_config(reward_config)

    task_reward = float(osworld_meta.get("task_reward", 0.0))
    step_signals = osworld_meta.get("step_signals", [])
    num_steps = osworld_meta.get("turns", len(step_signals))
    max_steps = int(osworld_meta.get("max_steps", 15))
    a11y_mode = osworld_meta.get("a11y_mode", "full")

    partial_score, components = compute_hybrid_partial_score(
        step_signals=step_signals,
        num_steps=num_steps,
        weights=weights,
        max_steps=max_steps,
        task_complexity=task_complexity or "medium",
        a11y_mode=a11y_mode,
    )
    components["fallback_penalty"] = compute_fallback_penalty(step_signals)
    components["wait_penalty"] = compute_wait_penalty(step_signals)

    shaped_reward = compute_shaped_reward(task_reward, partial_score, step_signals, alpha=alpha)

    breakdown = {
        "task_reward": task_reward,
        "partial_score": partial_score,
        "shaped_reward": shaped_reward,
        "components": components,
    }
    _attach_reward_breakdown(sample, breakdown)
    _maybe_print_reward_debug(task_reward, partial_score, components, step_signals)
    osworld_meta["reward_breakdown"] = breakdown
    osworld_meta["raw_reward"] = task_reward
    metadata["osworld"] = osworld_meta
    metadata["raw_reward"] = task_reward
    metadata["shaped_reward"] = shaped_reward
    sample.metadata = metadata

    return shaped_reward


async def async_compute_reward(args: Any, sample: Sample, **kwargs) -> float:
    """Async reward computation."""
    metadata = sample.metadata or {}

    # Replay samples have pre-computed rewards - don't overwrite
    if metadata.get("is_replay"):
        return sample.reward or metadata.get("replay_reward", 1.0)

    osworld_meta = metadata.get("osworld", {})

    reward_config = osworld_meta.get("reward_config")
    alpha, weights, task_complexity = _merge_reward_config(reward_config)

    task_reward = float(osworld_meta.get("task_reward", 0.0))
    step_signals = osworld_meta.get("step_signals", [])
    num_steps = osworld_meta.get("turns", len(step_signals))
    max_steps = int(osworld_meta.get("max_steps", 15))
    a11y_mode = osworld_meta.get("a11y_mode", "full")

    partial_score, components = compute_hybrid_partial_score(
        step_signals=step_signals,
        num_steps=num_steps,
        weights=weights,
        max_steps=max_steps,
        task_complexity=task_complexity or "medium",
        a11y_mode=a11y_mode,
    )
    components["fallback_penalty"] = compute_fallback_penalty(step_signals)
    components["wait_penalty"] = compute_wait_penalty(step_signals)

    shaped_reward = compute_shaped_reward(task_reward, partial_score, step_signals, alpha=alpha)

    breakdown = {
        "task_reward": task_reward,
        "partial_score": partial_score,
        "shaped_reward": shaped_reward,
        "components": components,
    }
    _attach_reward_breakdown(sample, breakdown)
    _maybe_print_reward_debug(task_reward, partial_score, components, step_signals)
    osworld_meta["reward_breakdown"] = breakdown
    osworld_meta["raw_reward"] = task_reward
    metadata["osworld"] = osworld_meta
    metadata["raw_reward"] = task_reward
    metadata["shaped_reward"] = shaped_reward
    sample.metadata = metadata

    return shaped_reward
