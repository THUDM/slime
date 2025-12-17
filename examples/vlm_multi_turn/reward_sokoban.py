from __future__ import annotations

from typing import Any

from slime.utils.types import Sample


def _distance_delta(dist0: float | None, dist1: float | None) -> float:
    if dist0 is None or dist1 is None:
        return 0.0
    return float(dist0 - dist1)


def calculate_sokoban_reward(
    turn_rewards: list[float],
    final_bonus: float | None,
    dist0: float | None,
    dist1: float | None,
) -> tuple[float, dict[str, Any]]:
    """
    Combine average per-turn reward with terminal bonus and optional distance shaping.

    Args:
        turn_rewards: rewards emitted by the environment at each turn (used for averaging).
        final_bonus: terminal reward (e.g., success bonus).
        dist0: initial box-target distance.
        dist1: final box-target distance.

    Returns:
        total reward and a breakdown dictionary for logging.
    """
    turn_reward_total = float(sum(turn_rewards))
    turn_reward_avg = turn_reward_total / max(1, len(turn_rewards))
    final_bonus = float(final_bonus or 0.0)
    distance_term = _distance_delta(dist0, dist1)
    total_reward = turn_reward_avg + final_bonus + distance_term
    breakdown = {
        "turn_reward_avg": turn_reward_avg,
        "turn_reward_total": turn_reward_total,
        "final_bonus": final_bonus,
        "dist0": dist0,
        "dist1": dist1,
        "distance_delta": distance_term,
        "total_reward": total_reward,
    }
    return total_reward, breakdown


def compute_reward_from_metadata(sample: Sample) -> float:
    """
    Convenience helper so the same reward logic can be used as a custom RM.
    """
    metadata = sample.metadata or {}
    sokoban_meta = metadata.get("sokoban", {})
    turn_rewards = sokoban_meta.get("turn_rewards") or []
    final_bonus = sokoban_meta.get("final_bonus", 0.0)
    dist0 = sokoban_meta.get("dist0")
    dist1 = sokoban_meta.get("dist1")

    reward, breakdown = calculate_sokoban_reward(turn_rewards, final_bonus, dist0, dist1)
    sokoban_meta.setdefault("reward_breakdown", breakdown)
    metadata["sokoban"] = sokoban_meta
    sample.metadata = metadata
    return reward


async def async_compute_reward(args, sample: Sample, **kwargs):
    """
    Async wrapper compatible with slime's custom_rm_path hook.
    """
    return compute_reward_from_metadata(sample)
