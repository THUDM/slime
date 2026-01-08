"""Reward shaping for tau2-bench rollouts.

Entry point: `--custom-reward-post-process-path reward.tau2_reward_post_process`
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict

import torch

from env import PartialScoreWeights, compute_partial_score_from_reward_info, parse_reward_info
from slime.utils.types import Sample


CURRICULUM_SOLVED_THRESHOLD = 0.75
CURRICULUM_HARD_THRESHOLD = 0.25


class _Curriculum:
    """Process-local curriculum tracker.

    Note: In distributed Ray settings, each worker maintains its own tracker
    instance. This is a best-effort heuristic; disable via TAU2_USE_CURRICULUM=0
    for fully deterministic behavior across workers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = defaultdict(int)
        self._successes: dict[str, int] = defaultdict(int)

    def enabled(self) -> bool:
        return os.environ.get("TAU2_USE_CURRICULUM", "1") == "1"

    def min_attempts(self) -> int:
        return int(os.environ.get("TAU2_CURRICULUM_MIN_ATTEMPTS", "5"))

    def solved_weight(self) -> float:
        return float(os.environ.get("TAU2_CURRICULUM_SOLVED_WEIGHT", "0.1"))

    def hard_weight(self) -> float:
        return float(os.environ.get("TAU2_CURRICULUM_HARD_WEIGHT", "0.5"))

    def update(self, task_id: str, reward: float) -> None:
        if not self.enabled() or not task_id:
            return
        with self._lock:
            self._attempts[task_id] += 1
            if reward >= 1.0:
                self._successes[task_id] += 1

    def weight(self, task_id: str) -> float:
        if not self.enabled() or not task_id:
            return 1.0
        with self._lock:
            attempts = self._attempts.get(task_id, 0)
            if attempts < self.min_attempts():
                return 1.0
            rate = self._successes.get(task_id, 0) / attempts if attempts else 0.0
        if rate > CURRICULUM_SOLVED_THRESHOLD:
            return self.solved_weight()
        if rate < CURRICULUM_HARD_THRESHOLD:
            return self.hard_weight()
        return 1.0


_curriculum = _Curriculum()


def _alpha(domain: str | None) -> float:
    base = float(os.environ.get("TAU2_REWARD_ALPHA", "0.25"))
    if os.environ.get("TAU2_DOMAIN_ADAPTIVE_ALPHA", "1") != "1" or not domain:
        return base
    # Telecom uses a higher multiplier to compensate for dual-control communication overhead.
    mult = {"retail": 0.8, "airline": 1.0, "telecom": 1.6}.get(domain, 1.0)
    return base * mult


def _partial_weights(domain: str | None) -> PartialScoreWeights:
    action = float(os.environ.get("TAU2_PARTIAL_ACTION_WEIGHT", "0.5"))
    communicate = float(os.environ.get("TAU2_PARTIAL_COMMUNICATE_WEIGHT", "0.15"))
    env_assertion = float(os.environ.get("TAU2_PARTIAL_ENV_ASSERTION_WEIGHT", "0.35"))
    db = float(os.environ.get("TAU2_PARTIAL_DB_WEIGHT", "0.0"))

    if domain == "telecom" and os.environ.get("TAU2_TELECOM_COMMUNICATION_BOOST", "1") == "1":
        return PartialScoreWeights(action=0.35, communicate=0.35, env_assertion=0.30, db=0.0)

    return PartialScoreWeights(action=action, communicate=communicate, env_assertion=env_assertion, db=db)


def _flatten(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], list):
        return [s for group in samples for s in group]
    return list(samples)


def _normalize_rewards(
    rewards: list[float],
    *,
    valid_mask: list[float],
    n_samples_per_prompt: int,
    apply_std: bool,
) -> list[float]:
    if not rewards:
        return []
    if n_samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be >= 1 (got {n_samples_per_prompt})")
    if len(rewards) % n_samples_per_prompt != 0:
        raise ValueError(
            "Reward count must be a multiple of n_samples_per_prompt "
            f"(count={len(rewards)}, n_samples_per_prompt={n_samples_per_prompt})."
        )

    rewards_t = torch.tensor(rewards, dtype=torch.float).view(-1, n_samples_per_prompt)
    mask_t = torch.tensor(valid_mask, dtype=torch.float).view(-1, n_samples_per_prompt)

    denom = mask_t.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (rewards_t * mask_t).sum(dim=-1, keepdim=True) / denom
    centered = (rewards_t - mean) * mask_t

    if apply_std:
        var = (centered**2).sum(dim=-1, keepdim=True) / denom
        centered = centered / (torch.sqrt(var) + 1e-6)

    return centered.flatten().tolist()


def tau2_reward_post_process(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    flat = _flatten(samples)

    raw_rewards: list[float] = []
    shaped_rewards: list[float] = []
    sample_weights: list[float] = []
    valid_mask: list[float] = []

    for sample in flat:
        task_reward = float(sample.get_reward_value(args))
        raw_rewards.append(task_reward)

        metadata = sample.metadata or {}
        domain = metadata.get("domain")
        task_id = metadata.get("task_id") or metadata.get("tau2_task_id") or ""

        reward_info = parse_reward_info({"reward_info": metadata.get("reward_info")})
        partial_score, partial_components = compute_partial_score_from_reward_info(
            reward_info,
            weights=_partial_weights(domain),
            normalize_over_present=True,
        )

        shaped = task_reward + _alpha(domain) * partial_score
        is_valid = not sample.remove_sample
        valid_mask.append(1.0 if is_valid else 0.0)
        shaped_rewards.append(float(shaped) if is_valid else 0.0)

        _curriculum.update(task_id, task_reward)
        w = _curriculum.weight(task_id) if is_valid else 0.0
        sample_weights.append(w)

        sample.metadata = {
            **metadata,
            "raw_reward": task_reward,
            "partial_score": partial_score,
            "partial_components": partial_components,
            "shaped_reward": shaped,
            "curriculum_weight": w,
        }

    if (
        args.advantage_estimator in {"grpo", "gspo", "reinforce_plus_plus_baseline"}
        and args.rewards_normalization
        and shaped_rewards
    ):
        rewards = _normalize_rewards(
            shaped_rewards,
            valid_mask=valid_mask,
            n_samples_per_prompt=args.n_samples_per_prompt,
            apply_std=(args.advantage_estimator in {"grpo", "gspo"} and args.grpo_std_normalization),
        )

        if os.environ.get("TAU2_APPLY_CURRICULUM_WEIGHTS", "1") == "1":
            weights = torch.tensor(sample_weights, dtype=torch.float).view(-1, args.n_samples_per_prompt)
            rewards_t = torch.tensor(rewards, dtype=torch.float).view_as(weights)
            rewards = (rewards_t * weights).flatten().tolist()

        return raw_rewards, rewards

    if os.environ.get("TAU2_APPLY_CURRICULUM_WEIGHTS", "1") == "1":
        shaped_rewards = [r * sample_weights[i] for i, r in enumerate(shaped_rewards)]

    return raw_rewards, shaped_rewards
