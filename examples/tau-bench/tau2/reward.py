"""Reward shaping for tau2-bench rollouts.

Usage: --custom-reward-post-process-path reward.tau2_reward_post_process

shaped_reward = task_reward + alpha * partial_score

Includes a small task curriculum for downsampling tasks outside the learning zone.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from typing import cast

import torch

from env import PartialScoreWeights, compute_partial_score_from_reward_info, parse_reward_info

# Import Sample at runtime for cast()
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Track iteration for curriculum logging
_iteration_counter = 0


class _TaskCurriculumTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict[str, int] = defaultdict(int)
        self._successes: dict[str, int] = defaultdict(int)
        self._partial_scores: dict[str, list[float]] = defaultdict(list)
        self._stats_logged_at = 0

    def _enabled(self) -> bool:
        return os.environ.get("TAU2_USE_CURRICULUM", "1") == "1"

    def _min_attempts(self) -> int:
        return int(os.environ.get("TAU2_CURRICULUM_MIN_ATTEMPTS", "5"))

    def _solved_weight(self) -> float:
        return float(os.environ.get("TAU2_CURRICULUM_SOLVED_WEIGHT", "0.1"))

    def _hard_weight(self) -> float:
        return float(os.environ.get("TAU2_CURRICULUM_HARD_WEIGHT", "0.5"))

    def _use_partial(self) -> bool:
        return os.environ.get("TAU2_CURRICULUM_USE_PARTIAL", "1") == "1"

    def update(self, task_id: str, reward: float, *, partial_score: float | None = None, success_threshold: float = 1.0) -> None:
        if not self._enabled() or not task_id:
            return

        with self._lock:
            self._attempts[task_id] += 1
            if reward >= success_threshold:
                self._successes[task_id] += 1
            if partial_score is not None:
                scores = self._partial_scores[task_id]
                scores.append(float(partial_score))
                if len(scores) > 20:
                    del scores[:-20]

    def get_sample_weight(self, task_id: str) -> float:
        if not self._enabled() or not task_id:
            return 1.0

        with self._lock:
            attempts = self._attempts.get(task_id, 0)
            min_attempts = self._min_attempts()
            if attempts < min_attempts:
                return 1.0

            progress_rate: float
            if self._use_partial() and task_id in self._partial_scores:
                scores = self._partial_scores[task_id]
                if len(scores) >= min_attempts:
                    progress_rate = sum(scores) / len(scores)
                else:
                    progress_rate = self._successes.get(task_id, 0) / attempts
            else:
                progress_rate = self._successes.get(task_id, 0) / attempts

            if progress_rate > 0.75:
                return self._solved_weight()
            if progress_rate < 0.25:
                return self._hard_weight()
            return 1.0

    def log_summary(self, iteration: int, *, log_interval: int = 10) -> str | None:
        if not self._enabled():
            return None
        if iteration - self._stats_logged_at < log_interval:
            return None

        with self._lock:
            self._stats_logged_at = iteration
            min_attempts = self._min_attempts()
            eligible = [tid for tid, a in self._attempts.items() if a >= min_attempts]
            if not eligible:
                return None

            solved = 0
            hard = 0
            learning = 0
            rates: list[float] = []
            for tid in eligible:
                attempts = self._attempts[tid]
                rate = self._successes.get(tid, 0) / attempts if attempts else 0.0
                rates.append(rate)
                if rate > 0.75:
                    solved += 1
                elif rate < 0.25:
                    hard += 1
                else:
                    learning += 1

            avg_solve = sum(rates) / len(rates) if rates else 0.0

        return (
            f"[Curriculum] iter={iteration} tasks={len(eligible)} "
            f"avg_solve={avg_solve:.1%} solved={solved} hard={hard} learning={learning}"
        )


_curriculum_tracker = _TaskCurriculumTracker()


# Diagnostic tools for telecom bootstrap scoring.
TELECOM_DIAGNOSTIC_TOOLS = frozenset({
    "check_network_status",
    "check_sim_status",
    "check_apn_settings",
    "toggle_airplane_mode",
    "reseat_sim_card",
    "reset_apn_settings",
    "reboot_device",
    "check_data_usage",
    "enable_roaming",
    "disable_roaming",
    "check_status_bar",
    "run_speed_test",
    "get_data_usage",
})


def _get_alpha(domain: str | None = None) -> float:
    """Get shaping coefficient, optionally domain-adaptive."""
    base_alpha = float(os.environ.get("TAU2_REWARD_ALPHA", "0.25"))

    if os.environ.get("TAU2_DOMAIN_ADAPTIVE_ALPHA", "1") == "1" and domain:
        domain_multipliers = {"retail": 0.8, "airline": 1.0, "telecom": 1.6}
        return base_alpha * domain_multipliers.get(domain, 1.0)

    return base_alpha


def _get_partial_weights(domain: str | None = None) -> PartialScoreWeights:
    """Get component weights for partial score computation.

    For telecom domain, communication is weighted higher because the model must
    INSTRUCT users to run diagnostic tools (they're user-only in dual-control mode).
    """
    base_action = float(os.environ.get("TAU2_PARTIAL_ACTION_WEIGHT", "0.5"))
    base_communicate = float(os.environ.get("TAU2_PARTIAL_COMMUNICATE_WEIGHT", "0.15"))
    base_env_assertion = float(os.environ.get("TAU2_PARTIAL_ENV_ASSERTION_WEIGHT", "0.35"))
    base_db = float(os.environ.get("TAU2_PARTIAL_DB_WEIGHT", "0.0"))

    # Domain-specific adjustments for telecom: emphasize communication quality
    # since the model must guide users through diagnostics, not call tools directly.
    if domain == "telecom" and os.environ.get("TAU2_TELECOM_COMMUNICATION_BOOST", "1") == "1":
        # Telecom: communication is critical (user coaching)
        # Shift weight from action → communicate
        return PartialScoreWeights(
            action=0.35,
            communicate=0.35,
            env_assertion=0.30,
            db=0.0,
        )

    return PartialScoreWeights(
        action=base_action,
        communicate=base_communicate,
        env_assertion=base_env_assertion,
        db=base_db,
    )


def _compute_telecom_bootstrap(tool_sequence: list[str] | None) -> float:
    """Bootstrap score based on diagnostic tool diversity (for cold-start)."""
    if not tool_sequence:
        return 0.0
    used_diagnostics = set(tool_sequence) & TELECOM_DIAGNOSTIC_TOOLS
    if not used_diagnostics:
        return 0.0
    return min(len(used_diagnostics) / 4.0, 1.0)


def tau2_reward_post_process(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    """Compute shaped rewards with domain-adaptive alpha, telecom bootstrap, and curriculum tracking."""
    global _iteration_counter
    _iteration_counter += 1

    if samples and isinstance(samples[0], list):
        flat_samples = [s for group in samples for s in group]
    else:
        flat_samples = cast(list[Sample], samples)

    raw_rewards: list[float] = []
    shaped_rewards: list[float] = []

    for sample in flat_samples:
        task_reward = float(sample.get_reward_value(args))
        raw_rewards.append(task_reward)

        if sample.metadata is None:
            sample.metadata = {}

        domain = sample.metadata.get("domain")
        tool_sequence = sample.metadata.get("tool_sequence") or []
        reward_info = parse_reward_info({"reward_info": sample.metadata.get("reward_info")})

        # Extract task_id for curriculum tracking
        task_id = sample.metadata.get("tau2_task_id") or sample.metadata.get("task_id", "")
        # Normalize task_id format: "[domain]task_idx"
        if task_id and domain and not task_id.startswith("["):
            task_id = f"[{domain}]{task_id}"

        # Get domain-specific weights (telecom emphasizes communication for user coaching)
        weights = _get_partial_weights(domain)
        partial_score, partial_components = compute_partial_score_from_reward_info(
            reward_info, weights=weights, normalize_over_present=True
        )

        # Telecom bootstrap for cold-start (only applies when using custom tau2-bench patch
        # with assistant_can_use_user_tools=True). With benchmark-compliant rollouts,
        # diagnostic tools are user-only so the model can't call them and this won't trigger.
        # Kept for backwards compatibility with custom training setups.
        bootstrap_score = 0.0
        if domain == "telecom" and partial_score < 0.01 and task_reward < 0.5:
            bootstrap_score = _compute_telecom_bootstrap(tool_sequence)
            if bootstrap_score > 0:
                partial_score = bootstrap_score
                partial_components["diagnostic_bootstrap"] = bootstrap_score

        alpha = _get_alpha(domain)

        shaped = task_reward + alpha * partial_score
        shaped_rewards.append(float(shaped))

        # Update curriculum tracker with task result and partial progress
        _curriculum_tracker.update(task_id, task_reward, partial_score=partial_score)
        sample_weight = _curriculum_tracker.get_sample_weight(task_id)

        sample.metadata["raw_reward"] = task_reward
        sample.metadata["partial_score"] = partial_score
        sample.metadata["partial_components"] = partial_components
        sample.metadata["shaped_reward"] = shaped
        sample.metadata["alpha_used"] = alpha
        sample.metadata["curriculum_weight"] = sample_weight
        if bootstrap_score > 0:
            sample.metadata["bootstrap_applied"] = True

    # Log curriculum statistics periodically
    curriculum_msg = _curriculum_tracker.log_summary(_iteration_counter, log_interval=10)
    if curriculum_msg:
        logger.info(curriculum_msg)

    # Log domain-wise metrics to WandB for training observability
    try:
        import wandb

        if wandb.run is not None:
            domain_rewards: dict[str, list[float]] = defaultdict(list)
            domain_partial: dict[str, list[float]] = defaultdict(list)
            domain_shaped: dict[str, list[float]] = defaultdict(list)

            for sample in flat_samples:
                domain = sample.metadata.get("domain", "unknown")
                domain_rewards[domain].append(float(sample.get_reward_value(args)))
                domain_partial[domain].append(sample.metadata.get("partial_score", 0.0))
                domain_shaped[domain].append(sample.metadata.get("shaped_reward", 0.0))

            metrics: dict[str, float] = {}
            for domain in domain_rewards:
                rewards = domain_rewards[domain]
                partials = domain_partial[domain]
                shaped = domain_shaped[domain]
                n = len(rewards)
                if n > 0:
                    metrics[f"rollout/domain/{domain}/task_reward"] = sum(rewards) / n
                    metrics[f"rollout/domain/{domain}/partial_score"] = sum(partials) / n
                    metrics[f"rollout/domain/{domain}/shaped_reward"] = sum(shaped) / n
                    metrics[f"rollout/domain/{domain}/success_rate"] = sum(1 for r in rewards if r >= 1.0) / n

            # Curriculum stats (every log interval)
            if curriculum_msg:
                with _curriculum_tracker._lock:
                    min_attempts = _curriculum_tracker._min_attempts()
                    eligible = [
                        t for t, a in _curriculum_tracker._attempts.items() if a >= min_attempts
                    ]
                    if eligible:
                        solved = sum(
                            1
                            for t in eligible
                            if _curriculum_tracker._successes.get(t, 0) / _curriculum_tracker._attempts[t]
                            > 0.75
                        )
                        hard = sum(
                            1
                            for t in eligible
                            if _curriculum_tracker._successes.get(t, 0) / _curriculum_tracker._attempts[t]
                            < 0.25
                        )
                        metrics["curriculum/solved_tasks"] = solved
                        metrics["curriculum/hard_tasks"] = hard
                        metrics["curriculum/learning_zone_tasks"] = len(eligible) - solved - hard
                        metrics["curriculum/total_tracked_tasks"] = len(eligible)

            if metrics:
                wandb.log(metrics, commit=True)
    except ImportError:
        pass

    # GRPO group-normalization
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
        and flat_samples
    ):
        rewards = torch.tensor(shaped_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        # Apply curriculum weights post-normalization to scale gradient contribution
        if os.environ.get("TAU2_APPLY_CURRICULUM_WEIGHTS", "1") == "1":
            curriculum_weights = torch.tensor(
                [s.metadata.get("curriculum_weight", 1.0) for s in flat_samples],
                dtype=torch.float,
            )
            curriculum_weights = curriculum_weights.view(rewards.shape)
            rewards = rewards * curriculum_weights

        return raw_rewards, rewards.flatten().tolist()

    # Non-GRPO path: apply curriculum weights directly to shaped rewards
    if os.environ.get("TAU2_APPLY_CURRICULUM_WEIGHTS", "1") == "1":
        shaped_rewards = [
            r * flat_samples[i].metadata.get("curriculum_weight", 1.0)
            for i, r in enumerate(shaped_rewards)
        ]

    return raw_rewards, shaped_rewards
