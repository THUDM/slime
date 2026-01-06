"""
Off-policy utilities for GRPO with decoupled PPO objective.

This module implements the decoupled PPO objective from AREAL paper:
    J(θ) = E[π_prox/π_behav * min(u_t^prox(θ) A_t, clip(u_t^prox(θ), 1-ε, 1+ε) A_t)]

where:
    - π_behav: behavior policy (used for sampling trajectories)
    - π_prox: proximal policy (regularization target, previous training step)
    - π_θ: current policy being updated
    - u_t^prox(θ) = π_θ / π_prox

Key features:
1. Supports staleness control with max_staleness parameter (η)
2. Handles inconsistent policy versions within a trajectory
3. Compatible with existing GRPO implementation
"""

from typing import List, Optional, Tuple
import torch
import torch.distributed as dist


@torch.compile(dynamic=True)
def compute_offpolicy_importance_weights(
    proximal_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute importance sampling weights: π_prox / π_behav

    Args:
        proximal_log_probs: Log probs from proximal policy [num_tokens]
        behavior_log_probs: Log probs from behavior policy [num_tokens]
        clip_min: Optional minimum clipping value
        clip_max: Optional maximum clipping value

    Returns:
        Importance weights [num_tokens]
    """
    log_ratio = proximal_log_probs - behavior_log_probs
    importance_weights = log_ratio.exp()

    if clip_min is not None or clip_max is not None:
        importance_weights = torch.clamp(
            importance_weights,
            min=clip_min if clip_min is not None else float('-inf'),
            max=clip_max if clip_max is not None else float('inf')
        )

    return importance_weights


@torch.compile(dynamic=True)
def compute_decoupled_policy_loss(
    log_ratio_intermediate: torch.Tensor,  # log(π_prox) - log(π_θ) from loss.py
    importance_weights: torch.Tensor,  # π_prox / π_behav
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute decoupled PPO policy loss with importance sampling.

    Loss = π_prox/π_behav * min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    where ratio = π_θ / π_prox

    Args:
        log_ratio_intermediate: log(π_prox) - log(π_θ), computed in loss.py
            Note: This is the NEGATIVE of what we want for the ratio, so we negate it
        importance_weights: π_prox / π_behav
        advantages: Advantage values [num_tokens]
        eps_clip: Lower clip threshold
        eps_clip_high: Upper clip threshold
        eps_clip_c: Optional dual-clip parameter

    Returns:
        pg_losses: Per-token losses [num_tokens]
        clipfrac: Clipping fraction (for monitoring)
    """
    # Compute ratio: π_θ / π_prox = exp(-(log(π_prox) - log(π_θ)))
    #                              = exp(log(π_θ) - log(π_prox))
    #                              = π_θ / π_prox
    ratio_prox = (-log_ratio_intermediate).exp()

    # Unclipped loss
    pg_losses1 = -ratio_prox * advantages

    # Clipped loss
    ratio_prox_clipped = ratio_prox.clamp(1 - eps_clip, 1 + eps_clip_high)
    pg_losses2 = -ratio_prox_clipped * advantages

    # Take maximum (pessimistic update)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # Clipping fraction
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    # Apply dual-clip if specified
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    # Apply importance weights: π_prox / π_behav
    pg_losses = pg_losses * importance_weights

    return pg_losses, clipfrac


def get_grpo_offpolicy_advantages(
    rewards: torch.Tensor,
    kl: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Compute advantages for off-policy GRPO.

    Same as on-policy GRPO, but will be used with importance sampling.

    Args:
        rewards: Reward values [num_samples]
        kl: List of KL divergence tensors per sample

    Returns:
        List of advantage tensors (one per sample)
    """
    from slime.utils.ppo_utils import get_grpo_returns
    return get_grpo_returns(rewards, kl)


class StalenessController:
    """
    Controls the staleness of training data in off-policy RL.

    Enforces the constraint: ⌊(N_r - 1) / B⌋ ≤ i + η
    where:
        - N_r: number of generated trajectories
        - B: batch size
        - i: current policy version
        - η: max staleness parameter
    """

    def __init__(self, max_staleness: int, batch_size: int):
        """
        Args:
            max_staleness: Maximum allowed staleness (η)
            batch_size: Training batch size (B)
        """
        self.max_staleness = max_staleness
        self.batch_size = batch_size
        self.num_generated = 0
        self.current_policy_version = 0

    def can_submit_request(self) -> bool:
        """
        Check if a new generation request can be submitted.

        Returns:
            True if request can be submitted without violating staleness constraint
        """
        if self.max_staleness < 0:  # No constraint
            return True

        max_allowed_generated = (self.current_policy_version + self.max_staleness) * self.batch_size + 1
        return self.num_generated < max_allowed_generated

    def on_generation_completed(self, num_samples: int = 1):
        """Record that generation has completed."""
        self.num_generated += num_samples

    def on_training_step(self):
        """Record that a training step has completed (policy version incremented)."""
        self.current_policy_version += 1

    def get_current_staleness(self) -> int:
        """
        Get the current maximum staleness in the buffer.

        Returns:
            Current staleness level
        """
        if self.batch_size == 0:
            return 0
        return (self.num_generated - 1) // self.batch_size - self.current_policy_version

    def reset(self):
        """Reset the controller."""
        self.num_generated = 0
        self.current_policy_version = 0

    def get_stats(self) -> dict:
        """Get statistics for logging."""
        return {
            "num_generated": self.num_generated,
            "policy_version": self.current_policy_version,
            "current_staleness": self.get_current_staleness(),
            "max_staleness": self.max_staleness,
        }


def compute_policy_version_staleness(
    policy_versions: List[int],
    current_version: int,
) -> torch.Tensor:
    """
    Compute staleness for each sample based on policy version.

    Args:
        policy_versions: Policy version for each sample
        current_version: Current policy version

    Returns:
        Staleness values [num_samples]
    """
    policy_versions_tensor = torch.tensor(policy_versions, dtype=torch.long)
    staleness = current_version - policy_versions_tensor
    return staleness


def validate_offpolicy_batch(
    batch: dict,
    max_staleness: int,
    current_policy_version: int,
    raise_error: bool = True,
) -> Tuple[bool, dict]:
    """
    Validate that a batch satisfies staleness constraints.

    Args:
        batch: Training batch with 'policy_versions' field
        max_staleness: Maximum allowed staleness
        current_policy_version: Current policy version
        raise_error: Whether to raise error on validation failure

    Returns:
        (is_valid, stats_dict)
    """
    if "policy_versions" not in batch:
        if raise_error:
            raise ValueError("Batch must contain 'policy_versions' for off-policy training")
        return False, {}

    policy_versions = batch["policy_versions"]
    staleness = compute_policy_version_staleness(policy_versions, current_policy_version)

    max_batch_staleness = staleness.max().item()
    min_batch_staleness = staleness.min().item()
    mean_batch_staleness = staleness.float().mean().item()

    is_valid = max_batch_staleness <= max_staleness

    stats = {
        "max_batch_staleness": max_batch_staleness,
        "min_batch_staleness": min_batch_staleness,
        "mean_batch_staleness": mean_batch_staleness,
        "is_valid": is_valid,
    }

    if not is_valid and raise_error:
        raise ValueError(
            f"Batch violates staleness constraint: max_staleness={max_batch_staleness} > {max_staleness}"
        )

    return is_valid, stats


def aggregate_importance_weights(
    importance_weights: torch.Tensor,
    loss_masks: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Aggregate importance weights for monitoring.

    Args:
        importance_weights: Importance weights [num_tokens]
        loss_masks: Loss masks [num_tokens]
        method: Aggregation method ('mean', 'max', 'min', 'effective_sample_size')

    Returns:
        Aggregated value (scalar)
    """
    masked_weights = importance_weights * loss_masks
    num_valid = loss_masks.sum()

    if num_valid == 0:
        return torch.tensor(0.0, device=importance_weights.device)

    if method == "mean":
        return masked_weights.sum() / num_valid
    elif method == "max":
        return masked_weights.max()
    elif method == "min":
        return masked_weights[loss_masks.bool()].min() if num_valid > 0 else torch.tensor(0.0)
    elif method == "effective_sample_size":
        # ESS = (Σw)^2 / Σw^2
        sum_w = masked_weights.sum()
        sum_w2 = (masked_weights ** 2).sum()
        if sum_w2 > 0:
            return sum_w ** 2 / sum_w2
        else:
            return torch.tensor(0.0, device=importance_weights.device)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
