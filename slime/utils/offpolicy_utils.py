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


def approximate_proximal_log_probs(
    rollout_log_probs: List[torch.Tensor],
    old_actor_log_probs: List[torch.Tensor],
    method: str = "linear",
    alpha: float = 0.5,
) -> List[torch.Tensor]:
    """
    Approximate proximal policy log probs without forward pass.

    This function approximates π_prox based on existing log probs to save computation.

    Supported methods:
    - "linear": π_prox ≈ π_rollout + α * (π_old - π_rollout)
    - "old_actor": π_prox ≈ π_old (use old_actor as proximal policy)
    - "rollout": π_prox ≈ π_rollout (use rollout policy as proximal policy)

    Args:
        rollout_log_probs: Log probs from rollout policy (behavior policy) [list of tensors]
        old_actor_log_probs: Log probs from old actor (previous training policy) [list of tensors]
        method: Approximation method
        alpha: Interpolation coefficient for linear method (0 = rollout, 1 = old_actor)

    Returns:
        Approximated proximal log probs [list of tensors]
    """
    if method == "old_actor":
        # Simple: use old_actor as proximal policy
        return old_actor_log_probs
    elif method == "rollout":
        # Simple: use rollout policy as proximal policy
        return rollout_log_probs
    elif method == "linear":
        # Linear interpolation: π_prox ≈ π_rollout + α * (π_old - π_rollout)
        proximal_log_probs = []
        for rollout_logp, old_logp in zip(rollout_log_probs, old_actor_log_probs):
            # Ensure same device and dtype
            if rollout_logp.device != old_logp.device:
                old_logp = old_logp.to(device=rollout_logp.device)
            if rollout_logp.dtype != old_logp.dtype:
                old_logp = old_logp.to(dtype=rollout_logp.dtype)

            # Linear interpolation in log space
            approx_logp = rollout_logp + alpha * (old_logp - rollout_logp)
            proximal_log_probs.append(approx_logp)
        return proximal_log_probs
    else:
        raise ValueError(f"Unknown approximation method: {method}. "
                        f"Supported methods: 'linear', 'old_actor', 'rollout'")


def apply_m2po_filtering(
    m2: torch.Tensor,
    loss_masks: List[torch.Tensor],
    threshold: float = 0.1,
    policy_version_gaps: Optional[List[int]] = None,
    min_gap_for_filtering: int = 0,  # Changed from 2 to 0: no gap restriction by default
) -> Tuple[List[torch.Tensor], int]:
    """
    Apply M2PO (Second-Momentum PPO) filtering to remove high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    behavior and proximal log-probabilities) to reduce gradient variance.

    Filtering logic (from AReaL):
    1. Sort tokens by m2 in descending order
    2. Find cutoff point where average m2 of remaining tokens < threshold
    3. Filter out tokens with highest m2 values

    Only applies filtering when policy_version_gap >= min_gap_for_filtering.

    Args:
        m2: Second-momentum values (log(π_behave) - log(π_prox))^2 [num_tokens]
        loss_masks: Loss masks for each sample [list of tensors]
        threshold: M2 threshold (default: 0.1 from AReaL paper)
        policy_version_gaps: Policy version gap for each sample [list of ints]
        min_gap_for_filtering: Minimum policy version gap to trigger filtering (default: 0, no restriction)

    Returns:
        modified_loss_masks: Updated loss masks with filtered tokens set to 0
        num_filtered: Number of tokens that were filtered out
    """
    # Flatten all masks to match m2 shape
    all_masks_flat = torch.cat(loss_masks, dim=0)

    # Validate shapes
    if m2.shape[0] != all_masks_flat.shape[0]:
        raise ValueError(
            f"Shape mismatch: m2 {m2.shape} vs loss_masks {all_masks_flat.shape}"
        )

    # Create per-token gap condition if provided
    gap_condition = None
    if policy_version_gaps is not None and min_gap_for_filtering > 0:
        # Create per-token gap tensor
        token_gaps = []
        for gap, mask in zip(policy_version_gaps, loss_masks):
            token_gaps.append(torch.full_like(mask, gap, dtype=torch.long))
        token_gaps_flat = torch.cat(token_gaps, dim=0)

        # Only apply filtering where gap >= min_gap_for_filtering
        gap_condition = token_gaps_flat >= min_gap_for_filtering

    # Apply M2PO filtering logic (from AReaL)
    mask_flat = all_masks_flat.bool()
    m2_selected = m2.view(-1)[mask_flat]

    # DEBUG: Print filtering statistics
    print(f"[M2PO FILTER DEBUG] Total tokens: {m2.numel()}, Valid tokens (mask=1): {m2_selected.numel()}")
    if m2_selected.numel() > 0:
        print(f"[M2PO FILTER DEBUG] m2_selected min/max/mean: {m2_selected.min().item():.6f} / {m2_selected.max().item():.6f} / {m2_selected.mean().item():.6f}")
        print(f"[M2PO FILTER DEBUG] threshold: {threshold}, threshold*0.1: {threshold*0.1}")

    if m2_selected.numel() == 0:
        # No valid tokens to filter
        print(f"[M2PO FILTER DEBUG] Early exit: no valid tokens")
        return loss_masks, 0

    # Early exit optimization: if all m2 values are very small, skip filtering
    # This handles the case when gap = 0 or 1 where m2 ≈ 0
    if m2_selected.max() < threshold * 0.1:
        # All m2 values are far below threshold, no need to filter
        print(f"[M2PO FILTER DEBUG] Early exit: max m2 ({m2_selected.max().item():.6f}) < threshold*0.1 ({threshold*0.1:.6f})")
        return loss_masks, 0

    # Sort m2 values in descending order
    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)

    # Get M2PO loss mask based on threshold
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2,
        m2_threshold=threshold
    )

    # DEBUG: Print mask statistics
    num_to_filter = (~sorted_m2_loss_mask).sum().item()
    print(f"[M2PO FILTER DEBUG] After _get_m2po_loss_mask: will filter {num_to_filter}/{sorted_m2_loss_mask.numel()} tokens")
    print(f"[M2PO FILTER DEBUG] sorted_m2 top 5: {sorted_m2[:5].tolist()}")
    print(f"[M2PO FILTER DEBUG] sorted_m2_loss_mask top 5: {sorted_m2_loss_mask[:5].tolist()}")

    # Restore original order
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    # Create full mask
    m2_full_flat = torch.ones_like(mask_flat, dtype=torch.bool, device=m2.device)
    m2_full_flat[mask_flat] = m2_selected_mask

    # Apply gap condition if provided
    if gap_condition is not None:
        # Only filter tokens where gap >= min_gap_for_filtering
        # For tokens with gap < min_gap, keep them (set mask to True)
        m2_full_flat = m2_full_flat | (~gap_condition)

    # Count filtered tokens
    num_filtered = ((~m2_full_flat) & mask_flat).sum().item()

    # Apply filtering: convert bool mask to float (True=1, False=0)
    filtered_masks_flat = all_masks_flat.clone()
    filtered_masks_flat[~m2_full_flat] = 0

    # Split back into list of tensors with original shapes
    modified_loss_masks = []
    offset = 0
    for mask in loss_masks:
        mask_len = mask.shape[0]
        modified_mask = filtered_masks_flat[offset:offset + mask_len]
        modified_loss_masks.append(modified_mask)
        offset += mask_len

    return modified_loss_masks, num_filtered


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.

    Mask the tokens whose second-momentum is the largest, until the average
    second-momentum of remaining tokens is below the threshold.

    This is a direct port from AReaL's implementation.

    Args:
        sorted_m2: M2 values sorted in descending order [num_tokens]
        m2_threshold: Threshold for average m2

    Returns:
        Boolean mask where True means keep the token, False means filter it out
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask
