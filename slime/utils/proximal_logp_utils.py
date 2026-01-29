"""
Proximal Policy Log-Probability Approximation Utilities

This module implements version-aware approximation methods for computing
proximal policy log-probabilities in off-policy GRPO, inspired by AReaL.

Key Features:
1. Version-aware interpolation between behavior and current policies
2. Multiple approximation methods (loglinear, linear, rollout)
3. Detailed error metrics for approximation quality assessment
4. M2PO (Second-Momentum PPO) filtering for variance reduction

References:
- AReaL: /mnt/shared-storage-user/puyuan/code/AReaL/areal/engine/ppo/actor.py
- Analysis: /mnt/shared-storage-user/puyuan/code/slime/AREAL_PROXIMAL_LOGP_COMPARISON.md
"""

from typing import Optional, Dict
import torch


# =============================================================================
# Constants
# =============================================================================

PROX_APPROX_METHOD_LOGLINEAR = "loglinear"
PROX_APPROX_METHOD_LINEAR = "linear"
PROX_APPROX_METHOD_ROLLOUT = "rollout"
PROX_APPROX_METHODS_ALL = [
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
]

PROX_LOGP_METHOD_RECOMPUTE = "recompute"
PROX_LOGP_METHOD_LOGLINEAR = "loglinear"
PROX_LOGP_METHOD_LINEAR = "linear"
PROX_LOGP_METHOD_METRICS = "metrics"
PROX_LOGP_METHODS_ALL = [
    PROX_LOGP_METHOD_RECOMPUTE,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_LINEAR,
    PROX_LOGP_METHOD_METRICS,
]


# =============================================================================
# Core Approximation Functions
# =============================================================================

@torch.compile(dynamic=True)
def compute_proximal_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Mathematical Background:
    - v_proximal = current_version - 1 (proximal policy is the last training step)
    - v_behave = versions (policy version that generated each token)
    - v_theta = current_version (current training policy)
    - alpha = (v_proximal - v_behave) / (v_theta - v_behave)

    When alpha = 0 (v_behave == v_proximal): use old_logp directly (no approximation needed)
    When alpha = 1 (v_behave == v_theta): use logprobs directly (fresh sample)
    Otherwise: interpolate between old_logp and logprobs

    Args:
        old_logp: log_p_behave from rollout (behavior policy) [num_tokens]
        logprobs: log_p_theta from current training forward pass [num_tokens]
        versions: per-token policy versions from rollout (v_behave for each token) [num_tokens]
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.

    Example:
        >>> old_logp = torch.tensor([-1.5, -2.0, -1.8])
        >>> logprobs = torch.tensor([-1.3, -1.9, -1.7])
        >>> versions = torch.tensor([8, 9, 10])
        >>> current_version = 10
        >>> approx = compute_proximal_logp_approximations(
        ...     old_logp, logprobs, versions, current_version, method="loglinear"
        ... )
        >>> # Returns {"loglinear": tensor(...)} with interpolated values
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In SLIME, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave

    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            # This is equivalent to: p_prox = p_behave^(1-α) * p_theta^α
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison (baseline)
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def resolve_proximal_logp(
    prox_logp_gt: Optional[torch.Tensor],
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: Optional[torch.Tensor],
    current_version: Optional[int],
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, linear, metrics).
        old_logp: Behavior policy log-probabilities [num_tokens].
        logprobs: Current policy log-probabilities (should be detached) [num_tokens].
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor [num_tokens].

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).

    Example:
        >>> # Scenario 1: RECOMPUTE method (standard)
        >>> prox_logp = resolve_proximal_logp(
        ...     prox_logp_gt=computed_prox_logp,  # from forward pass
        ...     prox_logp_method="recompute",
        ...     old_logp=old_logp,
        ...     logprobs=logprobs,
        ...     versions=None,  # not needed for recompute
        ...     current_version=None,  # not needed for recompute
        ... )
        >>> # Returns: computed_prox_logp

        >>> # Scenario 2: LOGLINEAR approximation (fast)
        >>> prox_logp = resolve_proximal_logp(
        ...     prox_logp_gt=None,  # skip forward pass
        ...     prox_logp_method="loglinear",
        ...     old_logp=old_logp,
        ...     logprobs=logprobs,
        ...     versions=versions,  # required for approximation
        ...     current_version=10,  # required for approximation
        ... )
        >>> # Returns: approximated prox_logp via loglinear interpolation
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if prox_logp_method == PROX_LOGP_METHOD_RECOMPUTE:
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None or current_version is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions or current_version not available. "
                "Cannot proceed without either ground truth or approximation data."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_proximal_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]

    elif prox_logp_method == PROX_LOGP_METHOD_LINEAR:
        # Use linear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_proximal_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LINEAR]

    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)

    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


# =============================================================================
# M2PO (Second-Momentum PPO) Filtering
# =============================================================================

def apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance and improve
    training stability.

    Mathematical Background:
    - Second-momentum: m2 = (old_logp - prox_logp)^2
    - Sort tokens by m2 in descending order
    - Mask tokens until average m2 of remaining tokens < threshold

    This adaptively removes tokens that have the largest divergence between
    behavior and proximal policies, which typically correspond to high-variance
    importance weights that can destabilize training.

    Args:
        old_logp: Behavior policy log-probabilities [batch, seq_len].
        prox_logp: Proximal policy log-probabilities [batch, seq_len].
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering (e.g., 0.1-0.5).

    Returns:
        Updated loss mask with M2PO filtering applied [batch, seq_len].

    Example:
        >>> old_logp = torch.tensor([[-1.5, -2.0], [-1.8, -2.5]])
        >>> prox_logp = torch.tensor([[-1.3, -1.9], [-1.7, -2.4]])
        >>> loss_mask = torch.ones_like(old_logp, dtype=torch.bool)
        >>> filtered_mask = apply_m2po_masking(
        ...     old_logp, prox_logp, loss_mask, m2_threshold=0.1
        ... )
        >>> # Returns: mask with high-variance tokens filtered out
    """
    # Compute second-momentum
    delta = old_logp - prox_logp
    m2 = delta * delta

    # Flatten masks and m2 for processing
    mask_flat = loss_mask.view(-1)
    m2_flat = m2.view(-1)

    # Only process tokens that are in the original loss mask
    m2_selected = m2_flat[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    # Sort by m2 in descending order
    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)

    # Get M2PO loss mask
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )

    # Restore original order
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    # Apply to full mask
    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.

    Mask the tokens whose second-momentum is the largest, until the average
    second-momentum of remaining tokens is below the threshold.

    Args:
        sorted_m2: Second-momentum values sorted in descending order [n].
        m2_threshold: Threshold for average m2.

    Returns:
        Boolean mask [n] where True = keep token, False = filter out.

    Algorithm:
        1. Compute suffix sums: S[i] = sum(sorted_m2[i:])
        2. Compute suffix counts: N[i] = n - i
        3. Compute suffix averages: A[i] = S[i] / N[i]
        4. Find first index k where A[k] < threshold
        5. Mask out first k tokens (highest m2 values)
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
        # All suffix averages are >= threshold. Mask all but one to avoid empty mask.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    # Safety check: ensure at least one token remains
    if loss_mask.sum() == 0:
        raise RuntimeError(
            "All tokens are masked out when applying M2PO filter. "
            f"m2_threshold={m2_threshold} is too strict."
        )

    return loss_mask


# =============================================================================
# Monitoring and Metrics
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Args:
        ground_truth: Ground truth values [num_tokens].
        approximation: Approximated values [num_tokens].

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.

    Example:
        >>> gt = torch.tensor([-1.5, -2.0, -1.8])
        >>> approx = torch.tensor([-1.4, -2.1, -1.9])
        >>> errors = compute_approximation_errors(gt, approx)
        >>> errors.keys()
        dict_keys(['abs_error', 'rel_error', 'squared_error'])
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """
    Compute importance weight as exp(logp_num - logp_denom).

    Args:
        logp_numerator: Log-probabilities in numerator [num_tokens].
        logp_denominator: Log-probabilities in denominator [num_tokens].

    Returns:
        Importance weights [num_tokens].
    """
    return torch.exp(logp_numerator - logp_denominator).float()


def compute_version_staleness(
    versions: torch.Tensor,
    current_version: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute staleness statistics based on policy versions.

    Args:
        versions: Per-token policy versions [num_tokens].
        current_version: Current training version.
        valid_mask: Optional mask for valid tokens [num_tokens].

    Returns:
        Dictionary with staleness statistics (avg, max, min).

    Example:
        >>> versions = torch.tensor([8, 9, 10, 10])
        >>> stats = compute_version_staleness(versions, current_version=10)
        >>> stats['avg']  # Average staleness
        1.0
    """
    v_proximal = current_version - 1
    v_theta = current_version
    v_behave = versions.float()

    # Filter to generated tokens only (version >= 0)
    if valid_mask is not None:
        valid_generated_mask = valid_mask & (versions >= 0)
    else:
        valid_generated_mask = versions >= 0

    if not valid_generated_mask.any():
        return {
            "proximal_avg": 0.0,
            "proximal_max": 0.0,
            "proximal_min": 0.0,
            "theta_avg": 0.0,
            "theta_max": 0.0,
            "theta_min": 0.0,
        }

    # Compute staleness for valid tokens
    staleness_proximal = (v_proximal - v_behave)[valid_generated_mask]
    staleness_theta = (v_theta - v_behave)[valid_generated_mask]

    return {
        "proximal_avg": staleness_proximal.mean().item(),
        "proximal_max": staleness_proximal.max().item(),
        "proximal_min": staleness_proximal.min().item(),
        "theta_avg": staleness_theta.mean().item(),
        "theta_max": staleness_theta.max().item(),
        "theta_min": staleness_theta.min().item(),
    }
