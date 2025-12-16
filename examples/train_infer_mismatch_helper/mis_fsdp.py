from typing import Any

import torch

from .mis import compute_mis_weights


def compute_mis_weights_fsdp(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Compute masked importance sampling weights for FSDP. No context parallelism.

    Args:
        args: Arguments containing MIS settings (use_tis, tis_mode, etc.)
        pg_loss: Policy gradient loss, flattened tensor [total_tokens]
        train_log_probs: Training log probs, list of 1D tensors per sequence
        rollout_log_probs: Rollout log probs, list of 1D tensors per sequence
        loss_masks: Loss masks, list of 1D tensors per sequence
        **kwargs: Additional arguments (cp_rank, cp_size, etc.) for compatibility

    Returns:
        pg_loss: Policy gradient loss with IS weights applied
        modified_masks: Modified loss masks after rejection sampling
        mis_metrics: Metrics dict with flattened tensors
    """
    is_weights, modified_masks, is_metrics = compute_mis_weights(
        args=args,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    result_metrics = {}
    if is_weights is not None:
        is_weights_flat = torch.cat(is_weights, dim=0)
        pg_loss = pg_loss * is_weights_flat

    for key, values in is_metrics.items():
        result_metrics[f"mis_{key}"] = torch.cat(values, dim=0)

    return pg_loss, modified_masks, result_metrics
