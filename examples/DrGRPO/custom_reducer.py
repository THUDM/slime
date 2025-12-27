"""Custom pg_loss reducer for Dr.GRPO.

This module provides a custom reducer that divides by a constant instead of
the number of effective tokens. This is useful for Dr.GRPO algorithm.

Usage:
    --custom-pg-loss-reducer-function-path examples.Dr.GRPO.custom_reducer:get_pg_loss_reducer
"""

from collections.abc import Callable

import torch
from megatron.core import mpu

# Constant divisor instead of effective token count
DIVISOR = 1000.0


def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Custom reducer for pg_loss only. Divides by a constant (DIVISOR)
    instead of the number of effective tokens.

    This function is designed to be used with --custom-pg-loss-reducer-function-path
    so that only pg_loss uses this custom reducer, while other metrics
    (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean.

    Note: This implementation only supports cp_size == 1 (no context parallelism).

    Args:
        total_lengths: List of total sequence lengths (prompt + response). Unused but kept for API compatibility.
        response_lengths: List of response lengths.
        loss_masks: List of loss masks for each sample.
        calculate_per_token_loss: If True, return sum_of_token (no division).
                                 If False, return sum_of_sample_mean with constant divisor.

    Returns:
        A callable function that takes a tensor and returns a scalar tensor.
    """
    assert mpu.get_context_parallel_world_size() == 1, "This custom reducer only supports cp_size == 1"

    if calculate_per_token_loss:

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

        return sum_of_token

    def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
        return sum(
            [
                (x_i * loss_mask_i).sum() / DIVISOR
                for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
            ]
        )

    return sum_of_sample_mean
