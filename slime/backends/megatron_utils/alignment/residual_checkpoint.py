"""Preserve the aligned FP32 residual across MCore activation checkpointing.

The Qwen3 alignment path stores the exact residual sum on the BF16 hidden
state as a tensor attribute. MCore checkpointing detaches tensor inputs before
recomputation, which drops that attribute. The recomputed layers then
normalize the rounded BF16 state and silently lose the intended norm/upstream
gradient path.

This wrapper carries the exact residual as an explicit checkpoint input. A
straight-through autograd boundary exposes its exact forward value while
routing its gradient to the ordinary hidden-state checkpoint input. The
explicit residual is detached so the checkpoint boundary is not counted
twice.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch

EXACT_RESIDUAL_ATTRIBUTE = "_sglang_residual_sum_fp32"
_INSTALL_MARKER = "_slime_exact_residual_checkpoint"


class _ExactResidualBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
        ctx.hidden_dtype = hidden.dtype
        return exact

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad.to(ctx.hidden_dtype), None


def wrap_checkpoint(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an MCore checkpoint function without changing unaligned calls."""

    @wraps(original)
    def checkpoint(function, distribute_saved_activations, *args):
        if not args or not torch.is_tensor(args[0]):
            return original(function, distribute_saved_activations, *args)

        hidden = args[0]
        exact = getattr(hidden, EXACT_RESIDUAL_ATTRIBUTE, None)
        if exact is None:
            return original(function, distribute_saved_activations, *args)
        if distribute_saved_activations:
            raise RuntimeError("Exact residual checkpointing requires undistributed saved activations")
        if exact.shape != hidden.shape or exact.dtype != torch.float32:
            raise RuntimeError("Aligned exact residual must match the hidden-state shape and use FP32")

        def with_exact_residual(*extended_args):
            recompute_hidden = extended_args[0]
            previous = getattr(recompute_hidden, EXACT_RESIDUAL_ATTRIBUTE, None)
            setattr(
                recompute_hidden,
                EXACT_RESIDUAL_ATTRIBUTE,
                _ExactResidualBoundary.apply(recompute_hidden, extended_args[-1]),
            )
            try:
                return function(*extended_args[:-1])
            finally:
                if previous is None:
                    delattr(recompute_hidden, EXACT_RESIDUAL_ATTRIBUTE)
                else:
                    setattr(recompute_hidden, EXACT_RESIDUAL_ATTRIBUTE, previous)

        return original(
            with_exact_residual,
            distribute_saved_activations,
            *args,
            exact.detach(),
        )

    setattr(checkpoint, _INSTALL_MARKER, True)
    return checkpoint


def enable_exact_residual_checkpointing() -> None:
    """Install the process-global wrapper once for both MCore entry points."""
    from megatron.core import tensor_parallel
    from megatron.core.tensor_parallel import random

    if getattr(tensor_parallel.checkpoint, _INSTALL_MARKER, False):
        return

    wrapped = wrap_checkpoint(tensor_parallel.checkpoint)
    tensor_parallel.checkpoint = wrapped
    random.checkpoint = wrapped
