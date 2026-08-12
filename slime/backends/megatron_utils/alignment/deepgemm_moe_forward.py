"""Align Megatron TEGroupedMLP with SGLang's DeepGEMM MoE path.

    block-FP8 fc1 -> SwiGLU -> block-FP8 fc2 -> FP32 router-probability multiply

Moving the probability multiply after fc2 matches SGLang more closely than
wrapping the two grouped linears independently.  The wrapper installs a custom
autograd function:

    forward  = block-FP8 grouped DeepGEMM;
    backward = grouped BF16 dgrad/wgrad GEMMs plus analytic
               SwiGLU/router-probability gradients.
"""

from __future__ import annotations

import logging
import math
import os
import re
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core import parallel_state

from slime.backends.megatron_utils.alignment.deepgemm_forward import (
    _deepgemm_bf16_gemm_nn,
    _deepgemm_bf16_gemm_nt,
    _deepgemm_bf16_gemm_tn,
    _format_int_ranges,
    _should_log_deepgemm_summary,
    _sum_to_parameter_dtype,
)
from slime.backends.sglang_utils.compat import import_sglang_module

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 128
_GROUPED_M_ALIGNMENT = 128
_ROUTER_PROBABILITY_CHUNK_ROWS = 1024
_UNPAD_CHUNK_ROWS = 1024
_BACKWARD_CHUNK_ROWS = 1024
_SWIGLU_POINTWISE_CHUNK_ROWS = 512
_DEFAULT_EXPERTS_PER_GROUP = 4
_DEFAULT_BACKWARD_EXPERTS_PER_GROUP = 4
_DEFAULT_BACKWARD_MAX_PADDED_BYTES = 256 * 1024 * 1024
_DEFAULT_TARGET_SUFFIXES = ("mlp.experts",)
_PREALLOCATED_COMBINE_BUFFER_ATTR = "_slime_preallocated_combine_buffer"
_PREALLOCATED_TOKEN_COMBINE_ATTR = "_slime_preallocated_token_combine"
_COMBINE_WORKSPACE_ATTR = "_slime_combine_workspace"
_LAYER_PATH_RE = re.compile(r"^(?P<layer_path>(?:.*\.)?decoder\.layers\.(?P<local_layer_index>\d+))(?:\.|$)")


@dataclass(frozen=True)
class _MoELayout:
    num_local_experts: int
    hidden_size: int
    ffn_hidden_size: int

    @property
    def fc1_weight_shape(self) -> tuple[int, int]:
        return (2 * self.ffn_hidden_size, self.hidden_size)

    @property
    def fc2_weight_shape(self) -> tuple[int, int]:
        return (self.hidden_size, self.ffn_hidden_size)


@dataclass(frozen=True)
class _DeepGEMMOps:
    quantize_weight: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    quantize_activation: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    align_input_scale: Callable[[torch.Tensor], torch.Tensor]
    grouped_gemm: Callable[..., Any]
    silu_and_mul: Callable[[torch.Tensor, torch.Tensor], Any]
    # Blackwell (sm100+) uses UE8M0 (power-of-two) block scales; Hopper (sm90)
    # uses FP32 block scales. When ``scale_ue8m0`` is False the H100 path below
    # is byte-for-byte unchanged.
    scale_ue8m0: bool = False
    need_tma_aligned_scales: bool = True
    transform_weight_scale: Callable[..., torch.Tensor] | None = None


def _apply_router_probability_fp32_inplace(
    down_output: torch.Tensor,
    permuted_probs: torch.Tensor,
) -> torch.Tensor:
    """Apply the post-fc2 router probability without a full-size FP32 temporary.

    SGLang performs this multiply in FP32 and casts the result back to the
    activation dtype.  A single expression such as
    ``(down_output.float() * probs.float()).to(dtype)`` temporarily materializes
    the entire MoE output in FP32.  For a packed 4x4 rollout this can exceed
    13 GiB per rank.  Processing independent row chunks is numerically identical
    while keeping the FP32 workspace bounded.

    The DeepGEMM output is scratch storage owned by this forward, so updating it
    in place also avoids allocating a second full-size BF16 tensor.
    """
    if down_output.ndim != 2:
        raise RuntimeError(f"Expected a 2D MoE output, got shape {tuple(down_output.shape)}")
    probabilities_fp32 = permuted_probs.detach().reshape(-1, 1).to(torch.float32)
    if probabilities_fp32.shape[0] != down_output.shape[0]:
        raise RuntimeError(
            "MoE output/router probability row mismatch: " f"{down_output.shape[0]} != {probabilities_fp32.shape[0]}"
        )

    for start in range(0, down_output.shape[0], _ROUTER_PROBABILITY_CHUNK_ROWS):
        end = min(start + _ROUTER_PROBABILITY_CHUNK_ROWS, down_output.shape[0])
        scaled = (down_output[start:end].to(torch.float32) * probabilities_fp32[start:end]).to(down_output.dtype)
        down_output[start:end].copy_(scaled)
    return down_output


def _router_probability_grad_fp32_chunked(
    grad_output: torch.Tensor,
    down_output: torch.Tensor,
) -> torch.Tensor:
    """Compute the per-row router-probability gradient with bounded scratch.

    Each output row is an independent hidden-dimension dot product.  Row
    chunking therefore preserves the exact FP32 reduction performed by the
    unchunked expression while avoiding two route-sized FP32 casts plus their
    product being live at once.
    """
    if grad_output.ndim != 2 or down_output.ndim != 2:
        raise RuntimeError(
            "Expected 2D router-gradient inputs, got " f"{tuple(grad_output.shape)} and {tuple(down_output.shape)}"
        )
    if grad_output.shape != down_output.shape:
        raise RuntimeError(
            "Router-gradient input shape mismatch: " f"{tuple(grad_output.shape)} != {tuple(down_output.shape)}"
        )

    result = torch.empty(
        (grad_output.shape[0], 1),
        dtype=torch.float32,
        device=grad_output.device,
    )
    for start in range(0, grad_output.shape[0], _ROUTER_PROBABILITY_CHUNK_ROWS):
        end = min(start + _ROUTER_PROBABILITY_CHUNK_ROWS, grad_output.shape[0])
        result[start:end].copy_(
            (grad_output[start:end].float() * down_output[start:end].float()).sum(
                dim=-1,
                keepdim=True,
            )
        )
    return result


def _compact_valid_rows_inplace(
    padded_value: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    """Remove per-expert padding without a second full-size activation tensor.

    ``valid_rows`` is produced in ascending expert/row order and always
    satisfies ``valid_rows[i] >= i``.  Therefore copying ascending chunks into
    the prefix cannot overwrite a source needed by a later chunk.  Each
    ``index_select`` only materializes a bounded temporary, and the returned
    prefix view keeps ownership of the original DeepGEMM output storage.
    """
    if padded_value.ndim != 2 or valid_rows.ndim != 1:
        raise RuntimeError(
            "Expected a 2D padded value and 1D valid-row indices, got "
            f"{tuple(padded_value.shape)} and {tuple(valid_rows.shape)}"
        )
    if valid_rows.numel() > padded_value.shape[0]:
        raise RuntimeError(f"Valid-row count {valid_rows.numel()} exceeds padded rows {padded_value.shape[0]}")
    if valid_rows.numel() == 0:
        return padded_value.narrow(0, 0, 0)

    # Avoid adding device synchronizations to every MoE layer.  CUDA indices
    # come exclusively from _pad_expert_rows, which guarantees these
    # invariants by construction; retain the defensive validation for CPU
    # callers and unit tests.
    if not valid_rows.is_cuda:
        expected_positions = torch.arange(
            valid_rows.numel(),
            device=valid_rows.device,
            dtype=valid_rows.dtype,
        )
        if bool(torch.any(valid_rows < expected_positions)):
            raise RuntimeError("Valid-row indices cannot move a row backward into a future source")
        if bool(torch.any(valid_rows[1:] <= valid_rows[:-1])):
            raise RuntimeError("Valid-row indices must be strictly increasing")
        if int(valid_rows[-1].item()) >= padded_value.shape[0]:
            raise RuntimeError(
                f"Valid-row index {int(valid_rows[-1].item())} exceeds padded rows {padded_value.shape[0]}"
            )

    for start in range(0, valid_rows.numel(), _UNPAD_CHUNK_ROWS):
        end = min(start + _UNPAD_CHUNK_ROWS, valid_rows.numel())
        selected = padded_value.index_select(0, valid_rows[start:end])
        padded_value[start:end].copy_(selected)
    return padded_value.narrow(0, 0, valid_rows.numel())


def _swiglu_forward_chunked(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Evaluate FP32 SwiGLU into BF16 storage with bounded temporaries."""
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise RuntimeError(
            "SwiGLU gate/up mismatch: " f"{tuple(gate.shape)}/{gate.dtype} != {tuple(up.shape)}/{up.dtype}"
        )
    down_input = torch.empty_like(gate)
    for start in range(0, gate.shape[0], _SWIGLU_POINTWISE_CHUNK_ROWS):
        end = min(start + _SWIGLU_POINTWISE_CHUNK_ROWS, gate.shape[0])
        gate_f = gate[start:end].float()
        up_f = up[start:end].float()
        silu_gate = F.silu(gate_f)
        down_input[start:end].copy_((silu_gate * up_f).to(dtype=gate.dtype))
    return down_input


def _swiglu_backward_chunked(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_down_input: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the established FP32 SwiGLU derivative with bounded temporaries."""
    if gate.shape != up.shape or gate.shape != grad_down_input.shape:
        raise RuntimeError(
            "SwiGLU backward shape mismatch: "
            f"{tuple(gate.shape)}, {tuple(up.shape)}, {tuple(grad_down_input.shape)}"
        )
    if gate.dtype != up.dtype or gate.dtype != grad_down_input.dtype:
        raise RuntimeError("SwiGLU backward dtype mismatch: " f"{gate.dtype}, {up.dtype}, {grad_down_input.dtype}")

    grad_gate_up = torch.empty(
        (gate.shape[0], 2 * gate.shape[1]),
        device=gate.device,
        dtype=gate.dtype,
    )
    grad_gate_out, grad_up_out = grad_gate_up.chunk(2, dim=-1)
    for start in range(0, gate.shape[0], _SWIGLU_POINTWISE_CHUNK_ROWS):
        end = min(start + _SWIGLU_POINTWISE_CHUNK_ROWS, gate.shape[0])
        gate_f = gate[start:end].float()
        up_f = up[start:end].float()
        grad_down_input_f = grad_down_input[start:end].float()
        silu_gate = F.silu(gate_f)
        sigmoid_gate = torch.sigmoid(gate_f)
        grad_gate = grad_down_input_f * up_f * sigmoid_gate * (1.0 + gate_f * (1.0 - sigmoid_gate))
        grad_up = grad_down_input_f * silu_gate
        grad_gate_out[start:end].copy_(grad_gate.to(dtype=gate.dtype))
        grad_up_out[start:end].copy_(grad_up.to(dtype=gate.dtype))
    return grad_gate_up


def _grouped_bf16_backward_experts_per_group(num_local_experts: int) -> int:
    configured = os.environ.get("SLIME_DEEPGEMM_MOE_BF16_BACKWARD_EXPERTS_PER_GROUP")
    if configured is None:
        return min(_DEFAULT_BACKWARD_EXPERTS_PER_GROUP, num_local_experts)
    try:
        experts_per_group = int(configured)
    except ValueError as exc:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_BF16_BACKWARD_EXPERTS_PER_GROUP must be a " f"positive integer, got {configured!r}"
        ) from exc
    if experts_per_group <= 0:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_BF16_BACKWARD_EXPERTS_PER_GROUP must be a "
            f"positive integer, got {experts_per_group}"
        )
    return min(experts_per_group, num_local_experts)


def _grouped_bf16_backward_max_padded_bytes() -> int:
    configured = os.environ.get("SLIME_DEEPGEMM_MOE_BF16_BACKWARD_MAX_PADDED_BYTES")
    if configured is None:
        return _DEFAULT_BACKWARD_MAX_PADDED_BYTES
    try:
        max_padded_bytes = int(configured)
    except ValueError as exc:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_BF16_BACKWARD_MAX_PADDED_BYTES must be a " f"positive integer, got {configured!r}"
        ) from exc
    if max_padded_bytes <= 0:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_BF16_BACKWARD_MAX_PADDED_BYTES must be a " f"positive integer, got {max_padded_bytes}"
        )
    return max_padded_bytes


def _padded_expert_hidden_bytes(
    count: int,
    *,
    hidden_size: int,
    element_size: int,
) -> int:
    padded_count = ((count + _GROUPED_M_ALIGNMENT - 1) // _GROUPED_M_ALIGNMENT) * _GROUPED_M_ALIGNMENT if count else 0
    return padded_count * hidden_size * element_size


def _grouped_bf16_backward_expert_ranges(
    counts: tuple[int, ...],
    *,
    hidden_size: int,
    element_size: int,
) -> tuple[tuple[int, int], ...]:
    """Greedily group adjacent experts without exceeding the padded-input cap."""
    experts_per_group = _grouped_bf16_backward_experts_per_group(len(counts))
    max_padded_bytes = _grouped_bf16_backward_max_padded_bytes()
    ranges = []
    expert_start = 0
    while expert_start < len(counts):
        expert_end = expert_start
        group_padded_bytes = 0
        while expert_end < len(counts) and expert_end - expert_start < experts_per_group:
            expert_padded_bytes = _padded_expert_hidden_bytes(
                counts[expert_end],
                hidden_size=hidden_size,
                element_size=element_size,
            )
            if expert_end > expert_start and group_padded_bytes + expert_padded_bytes > max_padded_bytes:
                break
            group_padded_bytes += expert_padded_bytes
            expert_end += 1
        ranges.append((expert_start, expert_end))
        expert_start = expert_end
    return tuple(ranges)


def _use_grouped_bf16_backward(
    hidden_states: torch.Tensor,
    counts: tuple[int, ...],
    needs_fc1_weights: tuple[bool, ...],
    needs_fc2_weights: tuple[bool, ...],
) -> bool:
    enabled = os.environ.get("SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    max_padded_bytes = _grouped_bf16_backward_max_padded_bytes()
    largest_expert_bytes = max(
        (
            _padded_expert_hidden_bytes(
                count,
                hidden_size=hidden_states.shape[1],
                element_size=hidden_states.element_size(),
            )
            for count in counts
        ),
        default=0,
    )
    if (
        not enabled
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
        # A single expert cannot be split by the contiguous grouped kernel.
        # Fall back to the established 1024-row path for that rare hot-expert
        # case instead of risking a full-padded allocation OOM.
        or largest_expert_bytes > max_padded_bytes
    ):
        return False

    if any(needs_fc1_weights) or any(needs_fc2_weights):
        import deep_gemm

        if not hasattr(deep_gemm, "k_grouped_bf16_gemm_tn_contiguous"):
            return False
    return True


def _deepgemm_bf16_m_grouped_gemm_nt(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    """Compute contiguous expert-major ``lhs @ rhs[group].T`` in one launch."""
    if lhs.ndim != 2 or rhs.ndim != 3 or lhs.shape[1] != rhs.shape[2]:
        raise RuntimeError(f"Grouped BF16 NT GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if grouped_layout.shape != (lhs.shape[0],) or grouped_layout.dtype != torch.int32:
        raise RuntimeError(
            "Grouped BF16 NT layout mismatch: "
            f"{tuple(grouped_layout.shape)}/{grouped_layout.dtype} for {lhs.shape[0]} rows"
        )
    if not (lhs.is_cuda and rhs.is_cuda and grouped_layout.is_cuda):
        raise RuntimeError("Grouped BF16 backward requires CUDA tensors")
    import deep_gemm

    out = torch.empty((lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        lhs.contiguous(),
        rhs.contiguous(),
        out,
        grouped_layout.contiguous(),
    )
    return out


def _deepgemm_bf16_m_grouped_gemm_nn(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    grouped_layout: torch.Tensor,
) -> torch.Tensor:
    """Compute contiguous expert-major ``lhs @ rhs[group]`` in one launch."""
    if lhs.ndim != 2 or rhs.ndim != 3 or lhs.shape[1] != rhs.shape[1]:
        raise RuntimeError(f"Grouped BF16 NN GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if grouped_layout.shape != (lhs.shape[0],) or grouped_layout.dtype != torch.int32:
        raise RuntimeError(
            "Grouped BF16 NN layout mismatch: "
            f"{tuple(grouped_layout.shape)}/{grouped_layout.dtype} for {lhs.shape[0]} rows"
        )
    if not (lhs.is_cuda and rhs.is_cuda and grouped_layout.is_cuda):
        raise RuntimeError("Grouped BF16 backward requires CUDA tensors")
    import deep_gemm

    out = torch.empty((lhs.shape[0], rhs.shape[2]), device=lhs.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
        lhs.contiguous(),
        rhs.contiguous(),
        out,
        grouped_layout.contiguous(),
    )
    return out


def _deepgemm_bf16_k_grouped_gemm_tn(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    grouped_k: tuple[int, ...],
) -> torch.Tensor:
    """Compute per-expert ``lhs.T @ rhs`` weight gradients in one launch."""
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] != rhs.shape[0]:
        raise RuntimeError(f"Grouped BF16 TN GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if sum(grouped_k) != lhs.shape[0]:
        raise RuntimeError(
            "Grouped BF16 TN row-count mismatch: " f"sum({grouped_k})={sum(grouped_k)} != {lhs.shape[0]}"
        )
    if not grouped_k:
        raise RuntimeError("Grouped BF16 TN GEMM requires at least one expert")
    if any(k < 0 or k % _GROUPED_M_ALIGNMENT for k in grouped_k):
        raise RuntimeError(
            f"Grouped BF16 TN K sizes must be non-negative multiples of {_GROUPED_M_ALIGNMENT}: {grouped_k}"
        )
    if not (lhs.is_cuda and rhs.is_cuda):
        raise RuntimeError("Grouped BF16 TN GEMM requires CUDA tensors")
    if lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16:
        raise RuntimeError(f"Grouped BF16 TN GEMM requires BF16 tensors, got {lhs.dtype} and {rhs.dtype}")

    import deep_gemm

    # The final Megatron expert gradients are BF16.  Asking the grouped kernel
    # to write BF16 directly matches one full per-expert DeepGEMM TN launch and
    # avoids retaining an additional FP32 copy of every expert weight gradient.
    out = torch.zeros(
        (len(grouped_k), lhs.shape[1], rhs.shape[1]),
        device=lhs.device,
        dtype=torch.bfloat16,
    )
    grouped_k_tensor = torch.tensor(grouped_k, device=lhs.device, dtype=torch.int32)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        lhs.contiguous(),
        rhs.contiguous(),
        out,
        grouped_k,
        grouped_k_tensor,
        out,
    )
    return out


def _grouped_expert_backward(
    *,
    hidden_states: torch.Tensor,
    permuted_probs: torch.Tensor,
    grad_output: torch.Tensor,
    fc1_weights: tuple[torch.Tensor, ...],
    fc2_weights: tuple[torch.Tensor, ...],
    counts: tuple[int, ...],
    layout: _MoELayout,
    needs_hidden: bool,
    needs_probs: bool,
    needs_fc1_weights: tuple[bool, ...],
    needs_fc2_weights: tuple[bool, ...],
    defer_router_probabilities: bool,
    grad_hidden: torch.Tensor | None,
    grad_probs: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    list[torch.Tensor | None],
    list[torch.Tensor | None],
]:
    """Grouped-BF16 dgrad/wgrad without changing the aligned forward."""
    probabilities = permuted_probs.reshape(-1, 1)
    token_offset = 0
    grad_fc1_weights: list[torch.Tensor | None] = [None] * layout.num_local_experts
    grad_fc2_weights: list[torch.Tensor | None] = [None] * layout.num_local_experts

    expert_ranges = _grouped_bf16_backward_expert_ranges(
        counts,
        hidden_size=hidden_states.shape[1],
        element_size=hidden_states.element_size(),
    )
    for expert_start, expert_end in expert_ranges:
        group_counts = counts[expert_start:expert_end]
        group_tokens = sum(group_counts)
        if group_tokens == 0:
            for expert_index in range(expert_start, expert_end):
                if needs_fc1_weights[expert_index]:
                    grad_fc1_weights[expert_index] = torch.zeros_like(fc1_weights[expert_index])
                if needs_fc2_weights[expert_index]:
                    grad_fc2_weights[expert_index] = torch.zeros_like(fc2_weights[expert_index])
            continue

        hidden = hidden_states.narrow(0, token_offset, group_tokens)
        grad = grad_output.narrow(0, token_offset, group_tokens)
        probability = probabilities.narrow(0, token_offset, group_tokens)
        padded_hidden, padded_counts, valid_rows = _pad_expert_rows(hidden, group_counts)

        grouped_layout = _build_m_indices(padded_counts, device=hidden_states.device)
        fc1_group = torch.stack(fc1_weights[expert_start:expert_end], dim=0)
        gate_up = _deepgemm_bf16_m_grouped_gemm_nt(
            padded_hidden,
            fc1_group,
            grouped_layout,
        )
        needs_fc1_group = needs_fc1_weights[expert_start:expert_end]
        needs_fc2_group = needs_fc2_weights[expert_start:expert_end]
        if not any(needs_fc1_group):
            # The FC1 input is otherwise not needed again in backward.
            del padded_hidden
        gate, up = gate_up.chunk(2, dim=-1)

        fc2_group = torch.stack(fc2_weights[expert_start:expert_end], dim=0)
        padded_grad, grad_padded_counts, grad_valid_rows = _pad_expert_rows(grad, group_counts)
        if grad_padded_counts != padded_counts or (valid_rows is None) != (grad_valid_rows is None):
            raise RuntimeError("Grouped BF16 backward produced inconsistent gradient padding")

        padded_probability = None
        probability_valid_rows = None
        if not defer_router_probabilities:
            padded_probability, probability_padded_counts, probability_valid_rows = _pad_expert_rows(
                probability,
                group_counts,
            )
            if probability_padded_counts != padded_counts or (valid_rows is None) != (probability_valid_rows is None):
                raise RuntimeError("Grouped BF16 backward produced inconsistent probability padding")

        down_input = None
        if (needs_probs and not defer_router_probabilities) or any(needs_fc2_group):
            down_input = _swiglu_forward_chunked(gate, up)
        if needs_probs and not defer_router_probabilities:
            assert down_input is not None
            down_output = _deepgemm_bf16_m_grouped_gemm_nt(
                down_input,
                fc2_group,
                grouped_layout,
            )
            grad_probs_padded = (padded_grad.float() * down_output.float()).sum(dim=-1, keepdim=True)
            if valid_rows is not None:
                grad_probs_group = _compact_valid_rows_inplace(grad_probs_padded, valid_rows)
            else:
                grad_probs_group = grad_probs_padded
            assert grad_probs is not None
            grad_probs.narrow(0, token_offset, group_tokens).copy_(
                grad_probs_group.reshape_as(permuted_probs.narrow(0, token_offset, group_tokens)).to(
                    dtype=permuted_probs.dtype
                )
            )
            del down_output, grad_probs_padded, grad_probs_group

        if defer_router_probabilities:
            grad_down_output = padded_grad
        else:
            assert padded_probability is not None
            grad_down_output = (padded_grad.float() * padded_probability.float()).to(dtype=hidden_states.dtype)
        if any(needs_fc2_group):
            assert down_input is not None
            grad_fc2_group = _deepgemm_bf16_k_grouped_gemm_tn(
                grad_down_output,
                down_input,
                padded_counts,
            )
            for local_expert_index, needs_weight in enumerate(needs_fc2_group):
                if needs_weight:
                    grad_fc2_weights[expert_start + local_expert_index] = grad_fc2_group[local_expert_index]
            del grad_fc2_group
        del down_input
        del padded_probability
        grad_down_input = _deepgemm_bf16_m_grouped_gemm_nn(
            grad_down_output,
            fc2_group,
            grouped_layout,
        )
        del grad_down_output, padded_grad, fc2_group

        grad_gate_up = _swiglu_backward_chunked(gate, up, grad_down_input)
        del grad_down_input, gate, up, gate_up

        if any(needs_fc1_group):
            grad_fc1_group = _deepgemm_bf16_k_grouped_gemm_tn(
                grad_gate_up,
                padded_hidden,
                padded_counts,
            )
            for local_expert_index, needs_weight in enumerate(needs_fc1_group):
                if needs_weight:
                    grad_fc1_weights[expert_start + local_expert_index] = grad_fc1_group[local_expert_index]
            del grad_fc1_group, padded_hidden

        if needs_hidden:
            grad_hidden_padded = _deepgemm_bf16_m_grouped_gemm_nn(
                grad_gate_up,
                fc1_group,
                grouped_layout,
            )
            if valid_rows is not None:
                grad_hidden_group = _compact_valid_rows_inplace(grad_hidden_padded, valid_rows)
            else:
                grad_hidden_group = grad_hidden_padded
            assert grad_hidden is not None
            grad_hidden.narrow(0, token_offset, group_tokens).copy_(grad_hidden_group)
            del grad_hidden_padded, grad_hidden_group

        token_offset += group_tokens
        del (
            valid_rows,
            grad_valid_rows,
            probability_valid_rows,
            grouped_layout,
            fc1_group,
            grad_gate_up,
        )

    return (
        grad_hidden,
        None if defer_router_probabilities else grad_probs,
        grad_fc1_weights,
        grad_fc2_weights,
    )


def _ordered_route_backward(
    *,
    route_values: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    grad_output: torch.Tensor,
    grad_routes: torch.Tensor | None,
    grad_weights: torch.Tensor | None,
    static_mapping_valid: bool | None = None,
) -> None:
    """Differentiate the ordered top-k gather with an optional static fast path."""
    routes_alias_values = (
        grad_routes is not None
        and grad_routes.untyped_storage().data_ptr() == route_values.untyped_storage().data_ptr()
    )
    use_static_mapping = static_mapping_valid is not None
    if use_static_mapping and static_mapping_valid is None:
        # The hot DeepEP caller passes this bit from its forward scatter,
        # where torch.nonzero has already exposed the route count to Python.
        # Other internal callers retain a safe fallback instead of assuming
        # that padded token slots own an expert-output row.
        static_mapping_valid = bool(torch.all(output_index >= 0).item())
    if use_static_mapping and static_mapping_valid:
        # Dropless fixed-top-k routing produces exactly one valid route row for
        # every flattened [token, top-k] slot.  Express that static mapping
        # directly instead of materializing torch.nonzero's data-dependent
        # output and synchronizing once per MoE layer.
        num_routes = output_index.numel()
        topk = output_index.shape[1]
        flat_output_index = output_index.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_grad_weights = grad_weights.reshape(-1) if grad_weights is not None else None
        chunk_rows = _BACKWARD_CHUNK_ROWS
        # When the forward combine input is dead after this operation, its
        # route-sized storage can hold the route gradient.  Complete every
        # probability gradient first because it still reads the original
        # route values; only then overwrite that storage with route gradients.
        if routes_alias_values and flat_grad_weights is not None:
            for start in range(0, num_routes, chunk_rows):
                end = min(start + chunk_rows, num_routes)
                flat_positions = torch.arange(
                    start,
                    end,
                    device=output_index.device,
                    dtype=torch.long,
                )
                token_rows = torch.div(flat_positions, topk, rounding_mode="floor")
                route_rows = flat_output_index[start:end].to(dtype=torch.long)
                token_grads = grad_output.index_select(0, token_rows)
                selected_route_values = route_values.index_select(0, route_rows)
                weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
                flat_grad_weights[start:end].copy_(weight_grads)

        fused_route_grad = False
        if grad_routes is not None and grad_output.is_cuda:
            from slime.backends.megatron_utils.alignment.deterministic_route_kernels import ordered_route_grad

            ordered_route_grad(
                grad_output.contiguous(),
                topk_weights.contiguous(),
                output_index.contiguous(),
                grad_routes,
            )
            fused_route_grad = True

        if (grad_routes is not None and not fused_route_grad) or (
            flat_grad_weights is not None and not routes_alias_values
        ):
            for start in range(0, num_routes, chunk_rows):
                end = min(start + chunk_rows, num_routes)
                flat_positions = torch.arange(
                    start,
                    end,
                    device=output_index.device,
                    dtype=torch.long,
                )
                token_rows = torch.div(flat_positions, topk, rounding_mode="floor")
                route_rows = flat_output_index[start:end].to(dtype=torch.long)
                token_grads = grad_output.index_select(0, token_rows)
                weights = flat_weights[start:end]

                if grad_routes is not None and not fused_route_grad:
                    route_grads = (token_grads.float() * weights.float().unsqueeze(1)).to(dtype=grad_routes.dtype)
                    # Every static top-k slot maps to one distinct route row.
                    grad_routes.index_copy_(0, route_rows, route_grads)
                if flat_grad_weights is not None and not routes_alias_values:
                    selected_route_values = route_values.index_select(0, route_rows)
                    weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
                    flat_grad_weights[start:end].copy_(weight_grads)
        return

    if use_static_mapping:
        # Padding-aware fixed-shape path.  Mapping every masked slot to row 0
        # keeps all intermediates bounded by chunk_rows and avoids allocating
        # torch.nonzero's data-dependent route table.  Masked slots write an
        # exact zero; row 0 is restored in a final ordered kernel, so duplicate
        # sentinel writes cannot affect the visible gradient.
        num_slots = output_index.numel()
        num_route_rows = route_values.shape[0]
        flat_output_index = output_index.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_grad_weights = grad_weights.reshape(-1) if grad_weights is not None else None
        chunk_rows = _BACKWARD_CHUNK_ROWS

        if num_route_rows == 0:
            if grad_routes is not None:
                grad_routes.zero_()
            if flat_grad_weights is not None:
                flat_grad_weights.zero_()
            return

        def probability_grad_chunk(start: int, end: int) -> None:
            flat_positions = torch.arange(
                start,
                end,
                device=output_index.device,
                dtype=torch.long,
            )
            token_rows = torch.div(
                flat_positions,
                output_index.shape[1],
                rounding_mode="floor",
            )
            route_rows = flat_output_index[start:end].to(dtype=torch.long)
            valid_rows = route_rows >= 0
            safe_route_rows = route_rows.clamp(min=0, max=num_route_rows - 1)
            token_grads = grad_output.index_select(0, token_rows)
            selected_route_values = route_values.index_select(0, safe_route_rows)
            weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
            weight_grads.masked_fill_(~valid_rows, 0)
            flat_grad_weights[start:end].copy_(weight_grads)

        if routes_alias_values and flat_grad_weights is not None:
            for start in range(0, num_slots, chunk_rows):
                probability_grad_chunk(start, min(start + chunk_rows, num_slots))

        if grad_routes is not None and routes_alias_values:
            grad_routes.zero_()

        for start in range(0, num_slots, chunk_rows):
            end = min(start + chunk_rows, num_slots)
            flat_positions = torch.arange(
                start,
                end,
                device=output_index.device,
                dtype=torch.long,
            )
            token_rows = torch.div(
                flat_positions,
                output_index.shape[1],
                rounding_mode="floor",
            )
            route_rows = flat_output_index[start:end].to(dtype=torch.long)
            valid_rows = route_rows >= 0
            safe_route_rows = route_rows.clamp(min=0, max=num_route_rows - 1)
            token_grads = grad_output.index_select(0, token_rows)

            if grad_routes is not None:
                weights = flat_weights[start:end].float().masked_fill(~valid_rows, 0)
                route_grads = (token_grads.float() * weights.unsqueeze(1)).to(dtype=grad_routes.dtype)
                grad_routes.index_copy_(0, safe_route_rows, route_grads)
            if flat_grad_weights is not None and not routes_alias_values:
                selected_route_values = route_values.index_select(0, safe_route_rows)
                weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
                weight_grads.masked_fill_(~valid_rows, 0)
                flat_grad_weights[start:end].copy_(weight_grads)

        if grad_routes is not None:
            route_zero_matches = flat_output_index == 0
            torch._assert_async(
                torch.any(route_zero_matches),
                "non-empty expert output has no route mapped to row zero",
            )
            route_zero_position = torch.argmax(route_zero_matches.to(dtype=torch.int32)).reshape(1)
            route_zero_token = torch.div(
                route_zero_position,
                output_index.shape[1],
                rounding_mode="floor",
            )
            route_zero_weight = flat_weights.index_select(0, route_zero_position).float()
            route_zero_grad = (
                grad_output.index_select(0, route_zero_token).float() * route_zero_weight.unsqueeze(1)
            ).to(dtype=grad_routes.dtype)
            grad_routes.narrow(0, 0, 1).copy_(route_zero_grad)
        return

    valid_positions = torch.nonzero(output_index >= 0, as_tuple=False)
    # If the returned input gradient aliases route_values, finish every
    # probability gradient before clearing or overwriting that storage.
    # Padded DeepEP batches contain -1 output indices, and aligned expert rows
    # not referenced by a real token must receive a zero gradient.
    if routes_alias_values and grad_weights is not None:
        for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
            end = min(start + _BACKWARD_CHUNK_ROWS, valid_positions.shape[0])
            positions = valid_positions[start:end]
            token_rows = positions[:, 0]
            topk_columns = positions[:, 1]
            route_rows = output_index[token_rows, topk_columns].to(dtype=torch.long)
            token_grads = grad_output.index_select(0, token_rows)
            selected_route_values = route_values.index_select(0, route_rows)
            weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
            grad_weights[token_rows, topk_columns] = weight_grads

    if grad_routes is not None and routes_alias_values:
        grad_routes.zero_()

    for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
        end = min(start + _BACKWARD_CHUNK_ROWS, valid_positions.shape[0])
        positions = valid_positions[start:end]
        token_rows = positions[:, 0]
        topk_columns = positions[:, 1]
        route_rows = output_index[token_rows, topk_columns].to(dtype=torch.long)
        token_grads = grad_output.index_select(0, token_rows)

        if grad_routes is not None:
            route_grads = (token_grads.float() * topk_weights[token_rows, topk_columns].float().unsqueeze(1)).to(
                dtype=grad_routes.dtype
            )
            grad_routes.index_copy_(0, route_rows, route_grads)
        if grad_weights is not None and not routes_alias_values:
            selected_route_values = route_values.index_select(0, route_rows)
            weight_grads = (token_grads.float() * selected_route_values.float()).sum(dim=-1)
            grad_weights[token_rows, topk_columns] = weight_grads


class _DeepGEMMMoEWithBF16Backward(torch.autograd.Function):
    """FP8 grouped-DeepGEMM MoE forward with explicit BF16 backward."""

    @staticmethod
    def forward(
        ctx,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        module: torch.nn.Module,
        layout: _MoELayout,
        module_name: str,
        *weights: torch.Tensor,
    ) -> torch.Tensor:
        if len(weights) != 2 * layout.num_local_experts:
            raise RuntimeError(
                f"{module_name} expected {2 * layout.num_local_experts} expert weights, got {len(weights)}"
            )
        counts = _validate_routing_inputs(
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            layout,
        )
        output = _deepgemm_grouped_moe_forward(
            module,
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
            layout=layout,
            module_name=module_name,
            validated_counts=counts,
        )
        ctx.layout = layout
        ctx.counts = counts
        ctx.module_name = module_name
        ctx.defer_router_probabilities = bool(getattr(module, "_slime_defer_router_probabilities", False))
        ctx.reuse_expert_input_for_grad = bool(getattr(module, "_slime_reuse_expert_input_for_grad", False))
        ctx.grad_workspace = getattr(module, _COMBINE_WORKSPACE_ATTR, None)
        ctx.save_for_backward(permuted_local_hidden_states, permuted_probs, *weights)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        hidden_states = saved[0]
        permuted_probs = saved[1]
        weights = saved[2:]
        layout: _MoELayout = ctx.layout
        counts: tuple[int, ...] = ctx.counts

        fc1_weights = weights[: layout.num_local_experts]
        fc2_weights = weights[layout.num_local_experts :]
        if len(fc1_weights) != layout.num_local_experts or len(fc2_weights) != layout.num_local_experts:
            raise RuntimeError(f"{ctx.module_name} saved expert weight count mismatch")

        needs = ctx.needs_input_grad
        needs_hidden = needs[0]
        needs_probs = needs[2]
        needs_fc1_weights = needs[6 : 6 + layout.num_local_experts]
        needs_fc2_weights = needs[6 + layout.num_local_experts :]

        grad_hidden = None
        if needs_hidden:
            workspace = ctx.grad_workspace
            if workspace is None and ctx.reuse_expert_input_for_grad:
                # DeepEP's expert-major input has no later reader.  Each
                # expert/chunk finishes recompute and wgrad before writing its
                # dgrad, so that input storage can safely carry the gradient.
                grad_hidden = hidden_states.detach()
            elif workspace is None:
                grad_hidden = torch.empty_like(hidden_states)
            else:
                required_bytes = hidden_states.numel() * hidden_states.element_size()
                if required_bytes > workspace.numel():
                    raise RuntimeError(
                        "Shared MoE backward workspace is too small: "
                        f"need {required_bytes} bytes for {tuple(hidden_states.shape)}, "
                        f"have {workspace.numel()} bytes; increase "
                        "SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES"
                    )
                if workspace.device != hidden_states.device:
                    raise RuntimeError(
                        "Shared MoE backward workspace is on "
                        f"{workspace.device}, input is on {hidden_states.device}"
                    )
                grad_hidden = workspace.narrow(0, 0, required_bytes).view(hidden_states.dtype).view_as(hidden_states)
        grad_probs = torch.empty_like(permuted_probs) if needs_probs else None
        grad_fc1_weights: list[torch.Tensor | None] = [None] * layout.num_local_experts
        grad_fc2_weights: list[torch.Tensor | None] = [None] * layout.num_local_experts
        grad_output = grad_output.contiguous().to(dtype=hidden_states.dtype)
        probabilities = permuted_probs.reshape(-1, 1)
        defer_router_probabilities = ctx.defer_router_probabilities

        if _use_grouped_bf16_backward(
            hidden_states,
            counts,
            needs_fc1_weights,
            needs_fc2_weights,
        ):
            grad_hidden, grad_probs, grad_fc1_weights, grad_fc2_weights = _grouped_expert_backward(
                hidden_states=hidden_states,
                permuted_probs=permuted_probs,
                grad_output=grad_output,
                fc1_weights=fc1_weights,
                fc2_weights=fc2_weights,
                counts=counts,
                layout=layout,
                needs_hidden=needs_hidden,
                needs_probs=needs_probs,
                needs_fc1_weights=needs_fc1_weights,
                needs_fc2_weights=needs_fc2_weights,
                defer_router_probabilities=defer_router_probabilities,
                grad_hidden=grad_hidden,
                grad_probs=grad_probs,
            )
            return (
                grad_hidden,
                None,
                grad_probs,
                None,
                None,
                None,
                *grad_fc1_weights,
                *grad_fc2_weights,
            )

        offset = 0
        for expert_index, count in enumerate(counts):
            fc1_weight = fc1_weights[expert_index]
            fc2_weight = fc2_weights[expert_index]
            needs_fc1_weight = needs_fc1_weights[expert_index]
            needs_fc2_weight = needs_fc2_weights[expert_index]
            if count == 0:
                if needs_fc1_weight:
                    grad_fc1_weights[expert_index] = torch.zeros_like(fc1_weight)
                if needs_fc2_weight:
                    grad_fc2_weights[expert_index] = torch.zeros_like(fc2_weight)
                continue

            fc1_accumulator = torch.zeros_like(fc1_weight, dtype=torch.float32) if needs_fc1_weight else None
            fc2_accumulator = torch.zeros_like(fc2_weight, dtype=torch.float32) if needs_fc2_weight else None

            for chunk_start in range(0, count, _BACKWARD_CHUNK_ROWS):
                chunk_end = min(chunk_start + _BACKWARD_CHUNK_ROWS, count)
                global_start = offset + chunk_start
                global_end = offset + chunk_end
                hidden = hidden_states[global_start:global_end]
                grad = grad_output[global_start:global_end]
                probability = probabilities[global_start:global_end]

                gate_up = _deepgemm_bf16_gemm_nt(hidden, fc1_weight)
                gate, up = gate_up.chunk(2, dim=-1)
                gate_f = gate.float()
                up_f = up.float()
                silu_gate = F.silu(gate_f)
                down_input = (silu_gate * up_f).to(dtype=hidden_states.dtype)

                if needs_probs and not defer_router_probabilities:
                    down_output = _deepgemm_bf16_gemm_nt(down_input, fc2_weight)
                    grad_probs_chunk = _router_probability_grad_fp32_chunked(
                        grad,
                        down_output,
                    )
                    grad_probs[global_start:global_end].copy_(
                        grad_probs_chunk.reshape_as(permuted_probs[global_start:global_end]).to(
                            dtype=permuted_probs.dtype
                        )
                    )

                if defer_router_probabilities:
                    grad_down_output = grad
                else:
                    grad_down_output = (grad.float() * probability.float()).to(dtype=hidden_states.dtype)
                grad_down_input = _deepgemm_bf16_gemm_nn(
                    grad_down_output,
                    fc2_weight,
                )
                if fc2_accumulator is not None:
                    fc2_accumulator.add_(
                        _deepgemm_bf16_gemm_tn(
                            grad_down_output,
                            down_input,
                        )
                    )

                grad_down_input_f = grad_down_input.float()
                sigmoid_gate = torch.sigmoid(gate_f)
                grad_gate = grad_down_input_f * up_f * sigmoid_gate * (1.0 + gate_f * (1.0 - sigmoid_gate))
                grad_up = grad_down_input_f * silu_gate
                grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1).to(dtype=hidden_states.dtype)

                if fc1_accumulator is not None:
                    fc1_accumulator.add_(_deepgemm_bf16_gemm_tn(grad_gate_up, hidden))
                # Keep this after the final read from ``hidden`` because the
                # DeepEP path may reuse that storage for grad_hidden.
                if needs_hidden:
                    grad_hidden[global_start:global_end].copy_(_deepgemm_bf16_gemm_nn(grad_gate_up, fc1_weight))

            if fc1_accumulator is not None:
                grad_fc1_weights[expert_index] = _sum_to_parameter_dtype(
                    fc1_accumulator,
                    fc1_weight,
                )
            if fc2_accumulator is not None:
                grad_fc2_weights[expert_index] = _sum_to_parameter_dtype(
                    fc2_accumulator,
                    fc2_weight,
                )
            offset += count

        return (
            grad_hidden,
            None,
            None if defer_router_probabilities else grad_probs,
            None,
            None,
            None,
            *grad_fc1_weights,
            *grad_fc2_weights,
        )


def _configure_batch_invariant(deep_gemm: Any, deep_gemm_wrapper: Any) -> bool:
    enabled = os.environ.get("SGLANG_DEEPGEMM_BATCH_INVARIANT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # The patched SGLang wrapper exposes configure_deep_gemm_batch_invariant, which
    # is the H100 path and is used unchanged when present. Some SGLang builds (e.g.
    # the current B300/cu130 image, without sglang-deterministic.patch) do not ship
    # it; fall back to DeepGEMM's own set_batch_invariant so batch-invariant mode
    # can still be toggled on Blackwell.
    configure = getattr(deep_gemm_wrapper, "configure_deep_gemm_batch_invariant", None)
    if configure is not None:
        configure(enabled)
    else:
        setter = getattr(deep_gemm, "set_batch_invariant", None)
        if setter is None:
            raise RuntimeError(
                "Neither deep_gemm_wrapper.configure_deep_gemm_batch_invariant nor "
                "deep_gemm.set_batch_invariant is available; cannot configure "
                "batch-invariant DeepGEMM kernels"
            )
        setter(enabled)
    getter = getattr(deep_gemm, "get_batch_invariant", None)
    if enabled and (getter is None or not getter()):
        raise RuntimeError(
            "SGLANG_DEEPGEMM_BATCH_INVARIANT=1, but the Megatron actor's "
            "DeepGEMM runtime did not enable batch-invariant kernels"
        )
    return enabled


def _load_deepgemm_ops() -> _DeepGEMMOps:
    """Load CUDA-only dependencies lazily so CPU tests can mock this boundary."""
    import deep_gemm
    from sgl_kernel import silu_and_mul
    from sglang.srt.layers import deep_gemm_wrapper

    from slime.backends.megatron_utils.kernels.fp8_kernel import blockwise_cast_to_fp8_triton

    ep_moe_kernels = import_sglang_module(
        "sglang.kernels.ops.moe.ep_moe_kernels",
        "sglang.srt.layers.moe.ep_moe.kernels",
    )
    fp8_kernel = import_sglang_module(
        "sglang.kernels.ops.quantization.fp8_kernel",
        "sglang.srt.layers.quantization.fp8_kernel",
    )

    _configure_batch_invariant(deep_gemm, deep_gemm_wrapper)
    scale_ue8m0 = bool(deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0)
    m_alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    if m_alignment != _GROUPED_M_ALIGNMENT:
        raise RuntimeError(
            f"Unexpected DeepGEMM contiguous grouped M alignment: {m_alignment} != {_GROUPED_M_ALIGNMENT}"
        )

    if not scale_ue8m0:
        # Hopper (sm90): FP32 block scales; weights cast with the Triton kernel
        # and activation/weight scales TMA-aligned as a separate step. Unchanged.
        return _DeepGEMMOps(
            quantize_weight=blockwise_cast_to_fp8_triton,
            quantize_activation=fp8_kernel.sglang_per_token_group_quant_fp8,
            align_input_scale=ep_moe_kernels.tma_align_input_scale,
            grouped_gemm=deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig,
            silu_and_mul=silu_and_mul,
            scale_ue8m0=False,
            need_tma_aligned_scales=bool(deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES),
            transform_weight_scale=None,
        )

    # Blackwell (sm100+): UE8M0 (power-of-two) packed block scales. Weights are
    # quantized with the same per-block cast the SGLang rollout uses
    # (requant_weight_ue8m0 -> quant_weight_ue8m0), and the grouped weight scale
    # is repacked with transform_scale_ue8m0. Activation scales are produced
    # column-major / TMA-aligned / UE8M0 directly by the quant kernel.
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0

    def _quantize_weight_ue8m0(weight: torch.Tensor, block: tuple[int, int]):
        # Mirror the Hopper op signature (weight, (block_n, block_k)); UE8M0 quant
        # requires a [128, 128] block and a BF16 input.
        return quant_weight_ue8m0(weight, [int(block[0]), int(block[1])])

    return _DeepGEMMOps(
        quantize_weight=_quantize_weight_ue8m0,
        quantize_activation=fp8_kernel.sglang_per_token_group_quant_fp8,
        align_input_scale=ep_moe_kernels.tma_align_input_scale,
        grouped_gemm=deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig,
        silu_and_mul=silu_and_mul,
        scale_ue8m0=True,
        need_tma_aligned_scales=bool(deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES),
        transform_weight_scale=transform_scale_ue8m0,
    )


def _get_expert_weights(
    grouped_linear: torch.nn.Module,
    *,
    num_local_experts: int,
    expected_shape: tuple[int, int],
    module_name: str,
) -> list[torch.Tensor]:
    weights = []
    for expert_index in range(num_local_experts):
        attr_name = f"weight{expert_index}"
        weight = getattr(grouped_linear, attr_name, None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(f"{module_name} is missing tensor parameter {attr_name}")
        if tuple(weight.shape) != expected_shape:
            raise RuntimeError(
                f"{module_name}.{attr_name} has shape {tuple(weight.shape)}, "
                f"expected {expected_shape} ([out_features, in_features])"
            )
        weights.append(weight)
    return weights


def _validate_parallelism() -> None:
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size != 1:
        raise RuntimeError(f"TEGroupedMLP DeepGEMM alignment requires tensor model parallel size 1, got {tp_size}")
    expert_tp_size = parallel_state.get_expert_tensor_parallel_world_size()
    if expert_tp_size != 1:
        raise RuntimeError(
            f"TEGroupedMLP DeepGEMM alignment requires expert tensor parallel size 1, got {expert_tp_size}"
        )


def _validate_te_grouped_mlp(module: torch.nn.Module, module_name: str) -> _MoELayout:
    if type(module).__name__ != "TEGroupedMLP":
        raise RuntimeError(
            f"DeepGEMM MoE target {module_name} has unsupported class {type(module).__name__}; expected TEGroupedMLP"
        )

    config = getattr(module, "config", None)
    if config is None:
        raise RuntimeError(f"DeepGEMM MoE target {module_name} has no TransformerConfig")
    if getattr(config, "add_bias_linear", False):
        raise RuntimeError(f"DeepGEMM MoE target {module_name} must be bias-free")
    if not getattr(config, "gated_linear_unit", False):
        raise RuntimeError(f"DeepGEMM MoE target {module_name} must use a gated linear unit")
    if getattr(module, "activation_func", None) is not F.silu:
        raise RuntimeError(f"DeepGEMM MoE target {module_name} must use SiLU")
    if getattr(config, "moe_apply_probs_on_input", False):
        raise RuntimeError(f"DeepGEMM MoE target {module_name} cannot use moe_apply_probs_on_input")
    if getattr(config, "fp8", False):
        raise RuntimeError(f"DeepGEMM MoE target {module_name} cannot also enable Megatron FP8")
    if getattr(config, "qat", False):
        raise RuntimeError(f"DeepGEMM MoE target {module_name} cannot enable QAT")
    if getattr(config, "swiglu_clamp_limit", None) is not None:
        raise RuntimeError(
            f"DeepGEMM MoE target {module_name} has a SwiGLU clamp, "
            "which the SGLang DeepGEMM activation used here does not reproduce"
        )

    num_local_experts = int(getattr(module, "num_local_experts", 0))
    hidden_size = int(getattr(config, "hidden_size", 0))
    ffn_hidden_size = int(getattr(config, "moe_ffn_hidden_size", 0))
    if min(num_local_experts, hidden_size, ffn_hidden_size) <= 0:
        raise RuntimeError(
            f"DeepGEMM MoE target {module_name} has invalid layout: "
            f"experts={num_local_experts}, hidden={hidden_size}, ffn={ffn_hidden_size}"
        )
    if hidden_size % _BLOCK_SIZE or ffn_hidden_size % _BLOCK_SIZE:
        raise RuntimeError(
            f"DeepGEMM MoE target {module_name} requires hidden and MoE FFN sizes "
            f"divisible by {_BLOCK_SIZE}, got {hidden_size} and {ffn_hidden_size}"
        )

    fc1 = getattr(module, "linear_fc1", None)
    fc2 = getattr(module, "linear_fc2", None)
    if type(fc1).__name__ != "TEColumnParallelGroupedLinear":
        raise RuntimeError(f"{module_name}.linear_fc1 must be TEColumnParallelGroupedLinear, got {type(fc1).__name__}")
    if type(fc2).__name__ != "TERowParallelGroupedLinear":
        raise RuntimeError(f"{module_name}.linear_fc2 must be TERowParallelGroupedLinear, got {type(fc2).__name__}")
    if int(getattr(fc1, "num_gemms", -1)) != num_local_experts:
        raise RuntimeError(f"{module_name}.linear_fc1 num_gemms does not match local experts")
    if int(getattr(fc2, "num_gemms", -1)) != num_local_experts:
        raise RuntimeError(f"{module_name}.linear_fc2 num_gemms does not match local experts")
    if getattr(fc1, "use_bias", False) or getattr(fc2, "use_bias", False):
        raise RuntimeError(f"DeepGEMM MoE target {module_name} grouped linears must be bias-free")
    if getattr(fc1, "_slime_deepgemm_forward_wrapped", False) or getattr(
        fc2, "_slime_deepgemm_forward_wrapped", False
    ):
        raise RuntimeError(
            f"DeepGEMM MoE target {module_name} has an individually wrapped expert linear; "
            "remove that wrapper before installing the whole-MLP hook"
        )

    layout = _MoELayout(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
    )
    fc1_weights = _get_expert_weights(
        fc1,
        num_local_experts=num_local_experts,
        expected_shape=layout.fc1_weight_shape,
        module_name=f"{module_name}.linear_fc1",
    )
    fc2_weights = _get_expert_weights(
        fc2,
        num_local_experts=num_local_experts,
        expected_shape=layout.fc2_weight_shape,
        module_name=f"{module_name}.linear_fc2",
    )
    named_weights = [(f"linear_fc1.weight{i}", weight) for i, weight in enumerate(fc1_weights)]
    named_weights.extend((f"linear_fc2.weight{i}", weight) for i, weight in enumerate(fc2_weights))
    for weight_name, weight in named_weights:
        if weight.dtype != torch.bfloat16:
            raise RuntimeError(f"{module_name}.{weight_name} must be BF16, got {weight.dtype}")
    return layout


def _validate_routing_inputs(
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
    layout: _MoELayout,
) -> tuple[int, ...]:
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(f"DeepGEMM MoE alignment requires BF16 hidden states, got {hidden_states.dtype}")
    if hidden_states.ndim != 2 or hidden_states.shape[1] != layout.hidden_size:
        raise RuntimeError(
            "DeepGEMM MoE hidden-state shape mismatch: "
            f"got {tuple(hidden_states.shape)}, expected [tokens, {layout.hidden_size}]"
        )
    if not isinstance(tokens_per_expert, torch.Tensor):
        raise RuntimeError("tokens_per_expert must be a tensor")
    if tokens_per_expert.ndim != 1 or tokens_per_expert.numel() != layout.num_local_experts:
        raise RuntimeError(
            "tokens_per_expert must have one entry per local expert: "
            f"got {tuple(tokens_per_expert.shape)}, expected [{layout.num_local_experts}]"
        )
    if tokens_per_expert.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise RuntimeError(f"tokens_per_expert must have an integer dtype, got {tokens_per_expert.dtype}")

    counts = tuple(int(value) for value in tokens_per_expert.detach().cpu().tolist())
    if any(value < 0 for value in counts):
        raise RuntimeError(f"tokens_per_expert contains a negative count: {counts}")
    if sum(counts) != hidden_states.shape[0]:
        raise RuntimeError(
            "tokens_per_expert sum does not match permuted hidden-state rows: "
            f"{sum(counts)} != {hidden_states.shape[0]}"
        )
    if not isinstance(permuted_probs, torch.Tensor) or not permuted_probs.is_floating_point():
        raise RuntimeError("permuted_probs must be a floating-point tensor")
    if permuted_probs.ndim not in (1, 2) or permuted_probs.numel() != hidden_states.shape[0]:
        raise RuntimeError(
            "permuted_probs must contain one scalar per permuted row: "
            f"got {tuple(permuted_probs.shape)}, expected [{hidden_states.shape[0]}]"
        )
    if permuted_probs.ndim == 2 and permuted_probs.shape[1] != 1:
        raise RuntimeError(f"2D permuted_probs must have shape [tokens, 1], got {tuple(permuted_probs.shape)}")
    if permuted_probs.device != hidden_states.device:
        raise RuntimeError(
            "permuted_probs and hidden states must be on the same device: "
            f"{permuted_probs.device} != {hidden_states.device}"
        )
    return counts


def _quantize_grouped_weights(
    grouped_linear: torch.nn.Module,
    *,
    expected_shape: tuple[int, int],
    layout: _MoELayout,
    input_device: torch.device,
    module_name: str,
    ops: _DeepGEMMOps,
    expert_start: int = 0,
    expert_end: int | None = None,
    grouped_qweight_workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = _get_expert_weights(
        grouped_linear,
        num_local_experts=layout.num_local_experts,
        expected_shape=expected_shape,
        module_name=module_name,
    )
    expert_end = layout.num_local_experts if expert_end is None else expert_end
    if not 0 <= expert_start < expert_end <= layout.num_local_experts:
        raise RuntimeError(
            f"Invalid {module_name} expert range [{expert_start}, {expert_end}) "
            f"for {layout.num_local_experts} local experts"
        )
    weights = weights[expert_start:expert_end]
    num_group_experts = expert_end - expert_start

    grouped_qweight = None
    grouped_scale = None
    expected_scale_shape = (
        (expected_shape[0] + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
        (expected_shape[1] + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
    )
    for local_expert_index, weight in enumerate(weights):
        expert_index = expert_start + local_expert_index
        if weight.dtype != torch.bfloat16:
            raise RuntimeError(f"{module_name}.weight{expert_index} must be BF16, got {weight.dtype}")
        if weight.device != input_device:
            raise RuntimeError(
                f"{module_name}.weight{expert_index} and hidden states must be on the same "
                f"device: {weight.device} != {input_device}"
            )
        qweight, scale = ops.quantize_weight(
            weight.detach().contiguous(),
            (_BLOCK_SIZE, _BLOCK_SIZE),
        )
        if tuple(qweight.shape) != expected_shape:
            raise RuntimeError(
                f"quantized {module_name}.weight{expert_index} has shape "
                f"{tuple(qweight.shape)}, expected {expected_shape}"
            )
        if tuple(scale.shape) != expected_scale_shape or (not ops.scale_ue8m0 and scale.dtype != torch.float32):
            raise RuntimeError(
                f"quantized {module_name}.weight{expert_index} scale has shape/dtype "
                f"{tuple(scale.shape)}/{scale.dtype}, expected "
                f"{expected_scale_shape}/{torch.float32}"
            )
        if qweight.device != input_device or scale.device != input_device:
            raise RuntimeError(f"quantized {module_name}.weight{expert_index} is on the wrong device")

        if grouped_qweight is None:
            grouped_shape = (num_group_experts, *expected_shape)
            grouped_qweight = _storage_prefix_view(
                grouped_qweight_workspace,
                grouped_shape,
                qweight.dtype,
            )
            if grouped_qweight is None:
                grouped_qweight = qweight.new_empty(grouped_shape)
            grouped_scale = scale.new_empty((num_group_experts, *expected_scale_shape))
        elif qweight.dtype != grouped_qweight.dtype:
            raise RuntimeError(f"quantized {module_name} expert weights have inconsistent dtypes")
        grouped_qweight[local_expert_index].copy_(qweight)
        grouped_scale[local_expert_index].copy_(scale)
        # Do not retain the previous per-expert quantization while the next
        # expert is being quantized.  At GLM-750B dimensions each FC1 value is
        # 24 MiB, so even one stale loop value matters at this peak.
        del qweight, scale

    assert grouped_qweight is not None and grouped_scale is not None
    if ops.scale_ue8m0:
        # Blackwell: the SGLang rollout stores UE8M0 MoE weights by quantizing the
        # BF16 experts to FP8 and then *requantizing* the grouped weight in
        # process_weights_after_loading (requant_block_scale_ue8m0_for_deepgemm ->
        # requant_weight_ue8m0). That requant is lossy, so a single quant here
        # would not bit-match the rollout. Replicate quant -> requant on the
        # grouped [E, N, K] weight to align to ~e-7.
        from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

        grouped_qweight, grouped_scale = requant_weight_ue8m0(
            grouped_qweight, grouped_scale, [_BLOCK_SIZE, _BLOCK_SIZE]
        )
    return grouped_qweight, grouped_scale


def _storage_prefix_view(
    workspace: torch.Tensor | None,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return a typed prefix of dead contiguous tensor storage, if it fits."""
    if workspace is None or not workspace.is_contiguous():
        return None
    required_elements = math.prod(shape)
    required_bytes = required_elements * torch.empty((), dtype=dtype).element_size()
    available_bytes = workspace.numel() * workspace.element_size()
    if required_bytes > available_bytes:
        return None
    raw_workspace = workspace.view(torch.uint8).reshape(-1)
    return raw_workspace.narrow(0, 0, required_bytes).view(dtype).view(shape)


def _quantize_activation(
    value: torch.Tensor,
    ops: _DeepGEMMOps,
) -> tuple[torch.Tensor, torch.Tensor]:
    ue8m0 = ops.scale_ue8m0
    qvalue, scale = ops.quantize_activation(
        value.detach().contiguous(),
        _BLOCK_SIZE,
        column_major_scales=ue8m0,
        scale_tma_aligned=ue8m0,
        scale_ue8m0=ue8m0,
    )
    if qvalue.shape != value.shape:
        raise RuntimeError(f"quantized activation shape mismatch: {tuple(qvalue.shape)} != {tuple(value.shape)}")
    if not ue8m0:
        # Hopper: FP32 row-major block scales, TMA-aligned as a separate step.
        expected_scale_shape = (value.shape[0], value.shape[1] // _BLOCK_SIZE)
        if tuple(scale.shape) != expected_scale_shape or scale.dtype != torch.float32:
            raise RuntimeError(
                "quantized activation scale shape/dtype mismatch: "
                f"{tuple(scale.shape)}/{scale.dtype} != {expected_scale_shape}/{torch.float32}"
            )
        return qvalue, ops.align_input_scale(scale)
    # Blackwell: the quant kernel already produced column-major, TMA-aligned
    # UE8M0 (int32-packed) scales; DEEPGEMM_NEED_TMA_ALIGNED_SCALES is False so
    # no separate align step is applied (matches the SGLang rollout runner).
    return qvalue, scale


def _build_m_indices(
    counts: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    # Counts are already available as Python integers for the per-expert
    # grouped launches.  Filling the device output directly avoids both a
    # blocking host-to-device upload of a newly constructed repeats tensor and
    # repeat_interleave's data-dependent shape path.
    output = torch.empty(sum(counts), dtype=torch.int32, device=device)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            output.narrow(0, start, count).fill_(expert)
            start += count
    return output


def _experts_per_forward_group(num_local_experts: int) -> int:
    configured = os.environ.get("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP")
    if configured is None:
        batch_invariant = os.environ.get("SGLANG_DEEPGEMM_BATCH_INVARIANT", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return min(_DEFAULT_EXPERTS_PER_GROUP, num_local_experts) if batch_invariant else num_local_experts
    try:
        experts_per_group = int(configured)
    except ValueError as exc:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP must be a positive integer, " f"got {configured!r}"
        ) from exc
    if experts_per_group <= 0:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP must be a positive integer, " f"got {experts_per_group}"
        )
    return min(experts_per_group, num_local_experts)


def _sort_chunks_into(
    input_: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Megatron ``sort_chunks_by_idxs`` ordering with caller-owned storage.

    Keep the chunk metadata on the input device.  Calling ``.tolist()`` on the
    CUDA split/order tensors serializes the stream once per MoE permutation.
    The destination-row map below expresses the same chunk-stable permutation
    with device operations and writes directly into the caller's workspace.
    """
    if output.shape != input_.shape or output.dtype != input_.dtype or output.device != input_.device:
        raise RuntimeError(
            "Preallocated MoE combine buffer must match its input: "
            f"input={tuple(input_.shape)}/{input_.dtype}/{input_.device}, "
            f"output={tuple(output.shape)}/{output.dtype}/{output.device}"
        )
    if split_sizes.ndim != 1 or sorted_idxs.ndim != 1:
        raise RuntimeError(
            "MoE chunk sizes/order must be one-dimensional: "
            f"{tuple(split_sizes.shape)} and {tuple(sorted_idxs.shape)}"
        )
    if split_sizes.numel() != sorted_idxs.numel():
        raise RuntimeError("MoE chunk sizes/order length mismatch: " f"{split_sizes.numel()} != {sorted_idxs.numel()}")
    if input_.shape[0] == 0:
        return output

    # Without fused permutation Megatron has already copied this metadata to
    # CPU and synchronized its side stream.  Preserve the cheap host path in
    # that case; moving the metadata back to CUDA would add a new transfer.
    # The CUDA branch below is what removes the hot-path synchronization when
    # moe_permute_fusion keeps both tensors on device.
    if not split_sizes.is_cuda and not sorted_idxs.is_cuda:
        chunks = torch.split(input_, split_sizes.tolist(), dim=0)
        output_offset = 0
        for index in sorted_idxs.tolist():
            chunk = chunks[index]
            chunk_rows = chunk.shape[0]
            output.narrow(0, output_offset, chunk_rows).copy_(chunk)
            output_offset += chunk_rows
        if output_offset != input_.shape[0]:
            raise RuntimeError(f"Preallocated MoE combine copied {output_offset} rows, " f"expected {input_.shape[0]}")
        return output

    device = input_.device
    sizes = split_sizes.to(device=device, dtype=torch.long)
    order = sorted_idxs.to(device=device, dtype=torch.long)
    num_chunks = sizes.numel()

    # The metadata is generated by Megatron's dispatcher.  Keep defensive
    # checks asynchronous for CUDA callers so they do not recreate the host
    # synchronization this path is meant to remove.
    torch._assert_async(torch.all(sizes >= 0), "MoE chunk sizes cannot be negative")
    torch._assert_async(
        sizes.sum() == input_.shape[0],
        "MoE chunk sizes must sum to the input row count",
    )
    torch._assert_async(
        torch.all((order >= 0) & (order < num_chunks)),
        "MoE chunk order contains an out-of-range index",
    )

    source_chunks = torch.repeat_interleave(
        torch.arange(num_chunks, device=device, dtype=torch.long),
        sizes,
        output_size=input_.shape[0],
    )
    source_starts = torch.cumsum(sizes, dim=0) - sizes
    within_chunk = torch.arange(input_.shape[0], device=device, dtype=torch.long) - source_starts.index_select(
        0, source_chunks
    )

    sorted_sizes = sizes.index_select(0, order)
    sorted_starts = torch.cumsum(sorted_sizes, dim=0) - sorted_sizes
    destination_starts = torch.empty_like(sorted_starts)
    destination_starts.scatter_(0, order, sorted_starts)
    destination_rows = destination_starts.index_select(0, source_chunks) + within_chunk
    output.index_copy_(0, destination_rows, input_)
    return output


def _wrap_preallocated_combine_preprocess(dispatcher: torch.nn.Module) -> bool:
    """Consume the expert's early-allocated combine buffer without a peak allocation."""
    if getattr(dispatcher, "_slime_preallocated_combine_wrapped", False):
        return False
    if int(getattr(dispatcher, "tp_size", 1)) != 1:
        raise RuntimeError("Preallocated MoE combine currently requires tensor parallel size 1")
    if bool(getattr(dispatcher, "drop_and_pad", False)):
        raise RuntimeError("Preallocated MoE combine does not support moe_expert_capacity_factor")

    original_combine_preprocess = dispatcher.combine_preprocess

    def combine_preprocess(
        patched_dispatcher: torch.nn.Module,
        hidden_states: torch.Tensor,
    ):
        combine_buffer = getattr(hidden_states, _PREALLOCATED_COMBINE_BUFFER_ATTR, None)
        if combine_buffer is None:
            return original_combine_preprocess(hidden_states)
        if int(patched_dispatcher.num_local_experts) <= 1:
            return hidden_states
        combined = _sort_chunks_into(
            hidden_states,
            patched_dispatcher.num_global_tokens_per_local_expert.T.ravel(),
            patched_dispatcher.restore_output_by_local_experts,
            combine_buffer,
        )
        setattr(combined, _PREALLOCATED_TOKEN_COMBINE_ATTR, True)
        return combined

    dispatcher.combine_preprocess = types.MethodType(combine_preprocess, dispatcher)
    dispatcher._slime_preallocated_combine_wrapped = True
    return True


def _wrap_preallocated_dispatch_postprocess(dispatcher: torch.nn.Module) -> bool:
    """Use the shared workspace for no-grad dispatch permutation.

    The all-to-all output is kept as the later combine destination.  With TP=1,
    shared-expert overlap does not consume this tensor's values after
    ``linear_fc1_forward_and_act``; it only tags the tensor for backward order.
    """
    if getattr(dispatcher, "_slime_preallocated_dispatch_wrapped", False):
        return False
    if int(getattr(dispatcher, "tp_size", 1)) != 1:
        raise RuntimeError("Preallocated MoE dispatch currently requires tensor parallel size 1")
    if bool(getattr(dispatcher, "drop_and_pad", False)):
        raise RuntimeError("Preallocated MoE dispatch does not support moe_expert_capacity_factor")
    if bool(getattr(getattr(dispatcher, "config", None), "moe_permute_fusion", False)):
        raise RuntimeError("Preallocated MoE dispatch currently requires moe_permute_fusion disabled")

    original_dispatch_postprocess = dispatcher.dispatch_postprocess

    def dispatch_postprocess(
        patched_dispatcher: torch.nn.Module,
        global_input_tokens: torch.Tensor,
        global_probs: torch.Tensor,
    ):
        if int(patched_dispatcher.num_local_experts) <= 1:
            return original_dispatch_postprocess(global_input_tokens, global_probs)
        if torch.is_grad_enabled():
            dispatched_input, tokens_per_expert, permuted_probs = original_dispatch_postprocess(
                global_input_tokens, global_probs
            )
            # The dispatch all-to-all output is no longer read after permutation:
            # AllToAllBackward saves only the group/splits, CatBackward saves no
            # input values, and shared experts consume cached_fc1_input instead.
            # Keep its storage as the destination for the inverse expert sort.
            # This removes the otherwise full-sized torch.cat allocation during
            # checkpoint recomputation without overwriting the expert input that
            # the explicit BF16 backward must retain.
            setattr(
                dispatched_input,
                _PREALLOCATED_COMBINE_BUFFER_ATTR,
                global_input_tokens,
            )
            return dispatched_input, tokens_per_expert, permuted_probs
        workspace_output = _combine_workspace_view(
            patched_dispatcher,
            global_input_tokens,
        )
        if workspace_output is None:
            return original_dispatch_postprocess(global_input_tokens, global_probs)

        if patched_dispatcher.shared_experts is not None:
            patched_dispatcher.shared_experts.linear_fc1_forward_and_act(global_input_tokens)
        patched_dispatcher.tokens_per_expert = patched_dispatcher._maybe_dtoh_and_synchronize(
            "before_permutation_2",
            patched_dispatcher.tokens_per_expert,
        )
        split_sizes = patched_dispatcher.num_global_tokens_per_local_expert.ravel()
        workspace_output = _sort_chunks_into(
            global_input_tokens,
            split_sizes,
            patched_dispatcher.sort_input_by_local_experts,
            workspace_output,
        )
        sorted_probs = _sort_chunks_into(
            global_probs,
            split_sizes,
            patched_dispatcher.sort_input_by_local_experts,
            torch.empty_like(global_probs),
        )
        setattr(
            workspace_output,
            _PREALLOCATED_COMBINE_BUFFER_ATTR,
            global_input_tokens,
        )
        tokens_per_expert = patched_dispatcher._maybe_dtoh_and_synchronize(
            "before_finish",
            patched_dispatcher.tokens_per_expert,
        )
        patched_dispatcher.tokens_per_expert = None
        return workspace_output, tokens_per_expert, sorted_probs

    dispatcher.dispatch_postprocess = types.MethodType(dispatch_postprocess, dispatcher)
    dispatcher._slime_preallocated_dispatch_wrapped = True
    return True


def _wrap_preallocated_token_combine(dispatcher: torch.nn.Module) -> bool:
    """Write no-grad expert all-to-all results back into the shared workspace."""
    if getattr(dispatcher, "_slime_preallocated_token_combine_wrapped", False):
        return False
    original_token_combine = dispatcher.token_combine

    def token_combine(
        patched_dispatcher: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if torch.is_grad_enabled() or not bool(getattr(hidden_states, _PREALLOCATED_TOKEN_COMBINE_ATTR, False)):
            return original_token_combine(hidden_states, *args, **kwargs)
        if patched_dispatcher.ep_group.size() == 1:
            return hidden_states
        output_rows = sum(patched_dispatcher.input_splits)
        output = _combine_workspace_view(
            patched_dispatcher,
            hidden_states,
            shape=(output_rows, *hidden_states.shape[1:]),
        )
        if output is None:
            return original_token_combine(hidden_states, *args, **kwargs)
        torch.distributed.all_to_all_single(
            output,
            hidden_states,
            output_split_sizes=patched_dispatcher.input_splits,
            input_split_sizes=patched_dispatcher.output_splits,
            group=patched_dispatcher.ep_group,
        )
        return output

    dispatcher.token_combine = types.MethodType(token_combine, dispatcher)
    dispatcher._slime_preallocated_token_combine_wrapped = True
    return True


def _combine_workspace_bytes() -> int | None:
    configured = os.environ.get("SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES")
    if configured is None:
        return None
    try:
        workspace_bytes = int(configured)
    except ValueError as exc:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES must be a positive integer, " f"got {configured!r}"
        ) from exc
    if workspace_bytes <= 0:
        raise RuntimeError(
            "SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES must be a positive integer, " f"got {workspace_bytes}"
        )
    return workspace_bytes


def _combine_workspace_view(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    workspace = getattr(module, _COMBINE_WORKSPACE_ATTR, None)
    if workspace is None:
        return None
    output_shape = tuple(hidden_states.shape) if shape is None else shape
    required_elements = math.prod(output_shape)
    required_bytes = required_elements * hidden_states.element_size()
    if required_bytes > workspace.numel():
        raise RuntimeError(
            "Shared MoE combine workspace is too small: "
            f"need {required_bytes} bytes for {output_shape}, "
            f"have {workspace.numel()} bytes; increase "
            "SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES"
        )
    if workspace.device != hidden_states.device:
        raise RuntimeError(
            f"Shared MoE combine workspace is on {workspace.device}, input is on {hidden_states.device}"
        )
    return workspace.narrow(0, 0, required_bytes).view(hidden_states.dtype).view(output_shape)


def _moe_recompute_scratch_view(
    module: torch.nn.Module,
    reference: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor | None:
    """Return a temporary tensor backed by the shared MoE workspace when it fits.

    This is used only by the gradient-enabled checkpoint recomputation.  The
    initial no-grad forward keeps its dispatched expert input in the same
    workspace, so it must not use this scratch view.  During recomputation the
    workspace otherwise remains idle until the custom MoE backward writes
    ``grad_hidden`` into it.
    """
    workspace = getattr(module, _COMBINE_WORKSPACE_ATTR, None)
    if workspace is None or workspace.device != reference.device:
        return None
    required_bytes = math.prod(shape) * reference.element_size()
    if required_bytes > workspace.numel():
        return None
    return workspace.narrow(0, 0, required_bytes).view(reference.dtype).view(shape)


def _pad_expert_rows(
    value: torch.Tensor,
    counts: tuple[int, ...],
    *,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[int, ...], torch.Tensor | None]:
    """Pad every expert segment to the DeepGEMM contiguous-layout M alignment."""
    padded_counts = tuple(
        ((count + _GROUPED_M_ALIGNMENT - 1) // _GROUPED_M_ALIGNMENT) * _GROUPED_M_ALIGNMENT if count else 0
        for count in counts
    )
    if padded_counts == counts:
        return value, padded_counts, None

    padded_shape = (sum(padded_counts), value.shape[1])
    if output is None:
        padded_value = value.new_zeros(padded_shape)
    else:
        if tuple(output.shape) != padded_shape:
            raise RuntimeError(
                "Preallocated expert-row padding shape mismatch: " f"{tuple(output.shape)} != {padded_shape}"
            )
        if output.dtype != value.dtype or output.device != value.device:
            raise RuntimeError(
                "Preallocated expert-row padding dtype/device mismatch: "
                f"{output.dtype}/{output.device} != {value.dtype}/{value.device}"
            )
        padded_value = output
        padded_value.zero_()
    valid_ranges = []
    padded_start = 0
    for count, padded_count in zip(counts, padded_counts, strict=True):
        if count:
            valid_ranges.append(
                torch.arange(
                    padded_start,
                    padded_start + count,
                    device=value.device,
                    dtype=torch.long,
                )
            )
        padded_start += padded_count
    valid_rows = torch.cat(valid_ranges)
    padded_value.index_copy_(0, valid_rows, value)
    return padded_value, padded_counts, valid_rows


def _deepgemm_grouped_moe_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
    *,
    layout: _MoELayout,
    module_name: str,
    reuse_input_buffer: bool = False,
    validated_counts: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute the SGLang-style contiguous grouped-MoE forward without autograd."""
    counts = validated_counts
    if counts is None:
        counts = _validate_routing_inputs(
            hidden_states,
            tokens_per_expert,
            permuted_probs,
            layout,
        )
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return hidden_states.new_empty((0, layout.hidden_size))

    ops = _load_deepgemm_ops()
    experts_per_group = _experts_per_forward_group(layout.num_local_experts)
    # Checkpointed forward runs under no_grad and does not need the dispatched
    # expert-major input after each expert has consumed it.  Reusing that
    # storage removes one full [routed_tokens, hidden] allocation before
    # Megatron's equally large combine permutation.  Gradient-enabled
    # recomputation keeps the ordinary separate output because the custom
    # backward must save the original expert inputs.
    shared_workspace = getattr(module, _COMBINE_WORKSPACE_ATTR, None)
    direct_group_outputs = not reuse_input_buffer and shared_workspace is None
    output_storage = None
    if direct_group_outputs:
        # During gradient-enabled checkpoint recomputation the expert input
        # must remain live for the custom backward.  Previously this allocated
        # both the final [routes, hidden] output and another group-sized output
        # (1.6 GiB each for the EP32/8K workload).  Reserve only enough padding
        # after the compact prefix for the largest current group, write the
        # grouped GEMM there, then compact it in place.  The returned prefix
        # owns the slightly larger storage, but no second routed output exists.
        compact_offset = 0
        output_capacity = num_tokens
        for capacity_start in range(0, layout.num_local_experts, experts_per_group):
            capacity_end = min(
                capacity_start + experts_per_group,
                layout.num_local_experts,
            )
            capacity_counts = counts[capacity_start:capacity_end]
            padded_capacity = sum(
                ((count + _GROUPED_M_ALIGNMENT - 1) // _GROUPED_M_ALIGNMENT * _GROUPED_M_ALIGNMENT if count else 0)
                for count in capacity_counts
            )
            output_capacity = max(
                output_capacity,
                compact_offset + padded_capacity,
            )
            compact_offset += sum(capacity_counts)
        output_storage = hidden_states.new_empty((output_capacity, layout.hidden_size))
        output = output_storage.narrow(0, 0, num_tokens)
    else:
        output = hidden_states if reuse_input_buffer else hidden_states.new_empty((num_tokens, layout.hidden_size))
    defer_router_probabilities = bool(getattr(module, "_slime_defer_router_probabilities", False))

    token_offset = 0
    for expert_start in range(0, layout.num_local_experts, experts_per_group):
        expert_end = min(expert_start + experts_per_group, layout.num_local_experts)
        group_counts = counts[expert_start:expert_end]
        group_tokens = sum(group_counts)
        group_hidden_states = hidden_states.narrow(0, token_offset, group_tokens)
        group_probs = permuted_probs.reshape(-1).narrow(0, token_offset, group_tokens)

        # An all-empty expert group has no output rows and does not need weight
        # quantization or a DeepGEMM launch.
        if group_tokens == 0:
            continue

        padded_group_tokens = sum(
            ((count + _GROUPED_M_ALIGNMENT - 1) // _GROUPED_M_ALIGNMENT * _GROUPED_M_ALIGNMENT if count else 0)
            for count in group_counts
        )
        group_output = None
        if direct_group_outputs:
            assert output_storage is not None
            group_output = output_storage.narrow(
                0,
                token_offset,
                padded_group_tokens,
            )
        reuse_unpadded_input = reuse_input_buffer and padded_group_tokens == group_tokens
        if reuse_unpadded_input:
            # DeepEP normally supplies M-aligned expert counts.  In the
            # checkpoint/no-grad forward, activation quantization is the last
            # reader of this group input, so the same slice can become FC1
            # gate/up scratch, SiLU/down scratch, and finally FC2 output.
            group_output = group_hidden_states

        # In checkpoint recomputation the final-output slice is still unused
        # here.  Use it as the aligned BF16 expert input, quantize from it, and
        # let FC2 overwrite it later.  This removes another group-sized
        # temporary (about 1.9 GiB for the largest EP32/8K group).
        if reuse_unpadded_input:
            padded_hidden_states = group_hidden_states
            padded_counts = group_counts
            valid_rows = None
        else:
            padded_hidden_states, padded_counts, valid_rows = _pad_expert_rows(
                group_hidden_states,
                group_counts,
                output=group_output,
            )
            if reuse_input_buffer:
                # The padded expert-major input is necessary for DeepGEMM, but
                # it becomes dead as soon as activation quantization finishes.
                # Keep its storage as the FC1/activation/FC2 scratch instead
                # of allocating those tensors alongside it.  FC2 is compacted
                # back into the original dispatched-input slice below.
                group_output = padded_hidden_states
        padded_tokens = padded_hidden_states.shape[0]
        m_indices = _build_m_indices(padded_counts, device=hidden_states.device)
        if m_indices.numel() != padded_tokens or padded_tokens % _GROUPED_M_ALIGNMENT:
            raise RuntimeError(
                "DeepGEMM contiguous grouped layout is not M-aligned: "
                f"rows={padded_tokens}, indices={m_indices.numel()}, "
                f"alignment={_GROUPED_M_ALIGNMENT}"
            )

        hidden_q, hidden_scale = _quantize_activation(padded_hidden_states, ops)
        gate_up_shape = (padded_tokens, 2 * layout.ffn_hidden_size)
        down_input_shape = (padded_tokens, layout.ffn_hidden_size)
        gate_up = None
        down_input = None
        if group_output is not None:
            # The padded expert input above has already been quantized, so its
            # final-output destination is dead until FC2 writes the result.
            # GLM's [hidden=6144, ffn=2048] layout fits gate/up and the SiLU
            # result in two disjoint slices of that same BF16 storage.  This
            # removes the 628 MiB gate/up plus 314 MiB down-input allocations
            # at the EP32/8K checkpoint-recompute peak.
            gate_up_elements = math.prod(gate_up_shape)
            down_input_elements = math.prod(down_input_shape)
            if gate_up_elements + down_input_elements <= group_output.numel():
                output_scratch = group_output.reshape(-1)
                gate_up = output_scratch.narrow(0, 0, gate_up_elements).view(gate_up_shape)
                down_input = output_scratch.narrow(
                    0,
                    gate_up_elements,
                    down_input_elements,
                ).view(down_input_shape)
        # Activation quantization has consumed padded_hidden_states.  Until
        # FC1 completes, the future SiLU/down-input slice is dead storage and
        # is disjoint from FC1's gate/up output.  Hold the grouped FP8 FC1
        # weights there instead of allocating 96 MiB at the full-pipeline
        # checkpoint-recompute peak; stream ordering makes the slice reusable
        # by silu_and_mul immediately after grouped_gemm returns.
        fc1_qweight, fc1_scale = _quantize_grouped_weights(
            module.linear_fc1,
            expected_shape=layout.fc1_weight_shape,
            layout=layout,
            input_device=hidden_states.device,
            module_name=f"{module_name}.linear_fc1",
            ops=ops,
            expert_start=expert_start,
            expert_end=expert_end,
            grouped_qweight_workspace=down_input,
        )
        if gate_up is None and not reuse_input_buffer:
            gate_up = _moe_recompute_scratch_view(module, hidden_states, gate_up_shape)
        if gate_up is None:
            gate_up = hidden_states.new_empty(gate_up_shape)
        ops.grouped_gemm(
            (hidden_q, hidden_scale),
            (fc1_qweight, fc1_scale),
            gate_up,
            m_indices,
        )
        del hidden_q, hidden_scale, fc1_qweight, fc1_scale

        if down_input is None:
            down_input = hidden_states.new_empty(down_input_shape)
        ops.silu_and_mul(gate_up, down_input)
        del gate_up

        down_q, down_scale = _quantize_activation(down_input, ops)
        del down_input
        fc2_qweight, fc2_scale = _quantize_grouped_weights(
            module.linear_fc2,
            expected_shape=layout.fc2_weight_shape,
            layout=layout,
            input_device=hidden_states.device,
            module_name=f"{module_name}.linear_fc2",
            ops=ops,
            expert_start=expert_start,
            expert_end=expert_end,
        )
        group_output_shape = (padded_tokens, layout.hidden_size)
        if group_output is None and not reuse_input_buffer:
            group_output = (
                _moe_recompute_scratch_view(
                    module,
                    hidden_states,
                    group_output_shape,
                )
                if not reuse_input_buffer
                else None
            )
        if group_output is None:
            group_output = hidden_states.new_empty(group_output_shape)
        ops.grouped_gemm(
            (down_q, down_scale),
            (fc2_qweight, fc2_scale),
            group_output,
            m_indices,
        )
        del down_q, down_scale, fc2_qweight, fc2_scale

        if valid_rows is not None:
            group_output = _compact_valid_rows_inplace(group_output, valid_rows)
        if not defer_router_probabilities:
            group_output = _apply_router_probability_fp32_inplace(group_output, group_probs)
        if not direct_group_outputs:
            output_slice = output.narrow(0, token_offset, group_tokens)
            if output_slice.data_ptr() != group_output.data_ptr():
                output_slice.copy_(group_output)
        token_offset += group_tokens

    if token_offset != num_tokens:
        raise AssertionError(f"Grouped MoE output row mismatch: {token_offset} != {num_tokens}")
    return output


def _wrap_te_grouped_mlp(
    module: torch.nn.Module,
    module_name: str,
) -> bool:
    if getattr(module, "_slime_deepgemm_moe_forward_wrapped", False):
        return False

    _validate_parallelism()
    layout = _validate_te_grouped_mlp(module, module_name)
    fc1_weights = _get_expert_weights(
        module.linear_fc1,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc1_weight_shape,
        module_name=f"{module_name}.linear_fc1",
    )
    fc2_weights = _get_expert_weights(
        module.linear_fc2,
        num_local_experts=layout.num_local_experts,
        expected_shape=layout.fc2_weight_shape,
        module_name=f"{module_name}.linear_fc2",
    )
    if getattr(getattr(module, "config", None), "delay_wgrad_compute", False):
        raise RuntimeError(
            "DeepGEMM MoE custom backward does not support Megatron delay_wgrad_compute; "
            "disable delay_wgrad_compute."
        )

    def deepgemm_moe_forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        grad_enabled = torch.is_grad_enabled()
        combine_buffer = getattr(
            permuted_local_hidden_states,
            _PREALLOCATED_COMBINE_BUFFER_ATTR,
            None,
        )
        if grad_enabled:
            deepgemm_output = _DeepGEMMMoEWithBF16Backward.apply(
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs,
                self,
                layout,
                module_name,
                *fc1_weights,
                *fc2_weights,
            )
        else:
            deepgemm_output = _deepgemm_grouped_moe_forward(
                self,
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs,
                layout=layout,
                module_name=module_name,
                reuse_input_buffer=True,
            )
        if combine_buffer is not None:
            setattr(deepgemm_output, _PREALLOCATED_COMBINE_BUFFER_ATTR, combine_buffer)
        return deepgemm_output, getattr(self, "output_bias", None)

    module.forward = types.MethodType(deepgemm_moe_forward, module)
    module._slime_deepgemm_moe_forward_wrapped = True
    module._slime_deepgemm_moe_module_name = module_name
    module._slime_deepgemm_moe_layout = layout
    return True


def _get_global_layer_index(model_chunk: torch.nn.Module, module_name: str) -> int | None:
    match = _LAYER_PATH_RE.search(module_name)
    if match is None:
        return None
    layer_path = match.group("layer_path")
    layer = model_chunk.get_submodule(layer_path)
    layer_number = getattr(layer, "layer_number", None)
    if layer_number is None:
        raise RuntimeError(
            "DeepGEMM MoE alignment requires TransformerLayer.layer_number to select "
            f"global pipeline layers, but {layer_path} has no layer_number"
        )
    return int(layer_number) - 1


def _normalize_model_chunks(model) -> list[torch.nn.Module]:
    if isinstance(model, torch.nn.Module):
        return [model]
    chunks = list(model)
    if not all(isinstance(chunk, torch.nn.Module) for chunk in chunks):
        raise RuntimeError("model must be a Megatron module or an iterable of model chunks")
    return chunks


class _DeepEPScatterWithDeterministicBackward(torch.autograd.Function):
    """Scatter routes with a deterministic backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        output_index: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        output = hidden_states.new_zeros((int(total_rows), hidden_states.shape[1]))
        if hidden_states.is_cuda:
            from slime.backends.megatron_utils.alignment.deterministic_route_kernels import scatter_routes_forward

            scatter_routes_forward(
                hidden_states.contiguous(),
                output_index.contiguous(),
                output,
            )
        else:
            valid_positions = torch.nonzero(output_index >= 0, as_tuple=False)
            for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
                positions = valid_positions[start : start + _BACKWARD_CHUNK_ROWS]
                token_rows = positions[:, 0]
                topk_columns = positions[:, 1]
                destination_rows = output_index[token_rows, topk_columns].to(dtype=torch.long)
                output.index_copy_(
                    0,
                    destination_rows,
                    hidden_states.index_select(0, token_rows),
                )

        ctx.input_shape = tuple(hidden_states.shape)
        ctx.input_dtype = hidden_states.dtype
        ctx.input_device = hidden_states.device
        ctx.save_for_backward(output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None

        (output_index,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.input_shape,
            dtype=ctx.input_dtype,
            device=ctx.input_device,
        )
        if grad_output.is_cuda:
            from slime.backends.megatron_utils.alignment.deterministic_route_kernels import scatter_routes_backward

            scatter_routes_backward(
                grad_output.contiguous(),
                output_index.contiguous(),
                grad_input,
            )
        else:
            # Accumulate top-k slots in their original order.  Each token row
            # is owned by one thread here, avoiding the nondeterministic
            # atomics that an index_add over expert-major occurrences would
            # require.
            for start in range(0, output_index.shape[0], _BACKWARD_CHUNK_ROWS):
                end = min(start + _BACKWARD_CHUNK_ROWS, output_index.shape[0])
                grad_chunk = grad_input[start:end]
                for column in range(output_index.shape[1]):
                    route_rows = output_index[start:end, column].to(dtype=torch.long)
                    valid_rows = route_rows >= 0
                    safe_route_rows = route_rows.clamp(min=0, max=max(grad_output.shape[0] - 1, 0))
                    if grad_output.shape[0] == 0:
                        continue
                    selected = grad_output.index_select(0, safe_route_rows)
                    selected.masked_fill_(~valid_rows.unsqueeze(1), 0)
                    grad_chunk.add_(selected)
        return grad_input, None, None


def _scatter_deepep_routes_with_padding(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    return_route_positions: bool = False,
    expected_route_count: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    """Build SGLang's padded expert-major DeepEP layout deterministically."""
    if hidden_states.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("DeepEP scatter expects [tokens, hidden] and [tokens, topk]")
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("DeepEP top-k indices and weights must have identical shapes")
    if hidden_states.shape[0] != topk_indices.shape[0]:
        raise ValueError("DeepEP hidden-state and top-k token dimensions differ")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"DeepEP top-k indices must be integer, got {topk_indices.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"DeepEP top-k weights must be FP32, got {topk_weights.dtype}")

    if tokens_per_expert.device.type == "cpu":
        # Normal DeepEP returns this metadata as a CPU tensor constructed from
        # its receive-count list.  The output row count is consequently
        # already known to Python; do not upload the counts only to block on a
        # device sum immediately afterwards.
        count_values = tuple(int(value) for value in tokens_per_expert.reshape(-1).tolist())
        total_rows = sum(count_values)
    else:
        count_values = None

    num_experts = tokens_per_expert.numel()
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    sanitized_indices = topk_indices.masked_fill(~valid, -1)
    real_counts = torch.bincount(
        sanitized_indices.masked_select(valid).to(dtype=torch.long),
        minlength=num_experts,
    )

    # The route-preserving metadata handle gives Python the exact number of
    # received routes.  Normal DeepEP's CPU count list is unpadded in this
    # case, so ``real_counts`` is the same per-expert metadata already resident
    # on CUDA.  Reusing it avoids a blocking pageable-CPU-to-CUDA upload once
    # per MoE layer (and again during checkpoint recomputation).  Retain the
    # original upload for aligned/padded layouts and callers without the exact
    # route count.
    counts_are_exact_unpadded = (
        count_values is not None and expected_route_count is not None and total_rows == expected_route_count
    )
    if counts_are_exact_unpadded:
        counts = real_counts
        torch._assert_async(
            real_counts.sum() == expected_route_count,
            "DeepEP received route count differs from its route-preserving metadata",
        )
    else:
        counts = tokens_per_expert.to(
            device=topk_indices.device,
            dtype=torch.long,
        ).reshape(-1)
    torch._assert_async(
        torch.all(real_counts <= counts),
        "DeepEP real route count exceeds its aligned expert count",
    )

    if count_values is None:
        total_rows = int(counts.sum().item())
    permuted_probs = topk_weights.new_zeros((total_rows,))
    output_index = torch.full_like(topk_indices, -1)
    routing_map = torch.zeros(
        (topk_indices.shape[0], num_experts),
        device=topk_indices.device,
        dtype=torch.bool,
    )
    expert_offsets = torch.cumsum(counts, dim=0) - counts

    if expected_route_count is not None and valid.is_cuda:
        from slime.backends.megatron_utils.alignment.deterministic_route_kernels import compact_route_positions

        occurrences = compact_route_positions(valid.contiguous(), expected_route_count)
    else:
        occurrences = torch.nonzero(valid, as_tuple=False)
    # The metadata handle exposes the compact route count as tensor shape.  If
    # it is available, use that static value instead of forcing nonzero to
    # report its data-dependent output size to Python.
    route_count = occurrences.shape[0] if expected_route_count is None else expected_route_count
    all_routes_valid = route_count == topk_indices.numel()
    if occurrences.numel():
        token_rows = occurrences[:, 0]
        topk_columns = occurrences[:, 1]
        route_experts = sanitized_indices[token_rows, topk_columns].to(dtype=torch.long)
        expert_order = torch.argsort(route_experts, stable=True)
        token_rows = token_rows.index_select(0, expert_order)
        topk_columns = topk_columns.index_select(0, expert_order)
        route_experts = route_experts.index_select(0, expert_order)

        real_offsets = torch.cumsum(real_counts, dim=0) - real_counts
        within_expert = torch.arange(
            route_experts.numel(),
            device=hidden_states.device,
            dtype=torch.long,
        ) - real_offsets.index_select(0, route_experts)
        destination_rows = expert_offsets.index_select(0, route_experts) + within_expert
        permuted_probs.index_copy_(
            0,
            destination_rows,
            topk_weights[token_rows, topk_columns],
        )
        output_index[token_rows, topk_columns] = destination_rows.to(dtype=output_index.dtype)
        routing_map[token_rows, route_experts] = True

    torch._assert_async(
        torch.all(~valid | (output_index >= 0)),
        "A valid DeepEP route has no expert-major output row",
    )
    permuted_hidden = _DeepEPScatterWithDeterministicBackward.apply(
        hidden_states,
        output_index,
        total_rows,
    )
    result = (
        permuted_hidden,
        permuted_probs,
        output_index,
        sanitized_indices,
        routing_map,
        all_routes_valid,
    )
    if return_route_positions:
        return (*result, occurrences)
    return result


class _SGLangEPGatherWithBF16Backward(torch.autograd.Function):
    """SGLang's ordered FP32 gather with a deterministic BF16 backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        output_index: torch.Tensor,
        reuse_input_for_grad: bool,
        static_mapping_valid: bool | None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "SGLang ep_gather requires BF16 [expert_rows, hidden], got "
                f"{hidden_states.dtype} {tuple(hidden_states.shape)}"
            )
        if topk_indices.shape != topk_weights.shape or topk_indices.shape != output_index.shape:
            raise ValueError("DeepEP gather IDs, weights, and output indices must align")
        ep_moe_kernels = import_sglang_module(
            "sglang.kernels.ops.moe.ep_moe_kernels",
            "sglang.srt.layers.moe.ep_moe.kernels",
        )

        output_shape = (topk_indices.shape[0], hidden_states.shape[1])
        output = torch.empty(
            output_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        ep_moe_kernels.ep_gather(
            hidden_states,
            topk_indices,
            topk_weights,
            output_index,
            output,
        )
        ctx.reuse_input_for_grad = bool(reuse_input_for_grad)
        ctx.static_mapping_valid = static_mapping_valid
        ctx.save_for_backward(hidden_states, topk_weights, output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, topk_weights, output_index = ctx.saved_tensors
        needs_hidden = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[2]
        if needs_hidden and ctx.reuse_input_for_grad:
            # The caller guarantees this combine input has no forward consumer
            # after ep_gather.  Returning its detached storage as the input
            # gradient avoids a second route-sized BF16 allocation.
            grad_hidden = hidden_states.detach()
        else:
            grad_hidden = torch.zeros_like(hidden_states) if needs_hidden else None
        grad_weights = torch.zeros_like(topk_weights) if needs_weights else None

        _ordered_route_backward(
            route_values=hidden_states,
            topk_weights=topk_weights,
            output_index=output_index,
            grad_output=grad_output,
            grad_routes=grad_hidden,
            grad_weights=grad_weights,
            static_mapping_valid=ctx.static_mapping_valid,
        )

        return grad_hidden, None, grad_weights, None, None, None


def _compact_route_preserving_metadata_inputs(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    assume_all_routes_valid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Build the small route-level dispatch used by the normal-mode prototype.

    The real hidden state remains rank-deduplicated in the primary DeepEP
    dispatch.  This second dispatch only carries a sixteen-BF16 fingerprint and
    creates a normal DeepEP handle whose logical tokens are ``(token, slot)``.
    The handle can consequently transport the real route outputs during
    combine without multiplying dispatch hidden traffic by top-k.

    Sixteen BF16 values are the smallest payload accepted by DeepEP's normal
    intranode dispatch: its TMA path requires the hidden payload to be a
    multiple of 32 bytes.  Keeping that constraint explicit here prevents a
    device-side assertion for otherwise-valid route metadata.
    """
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "Route-preserving DeepEP metadata requires BF16 [tokens, hidden], "
            f"got {hidden_states.dtype} {tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[1] < 16:
        raise ValueError("Route-preserving DeepEP fingerprints require hidden >= 16")
    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("Route-preserving DeepEP top-k IDs and weights must align")
    if topk_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("Route-preserving DeepEP token counts do not align")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"Route-preserving DeepEP IDs must be integer, got {topk_indices.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"Route-preserving DeepEP weights must be FP32, got {topk_weights.dtype}")

    flat_indices = topk_indices.reshape(-1)
    if assume_all_routes_valid:
        # With no capacity factor and no router token mask, the router emits a
        # fixed valid top-k for every source token.  Keep this fast path tied
        # to that structural guarantee instead of synchronizing on nonzero's
        # data-dependent output once per MoE layer.
        torch._assert_async(
            torch.all(flat_indices >= 0),
            "Route-preserving DeepEP expected a valid fixed top-k",
        )
        compact_indices = flat_indices.reshape(-1, 1).contiguous()
        compact_weights = topk_weights.detach().reshape(-1, 1).contiguous()
        fingerprints = (
            hidden_states.detach()
            .narrow(1, 0, 16)
            .unsqueeze(1)
            .expand(-1, topk_indices.shape[1], -1)
            .reshape(-1, 16)
            .contiguous()
        )
        output_index = torch.arange(
            flat_indices.numel(),
            device=flat_indices.device,
            dtype=torch.long,
        ).reshape_as(topk_indices)
        return compact_indices, compact_weights, fingerprints, output_index, True

    valid_positions = torch.nonzero(flat_indices >= 0, as_tuple=False).reshape(-1)
    if valid_positions.numel() == 0:
        raise RuntimeError("Route-preserving DeepEP received no valid expert routes")
    compact_indices = flat_indices.index_select(0, valid_positions).reshape(-1, 1).contiguous()
    compact_weights = topk_weights.detach().reshape(-1).index_select(0, valid_positions).reshape(-1, 1).contiguous()
    token_rows = torch.div(valid_positions, topk_indices.shape[1], rounding_mode="floor")
    fingerprints = hidden_states.detach().narrow(1, 0, 16).index_select(0, token_rows).contiguous()
    output_index = torch.full_like(topk_indices, -1, dtype=torch.long)
    output_index.reshape(-1).index_copy_(
        0,
        valid_positions,
        torch.arange(valid_positions.numel(), device=valid_positions.device, dtype=torch.long),
    )
    all_routes_valid = valid_positions.numel() == topk_indices.numel()
    return compact_indices, compact_weights, fingerprints, output_index, all_routes_valid


def _deepep_route_handle_received_rows(handle: tuple) -> int:
    """Return the received route count encoded by a normal DeepEP handle."""
    if not isinstance(handle, tuple):
        raise TypeError(f"DeepEP route handle must be a tuple, got {type(handle).__name__}")
    if len(handle) == 6:
        # Intranode: (..., recv_src_idx, ...).
        received_metadata = handle[3]
    elif len(handle) == 10:
        # Internode: (..., recv_src_meta, ...).
        received_metadata = handle[7]
    else:
        raise ValueError(f"Unsupported normal DeepEP route handle length: {len(handle)}")
    if not isinstance(received_metadata, torch.Tensor) or received_metadata.ndim < 1:
        raise TypeError("DeepEP route handle has invalid received-source metadata")
    return received_metadata.shape[0]


def _dispatch_route_preserving_deepep_metadata(
    manager: object,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    assume_all_routes_valid: bool = False,
) -> tuple[tuple, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Create a route-level normal DeepEP handle with tiny payloads.

    This is intentionally a correctness prototype.  Once validated, the same
    route counts/source metadata are to be emitted by the primary aligned
    dispatch so this extra metadata-only communication disappears.
    """
    (
        route_indices,
        route_weights,
        route_fingerprints,
        source_output_index,
        all_routes_valid,
    ) = _compact_route_preserving_metadata_inputs(
        hidden_states,
        topk_indices,
        topk_weights,
        assume_all_routes_valid=assume_all_routes_valid,
    )
    from megatron.core.transformer.moe.fused_a2a import get_buffer, get_hidden_bytes

    group = manager.group
    buffer = get_buffer(group, get_hidden_bytes(route_fingerprints))
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        layout_event,
    ) = buffer.get_dispatch_layout(
        route_indices,
        int(manager.num_experts),
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    (
        recv_fingerprints,
        recv_route_indices,
        recv_route_weights,
        _,
        route_handle,
        _,
    ) = buffer.dispatch(
        route_fingerprints,
        topk_idx=route_indices,
        topk_weights=route_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=layout_event,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    if recv_route_indices is None or recv_route_weights is None:
        raise RuntimeError("Route-preserving DeepEP metadata dispatch dropped top-k metadata")
    return (
        route_handle,
        recv_fingerprints,
        recv_route_indices.reshape(-1),
        recv_route_weights.reshape(-1),
        source_output_index,
        all_routes_valid,
    )


def _validate_and_order_route_preserving_outputs(
    expert_outputs: torch.Tensor,
    received_tokens: torch.Tensor,
    received_topk_indices: torch.Tensor,
    received_topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    route_fingerprints: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    order_outputs: bool = True,
    route_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return expert outputs in the route handle's receive order.

    DeepEP currently produces the same source-token/slot order for the primary
    rank-deduplicated dispatch and the virtual-token metadata dispatch.  Do not
    merely assume that invariant: validate expert ID, exact FP32 weight and an
    sixteen-BF16 source fingerprint before using the route handle.
    """
    if expert_outputs.ndim != 2 or received_tokens.ndim != 2:
        raise ValueError("Route-preserving DeepEP expects 2D hidden tensors")
    if received_topk_indices.shape != received_topk_weights.shape:
        raise ValueError("Received DeepEP IDs and weights do not align")
    if output_index.shape != received_topk_indices.shape:
        raise ValueError("Received DeepEP route mapping does not align")

    positions = torch.nonzero(output_index >= 0, as_tuple=False) if route_positions is None else route_positions
    if positions.shape[0] != route_indices.numel():
        raise RuntimeError(
            "Route-preserving DeepEP route count mismatch: "
            f"primary={positions.shape[0]} metadata={route_indices.numel()}"
        )
    token_rows = positions[:, 0]
    topk_slots = positions[:, 1]
    expected_indices = received_topk_indices[token_rows, topk_slots].reshape(-1)
    expected_weights = received_topk_weights[token_rows, topk_slots].reshape(-1)
    expected_fingerprints = received_tokens.narrow(1, 0, 16).index_select(0, token_rows)
    if route_fingerprints.shape != expected_fingerprints.shape:
        raise RuntimeError(
            "Route-preserving DeepEP fingerprint shape mismatch: "
            f"{tuple(route_fingerprints.shape)} != {tuple(expected_fingerprints.shape)}"
        )

    torch._assert_async(
        torch.all(expected_indices == route_indices.to(dtype=expected_indices.dtype)),
        "Route-preserving DeepEP metadata changed local expert order",
    )
    torch._assert_async(
        torch.all(expected_weights == route_weights.to(dtype=expected_weights.dtype)),
        "Route-preserving DeepEP metadata changed route probability order",
    )
    torch._assert_async(
        torch.all(expected_fingerprints == route_fingerprints),
        "Route-preserving DeepEP metadata changed source-token order",
    )

    if not order_outputs:
        return expert_outputs
    route_rows = output_index[token_rows, topk_slots].to(dtype=torch.long)
    return expert_outputs.index_select(0, route_rows)


def _patch_sglang_deepep_layer(mlp: torch.nn.Module, global_layer: int) -> bool:
    """Match SGLang low-latency reduction over Megatron normal DeepEP."""
    if getattr(mlp, "_slime_sglang_deepep_alignment", False):
        return False

    router = getattr(mlp, "router", None)
    dispatcher = getattr(mlp, "token_dispatcher", None)
    experts = getattr(mlp, "experts", None)
    if router is None or dispatcher is None or experts is None:
        raise RuntimeError(f"Layer {global_layer} is not a complete MoE layer")
    manager = getattr(dispatcher, "_comm_manager", None)
    if manager is None or manager.__class__.__name__ != "_DeepepManager":
        raise RuntimeError(
            "SGLang DeepEP alignment requires MCore's flex _DeepepManager; "
            f"layer {global_layer} has {type(manager).__name__}"
        )
    scaling_factor = float(router.config.moe_router_topk_scaling_factor or 1.0)
    original_routing = router.routing

    def routing_without_final_scaling(
        patched_router: torch.nn.Module,
        *args: object,
        **kwargs: object,
    ):
        config = patched_router.config
        previous = config.moe_router_topk_scaling_factor
        config.moe_router_topk_scaling_factor = 1.0
        try:
            return original_routing(*args, **kwargs)
        finally:
            config.moe_router_topk_scaling_factor = previous

    router.routing = types.MethodType(routing_without_final_scaling, router)
    original_setup_metadata = manager.setup_metadata
    original_dispatch = manager.dispatch
    from slime.utils.routing_replay import consume_ordered_topk, register_ordered_topk_capture

    register_ordered_topk_capture(router)

    def setup_ordered_metadata(
        patched_manager,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        router_token_masks: torch.Tensor | None = None,
    ) -> None:
        patched_manager._slime_source_fixed_topk_valid = False
        ordered = consume_ordered_topk(router)
        if ordered is None:
            # Megatron 1dcf0dafa's DeepEP dispatcher setup_metadata takes only
            # (routing_map, probs); the router_token_masks padding-mask arg exists
            # on newer forks. Call the original with whatever arity it supports.
            if router_token_masks is None:
                original_setup_metadata(routing_map, probs)
            else:
                original_setup_metadata(routing_map, probs, router_token_masks)
            return

        num_tokens = routing_map.shape[0]
        if ordered.shape[0] != num_tokens:
            raise ValueError(
                "Ordered top-k token count differs from DeepEP input: " f"{ordered.shape[0]} != {num_tokens}"
            )
        ordered = ordered.to(
            device=probs.device,
            dtype=torch.int64,
            non_blocking=ordered.is_pinned(),
        )
        dense_probs = probs.reshape(num_tokens, patched_manager.num_experts)
        patched_manager.token_indices = ordered
        patched_manager.token_probs = dense_probs.gather(-1, ordered)
        patched_manager._slime_source_fixed_topk_valid = (
            patched_manager.capacity_factor is None and router_token_masks is None
        )
        if patched_manager.capacity_factor is not None:
            patched_manager.token_indices = patched_manager.token_indices.masked_fill(
                patched_manager.token_probs == 0,
                -1,
            )
        if router_token_masks is not None:
            patched_manager.token_indices = patched_manager.token_indices.masked_fill(
                router_token_masks.view(-1, 1),
                -1,
            )

    manager.setup_metadata = types.MethodType(
        setup_ordered_metadata,
        manager,
    )

    def dispatch_with_route_preserving_handle(
        patched_manager,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        if patched_manager.token_indices is None or patched_manager.token_probs is None:
            raise RuntimeError("Ordered source top-k metadata is unavailable before DeepEP dispatch")
        dispatched_hidden = original_dispatch(
            hidden_states,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # _DeepepManager normalizes probabilities to FP32 in its dispatch
        # preamble.  Capture the normalized tensors that were actually sent.
        source_topk_indices = patched_manager.token_indices
        source_topk_weights = patched_manager.token_probs
        source_fixed_topk_valid = bool(getattr(patched_manager, "_slime_source_fixed_topk_valid", False))
        (
            route_handle,
            recv_route_fingerprints,
            recv_route_indices,
            recv_route_weights,
            source_output_index,
            source_all_routes_valid,
        ) = _dispatch_route_preserving_deepep_metadata(
            patched_manager,
            hidden_states,
            source_topk_indices,
            source_topk_weights,
            assume_all_routes_valid=source_fixed_topk_valid,
        )
        route_metadata_prevalidated = False
        patched_manager._slime_route_handle = route_handle
        patched_manager._slime_route_recv_fingerprints = recv_route_fingerprints
        patched_manager._slime_route_recv_indices = recv_route_indices
        patched_manager._slime_route_recv_weights = recv_route_weights
        patched_manager._slime_route_metadata_prevalidated = route_metadata_prevalidated
        patched_manager._slime_route_source_topk_indices = source_topk_indices
        patched_manager._slime_route_source_topk_weights = source_topk_weights
        patched_manager._slime_route_source_output_index = source_output_index
        patched_manager._slime_route_source_all_valid = source_all_routes_valid
        return dispatched_hidden

    manager.dispatch = types.MethodType(
        dispatch_with_route_preserving_handle,
        manager,
    )

    def get_permuted_hidden_states_by_experts(
        patched_manager,
        hidden_states: torch.Tensor,
    ):
        topk_indices = patched_manager.dispatched_indices
        topk_weights = patched_manager.dispatched_probs
        if topk_indices is None or topk_weights is None:
            raise RuntimeError("DeepEP dispatch metadata is unavailable before local permutation")
        (
            permuted_hidden,
            permuted_probs,
            output_index,
            sanitized_indices,
            routing_map,
            all_routes_valid,
            route_positions,
        ) = _scatter_deepep_routes_with_padding(
            hidden_states,
            topk_indices,
            topk_weights,
            patched_manager.tokens_per_expert,
            return_route_positions=True,
            expected_route_count=_deepep_route_handle_received_rows(patched_manager._slime_route_handle),
        )
        patched_manager.hidden_shape_before_permute = hidden_states.shape
        patched_manager.dispatched_routing_map = routing_map
        patched_manager._slime_sglang_topk_indices = sanitized_indices
        patched_manager._slime_sglang_topk_weights = topk_weights
        patched_manager._slime_sglang_output_index = output_index
        patched_manager._slime_sglang_route_positions = route_positions
        patched_manager._slime_sglang_all_routes_valid = all_routes_valid
        patched_manager._slime_sglang_expert_inputs = permuted_hidden
        patched_manager._slime_sglang_expert_probs = permuted_probs
        patched_manager._slime_sglang_tokens_per_expert = patched_manager.tokens_per_expert
        route_fingerprints = getattr(
            patched_manager,
            "_slime_route_recv_fingerprints",
            None,
        )
        route_indices = getattr(patched_manager, "_slime_route_recv_indices", None)
        route_weights = getattr(patched_manager, "_slime_route_recv_weights", None)
        route_metadata_prevalidated = bool(getattr(patched_manager, "_slime_route_metadata_prevalidated", False))
        if route_metadata_prevalidated:
            if any(value is not None for value in (route_fingerprints, route_indices, route_weights)):
                raise RuntimeError("Cached DeepEP route metadata retained unexpected payloads")
        else:
            if route_fingerprints is None or route_indices is None or route_weights is None:
                raise RuntimeError("Route-preserving DeepEP metadata handle is unavailable")
            _validate_and_order_route_preserving_outputs(
                permuted_hidden,
                hidden_states,
                sanitized_indices,
                topk_weights,
                output_index,
                route_fingerprints,
                route_indices,
                route_weights,
                order_outputs=False,
                route_positions=route_positions,
            )
        del patched_manager._slime_route_recv_fingerprints
        del patched_manager._slime_route_recv_indices
        del patched_manager._slime_route_recv_weights
        del patched_manager._slime_route_metadata_prevalidated
        return permuted_hidden, permuted_probs

    def get_restored_hidden_states_by_experts(
        patched_manager,
        hidden_states: torch.Tensor,
    ):
        topk_indices = getattr(patched_manager, "_slime_sglang_topk_indices", None)
        topk_weights = getattr(patched_manager, "_slime_sglang_topk_weights", None)
        output_index = getattr(patched_manager, "_slime_sglang_output_index", None)
        route_positions = getattr(patched_manager, "_slime_sglang_route_positions", None)
        if topk_indices is None or topk_weights is None or output_index is None or route_positions is None:
            raise RuntimeError("Saved route-preserving DeepEP mapping is unavailable")
        token_rows = route_positions[:, 0]
        topk_slots = route_positions[:, 1]
        route_rows = output_index[token_rows, topk_slots].to(dtype=torch.long)
        output = hidden_states.index_select(0, route_rows)
        del patched_manager._slime_sglang_topk_indices
        del patched_manager._slime_sglang_topk_weights
        del patched_manager._slime_sglang_output_index
        del patched_manager._slime_sglang_route_positions
        del patched_manager._slime_sglang_all_routes_valid
        del patched_manager._slime_sglang_expert_inputs
        del patched_manager._slime_sglang_expert_probs
        del patched_manager._slime_sglang_tokens_per_expert
        return output

    manager.get_permuted_hidden_states_by_experts = types.MethodType(
        get_permuted_hidden_states_by_experts,
        manager,
    )
    manager.get_restored_hidden_states_by_experts = types.MethodType(
        get_restored_hidden_states_by_experts,
        manager,
    )

    # Megatron 1dcf0dafa's MoELayer.combine(output) runs only token_combine, with a
    # separate postprocess(output, shared_expert_output) doing combine_postprocess +
    # shared-expert add. Newer forks fuse all of that into
    # combine(output, shared_expert_output). Detect which convention applies here.
    import inspect

    combine_fuses_postprocess = len(inspect.signature(mlp.combine).parameters) >= 2

    if not combine_fuses_postprocess:
        # Megatron 1dcf0dafa split token-combine from postprocess.  Preserve
        # SGLang's single BF16 shared.add_(routed, alpha=scaling_factor)
        # operation here: scaling the routed value first would introduce an
        # extra BF16 rounding before the shared-expert addition.
        def postprocess(
            patched_mlp: torch.nn.Module,
            output: torch.Tensor,
            shared_expert_output: torch.Tensor | None,
        ) -> torch.Tensor:
            output = patched_mlp.token_dispatcher.combine_postprocess(output)
            if bool(getattr(patched_mlp.config, "moe_latent_size", None)):
                output, _ = patched_mlp.fc2_latent_proj(output)
            if shared_expert_output is not None:
                return torch.add(
                    shared_expert_output,
                    output,
                    alpha=scaling_factor,
                )
            if scaling_factor != 1.0:
                output = output * scaling_factor
            return output

        mlp.postprocess = types.MethodType(postprocess, mlp)

    def combine(
        patched_mlp: torch.nn.Module,
        output: torch.Tensor,
        shared_expert_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        route_handle = getattr(manager, "_slime_route_handle", None)
        source_topk_indices = getattr(manager, "_slime_route_source_topk_indices", None)
        source_topk_weights = getattr(manager, "_slime_route_source_topk_weights", None)
        source_output_index = getattr(manager, "_slime_route_source_output_index", None)
        source_all_routes_valid = getattr(manager, "_slime_route_source_all_valid", None)
        if (
            route_handle is None
            or source_topk_indices is None
            or source_topk_weights is None
            or source_output_index is None
            or source_all_routes_valid is None
        ):
            raise RuntimeError("Route-preserving DeepEP combine handle is incomplete")
        from megatron.core.transformer.moe.fused_a2a import fused_combine

        combined_routes, _ = fused_combine(
            output,
            manager.group,
            route_handle,
            async_finish=True,
            allocate_on_comm_stream=getattr(patched_mlp.token_dispatcher, "allocate_on_comm_stream", False),
        )
        output = _SGLangEPGatherWithBF16Backward.apply(
            combined_routes,
            source_topk_indices,
            source_topk_weights,
            source_output_index,
            True,
            source_all_routes_valid,
        )
        manager.handle = None
        del manager._slime_route_handle
        del manager._slime_route_source_topk_indices
        del manager._slime_route_source_topk_weights
        del manager._slime_route_source_output_index
        del manager._slime_route_source_all_valid
        if combine_fuses_postprocess:
            output = patched_mlp.token_dispatcher.combine_postprocess(output)
            if shared_expert_output is not None:
                output = torch.add(
                    shared_expert_output,
                    output,
                    alpha=scaling_factor,
                )
            elif scaling_factor != 1.0:
                output = output * scaling_factor
        # 1dcf0dafa applies reshape, routed scaling, and the shared-expert add
        # in the patched postprocess above so their rounding order stays fused.
        return output

    mlp.combine = types.MethodType(combine, mlp)
    experts._slime_defer_router_probabilities = True
    experts._slime_reuse_expert_input_for_grad = True
    mlp._slime_sglang_deepep_alignment = True
    mlp._slime_sglang_deepep_global_layer = global_layer
    return True


def enable_sglang_deepep_moe_alignment(args, model, store_prefix: str) -> None:
    """Install the exact SGLang DeepEP combine semantics on selected MoE layers."""
    del store_prefix
    if not bool(getattr(args, "deterministic_mode", False)):
        return
    if not bool(getattr(args, "moe_enable_deepep", False)):
        return
    selected_layers = {int(layer) for layer in (getattr(args, "megatron_deepgemm_moe_forward_layers", None) or ())}
    if not selected_layers:
        return

    patched = []
    for model_chunk in _normalize_model_chunks(model):
        for layer_name, layer in model_chunk.named_modules():
            if re.match(r"^(?:.*\.)?decoder\.layers\.\d+$", layer_name) is None:
                continue
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None:
                continue
            global_layer = int(layer_number) - 1
            if global_layer not in selected_layers:
                continue
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "experts"):
                continue
            if _patch_sglang_deepep_layer(mlp, global_layer):
                patched.append(global_layer)

    if patched and _should_log_deepgemm_summary():
        logger.info(
            "Enabled SGLang ordered FP32 DeepEP gather on %d local MoE layers " "(global layers=%s)",
            len(patched),
            _format_int_ranges(selected_layers),
        )


def install_deepgemm_moe_forward(
    model,
    global_layer_indices: Iterable[int],
    *,
    target_suffixes: Iterable[str] = _DEFAULT_TARGET_SUFFIXES,
) -> list[str]:
    """Wrap selected global MoE layers, using their ``layer_number`` metadata.

    ``global_layer_indices`` are zero-based global decoder-layer indices.  This
    remains correct with pipeline parallelism even though each model chunk's
    ``decoder.layers`` list starts at local index zero.
    """
    _validate_parallelism()
    selected_layers = {int(layer_index) for layer_index in global_layer_indices}
    if not selected_layers:
        raise RuntimeError("global_layer_indices must select at least one MoE layer")
    suffixes = tuple(target_suffixes)
    if not suffixes:
        raise RuntimeError("target_suffixes must select at least one module name")

    workspace_bytes = _combine_workspace_bytes()
    wrapped = []
    workspaces_by_device: dict[torch.device, torch.Tensor] = {}
    for model_chunk in _normalize_model_chunks(model):
        for name, module in model_chunk.named_modules():
            if not any(name.endswith(suffix) for suffix in suffixes):
                continue
            if _get_global_layer_index(model_chunk, name) not in selected_layers:
                continue
            if _wrap_te_grouped_mlp(module, name):
                mlp_name = name.rsplit(".", 1)[0]
                mlp = model_chunk.get_submodule(mlp_name)
                dispatcher = getattr(mlp, "token_dispatcher", None)
                if (
                    workspace_bytes is not None
                    and dispatcher is not None
                    and int(getattr(dispatcher, "num_local_experts", 1)) > 1
                ):
                    _wrap_preallocated_combine_preprocess(dispatcher)
                    parameter = next(module.parameters())
                    workspace = workspaces_by_device.get(parameter.device)
                    if workspace is None:
                        workspace = torch.empty(
                            workspace_bytes,
                            dtype=torch.uint8,
                            device=parameter.device,
                        )
                        workspaces_by_device[parameter.device] = workspace
                    setattr(dispatcher, _COMBINE_WORKSPACE_ATTR, workspace)
                    setattr(module, _COMBINE_WORKSPACE_ATTR, workspace)
                    _wrap_preallocated_dispatch_postprocess(dispatcher)
                    _wrap_preallocated_token_combine(dispatcher)
                wrapped.append(name)

    if wrapped and _should_log_deepgemm_summary():
        logger.info(
            "Enabled SGLang grouped DeepGEMM MoE forward+BF16-backward on %d " "TEGroupedMLPs (global layers=%s)",
            len(wrapped),
            _format_int_ranges(selected_layers),
        )
        logger.debug("DeepGEMM wrapped TEGroupedMLPs: %s", ", ".join(wrapped))
    else:
        # Most PP ranks legitimately do not own a requested global layer.
        logger.debug(
            "No TEGroupedMLP matched the requested DeepGEMM MoE layers %s and suffixes %s",
            sorted(selected_layers),
            suffixes,
        )
    return wrapped


def enable_deepgemm_moe_forward(args, model, store_prefix: str) -> None:
    """Install the grouped DeepGEMM forward-value probe on selected MoE layers."""
    del store_prefix
    layers = getattr(args, "megatron_deepgemm_moe_forward_layers", None)
    if layers is None:
        raise RuntimeError(
            "args.megatron_deepgemm_moe_forward_layers is required; pass --megatron-deepgemm-moe-forward-layers"
        )
    suffixes = getattr(args, "megatron_deepgemm_moe_forward_modules", None) or _DEFAULT_TARGET_SUFFIXES
    install_deepgemm_moe_forward(
        model,
        layers,
        target_suffixes=suffixes,
    )
