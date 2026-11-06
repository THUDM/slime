"""Shared BF16 MoE expert dgrad/wgrad for aligned Megatron MoE forwards.

Used by DeepGEMM FP8 MoE and cutlass NVFP4 MoE after DeepEP dispatch.  Lives
outside ``deepgemm_moe_forward`` so quantization-specific forward modules can
reuse the same backward without co-locating NVFP4 logic there.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from slime.backends.megatron_utils.alignment.deepgemm_forward import (
    _deepgemm_bf16_gemm_nn,
    _deepgemm_bf16_gemm_nt,
    _deepgemm_bf16_gemm_tn,
    _sum_to_parameter_dtype,
)


def moe_bf16_expert_backward(
    *,
    hidden_states: torch.Tensor,
    permuted_probs: torch.Tensor,
    grad_output: torch.Tensor,
    fc1_weights: tuple[torch.Tensor, ...],
    fc2_weights: tuple[torch.Tensor, ...],
    counts: tuple[int, ...],
    layout: Any,
    module_name: str,
    needs_hidden: bool,
    needs_probs: bool,
    needs_fc1_weights: tuple[bool, ...],
    needs_fc2_weights: tuple[bool, ...],
    defer_router_probabilities: bool,
    reuse_expert_input_for_grad: bool,
    grad_workspace: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    list[torch.Tensor | None],
    list[torch.Tensor | None],
]:
    """BF16 MoE dgrad/wgrad recompute (grouped or per-expert chunked)."""
    # Import DeepGEMM MoE helpers lazily to avoid an import cycle with
    # ``deepgemm_moe_forward`` (which calls this function from autograd.backward).
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
        _BACKWARD_CHUNK_ROWS,
        _grouped_expert_backward,
        _router_probability_grad_fp32_chunked,
        _use_grouped_bf16_backward,
    )

    del module_name
    if len(fc1_weights) != layout.num_local_experts or len(fc2_weights) != layout.num_local_experts:
        raise RuntimeError(
            "saved expert weight count mismatch: "
            f"fc1={len(fc1_weights)} fc2={len(fc2_weights)} expected={layout.num_local_experts}"
        )

    grad_hidden = None
    if needs_hidden:
        if grad_workspace is None and reuse_expert_input_for_grad:
            # DeepEP's expert-major input has no later reader.  Each
            # expert/chunk finishes recompute and wgrad before writing its
            # dgrad, so that input storage can safely carry the gradient.
            grad_hidden = hidden_states.detach()
        elif grad_workspace is None:
            grad_hidden = torch.empty_like(hidden_states)
        else:
            required_bytes = hidden_states.numel() * hidden_states.element_size()
            if required_bytes > grad_workspace.numel():
                raise RuntimeError(
                    "Shared MoE backward workspace is too small: "
                    f"need {required_bytes} bytes for {tuple(hidden_states.shape)}, "
                    f"have {grad_workspace.numel()} bytes; increase "
                    "SLIME_DEEPGEMM_MOE_COMBINE_WORKSPACE_BYTES"
                )
            if grad_workspace.device != hidden_states.device:
                raise RuntimeError(
                    "Shared MoE backward workspace is on "
                    f"{grad_workspace.device}, input is on {hidden_states.device}"
                )
            grad_hidden = grad_workspace.narrow(0, 0, required_bytes).view(hidden_states.dtype).view_as(hidden_states)
    grad_probs = torch.empty_like(permuted_probs) if needs_probs else None
    grad_fc1_weights: list[torch.Tensor | None] = [None] * layout.num_local_experts
    grad_fc2_weights: list[torch.Tensor | None] = [None] * layout.num_local_experts
    grad_output = grad_output.contiguous().to(dtype=hidden_states.dtype)
    probabilities = permuted_probs.reshape(-1, 1)

    if _use_grouped_bf16_backward(
        hidden_states,
        counts,
        needs_fc1_weights,
        needs_fc2_weights,
    ):
        return _grouped_expert_backward(
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
                assert grad_probs is not None
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
                assert grad_hidden is not None
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
        None if defer_router_probabilities else grad_probs,
        grad_fc1_weights,
        grad_fc2_weights,
    )
