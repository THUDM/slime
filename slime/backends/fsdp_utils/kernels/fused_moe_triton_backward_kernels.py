from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_backward_input_kernel(
    # Pointers to matrices
    grad_output_ptr,
    weight_ptr,
    grad_input_ptr,
    grad_topk_weights_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_we,
    stride_wn,
    stride_wk,
    stride_gim,
    stride_gik,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_input.

    Forward: output = input @ weight.T (optionally multiplied by topk_weights)
    Backward: grad_input = grad_output @ weight (optionally multiplied by topk_weights)

    This kernel computes: grad_input[token] = sum_over_N(grad_output[token, n] * weight[expert, n, :])
    If MUL_ROUTED_WEIGHT: grad_input[token] *= topk_weights[token]

    Parallelization: Similar to forward, parallel over M and N dimensions, loop over K.
    """
    # Map program ids to blocks (parallel over M and N, similar to forward)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Check bounds
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    if pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
        # Load token information
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        offs_token = offs_token.to(tl.int64)
        token_mask = offs_token < num_valid_tokens

        # Get expert ID for this block
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        # Only process if expert is valid
        if off_experts != -1:
            # Initialize offsets for N dimension (current block)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Load grad_output block: shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
            grad_output_ptrs = grad_output_ptr + (offs_token[:, None] * stride_gom + offs_n[None, :] * stride_gon)
            grad_out = tl.load(
                grad_output_ptrs,
                mask=token_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )

            # Apply topk_weights to grad_output if needed
            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                grad_out = grad_out * moe_weight[:, None]

            # Iterate over K dimension
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # Current K offsets
                curr_offs_k = k * BLOCK_SIZE_K + offs_k

                # Load weight block: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
                # weight: shape (E, N, K)
                weight_ptrs = (
                    weight_ptr
                    + off_experts * stride_we
                    + offs_n[:, None] * stride_wn
                    + curr_offs_k[None, :] * stride_wk
                )
                w = tl.load(
                    weight_ptrs,
                    mask=(offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                    other=0.0,
                )

                # Compute contribution: grad_out @ weight
                # grad_out: (BLOCK_SIZE_M, BLOCK_SIZE_N)
                # w: (BLOCK_SIZE_N, BLOCK_SIZE_K)
                # result: (BLOCK_SIZE_M, BLOCK_SIZE_K)
                contribution = tl.dot(grad_out, w)

                # Atomic add to grad_input because different N blocks contribute to same K
                grad_input_ptrs = grad_input_ptr + (
                    (offs_token[:, None] // top_k) * stride_gim + curr_offs_k[None, :] * stride_gik
                )
                grad_input_mask = token_mask[:, None] & (curr_offs_k[None, :] < K)
                tl.atomic_add(grad_input_ptrs, contribution.to(compute_type), mask=grad_input_mask)


@triton.jit
def fused_moe_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr,
    input_ptr,
    grad_weight_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_weight.

    Forward: output = input @ weight.T (optionally multiplied by topk_weights)
    Backward: grad_weight = input.T @ grad_output (optionally multiplied by topk_weights)

    This kernel computes: grad_weight[expert, n, k] = sum_over_tokens(input[token, k] * grad_output[token, n])
    If MUL_ROUTED_WEIGHT: the accumulation is weighted by topk_weights[token]

    Parallelization: Parallel over M and N dimensions with grouping, loop over K.
    """
    # Map program ids to blocks (parallel over M and N with grouping, similar to forward and backward_input)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Check bounds
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Get expert ID for this M block
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Only process if expert is valid
    if expert_id == -1:
        return

    # Load token information for this M block
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m.to(tl.int64)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=num_valid_tokens
    )
    offs_token = offs_token.to(tl.int64)
    token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

    # Clamp offs_token to valid range
    offs_token_clamped = tl.where(token_mask, offs_token, 0)

    # Determine input token indices based on MUL_ROUTED_WEIGHT
    if MUL_ROUTED_WEIGHT:
        input_token_idx = offs_token_clamped
        input_mask = token_mask
    else:
        input_token_idx = offs_token_clamped // top_k
        num_input_tokens = num_valid_tokens // top_k
        input_mask = token_mask & (input_token_idx < num_input_tokens)

    # Load topk_weights if needed
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token_clamped, mask=token_mask, other=0.0)

    # Current N offset for this program
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)

    # Load grad_output for this N block: shape (M, BLOCK_SIZE_N)
    # grad_output is always indexed by sorted_token_ids (offs_token_clamped)
    # because it has shape (num_tokens * topk, N)
    grad_output_ptrs = grad_output_ptr + (offs_token_clamped[:, None] * stride_gom + offs_n[None, :] * stride_gon)
    grad_out = tl.load(
        grad_output_ptrs,
        mask=token_mask[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )

    # Apply topk_weights if needed
    if MUL_ROUTED_WEIGHT:
        grad_out = grad_out * moe_weight[:, None]

    # Zero out padding tokens
    token_mask_col = token_mask[:, None]
    grad_out = grad_out * token_mask_col

    # Iterate over K blocks and accumulate
    for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

        # Load input for this K block
        input_ptrs = input_ptr + (input_token_idx[:, None] * stride_im + offs_k[None, :] * stride_ik)
        inp = tl.load(
            input_ptrs,
            mask=input_mask[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )

        # Zero out padding tokens - use input_mask for input, token_mask for grad_output
        input_mask_col = input_mask[:, None]
        inp = inp * input_mask_col

        # Compute grad_weight contribution: grad_out.T @ inp
        grad_w_contribution = tl.dot(grad_out.T, inp)

        # Write back using atomic add
        grad_weight_ptrs = (
            grad_weight_ptr + expert_id * stride_gwe + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        grad_weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.atomic_add(grad_weight_ptrs, grad_w_contribution.to(compute_type), mask=grad_weight_mask)


@triton.jit
def fused_moe_backward_topk_weights_kernel(
    # Pointers to matrices
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    grad_topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Backward kernel for computing grad_topk_weights.

    Forward: output = topk_weights * (input @ weight.T)
    Backward: grad_topk_weights = sum(grad_output * (input @ weight.T))

    This kernel computes the gradient of topk_weights by computing the dot product
    of grad_output with the forward output before weight multiplication.
    """
    # Map program id to token block
    pid = tl.program_id(axis=0)

    # Check bounds
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    # Only process if this block is valid
    if pid * BLOCK_SIZE_M < num_tokens_post_padded:
        # Load token information
        offs_token_id = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(
            sorted_token_ids_ptr + offs_token_id, mask=offs_token_id < num_tokens_post_padded, other=num_valid_tokens
        )
        offs_token = offs_token.to(tl.int64)
        token_mask = (offs_token_id < num_tokens_post_padded) & (offs_token < num_valid_tokens)

        # Clamp offs_token to valid range for safe pointer arithmetic
        offs_token_clamped = tl.where(token_mask, offs_token, 0)

        # Get expert ID for this block
        off_experts = tl.load(expert_ids_ptr + pid).to(tl.int64)

        # Only process if expert is valid
        if off_experts != -1:
            # Initialize offsets
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Accumulator for grad_topk_weights
            accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

            # Iterate over N and K dimensions to compute forward output and gradient
            for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
                # Current N offset
                curr_offs_n = n * BLOCK_SIZE_N + offs_n

                # Load grad_output block: (M, N)
                grad_output_ptrs = grad_output_ptr + (
                    offs_token_clamped[:, None] * stride_gom + curr_offs_n[None, :] * stride_gon
                )
                grad_out = tl.load(
                    grad_output_ptrs,
                    mask=token_mask[:, None] & (curr_offs_n[None, :] < N),
                    other=0.0,
                )

                # Compute forward output for this N block: input @ weight[:, n, :].T
                forward_output_n = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    # Current K offset
                    curr_offs_k = k * BLOCK_SIZE_K + offs_k

                    # Load input block: (M, K)
                    input_ptrs = input_ptr + (
                        (offs_token_clamped[:, None] // top_k) * stride_im + curr_offs_k[None, :] * stride_ik
                    )
                    inp = tl.load(
                        input_ptrs,
                        mask=token_mask[:, None] & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )

                    # Load weight block: (N, K)
                    weight_ptrs = (
                        weight_ptr
                        + off_experts * stride_we
                        + curr_offs_n[:, None] * stride_wn
                        + curr_offs_k[None, :] * stride_wk
                    )
                    w = tl.load(
                        weight_ptrs,
                        mask=(curr_offs_n[:, None] < N) & (curr_offs_k[None, :] < K),
                        other=0.0,
                    )

                    # Accumulate forward output: input @ weight.T
                    # inp: (M, K), w.T: (K, N) -> (M, N)
                    forward_output_n += tl.dot(inp, w.T)

                # Compute contribution to grad_topk_weights: sum(grad_out * forward_output)
                # Sum over N dimension
                accumulator += tl.sum(grad_out * forward_output_n, axis=1)

            # Write back grad_topk_weights using atomic add with clamped token indices
            tl.atomic_add(grad_topk_weights_ptr + offs_token_clamped, accumulator.to(compute_type), mask=token_mask)


def invoke_fused_moe_backward_kernel(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: torch.Tensor,
    grad_topk_weights: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """
    Invoke the fused MOE backward kernels to compute gradients.

    Args:
        grad_output: Gradient of output, shape (num_tokens * topk, N) or (num_tokens, topk, N)
        input: Input tensor, shape (num_tokens, K)
        weight: Weight tensor, shape (E, N, K)
        grad_input: Output gradient for input, shape (num_tokens, K)
        grad_weight: Output gradient for weight, shape (E, N, K)
        grad_topk_weights: Output gradient for topk_weights, shape (num_tokens, topk) or None
        topk_weights: Top-K routing weights, shape (num_tokens, topk)
        topk_ids: Top-K expert IDs, shape (num_tokens, topk)
        sorted_token_ids: Sorted token IDs
        expert_ids: Expert IDs for each block
        num_tokens_post_padded: Number of tokens after padding
        mul_routed_weight: Whether to multiply by routing weights
        top_k: Number of experts per token
        config: Kernel configuration
        compute_type: Computation data type
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    # Flatten grad_output if needed
    # Before: (num_tokens, topk, hidden_size)
    # After: (num_tokens * topk, hidden_size)
    if grad_output.ndim == 3:
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

    E, N, K = weight.shape

    # ===================== Compute grad_input =====================
    def grid_input(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    fused_moe_backward_input_kernel[grid_input](
        grad_output,
        weight,
        grad_input,
        grad_topk_weights if grad_topk_weights is not None else grad_input,  # dummy pointer
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        grad_input.stride(0),
        grad_input.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # ===================== Compute grad_weight =====================
    # Initialize grad_weight to zero
    grad_weight.zero_()

    # Use same grid configuration as forward kernel: encode both M and N dimensions
    def grid_weight(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    fused_moe_backward_weight_kernel[grid_weight](
        grad_output,
        input,
        grad_weight,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        grad_output.shape[0],
        grad_output.stride(0),
        grad_output.stride(1),
        input.stride(0),
        input.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
        grad_weight.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        **config,
    )

    # ===================== Compute grad_topk_weights (if needed) =====================
    if mul_routed_weight and grad_topk_weights is not None:

        def grid_topk(META):
            return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]),)

        fused_moe_backward_topk_weights_kernel[grid_topk](
            grad_output,
            input,
            weight,
            grad_topk_weights.view(-1),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            sorted_token_ids.shape[0],
            grad_output.shape[0],
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            top_k=top_k,
            compute_type=compute_type,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        )
