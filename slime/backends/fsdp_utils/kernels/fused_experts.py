import torch
import triton.language as tl
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce,
    silu_and_mul,
)


class GateUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, D_in = hidden_states.shape
        E, N, K = w1.shape
        assert D_in == K, f"hidden_states dim {D_in} != w1 dim {K}"

        topk = topk_ids.shape[1]

        # Output: (num_tokens * topk, N)
        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Python implementation: iterate over tokens and their topk experts
        # For each token t and expert k:
        #   intermediate_cache1[t*topk + k] = hidden_states[t] @ w1[expert_id].T
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_ids[t, k].item()
                x_t = hidden_states[t]  # shape: (D_in,)
                W1_e = w1[expert_id]    # shape: (N, K)
                # Matrix multiplication using torch.mul and torch.sum:
                # (D_in,) @ (N, K).T = sum((D_in,) * (N, K) over K dimension)
                intermediate_cache1[t * topk + k] = x_t @ W1_e.T

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for GateUpProjFunction.

        Forward: output = input @ w1 (without topk_weight multiplication)
        Backward:
            - grad_hidden_states = grad_output @ w1.T
            - grad_w1 = input.T @ grad_output (note: transposed)
            - grad_topk_weights = zeros (not needed in this stage)

        Args:
            grad_output: shape (num_tokens * topk, N)

        Returns:
            (grad_hidden_states, grad_w1, grad_topk_weights, None)
        """
        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        topk = ctx.topk

        num_tokens, D_in = hidden_states.shape
        E, N, _ = w1.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_hidden_states = torch.zeros_like(hidden_states)
        # Use float32 for grad_w1 accumulation to avoid bfloat16 precision loss
        grad_w1 = torch.zeros(w1.shape, dtype=torch.float32, device=w1.device)
        # GateUpProj stage doesn't compute topk_weights gradient
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            # 1. Calculate grad_hidden_states: grad_output @ w1
            # For each token t and expert k:
            #   grad_hidden_states[t] += grad_output[t*topk+k] @ w1[expert_id]
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t * topk + k]  # shape: (N,)
                    W1_e = w1[expert_id]  # shape: (N, D_in)
                    # grad_x: (N,) @ (N, D_in) -> (D_in,)
                    grad_hidden_states[begin_chunk_idx + t] += grad_y_tk @ W1_e

            # 2. Calculate grad_w1: input.T @ grad_output
            # For each token t and expert k:
            #   grad_w1[expert_id] += input[t].T @ grad_output[t*topk+k]
            # Which is: grad_w1[expert_id] += outer(grad_output[t*topk+k], input[t])
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    x_t = curr_hidden_states[t]  # shape: (D_in,)
                    grad_y_tk = curr_grad_output[t * topk + k]  # shape: (N,)
                    # grad_W1: outer(grad_y_tk, x_t) -> (N, D_in)
                    # Convert to float32 for accumulation to avoid bfloat16 precision loss
                    grad_w1[expert_id] = torch.outer(grad_y_tk, x_t).to(torch.float32)
        # Convert grad_w1 back to original dtype (bfloat16)
        grad_w1 = grad_w1.to(hidden_states.dtype)

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        num_tokens, N = intermediate_cache1.shape
        intermediate_cache2 = torch.empty(
            (num_tokens, N // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache1,) = ctx.saved_tensors
        N = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)

        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)

        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        total_tokens, intermediate_size = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens = total_tokens // topk
        E, hidden_size, K = w2.shape
        assert intermediate_size == K, f"intermediate_cache2 dim {intermediate_size} != w2 dim {K}"

        # Output: (num_tokens, topk, hidden_size)
        intermediate_cache3 = torch.empty(
            (num_tokens, topk, hidden_size),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        # Python implementation: iterate over tokens and their topk experts
        # For each token t and expert k:
        #   intermediate_cache3[t, k] = topk_weights[t, k] * (intermediate_cache2[t*topk+k] @ w2[expert_id].T)
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_ids[t, k].item()
                x_tk = intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
                W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)
                weight_tk = topk_weights[t, k]  # scalar

                # Matrix multiplication using torch.mul and torch.sum with weight
                # weight * (x @ W.T) = weight * sum(x * W over intermediate_size dimension)
                intermediate_cache3[t, k] = weight_tk * (x_tk @ W2_e.T)

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DownProjFunction.

        Forward: output = topk_weights * (input @ w2) (with topk_weight multiplication)
        Backward:
            - grad_intermediate_cache2 = topk_weights * (grad_output @ w2.T)
            - grad_w2 = topk_weights * (grad_output.T @ intermediate_cache2)

        Args:
            grad_output: shape (num_tokens, topk, hidden_size)

        Returns:
            (grad_intermediate_cache2, grad_w2, grad_topk_weights, None)
        """
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        # Use float32 for grad_w2 accumulation to avoid bfloat16 precision loss
        grad_w2 = torch.zeros(w2.shape, dtype=torch.float32, device=w2.device)
        # Compute grad_topk_weights in DownProjFunction backward
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            # 1. Calculate grad_intermediate_cache2: topk_weights * (grad_output @ w2)
            # Forward: y[t,k] = topk_weights[t,k] * (x[t*topk+k] @ w2[expert_id].T)
            # Backward: grad_x[t*topk+k] = topk_weights[t,k] * (grad_y[t,k] @ w2[expert_id])
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
                    W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)
                    weight_tk = curr_topk_weights[t, k]  # scalar

                    # grad_intermediate_cache2 using torch.mul: topk_weight * (grad_y @ W2)
                    # (hidden_size,) @ (hidden_size, intermediate_size) -> (intermediate_size,)
                    grad_intermediate_cache2[(begin_chunk_idx + t) * topk + k] = weight_tk * (grad_y_tk @ W2_e)


            # 2. Calculate grad_w2: topk_weights * (grad_output.T @ intermediate_cache2)
            # grad_w2[expert_id] += topk_weights[t,k] * (grad_output[t, k].T @ intermediate_cache2[t*topk + k])
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
                    x_tk = curr_intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
                    weight_tk = curr_topk_weights[t, k]  # scalar

                    # grad_w2 using torch.mul: topk_weight * outer(grad_y_tk, x_tk) -> (hidden_size, intermediate_size)
                    # Convert to float32 for accumulation to avoid bfloat16 precision loss
                    grad_w2[expert_id] += (weight_tk * torch.outer(grad_y_tk, x_tk)).to(torch.float32)


            # 3. Calculate grad_topk_weights: dot(grad_output, forward_output_before_weighting)
            # Forward: output[t, k] = topk_weights[t, k] * (intermediate_cache2[t*topk+k] @ w2[expert_id].T)
            # Backward: grad_topk_weights[t, k] = dot(grad_output[t, k], intermediate_cache2[t*topk+k] @ w2[expert_id].T)
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
                    x_tk = curr_intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
                    W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)

                    # Compute forward output before weighting using torch.mul and torch.sum
                    # x @ w2.T = sum(x * w2 over intermediate_size dimension)
                    forward_output_unweighted = torch.sum(torch.mul(x_tk.unsqueeze(0), W2_e), dim=1)  # shape: (hidden_size,)

                    # grad_topk_weights: dot product using torch.mul and torch.sum
                    grad_topk_weights[begin_chunk_idx + t, k] = torch.add(
                        grad_topk_weights[begin_chunk_idx + t, k],
                        torch.sum(torch.mul(grad_y_tk, forward_output_unweighted))
                    )

        # Convert grad_w2 back to original dtype (bfloat16)
        grad_w2 = grad_w2.to(intermediate_cache2.dtype)

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


class MoeSumReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache3: torch.Tensor,
        hidden_states_shape,
    ):
        out_hidden_states = torch.empty(
            hidden_states_shape, device=intermediate_cache3.device, dtype=intermediate_cache3.dtype
        )
        moe_sum_reduce(
            intermediate_cache3,
            out_hidden_states,
            1.0,
        )
        ctx.save_for_backward(intermediate_cache3)
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache3,) = ctx.saved_tensors
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None