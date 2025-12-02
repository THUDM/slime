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
        num_tokens, _ = hidden_states.shape
        E, N, _ = w1.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        topk = topk_ids.shape[1]

        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            cur_intermediate_cache1 = intermediate_cache1[begin_chunk_idx * topk : end_chunk_idx * topk]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                None,
                cur_intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk_ids.shape[1],
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        ctx.save_for_backward(hidden_states, w1, topk_weights)

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, w1, topk_weights = ctx.saved_tensors
        return torch.zeros_like(hidden_states), torch.zeros_like(w1), torch.zeros_like(topk_weights), None


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
        return torch.zeros_like(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens //= topk
        E, _, _ = w2.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            cur_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            cur_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
            invoke_fused_moe_kernel(
                cur_intermediate_cache2,
                w2,
                None,
                cur_intermediate_cache3,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                a_use_tma=False,
                b_use_tma=False,
            )

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights)

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        intermediate_cache2, w2, topk_weights = ctx.saved_tensors

        return torch.zeros_like(intermediate_cache2), torch.zeros_like(w2), torch.zeros_like(topk_weights), None


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
        return torch.zeros_like(intermediate_cache3), None
