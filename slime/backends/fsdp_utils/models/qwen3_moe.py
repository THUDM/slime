import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce,
    silu_and_mul,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    inplace: bool = True,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    no_combine: bool = False,
    routed_scaling_factor: float | None = None,
    gemm1_alpha: float | None = None,
    gemm1_limit: float | None = None,
    filter_expert: bool = True,
):
    padded_size = 0

    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    # default deterministic config
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    down_config = config

    down_moe_use_tma = False
    topk = topk_ids.shape[1]
    total_tokens = M * topk
    cache = torch.empty(
        total_tokens * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]),
    )

    compute_type = tl.bfloat16

    out_hidden_states = hidden_states

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk.
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        total_tokens = tokens_in_chunk * topk
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            b1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            c_sorted=down_moe_use_tma,
            filter_expert=filter_expert,
        )
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
            pass  # we write directly into out_hidden_states
        elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
            ).squeeze(dim=1)
        else:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
                routed_scaling_factor,
            )

    return out_hidden_states


class StandardDispatcher:
    def __init__(self, num_experts: int, num_local_experts: int):
        self.moe_ep_size = 1
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_rank = 0
        self.local_expert_mapping = None

        if self.moe_ep_size > 1:
            self.local_expert_mapping = torch.full((self.num_experts,), -1, dtype=torch.int32, device="cuda")
            self.local_expert_mapping[
                self.moe_ep_rank * self.num_local_experts : (self.moe_ep_rank + 1) * self.num_local_experts
            ] = torch.arange(0, self.num_local_experts, dtype=torch.int32, device="cuda")

    def dispatch(self, topk_ids) -> torch.Tensor:
        if self.local_expert_mapping is not None:
            return self.local_expert_mapping[topk_ids]
        return topk_ids


class FusedMoe(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
        )
        ctx.save_for_backward(hidden_states, w1, w2, topk_weights, topk_ids)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_states, w1, w2, topk_weights, topk_ids = ctx.saved_tensors

        # Prepare grad_output
        grad_output = grad_output.contiguous()

        # TODO: write the correct backward
        # Call fused_moe_triton backward
        (
            grad_hidden_states,
            grad_w1,
            grad_w2,
            grad_topk_weights,
        ) = (
            torch.zeros_like(hidden_states),
            torch.zeros_like(w1),
            torch.zeros_like(w2),
            torch.zeros_like(topk_weights),
        )

        return grad_hidden_states, grad_w1, grad_w2, grad_topk_weights, None


class Qwen3MoeSparseMoeBlock(nn.Module):
    dispatcher = None
    runner = None

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        if Qwen3MoeSparseMoeBlock.dispatcher is None:
            Qwen3MoeSparseMoeBlock.dispatcher = StandardDispatcher(
                num_experts=config.num_experts, num_local_experts=config.num_experts
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts = Qwen3MoeSparseMoeBlock.dispatcher.dispatch(selected_experts)

        w13_weight = torch.stack(
            [torch.cat([layer.gate_proj.weight, layer.up_proj.weight], dim=0) for layer in self.experts]
        )
        w2_weight = torch.stack([layer.down_proj.weight for layer in self.experts], dim=0)

        final_hidden_states = FusedMoe.apply(
            hidden_states.to(torch.bfloat16),
            w13_weight,
            w2_weight,
            routing_weights,
            selected_experts,
        )

        return final_hidden_states, router_logits


def apply_true_on_policy_patch_for_qwen3_moe():
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock
