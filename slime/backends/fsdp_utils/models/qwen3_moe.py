import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from slime.backends.fsdp_utils.kernels.fused_experts import (
    DownProjFunction,
    GateUpProjFunction,
    MoeSumReduceFunction,
    SiluAndMulFunction,
)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    intermediate_cache1 = GateUpProjFunction.apply(
        hidden_states,
        w1,
        topk_weights,
        topk_ids,
    )
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunction.apply(
        intermediate_cache2,
        w2,
        topk_weights,
        topk_ids,
    )
    output_hidden_states = MoeSumReduceFunction.apply(
        intermediate_cache3,
        hidden_states.shape,
    )
    return output_hidden_states


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

        final_hidden_states = fused_experts_impl(
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
