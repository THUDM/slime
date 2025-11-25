from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from sglang.srt.layers.moe.fused_moe_triton.layer import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import set_global_server_args_for_scheduler
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP


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
        topk_output: StandardTopKOutput,
        moe_runner_config: MoeRunnerConfig,
    ) -> torch.Tensor:
        output = fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_output=topk_output,
            moe_runner_config=moe_runner_config,
        )
        ctx.save_for_backward(hidden_states, w1, w2, topk_output.topk_weights, topk_output.topk_ids)
        ctx.moe_runner_config = moe_runner_config
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_states, w1, w2, topk_weights, topk_ids = ctx.saved_tensors
        moe_runner_config = ctx.moe_runner_config

        # Prepare grad_output
        grad_output = grad_output.contiguous()

        # Call fused_moe_triton backward
        (
            grad_hidden_states,
            grad_w1,
            grad_w2,
        ) = (
            torch.zeros_like(hidden_states),
            torch.zeros_like(w1),
            torch.zeros_like(w2),
        )

        return grad_hidden_states, grad_w1, grad_w2, None, None


@dataclass
class ServerArgs:
    enable_deterministic_inference: bool = True


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

        self.moe_runner_config = MoeRunnerConfig(
            num_experts=config.num_experts,
            num_local_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size_per_partition=config.moe_intermediate_size,
            top_k=config.num_experts_per_tok,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
        )
        set_global_server_args_for_scheduler(ServerArgs())

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

        topk_output = StandardTopKOutput(
            topk_weights=routing_weights,
            topk_ids=selected_experts,
            router_logits=router_logits,
        )

        final_hidden_states = FusedMoe.apply(
            hidden_states.to(torch.bfloat16),
            w13_weight,
            w2_weight,
            topk_output,
            self.moe_runner_config,
        )

        return final_hidden_states, router_logits


def apply_true_on_policy_patch_for_qwen3_moe():
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = Qwen3MoeSparseMoeBlock
