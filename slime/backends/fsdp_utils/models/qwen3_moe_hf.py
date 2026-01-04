import logging
import torch
import torch.cuda.nvtx as nvtx
from sonicmoe.functional import moe_general_routing_inputs
from sonicmoe.enums import ActivationType

from .qwen3_moe_utils import (
    qwen3_moe_routing,
    stack_expert_weights_for_sonicmoe,
    prepare_sonicmoe_routing_inputs,
)

log = logging.getLogger(__name__)


def apply_fsdp_moe_patch(args=None):

    from transformers.models.qwen3_moe import modeling_qwen3_moe

    def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        nvtx.range_push("MoE_forward")
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        nvtx.range_push("MoE_routing")
        router_logits = self.gate(hidden_states)

        # Qwen3-style routing: softmax → topk → renormalize
        routing_weights, selected_experts = qwen3_moe_routing(
            router_logits, self.top_k, self.norm_topk_prob, hidden_states.dtype
        )
        nvtx.range_pop()  # MoE_routing

        moe_impl = getattr(args, "fsdp_moe_impl", "torch") if args is not None else "torch"

        if moe_impl == "sonicmoe":
            log.info("Using SonicMoE MoE implementation")
            nvtx.range_push("MoE_sonicmoe_experts")
            # SonicMoE path: keep HF routing semantics, only swap the experts implementation.
            
            # Prepare expert weights in SonicMoE format
            nvtx.range_push("MoE_prepare_weights")
            w13_weight, w2_weight = stack_expert_weights_for_sonicmoe(self.experts)
            nvtx.range_pop()  # MoE_prepare_weights
            
            # Prepare routing inputs
            nvtx.range_push("MoE_prepare_routing_inputs")
            selected_E, router_scores_selected, sorted_selected_T = prepare_sonicmoe_routing_inputs(
                routing_weights, selected_experts, hidden_states.device
            )
            nvtx.range_pop()  # MoE_prepare_routing_inputs

            # Call SonicMoE kernel with detailed step-by-step comparison
            nvtx.range_push("MoE_sonicmoe_kernel")
            stream_id = int(torch.cuda.current_stream().cuda_stream)
            is_inference_mode_enabled = (not torch.is_grad_enabled()) or torch.is_inference_mode_enabled()

            final_hidden_states, _ = moe_general_routing_inputs(
                hidden_states,
                router_scores_selected,
                sorted_selected_T,
                selected_E,
                w13_weight,
                None,  # b1
                w2_weight,
                None,  # b2
                self.num_experts,
                stream_id,
                ActivationType.SWIGLU,
                is_inference_mode_enabled,
            )
        else:
            log.info("Using torch MoE implementation")
            nvtx.range_push("MoE_torch_experts")
            # Reference path: per-expert loop (correct but slower).
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all experts
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                if top_x.numel() > 0:
                    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                    current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                    final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                else:
                    # Force experts to participate in computation graph.
                    dummy_output = expert_layer(hidden_states[:1]) * 0.0
                    final_hidden_states[:1] = final_hidden_states[:1] + dummy_output
            nvtx.range_pop()  # MoE_torch_experts

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        nvtx.range_pop()  # MoE_forward
        return final_hidden_states, router_logits

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = _forward
