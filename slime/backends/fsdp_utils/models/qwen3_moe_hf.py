import torch
import torch.nn.functional as F


def apply_fsdp_moe_patch(args=None):

    from transformers.models.qwen3_moe import modeling_qwen3_moe

    def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        moe_impl = getattr(args, "fsdp_moe_impl", "torch") if args is not None else "torch"

        if moe_impl == "sonicmoe":
            # SonicMoE path: keep HF routing semantics, only swap the experts implementation.
            from ..kernels.sonic_experts import sonic_experts_impl  # pyright: ignore[reportMissingImports]

            # Build expert weights in HF layout:
            # - w1: (E, 2I, H) where 2I = [gate; up]
            # - w2: (E, H, I)
            w13_weight = torch.stack(
                [torch.cat([layer.gate_proj.weight, layer.up_proj.weight], dim=0) for layer in self.experts]
            ).contiguous()
            w2_weight = torch.stack([layer.down_proj.weight for layer in self.experts], dim=0).contiguous()

            final_hidden_states = sonic_experts_impl(
                hidden_states,
                w13_weight,
                w2_weight,
                routing_weights,
                selected_experts,
            )
        else:
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

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = _forward
