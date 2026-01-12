"""Qwen3-MoE utility functions: routing, and weight preparation."""

import torch
import torch.nn.functional as F


def qwen3_moe_routing(
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-style routing: softmax(all) → topk → renormalize."""
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(dtype)
    return routing_weights, selected_experts


def stack_expert_weights_for_sonicmoe(experts):
    # First, stack gate and up weights for each expert: [gate, up] -> (2*I, H)
    per_expert_w13 = [
        torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0)  # (2*I, H) where I=intermediate_dim
        for e in experts
    ]
    w13_base = torch.stack(per_expert_w13, dim=0)          # (E, 2*I, H) contiguous
    
    # Interleave gate/up: [E, 2*I, H] concat -> [E, 2*I, H] interleaved
    # Format: [gate_row0, up_row0, gate_row1, up_row1, ...]
    I = w13_base.shape[1] // 2
    gate_part = w13_base[:, :I, :]  # (E, I, H) - gate weights
    up_part = w13_base[:, I:, :]    # (E, I, H) - up weights
    # Stack along new dimension and reshape to interleave
    w13_interleaved = torch.stack([gate_part, up_part], dim=2).reshape(w13_base.shape[0], 2 * I, w13_base.shape[2])  # (E, 2*I, H) interleaved
    
    w13_weight = w13_interleaved.permute(1, 2, 0)          # (2*I, H, E) view
    assert w13_weight.stride(1) == 1, w13_weight.stride()

    per_expert_w2 = [e.down_proj.weight for e in experts]  # each (H, I)

    w2_base = torch.stack(per_expert_w2, dim=0)            # (E, H, I) contiguous

    w2_weight = w2_base.permute(1, 2, 0)                   # (H, I, E) view, NOT contiguous
    assert w2_weight.stride(1) == 1, w2_weight.stride()

    return w13_weight, w2_weight


def prepare_sonicmoe_routing_inputs(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten routing inputs for SonicMoE kernel."""
    T, K = selected_experts.shape
    selected_E = selected_experts.reshape(-1).to(torch.int32)
    router_scores_selected = routing_weights.reshape(-1)
    sorted_selected_T = torch.arange(T, device=device, dtype=torch.int32).repeat_interleave(K)
    return selected_E, router_scores_selected, sorted_selected_T

