"""Unit tests for ``Gemma4Router`` and the ``Gemma4MoELayer.route`` adapter.

These are pure-Python/CPU tests that exercise the routing arithmetic without
going through Megatron's MoE dispatch infrastructure.
"""

from types import SimpleNamespace

import pytest
import torch

from slime_plugins.models.gemma4 import Gemma4MoELayer, Gemma4Router


def _make_router_config(hidden_size=16, num_experts=8, top_k=2, eps=1e-6):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_moe_experts=num_experts,
        moe_router_topk=top_k,
        layernorm_epsilon=eps,
    )


def test_router_outputs_have_correct_shapes():
    torch.manual_seed(0)
    cfg = _make_router_config(num_experts=8, top_k=2)
    router = Gemma4Router(cfg)
    h = torch.randn(5, cfg.hidden_size)
    weights, idx = router(h)
    assert weights.shape == (5, cfg.moe_router_topk)
    assert idx.shape == (5, cfg.moe_router_topk)
    assert idx.min() >= 0 and idx.max() < cfg.num_moe_experts


def test_router_weights_sum_to_one_before_per_expert_scale():
    """With per_expert_scale all ones, top-k weights must sum to 1.0 per
    token — the router normalises internally."""
    torch.manual_seed(1)
    cfg = _make_router_config(num_experts=8, top_k=3)
    router = Gemma4Router(cfg)
    # Leave per_expert_scale as its default (all ones).
    h = torch.randn(6, cfg.hidden_size)
    weights, _idx = router(h)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_router_per_expert_scale_multiplies_output():
    """Setting ``per_expert_scale`` to a constant c should scale weights by c."""
    torch.manual_seed(2)
    cfg = _make_router_config(num_experts=4, top_k=2)
    router = Gemma4Router(cfg)
    # Fix per_expert_scale = 3.0 for all experts.
    with torch.no_grad():
        router.per_expert_scale.fill_(3.0)
    h = torch.randn(4, cfg.hidden_size)
    weights, _idx = router(h)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.full_like(sums, 3.0), atol=1e-6)


def _make_moe_route_stub():
    """Build a Gemma4MoELayer stub that only exposes the ``route`` method.

    We bypass ``__init__`` (which builds a full Megatron MoE dispatcher)
    and populate only the attributes read by ``route``: ``self.router``
    and ``self.config.num_moe_experts``."""
    obj = object.__new__(Gemma4MoELayer)
    torch.nn.Module.__init__(obj)
    cfg = _make_router_config(num_experts=6, top_k=2)
    obj.router = Gemma4Router(cfg)
    obj.config = cfg
    return obj, cfg


def test_moe_route_packs_topk_into_dense_probs_and_routing_map():
    """route() must produce (probs [T, E], routing_map [T, E]) from the
    compact (top_k_weights [T, K], top_k_index [T, K]) router output."""
    torch.manual_seed(3)
    obj, cfg = _make_moe_route_stub()
    h = torch.randn(4, cfg.hidden_size)
    probs, routing_map = obj.route(h)

    T, E = 4, cfg.num_moe_experts
    assert probs.shape == (T, E)
    assert routing_map.shape == (T, E)
    assert routing_map.dtype == torch.bool

    # Each row has exactly top_k nonzero entries; routing_map matches.
    assert (probs != 0).sum(dim=-1).eq(cfg.moe_router_topk).all()
    assert routing_map.eq(probs != 0).all()

    # Probs sum to the same total per row as the compact top-k weights (with
    # default per_expert_scale=1, that's 1.0 per row).
    expected_sums = probs.sum(dim=-1)
    assert torch.allclose(expected_sums, torch.ones(T), atol=1e-6)


def test_moe_route_accepts_3d_input_by_flattening():
    """route() must flatten [S, B, H] (or any prefix dims) to [T, H] before
    routing, so it works both in thd (2D) and [seq, batch, hidden] (3D)
    layouts. Only the output's leading dimension is exercised here."""
    torch.manual_seed(4)
    obj, cfg = _make_moe_route_stub()
    h = torch.randn(3, 2, cfg.hidden_size)  # [S, B, H]
    probs, routing_map = obj.route(h)
    # T = 3 * 2 = 6
    assert probs.shape == (6, cfg.num_moe_experts)
    assert routing_map.shape == (6, cfg.num_moe_experts)


def _hf_reference_router(h, proj_w, scale, per_expert_scale, top_k, eps=1e-6):
    """Reference implementation of the HF Gemma4 router equation:

        h_norm  = rmsnorm_noscale(h)              # no-learnable-scale RMSNorm
        h_norm2 = h_norm * scale / sqrt(H)        # per-hidden learnable scale
        logits  = proj_w @ h_norm2                # [T, E]
        probs   = softmax(logits)
        top_w, top_i = topk(probs, k=top_k)
        top_w   = top_w / sum(top_w)              # renormalize
        top_w   = top_w * per_expert_scale[top_i] # per-expert scale multiplier

    This closes the loop on what Gemma4Router computes: exercises every step
    (RMSNorm without scale, per-hidden scale, proj, softmax, topk, renormalise,
    per-expert scale) and guards against silent reordering of those ops in
    future refactors.
    """
    # RMSNorm (no scale), float-precision to match Gemma4Router.VNorm.
    h = h.float()
    norm = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    h_norm2 = norm * scale * (h.shape[-1] ** -0.5)
    logits = torch.nn.functional.linear(h_norm2, proj_w)
    probs = torch.softmax(logits, dim=-1)
    top_w, top_i = torch.topk(probs, k=top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w * per_expert_scale[top_i]
    return top_w, top_i


def test_router_matches_hf_reference_equation():
    """Gemma4Router.forward must produce the exact HF router output up to
    kernel noise. This covers the full router equation."""
    torch.manual_seed(42)
    cfg = _make_router_config(hidden_size=32, num_experts=8, top_k=2)
    router = Gemma4Router(cfg)
    # Use realistic, non-trivial weights.
    with torch.no_grad():
        router.scale.copy_(torch.randn(cfg.hidden_size) * 0.1 + 1.0)
        router.per_expert_scale.copy_(torch.randn(cfg.num_moe_experts) * 0.2 + 1.0)

    h = torch.randn(5, cfg.hidden_size)
    w, idx = router(h)
    w_ref, idx_ref = _hf_reference_router(
        h, router.proj.weight, router.scale, router.per_expert_scale,
        cfg.moe_router_topk, eps=cfg.layernorm_epsilon,
    )

    # topk indices must match exactly.
    assert torch.equal(idx, idx_ref), (
        f"router top-k indices diverge: ours={idx}, ref={idx_ref}"
    )
    assert torch.allclose(w.float(), w_ref, atol=1e-5), (
        "router top-k weights diverge from HF reference"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
