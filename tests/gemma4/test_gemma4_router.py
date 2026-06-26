from types import SimpleNamespace

import torch

try:
    from slime_plugins.models.gemma4 import Gemma4MoELayer, Gemma4Router
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not (missing == "megatron" or missing.startswith("megatron.") or missing == "mbridge"):
        raise
    from tests.gemma4._standalone_imports import load_gemma4_model_module

    _gemma4 = load_gemma4_model_module()
    Gemma4MoELayer = _gemma4.Gemma4MoELayer
    Gemma4Router = _gemma4.Gemma4Router


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
    torch.manual_seed(1)
    cfg = _make_router_config(num_experts=8, top_k=3)
    router = Gemma4Router(cfg)
    h = torch.randn(6, cfg.hidden_size)
    weights, _idx = router(h)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_router_per_expert_scale_multiplies_output():
    torch.manual_seed(2)
    cfg = _make_router_config(num_experts=4, top_k=2)
    router = Gemma4Router(cfg)
    with torch.no_grad():
        router.per_expert_scale.fill_(3.0)
    h = torch.randn(4, cfg.hidden_size)
    weights, _idx = router(h)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.full_like(sums, 3.0), atol=1e-6)


def _make_moe_route_stub():
    obj = object.__new__(Gemma4MoELayer)
    torch.nn.Module.__init__(obj)
    cfg = _make_router_config(num_experts=6, top_k=2)
    obj.router = Gemma4Router(cfg)
    obj.config = cfg
    return obj, cfg


def test_moe_route_packs_topk_into_dense_probs_and_routing_map():
    torch.manual_seed(3)
    obj, cfg = _make_moe_route_stub()
    h = torch.randn(4, cfg.hidden_size)
    probs, routing_map = obj.route(h)

    T, E = 4, cfg.num_moe_experts
    assert probs.shape == (T, E)
    assert routing_map.shape == (T, E)
    assert routing_map.dtype == torch.bool

    assert (probs != 0).sum(dim=-1).eq(cfg.moe_router_topk).all()
    assert routing_map.eq(probs != 0).all()

    expected_sums = probs.sum(dim=-1)
    assert torch.allclose(expected_sums, torch.ones(T), atol=1e-6)


def test_moe_route_accepts_3d_input_by_flattening():
    torch.manual_seed(4)
    obj, cfg = _make_moe_route_stub()
    h = torch.randn(3, 2, cfg.hidden_size)
    probs, routing_map = obj.route(h)
    assert probs.shape == (6, cfg.num_moe_experts)
    assert routing_map.shape == (6, cfg.num_moe_experts)


def test_moe_forward_uses_current_megatron_preprocess_contract():
    obj = object.__new__(Gemma4MoELayer)
    torch.nn.Module.__init__(obj)
    obj.config = SimpleNamespace(sequence_parallel=True)
    obj.attn_tp_group = SimpleNamespace(size=lambda: 1)

    calls = []

    def norm(hidden_states):
        calls.append(("norm", hidden_states))
        return "experts_in"

    def shared_experts_compute(experts_in):
        calls.append(("shared", experts_in))
        return None

    def route(router_in):
        calls.append(("route", router_in))
        return "probs", "routing_map"

    def preprocess(experts_in, probs, routing_map):
        calls.append(("preprocess", experts_in, probs, routing_map))
        return "preprocessed", "preprocessed_probs"

    def dispatch(experts_in, probs):
        calls.append(("dispatch", experts_in, probs))
        return "dispatched", "dispatched_probs"

    def routed_experts_compute(dispatched_input, probs):
        calls.append(("experts", dispatched_input, probs))
        return "expert_output", None

    def combine(output):
        calls.append(("combine", output))
        return "combined"

    def postprocess(output, shared_expert_output):
        calls.append(("postprocess", output, shared_expert_output))
        return "postprocessed"

    obj.pre_feedforward_layernorm_2 = norm
    obj.shared_experts_compute = shared_experts_compute
    obj.route = route
    obj.preprocess = preprocess
    obj.dispatch = dispatch
    obj.routed_experts_compute = routed_experts_compute
    obj.combine = combine
    obj.postprocess = postprocess

    output, bias = obj.forward("hidden", router_input="router")

    assert output == "postprocessed"
    assert bias is None
    assert calls == [
        ("norm", "hidden"),
        ("shared", "experts_in"),
        ("route", "router"),
        ("preprocess", "experts_in", "probs", "routing_map"),
        ("dispatch", "preprocessed", "preprocessed_probs"),
        ("experts", "dispatched", "dispatched_probs"),
        ("combine", "expert_output"),
        ("postprocess", "combined", None),
    ]


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
    torch.manual_seed(42)
    cfg = _make_router_config(hidden_size=32, num_experts=8, top_k=2)
    router = Gemma4Router(cfg)
    with torch.no_grad():
        router.scale.copy_(torch.randn(cfg.hidden_size) * 0.1 + 1.0)
        router.per_expert_scale.copy_(torch.randn(cfg.num_moe_experts) * 0.2 + 1.0)

    h = torch.randn(5, cfg.hidden_size)
    w, idx = router(h)
    w_ref, idx_ref = _hf_reference_router(
        h,
        router.proj.weight,
        router.scale,
        router.per_expert_scale,
        cfg.moe_router_topk,
        eps=cfg.layernorm_epsilon,
    )

    assert torch.equal(idx, idx_ref), f"router top-k indices diverge: ours={idx}, ref={idx_ref}"
    assert torch.allclose(w.float(), w_ref, atol=1e-5), "router top-k weights diverge from HF reference"
