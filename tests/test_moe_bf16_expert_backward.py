"""Scheme-1A: moe_bf16_expert_backward vs pure BF16 torch autograd.

Compares the hand-written BF16 MoE expert dgrad/wgrad used by DeepGEMM/NVFP4
aligned forwards against a reference graph built from F.linear + SwiGLU +
post-fc2 router multiply. No FP8, no STE — this only checks implementation
self-consistency of the recompute backward.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _MoELayout

try:
    from slime.backends.megatron_utils.alignment.moe_bf16_expert_backward import (
        moe_bf16_expert_backward,
    )
except ImportError:  # older trees keep the helper inside deepgemm_moe_forward
    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
        _moe_bf16_expert_backward as moe_bf16_expert_backward,
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="moe_bf16_expert_backward CUDA path is what production uses",
)


def _grad_report(name: str, actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.float().reshape(-1)
    r = reference.float().reshape(-1)
    diff = (a - r).abs()
    denom = r.abs().clamp_min(1e-6)
    cos = F.cosine_similarity(a, r, dim=0).item()
    sign_agree = float((a.sign() == r.sign()).float().mean())
    # Treat near-zero refs as agreeing if actual is also near-zero.
    near0 = r.abs() < 1e-5
    sign_agree_nz = float((a.sign() == r.sign())[~near0].float().mean()) if (~near0).any() else 1.0
    stats = {
        "name": name,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "rel_l2": float((a - r).norm() / r.norm().clamp_min(1e-6)),
        "cosine": cos,
        "sign_agree": sign_agree,
        "sign_agree_nz": sign_agree_nz,
        "ref_rms": float(r.square().mean().sqrt()),
        "act_rms": float(a.square().mean().sqrt()),
    }
    print(
        f"[moe-bwd] {name}: cosine={stats['cosine']:.8f} rel_l2={stats['rel_l2']:.4e} "
        f"max_abs={stats['max_abs']:.4e} sign_nz={stats['sign_agree_nz']:.4f} "
        f"rms_act/ref={stats['act_rms']:.4e}/{stats['ref_rms']:.4e}",
        flush=True,
    )
    return stats


def _bf16_moe_forward(
    hidden: torch.Tensor,
    probs: torch.Tensor,
    fc1_weights: tuple[torch.Tensor, ...],
    fc2_weights: tuple[torch.Tensor, ...],
    counts: tuple[int, ...],
) -> torch.Tensor:
    """Mirror moe_bf16_expert_backward's BF16 recompute graph (post-fc2 probs)."""
    chunks_h = torch.split(hidden, list(counts), dim=0)
    chunks_p = torch.split(probs.reshape(-1, 1), list(counts), dim=0)
    outs = []
    for h, p, w1, w2 in zip(chunks_h, chunks_p, fc1_weights, fc2_weights, strict=True):
        if h.numel() == 0:
            continue
        gate_up = F.linear(h, w1)
        gate, up = gate_up.chunk(2, dim=-1)
        down_in = (F.silu(gate.float()) * up.float()).to(dtype=h.dtype)
        down = F.linear(down_in, w2)
        outs.append((down.float() * p.float()).to(dtype=h.dtype))
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize("num_experts,counts", [(1, (32,)), (2, (20, 12)), (3, (8, 0, 16))])
def test_moe_bf16_expert_backward_matches_torch_autograd(monkeypatch, num_experts, counts):
    # Force the per-expert chunked path so we exercise the same loops as the
    # fallback (grouped path is covered separately in test_deepgemm_moe_forward).
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD", "0")
    from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as dmf

    monkeypatch.setattr(dmf, "_BACKWARD_CHUNK_ROWS", 8)

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size, ffn = 128, 256
    layout = _MoELayout(num_local_experts=num_experts, hidden_size=hidden_size, ffn_hidden_size=ffn)
    t = sum(counts)

    hidden = (torch.randn(t, hidden_size, device=device, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    probs = (torch.rand(t, device=device, dtype=torch.float32) * 1.5 + 0.1).to(torch.bfloat16)
    fc1 = tuple(
        (torch.randn(2 * ffn, hidden_size, device=device, dtype=torch.float32) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_(True)
        for _ in range(num_experts)
    )
    fc2 = tuple(
        (torch.randn(hidden_size, ffn, device=device, dtype=torch.float32) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_(True)
        for _ in range(num_experts)
    )
    grad_output = (torch.randn(t, hidden_size, device=device, dtype=torch.float32) * 0.2).to(torch.bfloat16)

    # --- reference: pure BF16 autograd ---
    h_ref = hidden.detach().clone().requires_grad_(True)
    p_ref = probs.detach().clone().requires_grad_(True)
    w1_ref = tuple(w.detach().clone().requires_grad_(True) for w in fc1)
    w2_ref = tuple(w.detach().clone().requires_grad_(True) for w in fc2)
    out_ref = _bf16_moe_forward(h_ref, p_ref, w1_ref, w2_ref, counts)
    out_ref.backward(grad_output)

    # --- implementation: hand-written recompute backward ---
    g_h, g_p, g_fc1, g_fc2 = moe_bf16_expert_backward(
        hidden_states=hidden.detach(),
        permuted_probs=probs.detach(),
        grad_output=grad_output,
        fc1_weights=tuple(w.detach() for w in fc1),
        fc2_weights=tuple(w.detach() for w in fc2),
        counts=counts,
        layout=layout,
        module_name="test.experts",
        needs_hidden=True,
        needs_probs=True,
        needs_fc1_weights=(True,) * num_experts,
        needs_fc2_weights=(True,) * num_experts,
        defer_router_probabilities=False,
        reuse_expert_input_for_grad=False,
        grad_workspace=None,
    )

    assert g_h is not None and g_p is not None
    reports = [
        _grad_report("d_hidden", g_h, h_ref.grad),
        _grad_report("d_probs", g_p, p_ref.grad),
    ]
    for i, count in enumerate(counts):
        assert g_fc1[i] is not None and g_fc2[i] is not None
        if count == 0:
            assert torch.count_nonzero(g_fc1[i]) == 0
            assert torch.count_nonzero(g_fc2[i]) == 0
            assert w1_ref[i].grad is None and w2_ref[i].grad is None
            continue
        reports.append(_grad_report(f"d_fc1[{i}]", g_fc1[i], w1_ref[i].grad))
        reports.append(_grad_report(f"d_fc2[{i}]", g_fc2[i], w2_ref[i].grad))

    for stats in reports:
        assert stats["cosine"] > 0.999, stats
        assert stats["rel_l2"] < 5e-2, stats
        assert stats["sign_agree_nz"] > 0.99, stats


def test_moe_bf16_expert_backward_grouped_path_matches_autograd(monkeypatch):
    """Same check with the production grouped BF16 backward enabled."""
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_GROUPED_BF16_BACKWARD", "1")
    monkeypatch.delenv("SLIME_DEEPGEMM_MOE_BF16_BACKWARD_MAX_PADDED_BYTES", raising=False)

    torch.manual_seed(1)
    device = torch.device("cuda")
    num_experts, counts = 2, (64, 48)
    hidden_size, ffn = 128, 128
    layout = _MoELayout(num_local_experts=num_experts, hidden_size=hidden_size, ffn_hidden_size=ffn)
    t = sum(counts)

    hidden = (torch.randn(t, hidden_size, device=device, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    probs = (torch.rand(t, device=device, dtype=torch.float32) * 1.5 + 0.1).to(torch.bfloat16)
    fc1 = tuple(
        (torch.randn(*layout.fc1_weight_shape, device=device, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        for _ in range(num_experts)
    )
    fc2 = tuple(
        (torch.randn(*layout.fc2_weight_shape, device=device, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        for _ in range(num_experts)
    )
    grad_output = (torch.randn(t, hidden_size, device=device, dtype=torch.float32) * 0.2).to(torch.bfloat16)

    from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as dmf

    assert dmf._use_grouped_bf16_backward(
        hidden, counts, (True,) * num_experts, (True,) * num_experts
    ), "expected grouped BF16 backward to be selected"

    h_ref = hidden.detach().clone().requires_grad_(True)
    p_ref = probs.detach().clone().requires_grad_(True)
    w1_ref = tuple(w.detach().clone().requires_grad_(True) for w in fc1)
    w2_ref = tuple(w.detach().clone().requires_grad_(True) for w in fc2)
    _bf16_moe_forward(h_ref, p_ref, w1_ref, w2_ref, counts).backward(grad_output)

    g_h, g_p, g_fc1, g_fc2 = moe_bf16_expert_backward(
        hidden_states=hidden.detach(),
        permuted_probs=probs.detach(),
        grad_output=grad_output,
        fc1_weights=fc1,
        fc2_weights=fc2,
        counts=counts,
        layout=layout,
        module_name="test.experts",
        needs_hidden=True,
        needs_probs=True,
        needs_fc1_weights=(True,) * num_experts,
        needs_fc2_weights=(True,) * num_experts,
        defer_router_probabilities=False,
        reuse_expert_input_for_grad=False,
        grad_workspace=None,
    )

    for name, actual, reference in [
        ("d_hidden", g_h, h_ref.grad),
        ("d_probs", g_p, p_ref.grad),
        ("d_fc1[0]", g_fc1[0], w1_ref[0].grad),
        ("d_fc1[1]", g_fc1[1], w1_ref[1].grad),
        ("d_fc2[0]", g_fc2[0], w2_ref[0].grad),
        ("d_fc2[1]", g_fc2[1], w2_ref[1].grad),
    ]:
        stats = _grad_report(name, actual, reference)
        assert stats["cosine"] > 0.999, stats
        assert stats["rel_l2"] < 5e-2, stats


def test_moe_ste_fp8_path_vs_bf16_autograd_report(monkeypatch):
    """Scheme-1B: real DeepGEMM FP8 STE forward + BF16 recompute bwd vs BF16 autograd.

    Expectation (by design): direction roughly matches BF16, but magnitudes may
    shrink (training dulling). Catastrophic cosine / sign flips would be a bug.
    """
    import importlib.util
    from pathlib import Path

    from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as dmf

    helper_path = Path(__file__).resolve().parent / "test_deepgemm_moe_forward.py"
    spec = importlib.util.spec_from_file_location("_deepgemm_moe_fwd_helpers", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helpers)

    helpers._require_cuda_deepgemm(monkeypatch)
    monkeypatch.setattr(dmf.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dmf.parallel_state, "get_expert_tensor_parallel_world_size", lambda: 1)

    torch.cuda.set_device(torch.cuda.current_device())
    torch.manual_seed(457)

    module = helpers.TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    reference_module = (
        helpers.TEGroupedMLP(num_experts=3, hidden_size=128, ffn_hidden_size=128).cuda().to(torch.bfloat16)
    )
    reference_module.load_state_dict(module.state_dict())

    tokens_per_expert = torch.tensor([129, 17, 260], device="cuda", dtype=torch.int32)
    num_tokens = int(tokens_per_expert.sum().item())
    hidden_states = (
        (torch.randn(num_tokens, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16).requires_grad_()
    )
    permuted_probs = (torch.rand(num_tokens, device="cuda", dtype=torch.float32) * 1.5 + 0.1).requires_grad_()
    grad_output = (torch.randn(num_tokens, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)

    # --- BF16 autograd reference ---
    reference_hidden = hidden_states.detach().clone().requires_grad_()
    reference_probs = permuted_probs.detach().clone().requires_grad_()
    reference_output = helpers._post_fc2_probability_reference(
        reference_module, reference_hidden, tokens_per_expert, reference_probs
    )
    reference_output.backward(grad_output)

    # --- STE path: FP8 DeepGEMM forward + BF16 recompute backward ---
    assert dmf._wrap_te_grouped_mlp(module, "decoder.layers.3.mlp.experts")
    output, _ = module(hidden_states, tokens_per_expert, permuted_probs)
    output.backward(grad_output)

    fwd_diff = (output.float() - reference_output.detach().float()).abs()
    print(
        f"[moe-ste] forward_vs_bf16: mean_abs={fwd_diff.mean().item():.4e} "
        f"max_abs={fwd_diff.max().item():.4e}",
        flush=True,
    )

    checks = {
        "d_hidden": (hidden_states.grad, reference_hidden.grad),
        "d_probs": (permuted_probs.grad, reference_probs.grad),
    }
    for (name, parameter), (_, reference_parameter) in zip(
        module.named_parameters(),
        reference_module.named_parameters(),
        strict=True,
    ):
        checks[f"d_{name}"] = (parameter.grad, reference_parameter.grad)

    reports = []
    for name, (actual, reference) in checks.items():
        assert actual is not None and reference is not None, name
        stats = _grad_report(name, actual, reference)
        stats["rms_ratio"] = stats["act_rms"] / max(stats["ref_rms"], 1e-12)
        print(f"[moe-ste] {name} rms_ratio(act/ref)={stats['rms_ratio']:.4f}", flush=True)
        reports.append(stats)

    # Soft gates: STE should stay directionally useful; not bit-exact.
    for stats in reports:
        assert stats["cosine"] > 0.90, stats
        assert stats["sign_agree_nz"] > 0.90, stats
