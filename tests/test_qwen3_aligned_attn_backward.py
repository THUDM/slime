"""Qwen3 aligned attention: SGL-kernel forward must still produce QKV grads."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused QK-RoPE / FA4 backward tests need CUDA",
)


def _positions(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    parts = [torch.arange(n, device=device, dtype=torch.int32) for n in seq_lens]
    return torch.cat(parts, dim=0)


def test_fused_qk_norm_rope_backward_reaches_qkv_and_weights():
    from slime_plugins.models.qwen3_attn_ops import fused_qk_norm_rope_with_grad

    torch.manual_seed(0)
    device = torch.device("cuda")
    t, nq, nkv, hd = 8, 4, 1, 64
    query = torch.randn(t, nq, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    key = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    value = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    q_weight = torch.nn.Parameter(torch.ones(hd, dtype=torch.bfloat16, device=device))
    k_weight = torch.nn.Parameter(torch.ones(hd, dtype=torch.bfloat16, device=device))
    positions = _positions([8], device)

    q_out, k_out, v_out = fused_qk_norm_rope_with_grad(
        query,
        key,
        value,
        q_weight,
        k_weight,
        positions,
        eps=1e-6,
        rotary_base=1_000_000.0,
    )
    (q_out.float().square().sum() + k_out.float().square().sum() + v_out.float().square().sum()).backward()

    assert query.grad is not None and query.grad.abs().sum() > 0
    assert key.grad is not None and key.grad.abs().sum() > 0
    assert value.grad is not None and value.grad.abs().sum() > 0
    assert q_weight.grad is not None and q_weight.grad.abs().sum() > 0
    assert k_weight.grad is not None and k_weight.grad.abs().sum() > 0


def test_fused_qk_norm_rope_grad_matches_rms_then_rope():
    from slime_plugins.models.qwen3_attn_ops import (
        fused_qk_norm_rope_with_grad,
        neox_rope,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    t, nq, nkv, hd = 6, 2, 1, 64
    query = torch.randn(t, nq, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    key = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    value = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    q_weight = torch.nn.Parameter(torch.rand(hd, dtype=torch.bfloat16, device=device) + 0.5)
    k_weight = torch.nn.Parameter(torch.rand(hd, dtype=torch.bfloat16, device=device) + 0.5)
    positions = _positions([3, 3], device)
    eps = 1e-6
    rotary_base = 1_000_000.0

    q_out, k_out, v_out = fused_qk_norm_rope_with_grad(
        query, key, value, q_weight, k_weight, positions, eps=eps, rotary_base=rotary_base
    )
    loss = q_out.float().sum() + k_out.float().sum() + v_out.float().sum()
    loss.backward()

    q2 = query.detach().clone().requires_grad_(True)
    k2 = key.detach().clone().requires_grad_(True)
    v2 = value.detach().clone().requires_grad_(True)
    qw2 = torch.nn.Parameter(q_weight.detach().clone())
    kw2 = torch.nn.Parameter(k_weight.detach().clone())
    qn = F.rms_norm(q2.float(), (hd,), qw2.float(), eps).to(torch.bfloat16)
    kn = F.rms_norm(k2.float(), (hd,), kw2.float(), eps).to(torch.bfloat16)
    qn = neox_rope(qn, positions, rotary_base)
    kn = neox_rope(kn, positions, rotary_base)
    (qn.float().sum() + kn.float().sum() + v2.float().sum()).backward()

    torch.testing.assert_close(query.grad.float(), q2.grad.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(key.grad.float(), k2.grad.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(value.grad.float(), v2.grad.float(), rtol=0, atol=0)
    torch.testing.assert_close(q_weight.grad.float(), qw2.grad.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_weight.grad.float(), kw2.grad.float(), rtol=1e-2, atol=1e-2)


def test_fa4_varlen_backward_reaches_qkv():
    from slime_plugins.models.qwen3_attn_ops import fa4_varlen_with_grad

    torch.manual_seed(2)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(6, 2, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(6, 2, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    cu = torch.tensor([0, 3, 6], device=device, dtype=torch.int32)
    scale = 64**-0.5

    out = fa4_varlen_with_grad(q, k, v, cu, cu, softmax_scale=scale)
    out.float().square().sum().backward()

    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0
    assert out.shape == q.shape


def _raw_fused_qk_norm_rope(query, key, value, q_weight, k_weight, positions, *, eps, rotary_base):
    from sglang.jit_kernel.fused_qknorm_rope import fused_qk_norm_rope

    t = query.shape[0]
    nq, hd = query.shape[1], query.shape[2]
    nkv = key.shape[1]
    qkv = torch.cat(
        [
            query.reshape(t, nq * hd),
            key.reshape(t, nkv * hd),
            value.reshape(t, nkv * hd),
        ],
        dim=-1,
    ).contiguous()
    fused_qk_norm_rope(
        qkv,
        nq,
        nkv,
        nkv,
        hd,
        eps,
        q_weight.to(torch.bfloat16),
        k_weight.to(torch.bfloat16),
        rotary_base,
        True,
        positions,
        1.0,
        0.0,
        0.0,
        1.0,
    )
    q_size = nq * hd
    kv_size = nkv * hd
    q_out, k_out, v_out = qkv.split([q_size, kv_size, kv_size], dim=-1)
    return q_out.reshape(t, nq, hd), k_out.reshape(t, nkv, hd), v_out.reshape(t, nkv, hd)


def test_fused_qk_norm_rope_forward_is_bit_exact_vs_raw_kernel():
    from slime_plugins.models.qwen3_attn_ops import fused_qk_norm_rope_with_grad

    torch.manual_seed(3)
    device = torch.device("cuda")
    t, nq, nkv, hd = 8, 4, 1, 64
    query = torch.randn(t, nq, hd, dtype=torch.bfloat16, device=device)
    key = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device)
    value = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device)
    q_weight = torch.nn.Parameter(torch.rand(hd, dtype=torch.bfloat16, device=device) + 0.5)
    k_weight = torch.nn.Parameter(torch.rand(hd, dtype=torch.bfloat16, device=device) + 0.5)
    positions = _positions([5, 3], device)
    kwargs = dict(eps=1e-6, rotary_base=1_000_000.0)

    raw_q, raw_k, raw_v = _raw_fused_qk_norm_rope(
        query.clone(), key.clone(), value.clone(), q_weight, k_weight, positions, **kwargs
    )
    wrap_q, wrap_k, wrap_v = fused_qk_norm_rope_with_grad(
        query.clone().requires_grad_(True),
        key.clone().requires_grad_(True),
        value.clone().requires_grad_(True),
        q_weight,
        k_weight,
        positions,
        **kwargs,
    )
    assert torch.equal(wrap_q, raw_q)
    assert torch.equal(wrap_k, raw_k)
    assert torch.equal(wrap_v, raw_v)


def test_fa4_forward_is_bit_exact_vs_raw_kernel():
    from slime_plugins.models.qwen3_attn_ops import _fa4_varlen_forward, fa4_varlen_with_grad

    torch.manual_seed(4)
    device = torch.device("cuda")
    q = torch.randn(6, 4, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(6, 2, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn(6, 2, 64, dtype=torch.bfloat16, device=device)
    cu = torch.tensor([0, 3, 6], device=device, dtype=torch.int32)
    scale = 64**-0.5

    raw = _fa4_varlen_forward(q.clone(), k.clone(), v.clone(), cu, cu, scale)
    wrapped = fa4_varlen_with_grad(
        q.clone().requires_grad_(True),
        k.clone().requires_grad_(True),
        v.clone().requires_grad_(True),
        cu,
        cu,
        softmax_scale=scale,
    )
    assert torch.equal(wrapped, raw)


def _grad_report(name: str, actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a = actual.float().reshape(-1)
    r = reference.float().reshape(-1)
    diff = (a - r).abs()
    denom = r.abs().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(a, r, dim=0).item()
    stats = {
        "name": name,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "cosine": cos,
        "ref_rms": float(r.square().mean().sqrt()),
        "act_rms": float(a.square().mean().sqrt()),
    }
    print(
        f"[bwd-proof] {name}: max_abs={stats['max_abs']:.4e} "
        f"max_rel={stats['max_rel']:.4e} cosine={stats['cosine']:.8f} "
        f"ref_rms={stats['ref_rms']:.4e}",
        flush=True,
    )
    return stats


def test_fused_qk_norm_rope_backward_report():
    from slime_plugins.models.qwen3_attn_ops import (
        fused_qk_norm_rope_with_grad,
        neox_rope,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    t, nq, nkv, hd = 6, 2, 1, 64
    query = torch.randn(t, nq, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    key = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    value = torch.randn(t, nkv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    q_weight = torch.nn.Parameter(torch.rand(hd, dtype=torch.bfloat16, device=device) + 0.5)
    k_weight = torch.nn.Parameter(torch.rand(hd, dtype=torch.bfloat16, device=device) + 0.5)
    positions = _positions([3, 3], device)
    eps = 1e-6
    rotary_base = 1_000_000.0

    q_out, k_out, v_out = fused_qk_norm_rope_with_grad(
        query, key, value, q_weight, k_weight, positions, eps=eps, rotary_base=rotary_base
    )
    (q_out.float().sum() + k_out.float().sum() + v_out.float().sum()).backward()

    q2 = query.detach().clone().requires_grad_(True)
    k2 = key.detach().clone().requires_grad_(True)
    v2 = value.detach().clone().requires_grad_(True)
    qw2 = torch.nn.Parameter(q_weight.detach().clone())
    kw2 = torch.nn.Parameter(k_weight.detach().clone())
    qn = neox_rope(F.rms_norm(q2.float(), (hd,), qw2.float(), eps).to(torch.bfloat16), positions, rotary_base)
    kn = neox_rope(F.rms_norm(k2.float(), (hd,), kw2.float(), eps).to(torch.bfloat16), positions, rotary_base)
    (qn.float().sum() + kn.float().sum() + v2.float().sum()).backward()

    q_stats = _grad_report("fused_rope/dq", query.grad, q2.grad)
    k_stats = _grad_report("fused_rope/dk", key.grad, k2.grad)
    v_stats = _grad_report("fused_rope/dv", value.grad, v2.grad)
    assert q_stats["cosine"] > 0.99
    assert k_stats["cosine"] > 0.99
    assert v_stats["cosine"] > 0.999
    assert torch.equal(value.grad, v2.grad)


def _math_gqa_causal(query, key, value, softmax_scale):
    seq, n_q, head_dim = query.shape
    n_kv = key.shape[1]
    if n_q % n_kv != 0:
        raise RuntimeError(f"GQA mismatch q={n_q} kv={n_kv}")
    repeat = n_q // n_kv
    key = key.repeat_interleave(repeat, dim=1)
    value = value.repeat_interleave(repeat, dim=1)
    qh = query.transpose(0, 1).float()
    kh = key.transpose(0, 1).float()
    vh = value.transpose(0, 1).float()
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * float(softmax_scale)
    causal = torch.triu(torch.ones(seq, seq, device=query.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)
    return out.transpose(0, 1).to(dtype=query.dtype)


def test_fa4_backward_matches_math_attention():
    from slime_plugins.models.qwen3_attn_ops import _fa_varlen_backward, fa4_varlen_with_grad

    torch.manual_seed(5)
    device = torch.device("cuda")
    seq, n_q, n_kv, hd = 8, 4, 2, 64
    q = torch.randn(seq, n_q, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(seq, n_kv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(seq, n_kv, hd, dtype=torch.bfloat16, device=device, requires_grad=True)
    cu = torch.tensor([0, seq], device=device, dtype=torch.int32)
    scale = hd**-0.5

    out = fa4_varlen_with_grad(q, k, v, cu, cu, softmax_scale=scale)
    dout = torch.randn_like(out)
    out.backward(dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    math_out = _math_gqa_causal(q_ref, k_ref, v_ref, scale)
    math_out.backward(dout)

    q_stats = _grad_report("fa4_vs_math/dq", q.grad, q_ref.grad)
    k_stats = _grad_report("fa4_vs_math/dk", k.grad, k_ref.grad)
    v_stats = _grad_report("fa4_vs_math/dv", v.grad, v_ref.grad)
    assert q_stats["cosine"] > 0.99
    assert k_stats["cosine"] > 0.99
    assert v_stats["cosine"] > 0.99

    dq2, dk2, dv2 = _fa_varlen_backward(q.detach(), k.detach(), v.detach(), dout, cu, cu, scale)
    torch.testing.assert_close(dq2.float(), q.grad.float(), rtol=0, atol=0)
