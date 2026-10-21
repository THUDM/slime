"""Precision and packed-sequence regression tests for the native GLM5-next plugin."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig
from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrVisionPatchEmbed

from slime_plugins.models.glm5_next.attention import KimiDeltaAttention
from slime_plugins.models.glm5_next.ops import ClampedSwiGLU
from slime_plugins.models.glm5_next.vision import Glm5NextPatchEmbed

NUM_GPUS = 1
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("offload_fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("param_update_in_fp32", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_checkpoint_reload_preserves_fp32_parameters_after_first_step(offload_fraction, param_update_in_fp32, overlap):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    # HybridDeviceOptimizer identifies CPU optimizers by their class. Keep the
    # two classes distinct, as with CPU AdamW and GPU FusedAdam in production.
    class CPUAdamW(torch.optim.AdamW):
        pass

    fp32 = torch.nn.Parameter(torch.zeros(8, device="cuda", dtype=torch.float32))
    bf16 = torch.nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    optimizer = HybridDeviceOptimizer(
        [fp32, bf16],
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=CPUAdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=param_update_in_fp32,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=1e-6,
        weight_decay=0,
    )
    # Direct HF loading happens after optimizer construction. mHC and KDA keep
    # FP32 parameters, whose offloaded copies also need checkpoint refresh.
    with torch.no_grad():
        fp32.fill_(7)
        bf16.fill_(3)
    optimizer.update_fp32_param_by_new_param()
    for param in (fp32, bf16):
        torch.testing.assert_close(
            optimizer.param_to_inner_param[param].float().cpu(), param.float().cpu(), rtol=0, atol=0
        )
        param.grad = torch.ones_like(param)
    optimizer.step()
    torch.cuda.synchronize()
    torch.testing.assert_close(fp32, torch.full_like(fp32, 7 - 1e-6), rtol=0, atol=1e-7)
    torch.testing.assert_close(bf16, torch.full_like(bf16, 3), rtol=0, atol=0)
    # Resuming an optimizer checkpoint must also handle the original FP32
    # parameters and reproduce an uninterrupted second update.
    saved = copy.deepcopy(optimizer.state_dict())
    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in (fp32, bf16)]
    restored = HybridDeviceOptimizer(
        restored_params,
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=CPUAdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=param_update_in_fp32,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=1e-6,
        weight_decay=0,
    )
    restored.load_state_dict(saved)
    for param in restored_params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    restored.step()
    torch.cuda.synchronize()
    for parameter, expected in zip(restored_params, (fp32, bf16), strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_clamped_swiglu_precision_and_saturated_gradients():
    torch.manual_seed(37)
    x = (torch.randn(64, 512, device="cuda", dtype=torch.bfloat16) * 12).requires_grad_()
    activation = ClampedSwiGLU(SimpleNamespace(activation_func_clamp_value=10))
    output = activation(x)
    reference_input = x.detach().double().requires_grad_()
    gate, up = reference_input.chunk(2, dim=-1)
    reference = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    torch.testing.assert_close(output, reference.bfloat16(), rtol=0.008, atol=0.002)
    output.float().sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(x.grad, reference_input.grad.bfloat16(), rtol=0.008, atol=0.002)
    assert torch.count_nonzero(x.grad[:, :256][x[:, :256] > 10]) == 0
    assert torch.count_nonzero(x.grad[:, 256:][x[:, 256:].abs() > 10]) == 0


def test_patch_gemm_precision_and_backward():
    torch.manual_seed(123)
    config = GlmOcrVisionConfig(hidden_size=128, patch_size=14, temporal_patch_size=2)
    original = GlmOcrVisionPatchEmbed(config).cuda().bfloat16()
    optimized = Glm5NextPatchEmbed(config).cuda().bfloat16()
    optimized.load_state_dict(original.state_dict())
    x = torch.randn(128, 1176, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    before, after = original(x), optimized(other)
    reference = torch.nn.functional.linear(
        x.double(), original.proj.weight.flatten(1).double(), original.proj.bias.double()
    )
    # Fused bias keeps its FP32 accumulator until the final BF16 rounding.
    assert (after.double() - reference).square().mean() <= (before.double() - reference).square().mean()
    grad = torch.randn_like(before)
    before.backward(grad)
    after.backward(grad)
    torch.testing.assert_close(x.grad, other.grad, rtol=0.008, atol=0.002)
    for (_, p), (_, q) in zip(original.named_parameters(), optimized.named_parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0.008, atol=0.008)


@pytest.mark.parametrize("length", [33, 65])
def test_sparse_mla_partial_tiles_match_fp64_attention_gradients(length):
    from slime_plugins.models.glm5.ops.sparse_mla import SparseMLA

    torch.manual_seed(92)
    q = (torch.randn(length, 64, 576, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    kv = (torch.randn(length, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
    indices = torch.arange(2112, device="cuda").expand(length, 2112).clone()
    positions = torch.arange(length, device="cuda")
    indices[indices > positions[:, None]] = -1
    actual, _ = SparseMLA.apply(q, kv, indices[:, None].int().contiguous(), 0.0625)
    grad = torch.randn_like(actual)
    actual.backward(grad)

    q_ref, kv_ref = (p.detach().double().requires_grad_() for p in (q, kv))
    logits = torch.einsum("qhd,kd->hqk", q_ref, kv_ref[:, 0]) * 0.0625
    logits = logits.masked_fill((positions[None, :] > positions[:, None])[None], -torch.inf)
    expected = torch.einsum("hqk,kd->qhd", logits.softmax(-1), kv_ref[:, 0, :512])
    expected.backward(grad.double())
    for value, reference in ((actual, expected), (q.grad, q_ref.grad), (kv.grad, kv_ref.grad)):
        assert (value.double() - reference).norm() / reference.norm() < 0.005


def test_kda_packed_sequences_have_independent_outputs_and_gradients():
    torch.manual_seed(42)
    config = SimpleNamespace(hidden_size=256)
    hf_config = SimpleNamespace(
        linear_attn_config={"num_heads": 4, "head_dim": 128, "safe_gate": True, "short_conv_kernel_size": 4},
        rms_norm_eps=1e-5,
    )
    model = KimiDeltaAttention(config, hf_config).cuda().bfloat16()
    assert model.A_log.dtype == model.dt_bias.dtype == torch.float32
    x = torch.randn(32, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 16, 32], dtype=torch.int32, device="cuda"))
    output, _ = model(x, packed_seq_params=packed)
    second = x[16:].detach().clone().requires_grad_()
    individual = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 16], dtype=torch.int32, device="cuda"))
    expected, _ = model(second, packed_seq_params=individual)
    torch.testing.assert_close(output[16:], expected, rtol=0.008, atol=0.002)
    # Independently evaluate the delta-rule recurrence, including safe gates.
    from fla.ops.kda.naive import naive_recurrent_kda

    with torch.no_grad():
        states = second.transpose(0, 1)
        qkv = [
            getattr(model, f"{name}_conv1d")(
                getattr(model, f"{name}_proj")(states), cu_seqlens=individual.cu_seqlens_q
            )[0].unflatten(-1, (4, 128))
            for name in ("q", "k", "v")
        ]
        q, k = [
            (t.float() * torch.rsqrt(t.float().square().sum(-1, keepdim=True) + 1e-6)).to(t.dtype) for t in qkv[:2]
        ]
        gate = model.f_b_proj(model.f_a_proj(states)).unflatten(-1, (4, 128)).float()
        gate = -5 * torch.sigmoid(model.A_log.exp()[:, None] * (gate + model.dt_bias.view(4, 128)))
        recurrent, _ = naive_recurrent_kda(q, k, qkv[2], gate, model.b_proj(states).sigmoid())
        output_gate = model.g_b_proj(model.g_a_proj(states)).reshape(-1, 128)
        recurrent = model.o_norm(recurrent.reshape(-1, 128), output_gate).reshape(1, 16, 512)
        recurrent = model.o_proj(recurrent).transpose(0, 1)
        assert (recurrent.float() - expected.float()).norm() / recurrent.float().norm() < 0.01
    output[16:].float().square().sum().backward()
    expected.float().square().sum().backward()
    assert torch.count_nonzero(x.grad[:16]) == 0
    torch.testing.assert_close(x.grad[16:], second.grad, rtol=0.02, atol=0.002)
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
