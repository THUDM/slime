"""Cache lifetime, invalidation, and real DeepGEMM forward/backward checks."""

from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.alignment import deepgemm_forward as dense
from slime.backends.megatron_utils.alignment import deepgemm_moe_forward as moe
from slime.backends.megatron_utils.alignment.weight_cache import (
    deepgemm_weight_cache_scope,
    get_quantized_weight,
    is_deepgemm_weight_cache_enabled,
)

NUM_GPUS = 1


def _quantizer(weight, calls):
    def quantize():
        calls.append(1)
        return weight.detach().clone(), torch.ones(1)

    return quantize


def test_disabled_and_unscoped_calls_do_not_cache():
    weight = torch.ones(4)
    calls = []
    quantize = _quantizer(weight, calls)
    for _ in range(2):
        get_quantized_weight("w", (weight,), quantize)
    with deepgemm_weight_cache_scope(False):
        for _ in range(2):
            get_quantized_weight("w", (weight,), quantize)
    assert len(calls) == 4


def test_microbatches_reuse_but_next_scope_sees_data_updates():
    weight = torch.nn.Parameter(torch.ones(4))
    calls = []
    quantize = _quantizer(weight, calls)
    with deepgemm_weight_cache_scope(True) as cache:
        first = get_quantized_weight("w", (weight,), quantize)
        for _ in range(5):
            assert get_quantized_weight("w", (weight,), quantize) is first
        assert (cache.misses, cache.hits) == (1, 5)
    assert not cache.entries
    # Fused optimizers and weight stores may bypass the Parameter version.
    version = weight._version
    weight.data.add_(2)
    assert weight._version == version
    with deepgemm_weight_cache_scope(True):
        updated = get_quantized_weight("w", (weight,), quantize)
    torch.testing.assert_close(updated[0], torch.full((4,), 3.0))
    assert len(calls) == 2


def test_mutation_replacement_and_format_invalidate_within_scope():
    weight = torch.nn.Parameter(torch.ones(4))
    calls = []
    with deepgemm_weight_cache_scope(True):
        get_quantized_weight("fp32-scale", (weight,), _quantizer(weight, calls))
        with torch.no_grad():
            weight.add_(1)
        changed = get_quantized_weight("fp32-scale", (weight,), _quantizer(weight, calls))
        torch.testing.assert_close(changed[0], torch.full((4,), 2.0))
        weight.data = torch.full((4,), 3.0)
        replaced = get_quantized_weight("fp32-scale", (weight,), _quantizer(weight, calls))
        torch.testing.assert_close(replaced[0], weight)
        get_quantized_weight("ue8m0-scale", (weight,), _quantizer(weight, calls))
    assert len(calls) == 4


def test_nested_models_and_exceptions_release_their_own_cache():
    actor = torch.ones(4)
    ref = torch.zeros(4)
    calls = []
    with deepgemm_weight_cache_scope(True) as outer:
        original = get_quantized_weight("w", (actor,), _quantizer(actor, calls))
        with pytest.raises(ValueError), deepgemm_weight_cache_scope(True) as inner:
            result = get_quantized_weight("w", (ref,), _quantizer(ref, calls))
            torch.testing.assert_close(result[0], ref)
            raise ValueError("schedule failed")
        assert not inner.entries
        assert get_quantized_weight("w", (actor,), _quantizer(actor, calls)) is original
    assert not outer.entries
    assert not is_deepgemm_weight_cache_enabled()


def test_inference_tensors_without_versions_bypass_cache():
    with torch.inference_mode():
        weight = torch.ones(4)
    calls = []
    with deepgemm_weight_cache_scope(True):
        for _ in range(2):
            get_quantized_weight("w", (weight,), _quantizer(weight, calls))
    assert len(calls) == 2


def test_detached_checkpoint_alias_reuses_quantized_weight():
    weight = torch.nn.Parameter(torch.ones(4))
    calls = []
    with deepgemm_weight_cache_scope(True):
        first = get_quantized_weight("w", (weight,), _quantizer(weight, calls))
        for _ in range(3):
            alias = weight.detach().requires_grad_()
            assert get_quantized_weight("w", (alias,), _quantizer(alias, calls)) is first
    assert len(calls) == 1


@pytest.mark.parametrize("ue8m0", [False, True])
def test_dense_caches_final_quantization_chain_but_not_activations(monkeypatch, ue8m0):
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization import fp8_utils

    from slime.backends.megatron_utils.kernels import fp8_kernel

    calls = {"quant": 0, "requant": 0, "linear": 0}

    def quant(weight, block):
        calls["quant"] += 1
        return weight.clone(), torch.ones(1)

    def requant(weight, scale, block):
        calls["requant"] += 1
        return weight + 1, scale

    def linear(x, weight, block, scale, bias):
        calls["linear"] += 1
        return x @ weight.t()

    monkeypatch.setattr(moe, "_configure_batch_invariant", lambda *args: True)
    monkeypatch.setattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", ue8m0)
    monkeypatch.setattr(fp8_kernel, "blockwise_cast_to_fp8_triton", quant)
    monkeypatch.setattr(fp8_utils, "quant_weight_ue8m0", quant)
    monkeypatch.setattr(fp8_utils, "requant_weight_ue8m0", requant)
    monkeypatch.setattr(fp8_utils, "deepgemm_w8a8_block_fp8_linear_with_fallback", linear)
    weight = torch.ones((8, 8), dtype=torch.bfloat16)
    x = torch.ones((2, 8), dtype=torch.bfloat16)
    with deepgemm_weight_cache_scope(True):
        y1 = dense._deepgemm_linear(x, weight)
        y2 = dense._deepgemm_linear(x * 2, weight)
    torch.testing.assert_close(y2, y1 * 2, rtol=0, atol=0)
    assert calls == {"quant": 1, "requant": int(ue8m0), "linear": 2}


def test_grouped_cache_does_not_alias_activation_workspace():
    module = torch.nn.Module()
    for i in range(3):
        module.register_parameter(f"weight{i}", torch.nn.Parameter(torch.full((128, 128), i + 1.0).bfloat16()))
    calls = []

    def quant(weight, block):
        calls.append(1)
        return weight.to(torch.float8_e4m3fn), torch.ones((1, 1))

    ops = SimpleNamespace(quantize_weight=quant, scale_ue8m0=False)
    kwargs = dict(
        expected_shape=(128, 128),
        layout=SimpleNamespace(num_local_experts=3),
        input_device=torch.device("cpu"),
        module_name="experts.fc1",
        ops=ops,
        expert_start=0,
        expert_end=2,
    )
    scratch = torch.empty(3 * 128 * 128, dtype=torch.bfloat16)
    with deepgemm_weight_cache_scope(True):
        first = moe._quantize_grouped_weights(module, grouped_qweight_workspace=scratch, **kwargs)
        expected = first[0].float().clone()
        assert first[0].untyped_storage().data_ptr() != scratch.untyped_storage().data_ptr()
        scratch.fill_(99)
        second = moe._quantize_grouped_weights(module, grouped_qweight_workspace=scratch, **kwargs)
        assert first is second
        torch.testing.assert_close(second[0].float(), expected, rtol=0, atol=0)
        assert len(calls) == 2
        # A distinct expert group must have its own entry.
        moe._quantize_grouped_weights(module, **{**kwargs, "expert_start": 2, "expert_end": 3})
        assert len(calls) == 3
        with torch.no_grad():
            module.weight0.add_(1)
        changed = moe._quantize_grouped_weights(module, **kwargs)
        torch.testing.assert_close(changed[0][0].float(), torch.full((128, 128), 2.0))
        assert len(calls) == 5
    with deepgemm_weight_cache_scope(True):
        moe._quantize_grouped_weights(module, **kwargs)
    assert len(calls) == 7


@pytest.mark.parametrize("enabled", [False, True])
def test_train_schedule_clears_cache_before_optimizer(monkeypatch, enabled):
    from slime.backends.megatron_utils import model as training

    args = SimpleNamespace(
        megatron_deepgemm_cache_weights=enabled,
        custom_megatron_before_train_step_hook_path=None,
        seq_length=128,
        micro_batch_size=1,
        decoder_seq_length=None,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
    )
    weight = torch.ones(4)
    calls = []
    quant = _quantizer(weight, calls)
    observed = []

    def schedule(**kwargs):
        for _ in range(kwargs["num_microbatches"]):
            observed.append(get_quantized_weight("w", (weight,), quant)[0].clone())
        return []

    def update():
        assert not is_deepgemm_weight_cache_enabled()
        weight.data.add_(1)
        return True, 0.0, 0

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_forward_backward_func", lambda: schedule)
    monkeypatch.setattr(training.mpu, "is_pipeline_last_stage", lambda **kwargs: False)
    optimizer = SimpleNamespace(zero_grad=lambda: None, step=update)
    scheduler = SimpleNamespace(step=lambda **kwargs: None)
    model = [SimpleNamespace(zero_grad_buffer=lambda: None)]
    for step in range(2):
        training.train_one_step(args, 0, step, [], model, optimizer, scheduler, 3, 3)
    assert len(calls) == (2 if enabled else 6)
    assert [v[0].item() for v in observed] == [1, 1, 1, 2, 2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DeepGEMM")
def test_cuda_dense_checkpoint_reuse_preserves_outputs_and_gradients(monkeypatch):
    from torch.utils.checkpoint import checkpoint

    from slime.backends.megatron_utils.kernels import fp8_kernel

    monkeypatch.setenv("SGLANG_DEEPGEMM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    calls = []
    original = fp8_kernel.blockwise_cast_to_fp8_triton

    def quant(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fp8_kernel, "blockwise_cast_to_fp8_triton", quant)
    torch.manual_seed(17)
    initial_w = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16) / 16
    initial_x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16) / 16
    results = []
    for enabled in (False, True):
        weight = initial_w.clone().requires_grad_()
        x = initial_x.clone().requires_grad_()
        calls.clear()
        with deepgemm_weight_cache_scope(enabled):
            outputs = []
            for _ in range(3):
                y = checkpoint(dense._DeepGEMMLinearWithBF16Backward.apply, x, weight, use_reentrant=True)
                outputs.append(y.detach().clone())
                y.float().square().mean().backward()
        torch.cuda.synchronize()
        results.append((outputs, x.grad.clone(), weight.grad.clone()))
        assert len(calls) == (1 if enabled else 6)
    for left, right in zip(results[0][0], results[1][0], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(results[0][1:], results[1][1:], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DeepGEMM")
def test_cuda_grouped_cache_survives_scratch_reuse_and_weight_updates(monkeypatch):
    from slime.backends.megatron_utils.kernels import fp8_kernel

    monkeypatch.setenv("SGLANG_DEEPGEMM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    monkeypatch.setenv("SLIME_DEEPGEMM_MOE_EXPERTS_PER_GROUP", "1")
    calls = []
    original = fp8_kernel.blockwise_cast_to_fp8_triton

    def quant(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fp8_kernel, "blockwise_cast_to_fp8_triton", quant)
    torch.manual_seed(21)
    module = torch.nn.Module()
    for name, shape in (("linear_fc1", (256, 512)), ("linear_fc2", (512, 128))):
        layer = torch.nn.Module()
        for expert in range(2):
            layer.register_parameter(
                f"weight{expert}",
                torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 32),
            )
        setattr(module, name, layer)
    layout = moe._MoELayout(num_local_experts=2, hidden_size=512, ffn_hidden_size=128)
    initial = {k: v.detach().clone() for k, v in module.state_dict().items()}
    hidden = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16) / 16
    probs = torch.ones(3, device="cuda", dtype=torch.float32)
    results = []
    for enabled in (False, True):
        module.load_state_dict(initial)
        calls.clear()
        outputs = []
        for _step in range(2):
            with torch.no_grad(), deepgemm_weight_cache_scope(enabled):
                for counts in ([1, 2], [3, 0]):
                    output = moe._deepgemm_grouped_moe_forward(
                        module,
                        hidden.clone(),
                        torch.tensor(counts, dtype=torch.int32),
                        probs,
                        layout=layout,
                        module_name="test.experts",
                        reuse_input_buffer=True,
                    )
                    outputs.append(output.clone())
            # Simulate an optimizer that writes through .data.
            for parameter in module.parameters():
                parameter.data.add_(0.015625)
        torch.cuda.synchronize()
        assert len(calls) == (8 if enabled else 12)
        results.append(outputs)
    for left, right in zip(*results, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
