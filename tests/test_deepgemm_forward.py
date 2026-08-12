import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.alignment import deepgemm_forward, deepgemm_moe_forward

NUM_GPUS = 1


class TELinear(torch.nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(width * width).view(width, width).float() / 10)
        self.use_bias = False

    def forward(self, input_):
        return torch.nn.functional.linear(input_, self.weight), None


class CountingTELinear(TELinear):
    def __init__(self, width: int = 4):
        super().__init__(width)
        self.forward_calls = 0

    def forward(self, input_):
        self.forward_calls += 1
        return super().forward(input_)


class TELayerNormLinear(TELinear):
    def __init__(self, width: int = 4):
        super().__init__(width)
        self.return_layernorm_output = False
        self.return_layernorm_output_gathered = False

    def forward(self, input_):
        normalized = input_ * 2
        output = torch.nn.functional.linear(normalized, self.weight)
        if self.return_layernorm_output:
            return (output, normalized), None
        return output, None


class RMSNormTELayerNormLinear(TELayerNormLinear):
    def __init__(self, width: int = 4):
        super().__init__(width)
        self.config = SimpleNamespace(
            normalization="RMSNorm",
            layernorm_epsilon=1e-5,
            layernorm_zero_centered_gamma=False,
            delay_wgrad_compute=False,
        )
        self.layer_norm_weight = torch.nn.Parameter(torch.ones(width))


def test_deepgemm_forward_skips_te_and_differentiates_linear(monkeypatch):
    module = CountingTELinear()
    input_ = torch.arange(8, dtype=torch.float32).view(2, 4).requires_grad_()
    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = module.weight.detach().clone().requires_grad_()
    fixed_grad = torch.arange(8, dtype=torch.float32).view(2, 4) / 11
    expected_output = torch.nn.functional.linear(expected_input, expected_weight)
    expected_output.backward(fixed_grad)

    monkeypatch.setattr(
        deepgemm_forward,
        "_deepgemm_linear",
        lambda input_value, weight, checkpoint_scale_inv=None: torch.nn.functional.linear(input_value, weight),
    )
    monkeypatch.setattr(
        deepgemm_forward,
        "_SUPPORTED_TE_CLASS_NAMES",
        {*deepgemm_forward._SUPPORTED_TE_CLASS_NAMES, "CountingTELinear"},
    )
    deepgemm_forward._wrap_te_linear(
        module,
        "decoder.layers.0.mlp.linear_fc1",
    )

    output, bias = module(input_)
    assert bias is None
    assert module.forward_calls == 0
    torch.testing.assert_close(output, expected_output.detach())
    output.backward(fixed_grad)
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(module.weight.grad, expected_weight.grad)


def test_deepgemm_backward_skips_frozen_linear_weight_gradient(monkeypatch):
    module = CountingTELinear()
    module.weight.requires_grad_(False)
    monkeypatch.setattr(
        deepgemm_forward,
        "_deepgemm_linear",
        lambda input_value, weight, checkpoint_scale_inv=None: torch.nn.functional.linear(
            input_value,
            weight,
        ),
    )
    monkeypatch.setattr(
        deepgemm_forward,
        "_deepgemm_bf16_gemm_tn",
        lambda *_args, **_kwargs: pytest.fail("frozen linear weight must not compute wgrad"),
    )
    monkeypatch.setattr(
        deepgemm_forward,
        "_SUPPORTED_TE_CLASS_NAMES",
        {*deepgemm_forward._SUPPORTED_TE_CLASS_NAMES, "CountingTELinear"},
    )
    deepgemm_forward._wrap_te_linear(
        module,
        "decoder.layers.0.mlp.linear_fc1",
    )

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4).requires_grad_()
    output, _ = module(input_)
    output.sum().backward()

    assert input_.grad is not None
    assert module.weight.grad is None


def test_deepgemm_layernorm_linear_skips_te_forward(monkeypatch):
    module = RMSNormTELayerNormLinear()
    input_ = (torch.arange(8, dtype=torch.float32).view(2, 4) / 7).requires_grad_()
    fixed_grad = torch.arange(8, dtype=torch.float32).view(2, 4) / 13

    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = module.weight.detach().clone().requires_grad_()
    expected_norm_weight = module.layer_norm_weight.detach().clone().requires_grad_()
    expected_norm = torch.nn.functional.rms_norm(
        expected_input,
        normalized_shape=(4,),
        weight=expected_norm_weight,
        eps=module.config.layernorm_epsilon,
    )
    expected_output = torch.nn.functional.linear(expected_norm, expected_weight)
    expected_output.backward(fixed_grad)

    monkeypatch.setattr(
        deepgemm_forward,
        "_deepgemm_linear",
        lambda input_value, weight, checkpoint_scale_inv=None: torch.nn.functional.linear(input_value, weight),
    )
    monkeypatch.setattr(
        deepgemm_forward,
        "_SUPPORTED_TE_CLASS_NAMES",
        {*deepgemm_forward._SUPPORTED_TE_CLASS_NAMES, "RMSNormTELayerNormLinear"},
    )
    deepgemm_forward._wrap_te_linear(
        module,
        "decoder.layers.0.self_attention.linear_q_up_proj",
    )

    output, bias = module(input_)
    assert bias is None
    torch.testing.assert_close(output, expected_output.detach())
    output.backward(fixed_grad)
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(module.weight.grad, expected_weight.grad)
    torch.testing.assert_close(module.layer_norm_weight.grad, expected_norm_weight.grad)


def test_deepgemm_layernorm_linear_does_not_normalize_twice(monkeypatch):
    module = RMSNormTELayerNormLinear()
    module._deepgemm_input_already_normalized = True
    input_ = (torch.arange(8, dtype=torch.float32).view(2, 4) / 7).requires_grad_()
    fixed_grad = torch.arange(8, dtype=torch.float32).view(2, 4) / 13

    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = module.weight.detach().clone().requires_grad_()
    expected_output = torch.nn.functional.linear(expected_input, expected_weight)
    expected_output.backward(fixed_grad)

    monkeypatch.setattr(
        deepgemm_forward,
        "_deepgemm_linear",
        lambda input_value, weight, checkpoint_scale_inv=None: torch.nn.functional.linear(input_value, weight),
    )
    monkeypatch.setattr(
        deepgemm_forward,
        "_SUPPORTED_TE_CLASS_NAMES",
        {*deepgemm_forward._SUPPORTED_TE_CLASS_NAMES, "RMSNormTELayerNormLinear"},
    )
    deepgemm_forward._wrap_te_linear(
        module,
        "decoder.layers.0.mlp.linear_fc1",
    )

    output, bias = module(input_)
    assert bias is None
    torch.testing.assert_close(output, expected_output.detach())
    output.backward(fixed_grad)
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(module.weight.grad, expected_weight.grad)
    assert module.layer_norm_weight.grad is None


def test_global_layer_index_uses_layer_number_not_local_module_list_index():
    class Layer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.mlp = torch.nn.Module()
            self.mlp.linear_fc1 = TELinear()

    model_chunk = torch.nn.Module()
    model_chunk.decoder = torch.nn.Module()
    model_chunk.decoder.layers = torch.nn.ModuleList([Layer(layer_number=21)])

    assert (
        deepgemm_forward._get_global_layer_index(
            model_chunk,
            "decoder.layers.0.mlp.linear_fc1",
        )
        == 20
    )


def test_global_layer_index_keeps_float16_and_ddp_wrapper_prefixes():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 37
            self.mlp = torch.nn.Module()
            self.mlp.linear_fc1 = TELinear()

    gpt = torch.nn.Module()
    gpt.decoder = torch.nn.Module()
    gpt.decoder.layers = torch.nn.ModuleList([Layer()])
    float16_wrapper = torch.nn.Module()
    float16_wrapper.module = gpt
    ddp_wrapper = torch.nn.Module()
    ddp_wrapper.module = float16_wrapper

    assert (
        deepgemm_forward._get_global_layer_index(
            ddp_wrapper,
            "module.module.decoder.layers.0.mlp.linear_fc1",
        )
        == 36
    )


def test_pipeline_stage_input_rmsnorm_uses_one_sglang_forward_and_analytic_backward(
    monkeypatch,
):
    class CountingRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.eps = 1e-5
            self.forward_calls = 0

        def forward(self, value):
            self.forward_calls += 1
            return value * self.weight

    class Layer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.input_layernorm = CountingRMSNorm()

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    # Model a nonzero PP stage: its first local layer is global layer 2.
    model.decoder.layers = torch.nn.ModuleList([Layer(3), Layer(4)])
    monkeypatch.setenv("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "1")
    monkeypatch.setattr(
        deepgemm_forward,
        "_sglang_batch_invariant_rmsnorm",
        lambda value, weight, eps: value.detach() + 17,
    )

    deepgemm_forward.enable_sglang_layer0_input_rmsnorm(None, model, "")
    input_ = torch.tensor([[5.0, 7.0]], requires_grad=True)
    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = model.decoder.layers[0].input_layernorm.weight.detach().clone().requires_grad_()
    expected = torch.nn.functional.rms_norm(
        expected_input,
        normalized_shape=(2,),
        weight=expected_weight,
        eps=model.decoder.layers[0].input_layernorm.eps,
    )
    expected.sum().backward()
    output = model.decoder.layers[0].input_layernorm(input_)

    torch.testing.assert_close(output, input_.detach() + 17)
    output.sum().backward()
    assert model.decoder.layers[0].input_layernorm.forward_calls == 0
    assert model.decoder.layers[1].input_layernorm.forward_calls == 0
    assert not hasattr(
        model.decoder.layers[1].input_layernorm,
        "_slime_sglang_pipeline_input_rmsnorm_wrapped",
    )
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(
        model.decoder.layers[0].input_layernorm.weight.grad,
        expected_weight.grad,
    )


def test_pipeline_stage_input_rmsnorm_is_opt_in(monkeypatch):
    model = torch.nn.Module()
    monkeypatch.delenv("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", raising=False)

    deepgemm_forward.enable_sglang_layer0_input_rmsnorm(None, model, "")

    assert not hasattr(model, "_slime_sglang_layer0_input_rmsnorm_wrapped")


def test_absorbed_kv_rmsnorm_uses_one_sglang_forward_and_analytic_backward(
    monkeypatch,
):
    class KVUp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm_weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_kv_up_proj = KVUp()
            self.config = SimpleNamespace(layernorm_epsilon=1e-5)
            self.forward_calls = 0

        def forward(self, value):
            self.forward_calls += 1
            return torch.nn.functional.rms_norm(
                value.float(),
                normalized_shape=(2,),
                weight=self.linear_kv_up_proj.layer_norm_weight.float(),
                eps=self.config.layernorm_epsilon,
            ).to(value.dtype)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 1
            self.self_attention = Attention()

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([Layer()])
    args = SimpleNamespace(megatron_deepgemm_forward_layers=[0])
    monkeypatch.setenv("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "1")
    monkeypatch.setattr(
        deepgemm_forward,
        "_sglang_batch_invariant_rmsnorm",
        lambda value, weight, eps: value.detach() + 23,
    )

    deepgemm_forward.enable_sglang_absorbed_kv_rmsnorm(args, model, "")
    input_ = torch.tensor([[5.0, 7.0]], requires_grad=True)
    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = (
        model.decoder.layers[0].self_attention.linear_kv_up_proj.layer_norm_weight.detach().clone().requires_grad_()
    )
    expected = torch.nn.functional.rms_norm(
        expected_input.float(),
        normalized_shape=(2,),
        weight=expected_weight.float(),
        eps=model.decoder.layers[0].self_attention.config.layernorm_epsilon,
    ).to(input_.dtype)
    expected.sum().backward()
    output = model.decoder.layers[0].self_attention(input_)

    torch.testing.assert_close(output, input_.detach() + 23)
    output.sum().backward()
    attention = model.decoder.layers[0].self_attention
    assert attention.forward_calls == 1
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(
        attention.linear_kv_up_proj.layer_norm_weight.grad,
        expected_weight.grad,
    )
    assert torch.nn.functional.rms_norm is not None


def test_final_rmsnorm_uses_exact_fp32_residual_and_analytic_backward(monkeypatch):
    class CountingRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.eps = 0.0
            self.forward_calls = 0

        def forward(self, value):
            self.forward_calls += 1
            return value * self.weight

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 78

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([Layer()])
    model.decoder.final_layernorm = CountingRMSNorm()
    model.decoder.layers[-1]._sglang_residual_sum_fp32 = torch.tensor([[3.0, 4.0]])
    monkeypatch.setenv("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "1")

    deepgemm_forward.enable_sglang_final_rmsnorm(None, model, "")
    input_ = torch.tensor([[5.0, 7.0]], requires_grad=True)
    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = model.decoder.final_layernorm.weight.detach().clone().requires_grad_()
    expected_backward = torch.nn.functional.rms_norm(
        expected_input,
        normalized_shape=(2,),
        weight=expected_weight,
        eps=model.decoder.final_layernorm.eps,
    )
    expected_backward.sum().backward()
    output = model.decoder.final_layernorm(input_)
    rms = (torch.tensor(12.5)).sqrt()
    expected = torch.tensor([[3.0 / rms * 2.0, 4.0 / rms * 3.0]])

    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert model.decoder.final_layernorm.forward_calls == 0
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(
        model.decoder.final_layernorm.weight.grad,
        expected_weight.grad,
    )


def test_enable_wraps_only_selected_global_layer(monkeypatch):
    class Layer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number
            self.mlp = torch.nn.Module()
            self.mlp.linear_fc1 = TELinear()

    model_chunk = torch.nn.Module()
    model_chunk.decoder = torch.nn.Module()
    model_chunk.decoder.layers = torch.nn.ModuleList([Layer(layer_number=1), Layer(layer_number=11)])
    args = SimpleNamespace(
        megatron_deepgemm_forward_layers=[0],
        megatron_deepgemm_forward_modules=["mlp.linear_fc1"],
    )
    monkeypatch.setattr(
        deepgemm_forward.parallel_state,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    deepgemm_forward.enable_deepgemm_forward(args, [model_chunk], store_prefix="")

    assert model_chunk.decoder.layers[0].mlp.linear_fc1._slime_deepgemm_forward_wrapped
    assert not hasattr(
        model_chunk.decoder.layers[1].mlp.linear_fc1,
        "_slime_deepgemm_forward_wrapped",
    )


def test_default_targets_cover_indexer_and_shared_expert_fp8_linears():
    assert {
        "self_attention.wq_b",
        "self_attention.wk",
        "mlp.shared_experts.linear_fc1",
        "mlp.shared_experts.linear_fc2",
    }.issubset(deepgemm_forward._DEFAULT_TARGET_SUFFIXES)


def test_sglang_router_value_retains_megatron_backward(monkeypatch):
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(12).view(3, 4).float() / 10)
            self.forward_calls = 0

        def gating(self, value):
            self.forward_calls += 1
            return torch.nn.functional.linear(value, self.weight)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number = 4
            self.mlp = torch.nn.Module()
            self.mlp.router = Router()

    model_chunk = torch.nn.Module()
    model_chunk.decoder = torch.nn.Module()
    model_chunk.decoder.layers = torch.nn.ModuleList([Layer()])
    args = SimpleNamespace(
        megatron_deepgemm_moe_forward_layers=[3],
        sglang_enable_fp32_moe_router=True,
    )
    input_ = torch.arange(8, dtype=torch.float32).view(2, 4).requires_grad_()
    expected_input = input_.detach().clone().requires_grad_()
    expected_weight = model_chunk.decoder.layers[0].mlp.router.weight.detach().clone().requires_grad_()
    torch.nn.functional.linear(expected_input, expected_weight).sum().backward()

    batch_invariant_ops = importlib.import_module("sglang.srt.batch_invariant_ops")
    monkeypatch.setattr(
        batch_invariant_ops,
        "router_gemm_batch_invariant",
        lambda value, weight: torch.full(
            (value.shape[0], weight.shape[0]),
            17.0,
            dtype=torch.float32,
            device=value.device,
        ),
        raising=False,
    )

    deepgemm_forward.enable_sglang_router_gemm(
        args,
        [model_chunk],
        store_prefix="",
    )
    output = model_chunk.decoder.layers[0].mlp.router.gating(input_)

    torch.testing.assert_close(output, torch.full((2, 3), 17.0))
    assert model_chunk.decoder.layers[0].mlp.router.forward_calls == 0
    output.sum().backward()
    torch.testing.assert_close(input_.grad, expected_input.grad)
    torch.testing.assert_close(
        model_chunk.decoder.layers[0].mlp.router.weight.grad,
        expected_weight.grad,
    )


def test_sglang_router_value_requires_matching_sglang_fp32_router():
    args = SimpleNamespace(
        megatron_deepgemm_moe_forward_layers=[3],
        sglang_enable_fp32_moe_router=False,
    )

    with pytest.raises(RuntimeError, match="--sglang-enable-fp32-moe-router"):
        deepgemm_forward.enable_sglang_router_gemm(
            args,
            [],
            store_prefix="",
        )


def test_sglang_router_value_allows_saved_rollout_replay_without_live_sglang():
    args = SimpleNamespace(
        megatron_deepgemm_moe_forward_layers=[3],
        sglang_enable_fp32_moe_router=False,
        debug_train_only=True,
        load_debug_rollout_data="/tmp/rollout.pt",
    )

    deepgemm_forward.enable_sglang_router_gemm(args, [], store_prefix="")


def test_sglang_swiglu_uses_one_forward_and_analytic_backward(monkeypatch):
    value = torch.arange(24, dtype=torch.float32).view(3, 8).div(10).requires_grad_()
    expected_value = value.detach().clone().requires_grad_()
    gate, up = expected_value.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    grad_output = torch.arange(12, dtype=torch.float32).view(3, 4).div(13)
    expected.backward(grad_output)

    calls = {"count": 0}
    fake_sgl_kernel = types.ModuleType("sgl_kernel")

    def one_forward(input_, out):
        calls["count"] += 1
        input_gate, input_up = input_.chunk(2, dim=-1)
        out.copy_(torch.nn.functional.silu(input_gate) * input_up + 19)
        return out

    fake_sgl_kernel.silu_and_mul = one_forward
    monkeypatch.setitem(sys.modules, "sgl_kernel", fake_sgl_kernel)
    output = deepgemm_forward._sglang_swiglu_with_megatron_backward(value)

    torch.testing.assert_close(output, expected.detach() + 19)
    assert calls["count"] == 1
    output.backward(grad_output)
    torch.testing.assert_close(value.grad, expected_value.grad)


def test_sglang_swiglu_mlp_accepts_current_megatron_padding_mask(monkeypatch):
    class PairLinear(torch.nn.Module):
        def __init__(self, width_in, width_out):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(width_out, width_in))

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight), None

    mlp = torch.nn.Module()
    mlp.linear_fc1 = PairLinear(2, 4)
    mlp.linear_fc2 = PairLinear(2, 2)
    monkeypatch.setattr(
        deepgemm_forward,
        "_sglang_swiglu_with_megatron_backward",
        lambda value: value[..., :2],
    )

    assert deepgemm_forward._wrap_sglang_swiglu_mlp(
        mlp,
        "decoder.layers.0.mlp",
        return_tuple=True,
    )
    hidden = torch.tensor([[1.0, 2.0]])
    output, bias = mlp(hidden, padding_mask=torch.tensor([False]))

    torch.testing.assert_close(output, hidden)
    assert bias is None


def test_combined_hook_installs_linears_and_moe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        deepgemm_forward,
        "enable_deepgemm_forward",
        lambda args, model, store_prefix: calls.append(("linears", args, model, store_prefix)),
    )
    monkeypatch.setattr(
        deepgemm_moe_forward,
        "enable_deepgemm_moe_forward",
        lambda args, model, store_prefix: calls.append(("moe", args, model, store_prefix)),
    )
    args = SimpleNamespace(
        megatron_deepgemm_forward_layers=[0, 1],
        megatron_deepgemm_moe_forward_layers=[3],
    )
    model = object()

    deepgemm_forward.enable_deepgemm_all_forward(
        args,
        model,
        store_prefix="actor_",
    )

    assert calls == [
        ("linears", args, model, "actor_"),
        ("moe", args, model, "actor_"),
    ]


def test_combined_before_train_hook_installs_alignment(monkeypatch):
    state = {"calls": []}
    monkeypatch.setattr(
        deepgemm_forward,
        "enable_deepgemm_all_forward",
        lambda args, model, store_prefix: state["calls"].append((args, model, store_prefix)),
    )

    deepgemm_forward.enable_deepgemm_all_forward_before_train_step(
        args=object(),
        rollout_id=2,
        step_id=3,
        model="model",
        optimizer=object(),
        opt_param_scheduler=object(),
    )

    assert len(state["calls"]) == 1
    assert state["calls"][0][1:] == ("model", "")


def _require_cuda_deepgemm(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real DeepGEMM diff tests")
    monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
    for module_name in (
        "deep_gemm",
        "sglang.srt.layers.deep_gemm_wrapper",
        "sglang.srt.layers.quantization.fp8_utils",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            pytest.skip(f"{module_name} is unavailable: {exc}")


def test_cuda_deepgemm_dense_forward_diff_against_bf16_reference(monkeypatch):
    _require_cuda_deepgemm(monkeypatch)
    torch.cuda.set_device(0)
    torch.manual_seed(123)
    input_ = (torch.randn(257, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)
    weight = (torch.randn(256, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)

    output = deepgemm_forward._deepgemm_linear(input_, weight)
    reference = torch.nn.functional.linear(input_, weight)
    diff = (output.float() - reference.float()).abs()

    assert diff.mean().item() < 0.03
    assert diff.max().item() < 0.15


def test_cuda_deepgemm_dense_backward_diff_against_bf16_reference(monkeypatch):
    _require_cuda_deepgemm(monkeypatch)
    torch.cuda.set_device(0)
    torch.manual_seed(124)

    module = CountingTELinear(width=128).cuda().to(torch.bfloat16)
    with torch.no_grad():
        module.weight.copy_((torch.randn_like(module.weight, dtype=torch.float32) * 0.2).to(torch.bfloat16))
    input_ = (torch.randn(257, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16).requires_grad_()
    grad_output = (torch.randn(257, 128, device="cuda", dtype=torch.float32) * 0.2).to(torch.bfloat16)

    reference_input = input_.detach().clone().requires_grad_()
    reference_weight = module.weight.detach().clone().requires_grad_()
    reference_output = torch.nn.functional.linear(
        reference_input,
        reference_weight,
    )
    reference_output.backward(grad_output)

    monkeypatch.setattr(
        deepgemm_forward,
        "_SUPPORTED_TE_CLASS_NAMES",
        {*deepgemm_forward._SUPPORTED_TE_CLASS_NAMES, "CountingTELinear"},
    )
    deepgemm_forward._wrap_te_linear(
        module,
        "decoder.layers.0.mlp.linear_fc1",
    )
    output, bias = module(input_)
    assert bias is None
    assert module.forward_calls == 0
    output.backward(grad_output)

    checks = {
        "output": (output, reference_output, 0.03, 0.15),
        "input_grad": (input_.grad, reference_input.grad, 0.01, 0.08),
        "weight_grad": (module.weight.grad, reference_weight.grad, 0.01, 0.08),
    }
    for name, (actual, reference, mean_limit, max_limit) in checks.items():
        diff = (actual.float() - reference.float()).abs()
        assert diff.mean().item() < mean_limit, f"{name} mean diff {diff.mean().item()} >= {mean_limit}"
        assert diff.max().item() < max_limit, f"{name} max diff {diff.max().item()} >= {max_limit}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
