from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def install_megatron_stubs() -> None:
    if "megatron" in sys.modules:
        return

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    inference_mod = types.ModuleType("megatron.core.inference")
    inference_contexts_mod = types.ModuleType("megatron.core.inference.contexts")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_module_mod = types.ModuleType("megatron.core.transformer.module")
    spec_utils_mod = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_block_mod = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")

    class PackedSeqParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class ModuleSpec:
        def __init__(self, module=None, params=None):
            self.module = module
            self.params = params or {}

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    tensor_parallel_stub = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )

    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    inference_contexts_mod.BaseInferenceContext = type("BaseInferenceContext", (), {})
    packed_seq_mod.PackedSeqParams = PackedSeqParams
    transformer_module_mod.MegatronModule = MegatronModule
    spec_utils_mod.ModuleSpec = ModuleSpec
    transformer_block_mod.get_num_layers_to_build = lambda *args, **kwargs: 0
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0

    core_mod.mpu = mpu_stub
    core_mod.tensor_parallel = tensor_parallel_stub

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["megatron.core.inference"] = inference_mod
    sys.modules["megatron.core.inference.contexts"] = inference_contexts_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod
    sys.modules["megatron.core.transformer"] = transformer_mod
    sys.modules["megatron.core.transformer.module"] = transformer_module_mod
    sys.modules["megatron.core.transformer.spec_utils"] = spec_utils_mod
    sys.modules["megatron.core.transformer.transformer_block"] = transformer_block_mod
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_mod


class FakeShortConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, cu_seqlens=None, **kwargs):
        return x, None


class FakeShortConvolutionWithWeights(FakeShortConvolution):
    def __init__(self, *args, **kwargs):
        super().__init__()
        hidden_size = kwargs["hidden_size"]
        kernel_size = kwargs["kernel_size"]
        self.weight = nn.Parameter(torch.ones(hidden_size, 1, kernel_size))
        self.bias = None


class FakeFusedRMSNormGated(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, z):
        return x


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.float32,
    )


def load_module(module_name: str):
    install_megatron_stubs()
    sys.modules.pop("slime_plugins.models.hf_attention", None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("slime_plugins.models.qwen3_5", "Qwen3_5GatedDeltaNet"),
        ("slime_plugins.models.qwen3_next", "Qwen3NextGatedDeltaNet"),
    ],
)
def test_linear_attention_forwards_cu_seqlens_to_chunk_kernel(monkeypatch, module_name: str, class_name: str):
    module = load_module(module_name)

    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(module, "ShortConvolution", FakeShortConvolution)
    monkeypatch.setattr(module, "FusedRMSNormGated", FakeFusedRMSNormGated)

    chunk_calls = []

    def fake_chunk_gated_delta_rule(
        q,
        k,
        v,
        *,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens=None,
        **kwargs,
    ):
        chunk_calls.append(cu_seqlens.clone() if cu_seqlens is not None else None)
        assert q.shape[0] == 1
        assert cu_seqlens is not None
        return torch.zeros_like(v), None

    monkeypatch.setattr(module, "chunk_gated_delta_rule", fake_chunk_gated_delta_rule)

    layer = getattr(module, class_name)(make_config(), layer_idx=0)
    hidden_states = torch.randn(1, 7, 32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    output = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert output.shape == hidden_states.shape
    assert len(chunk_calls) == 1
    assert torch.equal(chunk_calls[0], cu_seqlens)


@pytest.mark.unit
def test_qwen35_linear_attention_builds_cp_context(monkeypatch):
    module = load_module("slime_plugins.models.qwen3_5")

    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(module, "ShortConvolution", FakeShortConvolutionWithWeights)
    monkeypatch.setattr(module, "FusedRMSNormGated", FakeFusedRMSNormGated)

    cp_context_calls = {}

    def fake_build_cp_context(*, cu_seqlens, group, conv1d_kernel_size):
        cp_context_calls["cu_seqlens"] = cu_seqlens.clone()
        cp_context_calls["group"] = group
        cp_context_calls["conv1d_kernel_size"] = conv1d_kernel_size
        return SimpleNamespace(cu_seqlens=cu_seqlens)

    def fake_causal_conv1d(x, **kwargs):
        cp_context_calls["conv_weight_shape"] = tuple(kwargs["weight"].shape)
        cp_context_calls["conv_cp_context"] = kwargs["cp_context"]
        return x, None

    def fake_chunk_gated_delta_rule(
        q,
        k,
        v,
        *,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens=None,
        cp_context=None,
        **kwargs,
    ):
        cp_context_calls["gdn_cu_seqlens"] = cu_seqlens.clone() if cu_seqlens is not None else None
        cp_context_calls["gdn_cp_context"] = cp_context
        return torch.zeros_like(v), None

    monkeypatch.setattr(module, "build_cp_context", fake_build_cp_context)
    monkeypatch.setattr(module, "causal_conv1d", fake_causal_conv1d)
    monkeypatch.setattr(module, "chunk_gated_delta_rule", fake_chunk_gated_delta_rule)

    layer = module.Qwen3_5GatedDeltaNet(make_config(), layer_idx=0)
    layer.cp_group = "fake-cp-group"
    layer.cp_world_size = 4
    hidden_states = torch.randn(1, 8, 32)

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.equal(cp_context_calls["cu_seqlens"], torch.tensor([0, 32], dtype=torch.int32))
    assert cp_context_calls["group"] == "fake-cp-group"
    assert cp_context_calls["conv1d_kernel_size"] == 4
    assert cp_context_calls["conv_weight_shape"] == (32, 4)
    assert cp_context_calls["conv_cp_context"] is cp_context_calls["gdn_cp_context"]
    assert torch.equal(cp_context_calls["gdn_cu_seqlens"], torch.tensor([0, 32], dtype=torch.int32))


@pytest.mark.unit
def test_qwen35_attention_supports_bshd_without_packed_seq_params(monkeypatch):
    module = load_module("slime_plugins.models.qwen3_5")
    hf_attention = importlib.import_module("slime_plugins.models.hf_attention")

    monkeypatch.setattr(hf_attention, "_load_hf_config", lambda checkpoint_path: make_config())

    class FakeLinearAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.cp_group = None
            self.cp_rank = 0
            self.cp_world_size = 1

        def forward(self, hidden_states, cu_seqlens=None):
            self.calls.append(cu_seqlens)
            return hidden_states

    monkeypatch.setattr(module, "Qwen3_5GatedDeltaNet", lambda config, layer_idx: FakeLinearAttention())

    args = SimpleNamespace(sequence_parallel=False, qkv_format="bshd", hf_checkpoint="/tmp/unused")
    attention = module.Attention(args, config=SimpleNamespace(), layer_number=1)
    hidden_states = torch.randn(6, 2, 32)

    output, bias = attention(hidden_states, attention_mask=None, packed_seq_params=None)

    assert output.shape == hidden_states.shape
    assert bias is None
    assert attention.linear_attn.calls == [None]
