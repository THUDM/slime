from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest
import torch
import torch.nn as nn

NUM_GPUS = 0

# Loaded straight from its path: importing it as a package member would pull in the
# other bridge plugins, which need a real megatron.bridge.
PLUGIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "slime_plugins" / "megatron_bridge" / "qwen3_5_vl.py"
PLUGIN = "qwen3_5_vl_under_test"


class _StubGatedDeltaNet(nn.Module):
    """Stands in for megatron-core's GatedDeltaNet, including its packed refusal."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.sp_size = 1

    def forward(self, hidden_states, attention_mask=None, *args, packed_seq_params=None, **kwargs):
        if packed_seq_params is not None:
            raise NotImplementedError("GDN does not support packed sequence for now.")
        return hidden_states


class _StubSelfAttention(nn.Module):
    pass


class _ModuleSpec:
    def __init__(self, module=None, submodules=None):
        self.module = module
        self.submodules = submodules


class _Submodules:
    def __init__(self, self_attention=None, mtp_model_layer=None):
        self.self_attention = self_attention
        if mtp_model_layer is not None:
            self.mtp_model_layer = mtp_model_layer


class _BlockSpec:
    def __init__(self, layer_specs):
        self.layer_specs = layer_specs


def _install_stubs(monkeypatch, *, cp_size=1, micro_batch_size=1, use_dynamic_batch_size=False):
    """Install the minimal megatron / megatron.bridge surface the plugin imports."""
    registered: dict[str, str] = {}

    def module(name):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("megatron")
    core = module("megatron.core")
    module("megatron.core.ssm")
    gdn_mod = module("megatron.core.ssm.gated_delta_net")
    gdn_mod.GatedDeltaNet = _StubGatedDeltaNet

    mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: cp_size)
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)

    training = module("megatron.training")
    training.get_args = lambda: types.SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
    )

    module("megatron.bridge")
    models_mod = module("megatron.bridge.models")
    module("megatron.bridge.models.conversion")
    param_mapping = module("megatron.bridge.models.conversion.param_mapping")

    class AutoMapping:
        @classmethod
        def register_module_type(cls, name, parallelism_type):
            registered[name] = parallelism_type

    param_mapping.AutoMapping = AutoMapping

    gpt_provider = module("megatron.bridge.models.gpt_provider")
    gpt_provider.mtp_block_spec = lambda config, vp_stage=None: _make_block_spec()
    models_mod.gpt_provider = gpt_provider

    module("megatron.bridge.models.qwen_vl")
    provider_mod = module("megatron.bridge.models.qwen_vl.qwen35_vl_provider")
    provider_mod.get_transformer_block_with_experimental_attention_variant_spec = (
        lambda config, vp_stage=None: _make_block_spec()
    )

    class Qwen35VLModelProvider:
        """Mimics the real provider: builds its block spec from module globals."""

        transformer_layer_spec = staticmethod(lambda config, vp_stage=None: _make_block_spec())

        def provide(self, pre_process=None, post_process=None, vp_stage=None):
            block_spec = provider_mod.get_transformer_block_with_experimental_attention_variant_spec(
                self, vp_stage=vp_stage
            )
            mtp_spec = gpt_provider.mtp_block_spec(self, vp_stage=vp_stage)
            return block_spec, mtp_spec

        def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
            return self.transformer_layer_spec(self)

    class Qwen35VLMoEModelProvider(Qwen35VLModelProvider):
        pass

    provider_mod.Qwen35VLModelProvider = Qwen35VLModelProvider
    provider_mod.Qwen35VLMoEModelProvider = Qwen35VLMoEModelProvider

    sys.modules.pop(PLUGIN, None)
    spec = importlib.util.spec_from_file_location(PLUGIN, PLUGIN_PATH)
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, PLUGIN, plugin)
    spec.loader.exec_module(plugin)
    return plugin, provider_mod, registered


def _make_block_spec():
    """One GDN layer, one standard attention layer, and a GDN layer nested under MTP."""
    gdn_layer = _ModuleSpec(submodules=_Submodules(self_attention=_ModuleSpec(module=_StubGatedDeltaNet)))
    attn_layer = _ModuleSpec(submodules=_Submodules(self_attention=_ModuleSpec(module=_StubSelfAttention)))
    mtp_layer = _ModuleSpec(
        submodules=_Submodules(
            self_attention=_ModuleSpec(module=_StubSelfAttention),
            mtp_model_layer=_ModuleSpec(submodules=_Submodules(self_attention=_ModuleSpec(module=_StubGatedDeltaNet))),
        )
    )
    return _BlockSpec([gdn_layer, attn_layer, mtp_layer])


def _packed_seq_params(total_tokens, qkv_format="thd"):
    return types.SimpleNamespace(
        qkv_format=qkv_format,
        cu_seqlens_q=torch.tensor([0, total_tokens], dtype=torch.int32),
        cu_seqlens_q_padded=None,
    )


def test_only_gdn_layers_are_replaced(monkeypatch):
    plugin, provider_mod, _ = _install_stubs(monkeypatch)

    spec, _mtp = provider_mod.Qwen35VLModelProvider().provide()
    gdn_layer, attn_layer, mtp_layer = spec.layer_specs

    assert gdn_layer.submodules.self_attention.module is plugin.SinglePackedSequenceGatedDeltaNet
    assert attn_layer.submodules.self_attention.module is _StubSelfAttention
    # nested MTP layers are reached too
    nested = mtp_layer.submodules.mtp_model_layer.submodules.self_attention
    assert nested.module is plugin.SinglePackedSequenceGatedDeltaNet


def test_moe_provider_is_patched_too(monkeypatch):
    plugin, provider_mod, _ = _install_stubs(monkeypatch)

    spec, _mtp = provider_mod.Qwen35VLMoEModelProvider().provide()
    assert spec.layer_specs[0].submodules.self_attention.module is plugin.SinglePackedSequenceGatedDeltaNet


def test_subclass_is_registered_for_weight_mapping(monkeypatch):
    plugin, _, registered = _install_stubs(monkeypatch)

    # AutoMapping dispatches on the exact class name, so the subclass must be registered
    # the same way megatron-bridge registers GatedDeltaNet, or weight conversion fails.
    assert registered[plugin.SinglePackedSequenceGatedDeltaNet.__name__] == "column"


def test_packed_single_sequence_is_accepted(monkeypatch):
    plugin, _, _ = _install_stubs(monkeypatch)

    layer = plugin.SinglePackedSequenceGatedDeltaNet()
    hidden = torch.zeros(8, 1, 4)

    # the unpatched parent would raise NotImplementedError here
    out = layer(hidden, None, packed_seq_params=_packed_seq_params(8))
    assert out.shape == hidden.shape


def test_unpacked_input_is_delegated_unchanged(monkeypatch):
    plugin, _, _ = _install_stubs(monkeypatch)

    layer = plugin.SinglePackedSequenceGatedDeltaNet()
    hidden = torch.zeros(8, 2, 4)
    assert layer(hidden, None).shape == hidden.shape


@pytest.mark.parametrize(
    "kwargs, packed, message",
    [
        ({"cp_size": 2}, _packed_seq_params(8), "context parallel"),
        ({}, _packed_seq_params(8, qkv_format="bshd"), "thd"),
        ({}, _packed_seq_params(5), "not a single packed sequence"),
    ],
)
def test_unsupported_packing_is_rejected(monkeypatch, kwargs, packed, message):
    plugin, _, _ = _install_stubs(monkeypatch, **kwargs)

    layer = plugin.SinglePackedSequenceGatedDeltaNet()
    with pytest.raises((NotImplementedError, RuntimeError), match=message):
        layer(torch.zeros(8, 1, 4), None, packed_seq_params=packed)


def test_batch_dimension_greater_than_one_is_rejected(monkeypatch):
    plugin, _, _ = _install_stubs(monkeypatch)

    layer = plugin.SinglePackedSequenceGatedDeltaNet()
    with pytest.raises(NotImplementedError, match="batch dimension"):
        layer(torch.zeros(8, 2, 4), None, packed_seq_params=_packed_seq_params(8))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"use_dynamic_batch_size": True}, "use-dynamic-batch-size"),
        ({"micro_batch_size": 4}, "micro-batch-size 1"),
    ],
)
def test_multi_sequence_microbatches_are_rejected_at_build_time(monkeypatch, kwargs, message):
    # These cannot be caught in forward: one sequence plus padding and two sequences
    # both look like two cu_seqlens segments.
    _, provider_mod, _ = _install_stubs(monkeypatch, **kwargs)

    with pytest.raises(NotImplementedError, match=message):
        provider_mod.Qwen35VLModelProvider().provide()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
