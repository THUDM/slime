"""CPU tests for the Nemotron 3.5 Super VL (`nemotron_h_omni`) weight mapping.

The loader (HF -> Megatron) and the exporter (Megatron -> HF, used by the RL weight
sync) are written as a pair and must stay exact inverses: the engine drops any key
it does not recognise without raising, so a wrong name in either direction shows up
only as stale weights in the rollout engine.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# CPU CI images do not install Megatron; mount the source package without running
# megatron_utils' runtime patch initialization.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules.setdefault("slime.backends.megatron_utils", _megatron_utils)

from slime.backends.megatron_utils.hf_to_megatron import _LOADERS
from slime.backends.megatron_utils.hf_to_megatron.nemotron_h import nemotron_h_hf_tensor
from slime.backends.megatron_utils.hf_to_megatron.nemotron_h_vl import _language_config, nemotron_h_vl_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from slime.backends.megatron_utils.megatron_to_hf import nemotron_h_vl as export_module
from slime.backends.megatron_utils.megatron_to_hf.nemotron_h_vl import convert_nemotron_h_vl_to_hf

NUM_GPUS = 0

RADIO = "vision_model.radio_model.model"


class Reader:
    def __init__(self, **tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def get_tensor(self, name):
        return self.tensors[name]


def _llm_config(**overrides):
    fields = dict(
        layers_block_type=["mamba", "attention", "moe"],
        mtp_layers_block_type=["attention", "moe"],
        expand=2,
        hidden_size=8,
        n_groups=2,
        ssm_state_size=2,
        mamba_num_heads=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        tie_word_embeddings=False,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _hf_config(**llm_overrides):
    return types.SimpleNamespace(llm_config=_llm_config(**llm_overrides))


def _export_args(pattern="M*E/*E"):
    return types.SimpleNamespace(
        tensor_model_parallel_size=1,
        hybrid_override_pattern=pattern,
        mamba_num_heads=4,
        mamba_head_dim=4,
        mamba_num_groups=2,
        mamba_state_dim=2,
        num_query_groups=2,
        num_attention_heads=4,
        kv_channels=2,
        q_lora_rank=None,
    )


def _load(name, tensors, hf_config=None):
    return nemotron_h_vl_hf_tensor(f"module.module.{name}", Reader(**tensors), hf_config or _hf_config())


def _export(name, param, args=None):
    return convert_nemotron_h_vl_to_hf(args or _export_args(), f"module.module.{name}", param)


@pytest.fixture(autouse=True)
def _empty_qkv_cache():
    export_module._QKV_CACHE.clear()
    yield
    export_module._QKV_CACHE.clear()


# --------------------------------------------------------------- registration


@pytest.mark.unit
def test_nemotron_families_are_registered_with_the_loader_table():
    assert _LOADERS["nemotron_h"] is nemotron_h_hf_tensor
    assert _LOADERS["nemotron_h_omni"] is nemotron_h_vl_hf_tensor


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name",
    [
        "nemotron_h_omni",  # the HF model_type
        "NemotronH_Omni_Reasoning_V3_Config",  # the config class slime passes when --model-name is unset
        "NemotronH_Nano_VL_V2_Config",
    ],
)
def test_vl_model_names_reach_the_vl_exporter(model_name):
    param = torch.randn(4, 4)
    converted = _convert_to_hf_core(
        _export_args(), model_name, "module.module.vision_model.embeddings.position_embedding", param
    )
    assert converted == [(f"{RADIO}.patch_generator.pos_embed", param)]


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["nemotron_h", "NemotronHConfig"])
def test_text_model_names_reach_the_text_exporter(model_name):
    param = torch.randn(4, 4)
    converted = _convert_to_hf_core(
        _export_args(), model_name, "module.module.decoder.layers.1.self_attention.linear_proj.weight", param
    )
    assert converted == [("backbone.layers.1.mixer.o_proj.weight", param)]

    # The text converter has no notion of a vision tower: a VL model routed here
    # by mistake would die on its first vision parameter.
    with pytest.raises(ValueError, match="Unsupported Nemotron-H"):
        _convert_to_hf_core(
            _export_args(), model_name, "module.module.vision_model.embeddings.position_embedding", param
        )


# ---------------------------------------------------------------- vision tower


VISION_NAMES = [
    ("embeddings.patch_projection.weight", f"{RADIO}.patch_generator.embedder.weight"),
    ("embeddings.video_patch_projection.weight", f"{RADIO}.patch_generator.video_embedder.weight"),
    ("embeddings.position_embedding", f"{RADIO}.patch_generator.pos_embed"),
    ("embeddings.cls_register_token", f"{RADIO}.patch_generator.cls_token.token"),
    ("encoder.layer.3.attention.output.dense.weight", f"{RADIO}.blocks.3.attn.proj.weight"),
    ("encoder.layer.3.attention.output.dense.bias", f"{RADIO}.blocks.3.attn.proj.bias"),
    # norm1 / norm2 / mlp.fc1 / mlp.fc2 are spelled the same on both sides.
    ("encoder.layer.3.norm1.weight", f"{RADIO}.blocks.3.norm1.weight"),
    ("encoder.layer.12.norm2.bias", f"{RADIO}.blocks.12.norm2.bias"),
    ("encoder.layer.3.mlp.fc1.weight", f"{RADIO}.blocks.3.mlp.fc1.weight"),
    ("encoder.layer.3.mlp.fc2.bias", f"{RADIO}.blocks.3.mlp.fc2.bias"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("radio_name", "hf_key"), VISION_NAMES)
def test_vision_names_map_to_the_released_checkpoint_keys_both_ways(radio_name, hf_key):
    tensor = torch.randn(4, 4)
    name = f"vision_model.{radio_name}"
    assert _load(name, {hf_key: tensor}) is tensor
    assert _export(name, tensor) == [(hf_key, tensor)]


@pytest.mark.unit
def test_loading_a_vision_name_the_checkpoint_lacks_fails_loudly():
    with pytest.raises(KeyError):
        _load("vision_model.embeddings.position_embedding", {})


@pytest.mark.unit
@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_fused_qkv_is_split_on_load_and_refused_on_export(suffix):
    q, k, v = (torch.randn(4, 5) if suffix == "weight" else torch.randn(4) for _ in range(3))
    fused = torch.cat([q, k, v], dim=0)
    key = f"{RADIO}.blocks.0.attn.qkv.{suffix}"

    for component, expected in (("query", q), ("key", k), ("value", v)):
        name = f"vision_model.encoder.layer.0.attention.attention.{component}.{suffix}"
        assert torch.equal(_load(name, {key: fused}), expected)

    # Export: three separate calls, one fused tensor -- emitted only on the third.
    names = {c: f"vision_model.encoder.layer.0.attention.attention.{c}.{suffix}" for c in ("query", "key", "value")}
    assert _export(names["query"], q) == []
    assert _export(names["key"], k) == []
    ((out_name, out),) = _export(names["value"], v)
    assert out_name == key
    assert torch.equal(out, fused)
    assert export_module._QKV_CACHE == {}


@pytest.mark.unit
def test_fused_qkv_export_is_independent_of_arrival_order_and_of_weight_vs_bias():
    q_w, k_w, v_w = (torch.randn(2, 3) for _ in range(3))
    q_b, k_b, v_b = (torch.randn(2) for _ in range(3))

    def name(component, suffix, block=1):
        return f"vision_model.encoder.layer.{block}.attention.attention.{component}.{suffix}"

    # Interleave a weight triple and a bias triple, out of order. Each completes
    # on its own third component and never on the other's.
    assert _export(name("value", "weight"), v_w) == []
    assert _export(name("query", "bias"), q_b) == []
    assert _export(name("key", "weight"), k_w) == []
    assert _export(name("value", "bias"), v_b) == []

    ((bias_name, bias),) = _export(name("key", "bias"), k_b)
    assert bias_name == f"{RADIO}.blocks.1.attn.qkv.bias"
    assert torch.equal(bias, torch.cat([q_b, k_b, v_b]))

    ((weight_name, weight),) = _export(name("query", "weight"), q_w)
    assert weight_name == f"{RADIO}.blocks.1.attn.qkv.weight"
    # q, k, v order regardless of arrival order.
    assert torch.equal(weight, torch.cat([q_w, k_w, v_w]))
    assert export_module._QKV_CACHE == {}


@pytest.mark.unit
def test_fused_qkv_blocks_do_not_mix():
    names = [
        f"vision_model.encoder.layer.{block}.attention.attention.{component}.weight"
        for block in (0, 1)
        for component in ("query", "key", "value")
    ]
    tensors = [torch.full((2, 2), float(index)) for index in range(6)]
    outputs = []
    for name, tensor in zip(names, tensors, strict=True):
        outputs += _export(name, tensor)
    assert [n for n, _ in outputs] == [f"{RADIO}.blocks.0.attn.qkv.weight", f"{RADIO}.blocks.1.attn.qkv.weight"]
    assert torch.equal(outputs[0][1], torch.cat(tensors[0:3]))
    assert torch.equal(outputs[1][1], torch.cat(tensors[3:6]))


# ------------------------------------------------------------------- projector


@pytest.mark.unit
@pytest.mark.parametrize(
    ("megatron_rest", "hf_key"),
    [
        ("mlp1.norm.weight", "mlp1.0.weight"),
        ("mlp1.norm.bias", "mlp1.0.bias"),
        ("mlp1.linear1.weight", "mlp1.1.weight"),
        ("mlp1.linear2.weight", "mlp1.3.weight"),  # index 2 is the activation
        ("vision_final_layernorm.weight", "vision_projector.vision_final_layernorm.weight"),
        ("vision_final_layernorm.bias", "vision_projector.vision_final_layernorm.bias"),
    ],
)
def test_projector_names_round_trip(megatron_rest, hf_key):
    tensor = torch.randn(4)
    name = f"vision_projector.{megatron_rest}"
    assert _load(name, {hf_key: tensor}) is tensor
    assert _export(name, tensor) == [(hf_key, tensor)]


@pytest.mark.unit
def test_unknown_projector_parameters_are_rejected():
    with pytest.raises(KeyError, match="projector"):
        _load("vision_projector.mlp1.linear3.weight", {})
    with pytest.raises(ValueError, match="projector"):
        _export("vision_projector.mlp1.linear3.weight", torch.randn(2))


# --------------------------------------------------------------------- decoder


@pytest.mark.unit
def test_decoder_parameters_gain_the_language_model_prefix_on_the_hf_side():
    tensor = torch.randn(8, 8)
    assert (
        _load(
            "language_model.decoder.layers.1.self_attention.linear_proj.weight",
            {"language_model.backbone.layers.1.mixer.o_proj.weight": tensor},
        )
        is tensor
    )
    assert _export("language_model.decoder.layers.1.self_attention.linear_proj.weight", tensor) == [
        ("language_model.backbone.layers.1.mixer.o_proj.weight", tensor)
    ]


@pytest.mark.unit
def test_decoder_mtp_parameters_keep_the_prefix_too():
    tensor = torch.randn(8, 16)
    assert (
        _load(
            "language_model.mtp.layers.0.eh_proj.weight",
            {"language_model.mtp.layers.0.eh_proj.weight": tensor},
        )
        is tensor
    )
    assert _export("language_model.mtp.layers.0.eh_proj.weight", tensor) == [
        ("language_model.mtp.layers.0.eh_proj.weight", tensor)
    ]


@pytest.mark.unit
def test_hybrid_pattern_is_derived_from_layers_block_type_without_mutating_the_config():
    hf_config = _hf_config()
    derived = _language_config(hf_config)

    assert derived.hybrid_override_pattern == "M*E"
    # The same config object is shared with the model provider.
    assert not hasattr(hf_config.llm_config, "hybrid_override_pattern")
    assert derived is not hf_config.llm_config


@pytest.mark.unit
def test_an_explicit_hybrid_pattern_wins_and_the_config_is_reused():
    hf_config = _hf_config(hybrid_override_pattern="EM")
    assert _language_config(hf_config) is hf_config.llm_config


@pytest.mark.unit
def test_language_config_errors_name_the_problem():
    with pytest.raises(KeyError, match="neither"):
        _language_config(types.SimpleNamespace(llm_config=types.SimpleNamespace()))
    with pytest.raises(KeyError, match="unknown layers_block_type entry 'rnn'"):
        _language_config(_hf_config(layers_block_type=["rnn"]))
