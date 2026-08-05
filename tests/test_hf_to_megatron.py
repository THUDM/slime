import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# These mapping tests also run in the CPU CI image, which does not install
# Megatron. In that environment, mount the source package without executing
# megatron_utils' runtime patch initialization.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules["slime.backends.megatron_utils"] = _megatron_utils

from slime.backends.megatron_utils.hf_to_megatron import _LOADERS
from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from slime.backends.megatron_utils.hf_to_megatron.deepseek import deepseek_hf_tensor
from slime.backends.megatron_utils.hf_to_megatron.glm import glm4_hf_tensor, glm4_moe_hf_tensor
from slime.backends.megatron_utils.hf_to_megatron.qwen import (
    mimo_hf_tensor,
    minimax_m2_hf_tensor,
    qwen_hf_tensor,
    qwen_moe_hf_tensor,
)
from slime.backends.megatron_utils.hf_to_megatron.qwen3_next import qwen3_next_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from slime.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf
from slime.backends.megatron_utils.megatron_to_hf.glm4 import convert_glm4_to_hf
from slime.backends.megatron_utils.megatron_to_hf.glm4moe import convert_glm4moe_to_hf
from slime.backends.megatron_utils.megatron_to_hf.mimo import convert_mimo_to_hf
from slime.backends.megatron_utils.megatron_to_hf.minimax_m2 import convert_minimax_m2_to_hf
from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf
from slime.backends.megatron_utils.megatron_to_hf.qwen3_next import convert_qwen3_next_to_hf
from slime.backends.megatron_utils.megatron_to_hf.qwen3moe import convert_qwen3moe_to_hf

NUM_GPUS = 0


class Reader:
    def __init__(self, **tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def get_tensor(self, name):
        return self.tensors[name]


def _config(model_type="qwen3"):
    return types.SimpleNamespace(
        model_type=model_type,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_hidden_layers=2,
        tie_word_embeddings=False,
    )


_EXPORT_ARGS = types.SimpleNamespace(
    kv_channels=2,
    hidden_size=8,
    num_attention_heads=4,
    num_query_groups=2,
    num_layers=2,
    q_lora_rank=None,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loader", "exporter", "model_type", "name", "shape"),
    [
        (
            qwen_hf_tensor,
            convert_qwen2_to_hf,
            "qwen3",
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            (16, 8),
        ),
        (
            qwen_moe_hf_tensor,
            convert_qwen3moe_to_hf,
            "qwen3_moe",
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
            (12, 8),
        ),
        (
            mimo_hf_tensor,
            convert_mimo_to_hf,
            "mimo",
            "module.module.mtp.layers.0.eh_proj.weight",
            (8, 8),
        ),
        (
            minimax_m2_hf_tensor,
            convert_minimax_m2_to_hf,
            "minimax_m2",
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
            (12, 8),
        ),
        (
            deepseek_hf_tensor,
            convert_deepseekv3_to_hf,
            "deepseek_v32",
            "module.module.decoder.layers.0.self_attention.wq_b.weight",
            (256, 8),
        ),
        (
            deepseek_hf_tensor,
            convert_deepseekv3_to_hf,
            "deepseek_v3",
            "module.module.mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight3",
            (12, 8),
        ),
        (
            glm4_hf_tensor,
            convert_glm4_to_hf,
            "glm4",
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            (16, 8),
        ),
        (
            glm4_moe_hf_tensor,
            convert_glm4moe_to_hf,
            "glm4_moe",
            "module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
            (12, 8),
        ),
        (
            qwen3_next_hf_tensor,
            convert_qwen3_next_to_hf,
            "qwen3_next",
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            (24, 8),
        ),
    ],
)
def test_hf_and_megatron_mappings_round_trip(loader, exporter, model_type, name, shape):
    parameter = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    hf_tensors = dict(exporter(_EXPORT_ARGS, name, parameter))
    config = types.SimpleNamespace(
        model_type=model_type,
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        num_hidden_layers=2,
        tie_word_embeddings=False,
    )

    loaded = loader(name, Reader(**hf_tensors), config)

    assert torch.equal(loaded, parameter)


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["deepseekv32config", "kimik2config"])
def test_deepseek_family_parameter_updates_use_the_direct_exporter(model_name):
    parameter = torch.randn(8, 8)

    converted = _convert_to_hf_core(
        _EXPORT_ARGS,
        model_name,
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        parameter,
    )

    assert len(converted) == 1
    assert converted[0][0] == "model.layers.0.self_attn.o_proj.weight"
    assert converted[0][1] is parameter


@pytest.mark.unit
def test_qwen2_moe_parameter_updates_use_the_moe_exporter():
    parameter = torch.randn(12, 8)

    converted = _convert_to_hf_core(
        _EXPORT_ARGS,
        "qwen2_moe",
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight3",
        parameter,
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
    ]
    assert torch.equal(converted[0][1], parameter[:6])
    assert torch.equal(converted[1][1], parameter[6:])


@pytest.mark.unit
def test_qwen_and_llama_share_the_basic_qkv_mapping():
    q = torch.arange(16).view(4, 4)
    k = torch.arange(8).view(2, 4) + 100
    v = torch.arange(8).view(2, 4) + 200
    reader = Reader(
        **{
            "model.layers.3.self_attn.q_proj.weight": q,
            "model.layers.3.self_attn.k_proj.weight": k,
            "model.layers.3.self_attn.v_proj.weight": v,
        }
    )

    loaded = qwen_hf_tensor(
        "module.module.decoder.layers.3.self_attention.linear_qkv.weight",
        reader,
        _config(),
    )

    assert torch.equal(loaded, torch.cat((q, k, v)))
    assert _LOADERS["qwen3"] is _LOADERS["llama"] is qwen_hf_tensor


@pytest.mark.unit
def test_qwen_moe_merges_one_global_expert():
    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    reader = Reader(
        **{
            "model.layers.1.mlp.experts.9.gate_proj.weight": gate,
            "model.layers.1.mlp.experts.9.up_proj.weight": up,
        }
    )

    loaded = qwen_moe_hf_tensor(
        "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight9",
        reader,
        _config(),
    )

    assert torch.equal(loaded, torch.cat((gate, up)))


@pytest.mark.unit
def test_deepseek_mapping_handles_kimi_and_dsa_layouts():
    mla = torch.randn(8, 4)
    kimi = deepseek_hf_tensor(
        "module.module.decoder.layers.0.self_attention.linear_kv_down_proj.weight",
        Reader(**{"model.layers.0.self_attn.kv_a_proj_with_mqa.weight": mla}),
        _config("kimi_k2"),
    )
    assert kimi is mla

    dsa = torch.arange(128 * 2).view(128, 2)
    reordered = deepseek_hf_tensor(
        "module.module.decoder.layers.0.self_attention.wk.weight",
        Reader(**{"model.layers.0.self_attn.indexer.wk.weight": dsa}),
        _config("glm_moe_dsa"),
    )
    assert torch.equal(reordered, torch.cat((dsa[64:], dsa[:64])))


@pytest.mark.unit
def test_glm_dense_and_moe_mtp_use_native_mappings():
    fused = torch.randn(8, 4)
    dense = glm4_hf_tensor(
        "module.module.decoder.layers.1.mlp.linear_fc1.weight",
        Reader(**{"model.layers.1.mlp.gate_up_proj.weight": fused}),
        _config("glm4"),
    )
    assert dense is fused

    mtp = torch.randn(4, 4)
    moe = glm4_moe_hf_tensor(
        "module.module.mtp.layers.0.eh_proj.weight",
        Reader(**{"model.layers.2.eh_proj.weight": mtp}),
        _config("glm4_moe"),
    )
    assert moe is mtp

    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    shared = glm4_moe_hf_tensor(
        "module.module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
        Reader(
            **{
                "model.layers.0.mlp.shared_experts.gate_proj.weight": gate,
                "model.layers.0.mlp.shared_experts.up_proj.weight": up,
            }
        ),
        _config("glm4_moe"),
    )
    assert torch.equal(shared, torch.cat((gate, up)))


@pytest.mark.unit
def test_minimax_and_mimo_keep_their_small_qwen_deltas():
    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    minimax = minimax_m2_hf_tensor(
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight2",
        Reader(
            **{
                "model.layers.0.block_sparse_moe.experts.2.w1.weight": gate,
                "model.layers.0.block_sparse_moe.experts.2.w3.weight": up,
            }
        ),
        _config("minimax_m2"),
    )
    assert torch.equal(minimax, torch.cat((gate, up)))

    hf_eh = torch.arange(24).view(3, 8)
    mimo = mimo_hf_tensor(
        "module.module.mtp.layers.0.eh_proj.weight",
        Reader(**{"model.mtp_layers.0.input_proj.weight": hf_eh}),
        _config("mimo"),
    )
    assert torch.equal(mimo, torch.cat((hf_eh[:, 4:], hf_eh[:, :4]), dim=1))


@pytest.mark.unit
def test_loader_scope_stays_explicit():
    assert set(_LOADERS) == {
        "deepseek_v3",
        "deepseek_v32",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "kimi_k2",
        "llama",
        "mimo",
        "minimax_m2",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_moe",
        "qwen3_next",
    }


@pytest.mark.unit
def test_reader_dequantizes_block_scaled_fp8(tmp_path):
    weight = torch.linspace(-2, 2, 128 * 128).view(128, 128).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0]])
    save_file(
        {"weight": weight, "weight_scale_inv": scale},
        tmp_path / "model.safetensors",
    )

    loaded = SafetensorReader(tmp_path).get_tensor("weight")

    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, weight.to(torch.bfloat16) * 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
