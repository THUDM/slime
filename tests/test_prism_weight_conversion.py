"""CPU weight-layout checks; optional real DCP/HF comparison uses local assets.

Set PRISM_NATIVE_MODEL_DIR to a selected native ``model/`` component and
PRISM_HF_DIR to its official HF export to verify every published tensor.
"""

import json
import os
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
from slime.backends.megatron_utils.megatron_to_hf.prism import convert_prism_to_hf, is_prism_model

NUM_GPUS = 0


def model_args(**overrides):
    values = {
        "hidden_size": 8,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "kv_channels": 2,
        "ffn_hidden_size": 12,
        "num_layers": 3,
        "padded_vocab_size": 16,
        "vocab_size": 13,
        "speech_vocab_size": 5,
        "n_vq": 4,
        "prism_mlp_only_layers": (2,),
        "prism_rvq_heads_per_execution_unit": (0, 1, 1, 2),
        "sglang_dtype": "auto",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("prefix", ["", "module.", "module.module."])
def test_group_major_qkv_preserves_each_query_group(prefix):
    args = model_args()
    q = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    k = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 100
    v = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 200
    packed = torch.cat(
        [
            torch.cat((q[group * 4 : (group + 1) * 4], k[group * 2 : (group + 1) * 2], v[group * 2 : (group + 1) * 2]))
            for group in range(2)
        ]
    )
    result = dict(convert_prism_to_hf(args, f"{prefix}decoder.layers.1.self_attention.linear_qkv.weight", packed))
    for projection, expected in (("q", q), ("k", k), ("v", v)):
        actual = result[f"transformer.layers.1.self_attn.{projection}_proj.weight"]
        assert torch.equal(actual, expected)
        assert actual.is_contiguous()
        # Linear projections provide a numerical check independent of naming.
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        assert torch.equal(inputs @ actual.T, inputs @ expected.T)


def test_swiglu_fc1_retains_global_gate_and_up_halves():
    args = model_args()
    gate = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    up = torch.arange(96, dtype=torch.float32).reshape(12, 8) + 1000
    result = dict(convert_prism_to_hf(args, "decoder.layers.2.mlp.linear_fc1.weight", torch.cat((gate, up))))
    assert torch.equal(result["transformer.layers.2.mlp.gate_proj.weight"], gate)
    assert torch.equal(result["transformer.layers.2.mlp.up_proj.weight"], up)
    x = torch.ones(2, 8)
    fused = (x @ torch.cat((gate, up)).T).chunk(2, dim=-1)
    split = (
        x @ result["transformer.layers.2.mlp.gate_proj.weight"].T,
        x @ result["transformer.layers.2.mlp.up_proj.weight"].T,
    )
    assert torch.equal(torch.nn.functional.silu(fused[0]) * fused[1], torch.nn.functional.silu(split[0]) * split[1])


@pytest.mark.parametrize("family", ["prism", "moss_tts_prism", "MossTTSPrismConfig", "MOSS-TTS-Prism"])
def test_explicit_prism_dispatch_preserves_padded_text_vocab(family):
    args = model_args()
    tensor = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    result = convert_to_hf(args, family, "module.module.text_embedding.word_embeddings.weight", tensor)
    assert result[0][0] == "transformer.embed_tokens.weight"
    assert result[0][1].shape == (16, 8)
    assert torch.equal(result[0][1], tensor)
    assert not is_prism_model("qwen3")
    assert not is_prism_model("qwen3prism")


@pytest.mark.parametrize(
    ("native", "target", "shape"),
    [
        ("audio_embeddings.3.word_embeddings.weight", "audio_embeddings.3.weight", (5, 8)),
        ("audio_lm_heads.3.weight", "audio_lm_heads.3.weight", (5, 8)),
        ("stop_head.weight", "stop_head.weight", (2, 8)),
        ("prediction_norms.2.weight", "prediction_norms.2.weight", (8,)),
        ("decoder.layers.1.self_attention.linear_proj.weight", "transformer.layers.1.self_attn.o_proj.weight", (8, 8)),
        (
            "decoder.layers.1.self_attention.linear_qkv.layer_norm_weight",
            "transformer.layers.1.input_layernorm.weight",
            (8,),
        ),
        ("decoder.layers.1.self_attention.q_layernorm.weight", "transformer.layers.1.self_attn.q_norm.weight", (2,)),
        ("decoder.layers.1.self_attention.k_layernorm.weight", "transformer.layers.1.self_attn.k_norm.weight", (2,)),
        (
            "decoder.layers.2.mlp.linear_fc1.layer_norm_weight",
            "transformer.layers.2.post_attention_layernorm.weight",
            (8,),
        ),
        ("decoder.layers.2.mlp.linear_fc2.weight", "transformer.layers.2.mlp.down_proj.weight", (8, 12)),
    ],
)
def test_parameter_names_shapes_and_explicit_bf16_cast(native, target, shape):
    param = torch.linspace(-1, 1, torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    original = param.clone()
    result = convert_prism_to_hf(model_args(sglang_dtype="bfloat16"), native, param)
    assert len(result) == 1 and result[0][0] == target
    assert result[0][1].dtype == torch.bfloat16
    assert torch.equal(result[0][1], original.to(torch.bfloat16))
    assert torch.equal(param, original) and param.dtype == torch.float32


def test_bf16_qkv_and_fc1_cast_all_output_fragments():
    args = model_args(sglang_dtype="bfloat16")
    for name, shape in [
        ("decoder.layers.1.self_attention.linear_qkv.weight", (16, 8)),
        ("decoder.layers.1.mlp.linear_fc1.weight", (24, 8)),
    ]:
        result = convert_prism_to_hf(args, name, torch.ones(shape))
        assert all(param.dtype == torch.bfloat16 for _, param in result)
        assert all(param.is_contiguous() for _, param in result)


@pytest.mark.parametrize(
    ("name", "shape", "message"),
    [
        ("text_embedding.word_embeddings.weight", (13, 8), "shape mismatch"),
        ("audio_embeddings.4.word_embeddings.weight", (5, 8), "unknown Prism"),
        ("audio_embeddings.0.weight", (5, 8), "unknown Prism"),
        ("prediction_norms.3.weight", (8,), "unknown Prism"),
        ("decoder.layers.3.mlp.linear_fc2.weight", (8, 12), "unknown Prism"),
        ("decoder.layers.2.self_attention.linear_proj.weight", (8, 8), "MLP-only"),
        ("decoder.layers.0.self_attention.linear_qkv.bias", (16,), "unknown Prism"),
        ("decoder.final_layernorm.weight", (8,), "unknown Prism"),
    ],
)
def test_incomplete_or_non_prism_parameters_are_rejected(name, shape, message):
    with pytest.raises(ValueError, match=message):
        convert_prism_to_hf(model_args(), name, torch.zeros(shape))


def test_invalid_gqa_dtype_and_quantization_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        convert_prism_to_hf(
            model_args(num_attention_heads=3), "decoder.layers.0.self_attention.linear_qkv.weight", torch.zeros(16, 8)
        )
    with pytest.raises(ValueError, match="inference dtype"):
        convert_prism_to_hf(model_args(sglang_dtype="int8"), "stop_head.weight", torch.zeros(2, 8))
    with pytest.raises(ValueError, match="trainable dtype"):
        convert_prism_to_hf(model_args(), "stop_head.weight", torch.zeros(2, 8, dtype=torch.int64))
    with pytest.raises(ValueError, match="quantization"):
        convert_to_hf(
            model_args(), "prism", "stop_head.weight", torch.zeros(2, 8), quantization_config={"quant_method": "fp8"}
        )


@pytest.mark.integration
def test_real_prism_dcp_matches_complete_official_hf_export():
    """Read at most one native parameter at a time; no GPU or full model load."""
    native, hf = os.environ.get("PRISM_NATIVE_MODEL_DIR"), os.environ.get("PRISM_HF_DIR")
    if not native and not hf:
        pytest.skip("set PRISM_NATIVE_MODEL_DIR and PRISM_HF_DIR for real asset verification")
    assert native and hf, "both native and HF asset paths must be set"
    from safetensors import safe_open
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    hf = Path(hf)
    config = json.loads((hf / "config.json").read_text())
    language = config["language_config"]
    args = model_args(
        hidden_size=language["hidden_size"],
        num_attention_heads=language["num_attention_heads"],
        num_query_groups=language["num_key_value_heads"],
        kv_channels=language["head_dim"],
        ffn_hidden_size=language["intermediate_size"],
        num_layers=language["num_hidden_layers"],
        padded_vocab_size=language["vocab_size"],
        speech_vocab_size=config["speech_vocab_size"],
        n_vq=config["n_vq"],
        prism_mlp_only_layers=config["prism_mlp_only_layers"],
        prism_rvq_heads_per_execution_unit=config["prism_rvq_heads_per_execution_unit"],
        sglang_dtype=config["dtype"],
    )
    index = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    reader = FileSystemReader(native)
    metadata = reader.read_metadata()
    native_tensors = {
        key: value for key, value in metadata.state_dict_metadata.items() if isinstance(value, TensorStorageMetadata)
    }
    assert native_tensors, "native model component has no tensors"
    seen = set()
    with ExitStack() as stack:
        shards = {
            name: stack.enter_context(safe_open(hf / name, framework="pt", device="cpu"))
            for name in set(index.values())
        }
        for name, description in native_tensors.items():
            state = {name: torch.empty(description.size, dtype=description.properties.dtype)}
            load(state, storage_reader=reader, no_dist=True)
            for hf_name, actual in convert_to_hf(args, "moss_tts_prism", name, state[name]):
                assert hf_name not in seen, f"duplicate physical parameter publication: {hf_name}"
                assert hf_name in index, f"unexpected published tensor: {hf_name}"
                expected = shards[index[hf_name]].get_tensor(hf_name)
                assert actual.shape == expected.shape and actual.dtype == expected.dtype, hf_name
                assert torch.equal(actual, expected), f"weight layout/value mismatch: {name} -> {hf_name}"
                seen.add(hf_name)
    assert seen == set(index), f"missing published tensors: {sorted(set(index) - seen)}"
