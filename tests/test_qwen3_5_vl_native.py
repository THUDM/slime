import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

# Keep this CPU test independent of the Megatron runtime package. The tested
# mapping and packed-sequence helpers themselves only depend on torch.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules["slime.backends.megatron_utils"] = _megatron_utils

from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader, _tensor_parallel_shard
from slime.backends.megatron_utils.hf_to_megatron.qwen3_5 import qwen3_5_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf
from slime_plugins.models.qwen3_5_vl_utils import build_packed_mrope_position_ids, get_packed_cp_local_indices

NUM_GPUS = 0


@pytest.mark.unit
def test_packed_mrope_resets_positions_for_each_sample():
    input_ids = torch.tensor(
        [[99, 10, 10, 10, 10, 98, 7, 99, 10, 10, 10, 10, 98, 8]],
        dtype=torch.long,
    )
    positions = build_packed_mrope_position_ids(
        input_ids,
        [0, 7, 14],
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
        video_grid_thw=None,
        image_token_id=10,
        video_token_id=20,
        vision_start_token_id=99,
        spatial_merge_size=2,
    )

    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 4],
            [0, 1, 1, 2, 2, 3, 4],
            [0, 1, 2, 1, 2, 3, 4],
        ]
    )
    assert torch.equal(positions[:, 0, :7], expected)
    assert torch.equal(positions[:, 0, 7:], expected)


@pytest.mark.unit
def test_packed_mrope_rejects_unused_grids():
    with pytest.raises(ValueError, match="Unused .* image grids"):
        build_packed_mrope_position_ids(
            torch.tensor([[1, 2, 3]]),
            [0, 3],
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            video_grid_thw=None,
            image_token_id=10,
            video_token_id=20,
            vision_start_token_id=99,
            spatial_merge_size=2,
        )


@pytest.mark.unit
def test_thd_cp_indices_select_two_chunks_per_packed_sequence():
    rank_0 = get_packed_cp_local_indices([0, 8, 16], cp_size=2, cp_rank=0, device=torch.device("cpu"))
    rank_1 = get_packed_cp_local_indices([0, 8, 16], cp_size=2, cp_rank=1, device=torch.device("cpu"))

    assert rank_0.tolist() == [0, 1, 6, 7, 8, 9, 14, 15]
    assert rank_1.tolist() == [2, 3, 4, 5, 10, 11, 12, 13]


@pytest.mark.unit
def test_raw_qkv_loader_is_inverse_of_exporter():
    hidden_size = 3
    q = torch.arange(16 * hidden_size).reshape(16, hidden_size)
    k = torch.arange(4 * hidden_size).reshape(4, hidden_size) + 1000
    v = torch.arange(4 * hidden_size).reshape(4, hidden_size) + 2000
    tensors = {
        "model.language_model.layers.0.self_attn.q_proj.weight": q,
        "model.language_model.layers.0.self_attn.k_proj.weight": k,
        "model.language_model.layers.0.self_attn.v_proj.weight": v,
    }
    reader = types.SimpleNamespace(get_tensor=tensors.__getitem__)
    text_config = types.SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
    )
    hf_config = types.SimpleNamespace(text_config=text_config, tie_word_embeddings=False)

    mcore_qkv = qwen3_5_hf_tensor(
        "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.weight",
        reader,
        hf_config,
    )
    converted = dict(
        convert_qwen3_5_to_hf(
            types.SimpleNamespace(
                kv_channels=2,
                hidden_size=hidden_size,
                num_attention_heads=4,
                num_query_groups=2,
            ),
            "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.weight",
            mcore_qkv,
        )
    )

    assert torch.equal(converted["model.language_model.layers.0.self_attn.q_proj.weight"], q)
    assert torch.equal(converted["model.language_model.layers.0.self_attn.k_proj.weight"], k)
    assert torch.equal(converted["model.language_model.layers.0.self_attn.v_proj.weight"], v)


@pytest.mark.unit
def test_raw_loader_uses_hf_vision_name_directly():
    weight = torch.randn(4, 3)
    key = "model.visual.patch_embed.proj.weight"
    reader = types.SimpleNamespace(get_tensor={key: weight}.__getitem__)
    hf_config = types.SimpleNamespace(tie_word_embeddings=False)

    loaded = qwen3_5_hf_tensor(
        "module.module.model.visual.patch_embed.proj.weight",
        reader,
        hf_config,
    )
    assert loaded is weight


@pytest.mark.unit
def test_raw_loader_uses_individual_mtp_expert_weights():
    gate = torch.randn(4, 3)
    up = torch.randn(4, 3)
    down = torch.randn(3, 4)
    tensors = {
        "mtp.layers.0.mlp.experts.7.gate_proj.weight": gate,
        "mtp.layers.0.mlp.experts.7.up_proj.weight": up,
        "mtp.layers.0.mlp.experts.7.down_proj.weight": down,
    }
    reader = types.SimpleNamespace(get_tensor=tensors.__getitem__)
    config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(tie_word_embeddings=False),
        tie_word_embeddings=False,
    )

    fc1 = qwen3_5_hf_tensor(
        "module.module.language_model.mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight7",
        reader,
        config,
    )
    fc2 = qwen3_5_hf_tensor(
        "module.module.language_model.mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight7",
        reader,
        config,
    )

    assert torch.equal(fc1, torch.cat((gate, up)))
    assert fc2 is down


@pytest.mark.unit
def test_raw_loader_shards_swiglu_and_grouped_moe_fc2():
    fc1 = torch.arange(8 * 3).reshape(8, 3)
    fc1_shard = _tensor_parallel_shard(
        "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
        fc1,
        parallel_size=2,
        parallel_rank=1,
        partition_dim=0,
        partition_stride=1,
    )
    assert torch.equal(fc1_shard, torch.cat((fc1[2:4], fc1[6:8])))

    fc2 = torch.arange(4 * 6).reshape(4, 6)
    fc2_shard = _tensor_parallel_shard(
        "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        fc2,
        parallel_size=2,
        parallel_rank=1,
        partition_dim=0,
        partition_stride=1,
    )
    assert torch.equal(fc2_shard, fc2[:, 3:])


@pytest.mark.unit
def test_safetensor_reader_caches_only_the_last_tensor(tmp_path):
    save_file({"first": torch.ones(2), "second": torch.zeros(2)}, tmp_path / "model.safetensors")
    reader = SafetensorReader(tmp_path)

    first = reader.get_tensor("first")
    assert reader.get_tensor("first") is first
    reader.get_tensor("second")
    assert reader.get_tensor("first") is not first


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
