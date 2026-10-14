"""CPU tests for the small helpers in megatron_utils/misc_utils.py."""

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

from slime.backends.megatron_utils.hf_to_megatron.common import _tensor_parallel_shard
from slime.backends.megatron_utils.misc_utils import collate_multimodal_train_inputs, has_gated_linear_unit

NUM_GPUS = 0


# ----------------------------------------------------------- has_gated_linear_unit


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["gated_linear_unit", "swiglu", "quick_geglu"])
def test_gated_activations_are_detected(flag):
    assert has_gated_linear_unit(types.SimpleNamespace(**{flag: True})) is True


@pytest.mark.unit
def test_squared_relu_and_bare_namespaces_are_ungated():
    # Nemotron: --squared-relu, no gate half in linear_fc1.
    assert has_gated_linear_unit(types.SimpleNamespace(squared_relu=True, swiglu=False)) is False
    assert has_gated_linear_unit(types.SimpleNamespace()) is False


@pytest.mark.unit
def test_gated_flag_decides_how_fc1_is_sharded_across_tp():
    fc1 = torch.arange(8 * 3).reshape(8, 3)
    name = "module.module.decoder.layers.0.mlp.linear_fc1.weight"
    shard = dict(parallel_size=2, parallel_rank=1, partition_dim=0, partition_stride=1)

    # Gated: rows are cat(gate, up); each rank takes its slice of BOTH halves.
    gated = _tensor_parallel_shard(name, fc1, gated_mlp=True, **shard)
    assert torch.equal(gated, torch.cat((fc1[2:4], fc1[6:8])))

    # Ungated: one projection, a plain contiguous block. Same shape, different
    # rows -- the caller's shape check cannot tell the two apart.
    ungated = _tensor_parallel_shard(name, fc1, gated_mlp=False, **shard)
    assert torch.equal(ungated, fc1[4:8])
    assert gated.shape == ungated.shape


def _gather_like_the_weight_sync(shards, gated):
    """What update_weight/common.py does with the per-rank shards of linear_fc1."""
    partitions = list(shards)
    if gated:
        partitions = [p.chunk(2, dim=0) for p in partitions]
        partitions = [p[0] for p in partitions] + [p[1] for p in partitions]
    return torch.cat(partitions, dim=0)


@pytest.mark.unit
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("tp", [2, 4])
def test_load_shard_and_weight_sync_gather_are_inverses(gated, tp):
    fc1 = torch.arange(16 * 3).reshape(16, 3)
    name = "module.module.decoder.layers.0.mlp.linear_fc1.weight"
    shards = [
        _tensor_parallel_shard(
            name, fc1, parallel_size=tp, parallel_rank=rank, partition_dim=0, partition_stride=1, gated_mlp=gated
        )
        for rank in range(tp)
    ]
    assert torch.equal(_gather_like_the_weight_sync(shards, gated), fc1)


@pytest.mark.unit
def test_fixing_only_one_half_of_the_round_trip_permutes_rows():
    fc1 = torch.arange(16 * 3).reshape(16, 3)
    name = "module.module.decoder.layers.0.mlp.linear_fc1.weight"
    shards = [
        _tensor_parallel_shard(
            name, fc1, parallel_size=2, parallel_rank=rank, partition_dim=0, partition_stride=1, gated_mlp=False
        )
        for rank in range(2)
    ]
    wrong = _gather_like_the_weight_sync(shards, gated=True)
    assert wrong.shape == fc1.shape
    assert not torch.equal(wrong, fc1)


# --------------------------------------------- collate_multimodal_train_inputs


@pytest.mark.unit
def test_ragged_patch_tensors_concatenate_along_dim0():
    # Qwen-VL style: (total_patches, patch_dim); only dim 0 varies.
    a, b = torch.randn(6, 4), torch.randn(10, 4)
    out = collate_multimodal_train_inputs([{"pixel_values": a}, {"pixel_values": b}])
    assert torch.equal(out["pixel_values"], torch.cat([a, b]))


@pytest.mark.unit
def test_images_of_different_resolution_are_passed_through_as_a_list():
    # Nemotron style: (tiles, 3, H, W) with H, W chosen per image.
    a, b = torch.randn(1, 3, 416, 32), torch.randn(1, 3, 384, 32)
    out = collate_multimodal_train_inputs([{"pixel_values": a}, {"pixel_values": b}])
    assert isinstance(out["pixel_values"], list)
    assert out["pixel_values"][0] is a and out["pixel_values"][1] is b


@pytest.mark.unit
def test_equal_resolution_images_still_concatenate():
    a, b = torch.randn(1, 3, 16, 16), torch.randn(2, 3, 16, 16)
    out = collate_multimodal_train_inputs([{"pixel_values": a}, {"pixel_values": b}])
    assert torch.equal(out["pixel_values"], torch.cat([a, b]))


@pytest.mark.unit
def test_keys_are_decided_independently():
    sample_a = {"pixel_values": torch.randn(1, 3, 8, 12), "imgs_sizes": torch.tensor([[8, 12]])}
    sample_b = {"pixel_values": torch.randn(1, 3, 10, 12), "imgs_sizes": torch.tensor([[10, 12]])}
    out = collate_multimodal_train_inputs([sample_a, sample_b])
    assert isinstance(out["pixel_values"], list)
    assert torch.equal(out["imgs_sizes"], torch.tensor([[8, 12], [10, 12]]))


@pytest.mark.unit
def test_none_entries_are_skipped_and_a_lone_tensor_passes_through():
    tensor = torch.randn(3, 4)
    out = collate_multimodal_train_inputs([None, {"pixel_values": tensor}, None])
    assert out["pixel_values"] is tensor
    assert collate_multimodal_train_inputs([None, None]) == {}
