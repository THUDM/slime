import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from slime.backends.megatron_utils import hf_checkpoint_saver
from slime.backends.megatron_utils.hf_checkpoint_saver import (
    _clear_existing_hf_weights,
    _copy_hf_assets,
    _finalize_local_shards,
    _raise_if_writer_misses_assets,
    _SafetensorShardWriter,
    _write_pending_chunk,
    save_hf_model_to_path,
)

NUM_GPUS = 0


def test_copy_hf_assets_keeps_quantized_config_and_skips_weights(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    config = {"model_type": "tiny", "quantization_config": {"quant_method": "fp8"}}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (src / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (src / "pytorch_model.bin").write_bytes(b"weight")

    _copy_hf_assets(str(src), dst)

    assert json.loads((dst / "config.json").read_text(encoding="utf-8")) == config
    assert (dst / "tokenizer.json").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    assert not (dst / "model-00001-of-00001.safetensors").exists()
    assert not (dst / "pytorch_model.bin").exists()


def test_clear_existing_hf_weights_removes_old_weight_files_only(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weight")

    _clear_existing_hf_weights(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert not (tmp_path / "model-00001-of-00001.safetensors").exists()
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_save_hf_model_to_path_rejects_origin_checkpoint(tmp_path: Path):
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))

    with pytest.raises(ValueError, match="same directory as --hf-checkpoint"):
        save_hf_model_to_path(args, tmp_path, model=None)


def test_safetensor_shard_writer_writes_hf_index(tmp_path: Path):
    writer = _SafetensorShardWriter(tmp_path, enabled=True)
    writer.write([("layers.0.weight", torch.ones(2, 2)), ("layers.0.weight_scale", torch.ones(1))], shard_idx=0)
    writer.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)
    writer.finalize()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 36
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.0.weight_scale": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_finalize_shard_files_merges_node_writer_states(tmp_path: Path):
    writer0 = _SafetensorShardWriter(tmp_path, enabled=True)
    writer1 = _SafetensorShardWriter(tmp_path, enabled=True)

    writer0.write([("layers.0.weight", torch.ones(2, 2))], shard_idx=0)
    writer1.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)

    # each rank renames its own files off the shared plan; rank 0 writes the index
    states = [writer0.state(), writer1.state()]
    for rank, state in enumerate(states):
        _finalize_local_shards(tmp_path, state, states, write_index=rank == 0)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 32
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }
    assert not (tmp_path / "model-00001.safetensors").exists()
    assert not (tmp_path / "model-00002.safetensors").exists()

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_pending_chunk_write_flushes_incomplete_node_group(tmp_path: Path):
    num_nodes = 3
    writers = [_SafetensorShardWriter(tmp_path, enabled=True) for _ in range(num_nodes)]
    pending_writes = [None] * num_nodes

    for chunk_idx in range(5):
        node_rank = chunk_idx % num_nodes
        pending_writes[node_rank] = (
            chunk_idx,
            [(f"layers.{chunk_idx}.weight", torch.full((1,), chunk_idx, dtype=torch.float32))],
        )

        if (chunk_idx + 1) % num_nodes == 0:
            for i, writer in enumerate(writers):
                pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    for i, writer in enumerate(writers):
        pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    states = [writer.state() for writer in writers]
    for rank, state in enumerate(states):
        _finalize_local_shards(tmp_path, state, states, write_index=rank == 0)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {f"layers.{i}.weight": f"model-{i + 1:05d}-of-00005.safetensors" for i in range(5)}


def test_writer_that_cannot_see_rank_zero_assets_fails(tmp_path: Path):
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "config.json").write_text("{}")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "config.json").write_text("{}")
    node_local = tmp_path / "node_local"
    node_local.mkdir()

    _raise_if_writer_misses_assets(str(origin), shared, is_writer_rank=True)
    _raise_if_writer_misses_assets(str(origin), node_local, is_writer_rank=False)
    with pytest.raises(RuntimeError, match="shared by every node"):
        _raise_if_writer_misses_assets(str(origin), node_local, is_writer_rank=True)


class _TwoChunkIterator:
    """Stand-in for HfWeightIteratorDirect that yields two small chunks without Megatron."""

    def __init__(self, **kwargs):
        pass

    def get_hf_weight_chunks(self, weights, progress_desc, should_convert_chunk):
        """Yield one tensor per chunk, converted only where the writer layout asks for it."""
        for idx in range(2):
            yield [(f"layers.{idx}.weight", torch.full((4,), float(idx)))] if should_convert_chunk(idx) else []


@pytest.fixture
def writer_without_rank_zero_assets(tmp_path: Path, monkeypatch):
    """Return save args whose writer cannot see the assets rank 0 copies, as on a filesystem that shows files to
    other nodes only after publishing them; conversion runs without Megatron."""
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "config.json").write_text('{"model_type": "qwen2"}')
    common = ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.named_params_and_buffers = lambda args, model: []
    direct = ModuleType("slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct")
    direct.HfWeightIteratorDirect = _TwoChunkIterator
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, direct.__name__, direct)
    monkeypatch.setattr(hf_checkpoint_saver, "_copy_hf_assets", lambda origin_dir, output_dir: None)
    return SimpleNamespace(
        hf_checkpoint=str(origin),
        model_name=None,
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        update_weight_disk_dir=str(tmp_path / "updates"),
        custom_update_weight_post_write_path=None,
    )


def test_hf_export_fails_when_a_writer_cannot_see_rank_zero_assets(tmp_path: Path, writer_without_rank_zero_assets):
    with pytest.raises(RuntimeError, match="shared by every node"):
        save_hf_model_to_path(writer_without_rank_zero_assets, tmp_path / "export", model=None, model_name="qwen2")


def test_disk_weight_sync_skips_the_shared_directory_check(
    tmp_path: Path, monkeypatch, writer_without_rank_zero_assets
):
    from slime.backends.megatron_utils.update_weight import update_weight_from_disk

    monkeypatch.setattr(update_weight_from_disk.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_weight_from_disk.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(update_weight_from_disk, "get_gloo_group", lambda: None)
    updater = update_weight_from_disk.UpdateWeightFromDisk(
        writer_without_rank_zero_assets, [], lambda: {}, model_name="qwen2", quantization_config=None
    )

    updater.update_weights()

    index = json.loads((tmp_path / "updates" / "weight_v000001" / "model.safetensors.index.json").read_text())
    assert sorted(index["weight_map"]) == ["layers.0.weight", "layers.1.weight"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
