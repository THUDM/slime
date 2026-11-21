"""Disk publication/restart contracts without Megatron, CUDA, or a serving process.

The GPU recovery suite additionally exercises SGLang's actual pull/reload path.
Here the filesystem and tensor encoders are real; only collectives and RPCs are
replaced so CI can cover lost replies and restored weights on a CPU runner.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import safetensors.numpy
import safetensors.torch
import torch
import zstandard

NUM_GPUS = 0


@pytest.fixture
def delta_module(monkeypatch):
    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    base = ModuleType(module_name)

    class Updater:
        def __init__(self, args, *a, **kw):
            self.args = args
            self.weight_version = 0

    base.UpdateWeightFromDistributed = Updater
    monkeypatch.setitem(sys.modules, module_name, base)
    megatron, core = ModuleType("megatron"), ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_data_parallel_rank=lambda **kw: 0, get_tensor_model_parallel_rank=lambda: 0)
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    name = "slime.backends.megatron_utils.update_weight._delta_recovery_test"
    path = (
        Path(__file__).resolve().parents[1]
        / "slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(module.dist, "barrier", lambda **kw: None)
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *a, **kw: None)
    monkeypatch.setattr(module.dist, "all_gather_object", lambda out, value, **kw: out.__setitem__(0, value))
    # A CPU regression must not allocate a CUDA context even on a GPU runner.
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("CPU test")
        return empty(*args, **kwargs)

    monkeypatch.setattr(module.torch, "empty", cpu_empty)
    return module


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_restart_publishes_restored_weights_above_every_attempt(tmp_path, delta_module, encoding):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"test"}')
    safetensors.torch.save_file({"w": torch.zeros(4)}, base / "model.safetensors")
    source = tmp_path / "versions"
    source.mkdir()
    # Version 7 reached a serving host before the old trainer lost its reply.
    (source / "weight_v000007").mkdir()
    args = SimpleNamespace(
        hf_checkpoint=str(base),
        update_weight_disk_dir=str(source),
        update_weight_delta_encoding=encoding,
        update_weight_delta_checksum="adler32",
        custom_update_weight_post_write_path=None,
    )
    current = {"w": torch.arange(4, dtype=torch.float32) + 5}
    applied = {}
    versions = []

    def make_updater():
        updater = delta_module.UpdateWeightFromDiskDelta(
            args, [], lambda: current, model_name="test", quantization_config=None
        )
        updater.connect_rollout_engines(["engine"], None)
        updater._iter_non_expert_chunks = lambda: iter([list(current.items())])
        updater._iter_expert_chunks = lambda: iter([])
        updater._record_metrics = lambda: None

        def reload():
            directory = Path(updater._version_dir)
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            versions.append(updater.weight_version)
            if "delta_encoding" not in index["metadata"]:
                applied.clear()
                for filename in set(index["weight_map"].values()):
                    for name, value in safetensors.torch.load_file(directory / filename).items():
                        applied[name] = value.view(torch.uint8).numpy().reshape(-1).copy()
            else:
                assert int(index["metadata"]["base_version"]) == versions[-2]
                for filename in set(index["weight_map"].values()):
                    for name, value in safetensors.numpy.load_file(directory / filename).items():
                        delta = np.frombuffer(zstandard.ZstdDecompressor().decompress(value), dtype=np.uint8)
                        if encoding == "xor":
                            applied[name] ^= delta
                        else:
                            count = int(np.frombuffer(delta[:4], dtype="<u4")[0])
                            positions = np.frombuffer(delta[4 : 4 + count * 4], dtype="<u4")
                            applied[name][positions] = delta[4 + count * 4 :]
            np.testing.assert_array_equal(applied["w"], current["w"].view(torch.uint8).numpy())

        updater._reload_engines = reload
        return updater

    updater = make_updater()
    updater.update_weights()
    assert versions == [8]  # Publish actual restored weights, not the original HF baseline.
    current["w"].add_(3)
    updater.update_weights()
    assert versions == [8, 9]
    current["w"].fill_(2)  # Optimizer/model rollback on a replacement trainer.
    replacement = make_updater()
    original_reload = replacement._reload_engines

    def lose_reply():
        original_reload()
        raise TimeoutError("GPU reload reply lost after local pull")

    replacement._reload_engines = lose_reply
    with pytest.raises(TimeoutError):
        replacement.update_weights()
    replacement._reload_engines = original_reload
    replacement.update_weights()
    current["w"].add_(1)
    replacement.update_weights()
    assert versions == [8, 9, 10, 11, 12]
    assert (source / "weight_v000007").exists()
    assert (source / "weight_v000008/config.json").read_text() == (base / "config.json").read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
