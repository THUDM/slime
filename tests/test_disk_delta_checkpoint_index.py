"""Disk delta must update the same checkpoint shards the model loader reads."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import safetensors.numpy
import zstandard

from slime.utils.disk_delta import checksum, overwrite_encode

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PATCHES = sorted((ROOT / "docker" / "patch").glob("*/sglang-pull_weights.patch"))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "hf_checkpoint_reader", ROOT / "slime/backends/megatron_utils/hf_to_megatron/common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_receiver(patch):
    # Test the exact standalone module shipped to SGLang in each patch bundle.
    text = patch.read_text()
    section = text.split("+++ b/python/sglang/srt/weight_sync/local_checkpoint.py\n", 1)[1]
    section = section.split("\ndiff --git ", 1)[0]
    source = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))
    module = ModuleType("local_checkpoint")
    exec(compile(source, str(patch), "exec"), module.__dict__)
    return module


def _write_checkpoint(path):
    path.mkdir()
    safetensors.numpy.save_file({"weight": np.ones((2, 2), dtype=np.float32)}, path / "model.safetensors")
    safetensors.numpy.save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, path / "stale.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))


@pytest.fixture(params=[None, *PATCHES], ids=["trainer", *(p.parent.name for p in PATCHES)])
def reader(request, monkeypatch):
    if request.param is None:
        module = _load_trainer()
        return lambda path: module.SafetensorReader(path).get_tensor("weight").numpy()
    module = _load_receiver(request.param)
    # scandir/glob order is unspecified; force the stale file to be visited last.
    original_glob = module.glob.glob
    monkeypatch.setattr(module.glob, "glob", lambda pattern: sorted(original_glob(pattern)))

    def read_weight(path):
        filename, _, _ = module._tensor_locations(str(path))["weight"]
        return safetensors.numpy.load_file(filename)["weight"]

    return read_weight


def test_index_excludes_stale_shards(reader, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    np.testing.assert_array_equal(reader(checkpoint), np.ones((2, 2), dtype=np.float32))


def test_missing_indexed_shard_does_not_fall_back_to_stale_weights(reader, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        reader(checkpoint)


def test_unindexed_checkpoint_remains_supported(reader, tmp_path):
    safetensors.numpy.save_file({"weight": np.ones(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    np.testing.assert_array_equal(reader(tmp_path), np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("patch", PATCHES, ids=[p.parent.name for p in PATCHES])
@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_delta_updates_indexed_weights(patch, encoding, tmp_path, monkeypatch):
    trainer, receiver = _load_trainer(), _load_receiver(patch)
    original_glob = receiver.glob.glob
    monkeypatch.setattr(receiver.glob, "glob", lambda pattern: sorted(original_glob(pattern)))
    base, local, stream = tmp_path / "base", tmp_path / "local", tmp_path / "stream"
    _write_checkpoint(base)
    stream.mkdir()
    receiver.pull(str(local), str(base), str(stream), 0)
    old = trainer.SafetensorReader(base).get_tensor("weight").numpy().view(np.uint8).reshape(-1)
    new = np.full((2, 2), 2, dtype=np.float32).view(np.uint8).reshape(-1)
    diff = new ^ old if encoding == "xor" else overwrite_encode(new, new != old)
    version = stream / "weight_v000001"
    version.mkdir()
    compressed = np.frombuffer(zstandard.ZstdCompressor().compress(diff), dtype=np.uint8)
    safetensors.numpy.save_file(
        {"weight": compressed}, version / "model.safetensors", metadata={"weight": checksum("adler32", new)}
    )
    (version / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": encoding,
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {"weight": "model.safetensors"},
            }
        )
    )
    receiver.pull(str(local), str(base), str(stream), 1)
    # The delta's own checksum can pass while the indexed model stays stale.
    loaded = safetensors.numpy.load_file(local / "model.safetensors")["weight"]
    np.testing.assert_array_equal(loaded.view(np.uint8).reshape(-1), new)
    np.testing.assert_array_equal(safetensors.numpy.load_file(local / "stale.safetensors")["weight"], 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
