"""Disk delta must update the same checkpoint shards the model loader reads."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import safetensors.numpy
import zstandard

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PATCHES = sorted((ROOT / "docker" / "patch").glob("*/sglang-pull_weights.patch"))


def _load_trainer():
    spec = importlib.util.spec_from_file_location("disk_delta", ROOT / "slime/utils/disk_delta.py")
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
    module = _load_trainer() if request.param is None else _load_receiver(request.param)
    # scandir/glob order is unspecified; force the stale file to be visited last.
    original_glob = module.glob.glob
    monkeypatch.setattr(module.glob, "glob", lambda pattern: sorted(original_glob(pattern)))
    return module


def test_index_excludes_stale_shards(reader, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    locations = reader._tensor_locations(str(checkpoint))
    assert Path(locations["weight"][0]).name == "model.safetensors"


def test_missing_indexed_shard_does_not_fall_back_to_stale_weights(reader, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        reader._tensor_locations(str(checkpoint))


def test_unindexed_checkpoint_remains_supported(reader, tmp_path):
    safetensors.numpy.save_file({"weight": np.ones(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    assert set(reader._tensor_locations(str(tmp_path))) == {"weight"}


@pytest.mark.parametrize("patch", PATCHES, ids=[p.parent.name for p in PATCHES])
@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_delta_updates_indexed_weights(patch, encoding, tmp_path, monkeypatch):
    trainer, receiver = _load_trainer(), _load_receiver(patch)
    original_glob = trainer.glob.glob
    monkeypatch.setattr(trainer.glob, "glob", lambda pattern: sorted(original_glob(pattern)))
    base, local, stream = tmp_path / "base", tmp_path / "local", tmp_path / "stream"
    _write_checkpoint(base)
    stream.mkdir()
    receiver.pull(str(local), str(base), str(stream), 0)
    read = trainer.make_tensor_reader(str(base))
    old = read("weight")
    new = np.full((2, 2), 2, dtype=np.float32).view(np.uint8).reshape(-1)
    diff = new ^ old if encoding == "xor" else trainer.overwrite_encode(new, new != old)
    version = stream / "weight_v000001"
    version.mkdir()
    compressed = np.frombuffer(zstandard.ZstdCompressor().compress(diff), dtype=np.uint8)
    safetensors.numpy.save_file(
        {"weight": compressed}, version / "model.safetensors", metadata={"weight": trainer.checksum("adler32", new)}
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
