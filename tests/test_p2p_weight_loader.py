import base64
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.backends.sglang_utils import p2p

NUM_GPUS = 0


def test_optional_dependency_is_only_required_when_using_mooncake(monkeypatch):
    import importlib

    monkeypatch.setitem(sys.modules, "mooncake.engine", None)
    importlib.reload(p2p)
    # Ordinary imports and control-plane serialization need no optional package.
    p2p.encode_manifest(manifest())
    from slime.weight_sync.p2p.transport import create_transport

    with pytest.raises(ImportError, match="requires Mooncake Transfer Engine"):
        create_transport("127.0.0.1", device="cpu")


def manifest():
    return {
        "endpoints": [{"host": "127.0.0.1", "port": 12345}],
        "version": "7",
        "consumer": "0:0",
        "timeout": 2,
    }


def test_manifest_is_self_contained_across_processes(monkeypatch):
    import subprocess

    # A training library may replace the storage reducer with its own loader.
    monkeypatch.setattr(torch.storage, "_load_from_bytes", lambda data: None)
    payload = p2p.encode_manifest(manifest())
    script = (
        "import base64,json,pickle,sys; "
        "weights=pickle.loads(base64.b64decode(sys.stdin.read())); "
        "assert weights[0][0]=='__p2p_manifest__'; "
        "assert json.loads(bytes(weights[0][1].tolist()))['version']=='7'"
    )
    subprocess.run([sys.executable, "-c", script], input=payload, text=True, check=True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_native_loader_can_retain_tensors_across_receive_buffers(monkeypatch, duplicate):
    closed = []

    class Receiver:
        def __init__(self, endpoints, **kwargs):
            assert kwargs["version"] == "7" and kwargs["consumer"] == "0:0"

        def __enter__(self):
            return self

        def __iter__(self):
            buffer = torch.ones(4)
            yield [("q", buffer)]
            buffer.fill_(2)
            yield [("q" if duplicate else "kv", buffer)]
            buffer.fill_(-1)

        def __exit__(self, *exc):
            closed.append(True)

    from slime.weight_sync import p2p as transport

    monkeypatch.setattr(transport, "Receiver", Receiver)
    monkeypatch.setattr(
        transport, "BufferPool", lambda host, device: SimpleNamespace(trim=lambda: closed.append("trim"))
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: closed.append("empty_cache"))
    retained = {}
    calls = []

    def load_weights(weights):
        calls.append(True)
        # Fused projection loaders cache tensors until their pair arrives.
        retained.update(weights)

    weights = pickle.loads(base64.b64decode(p2p.encode_manifest(manifest())))
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate P2P weight"):
            p2p.load_weights(SimpleNamespace(load_weights=load_weights), weights)
    else:
        p2p.load_weights(SimpleNamespace(load_weights=load_weights), weights)
        assert torch.equal(retained["q"], torch.ones(4))
        assert torch.equal(retained["kv"], torch.full((4,), 2.0))
    assert calls == [True] and closed == [True, "trim", "empty_cache"]


def test_owned_bucket_preserves_mixed_dtypes_offsets_and_empty_shapes():
    packed = torch.zeros(192, dtype=torch.uint8)
    matrix = packed[:24].view(torch.float32).view(2, 3)
    matrix.copy_(torch.arange(6).reshape(2, 3))
    bf16 = packed[64:70].view(torch.bfloat16)
    bf16.fill_(0.5)
    scalar = packed[128:136].view(torch.int64).view(())
    scalar.fill_(42)
    empty = packed[192:192].view(torch.float32).view(0, 3)
    batch = [("matrix", matrix), ("bf16", bf16), ("scalar", scalar), ("empty", empty)]
    expected = {name: tensor.clone() for name, tensor in batch}
    owned = dict(p2p._clone_bucket(batch))
    packed.zero_()
    for name, tensor in owned.items():
        assert tensor.dtype == expected[name].dtype
        assert tensor.shape == expected[name].shape
        assert torch.equal(tensor, expected[name])


@pytest.mark.parametrize("config", [{"pp_size": 2}, {"dp_size": 2}])
def test_serving_layout_rejected_before_loading(config):
    with pytest.raises(ValueError, match="P2P"):
        p2p.validate_server_args(config)


def test_dp_attention_layout_is_accepted():
    p2p.validate_server_args({"pp_size": 1, "dp_size": 2, "enable_dp_attention": True})


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("interval", [0, 1, 2])
def test_pd_requires_draining_and_flushing_each_update(role, interval):
    config = {"disaggregation_mode": role, "disaggregation_transfer_backend": "mooncake"}
    if interval == 1:
        p2p.validate_server_args(config, flush_cache_interval=interval)
    else:
        with pytest.raises(ValueError, match="PD requires --flush-cache-interval=1"):
            p2p.validate_server_args(config, flush_cache_interval=interval)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
