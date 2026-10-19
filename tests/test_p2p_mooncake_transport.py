import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.p2p.mooncake import MooncakeTransport

NUM_GPUS = 0


def test_reset_recreates_engine_without_rediscovering_gpu_affinity(monkeypatch):
    engines, discoveries = [], []

    class Engine:
        def __init__(self):
            self.port = 1234 + len(engines)
            engines.append(self)

        def get_local_topology(self, devices):
            discoveries.append(devices)
            # Logical cuda:0 can be physical GPU 7 in a restricted process.
            return json.dumps({"cuda:0": [["mlx5_1"], ["mlx5_0"]]})

        def initialize(self, host, metadata, protocol, devices):
            assert (host, metadata, protocol, devices) == ("host", "P2PHANDSHAKE", "rdma", "mlx5_1")
            return 0

        def get_rpc_port(self):
            return self.port

    monkeypatch.setitem(sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=Engine))
    transport = MooncakeTransport("host", device="cuda:0", device_names="mlx5_0,mlx5_1")
    first_session = transport.session
    transport.reset()
    assert transport._engine is None
    assert transport.session != first_session
    assert discoveries == ["mlx5_0,mlx5_1"]
    assert len(engines) == 2


def test_missing_rdma_devices_fail_before_native_initialize(monkeypatch):
    class Engine:
        def get_local_topology(self, devices):
            return json.dumps({"cuda:0": [[], []], "cpu:0": [[], []]})

        def initialize(self, *args):
            pytest.fail("must not silently fall back to a different data plane")

    monkeypatch.setitem(sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=Engine))
    with pytest.raises(RuntimeError, match="no usable RDMA devices"):
        MooncakeTransport("host", device="cuda:0")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
