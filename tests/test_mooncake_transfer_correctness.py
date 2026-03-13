"""
Test Mooncake hybrid rollout transfer correctness.

Verifies that data put via MooncakeHybridRolloutTransfer.put_rollout() and
retrieved via get_rollout() matches the original (deep equal).

Prerequisites:
- mooncake_master running (e.g. 127.0.0.1:50051)
- mooncake_client running with MC_STORE_LOCAL_HOT_CACHE_USE_SHM=1 (optional, for hot cache)
- MOONCAKE_MASTER, MOONCAKE_TE_META_DATA_SERVER env set (or defaults to 127.0.0.1)

Run: pytest tests/test_mooncake_transfer_correctness.py -v
Or:  python tests/test_mooncake_transfer_correctness.py
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slime.utils.mock_rollout import make_mock_rollout_data
from slime.utils.rollout_hybrid_transfer import MooncakeHybridRolloutTransfer


def _assert_equal_recursive(orig, new, path: str = "root") -> None:
    """Recursively assert orig == new with helpful error messages."""
    if isinstance(orig, dict):
        assert isinstance(new, dict), f"Type mismatch at {path}: expected dict, got {type(new)}"
        assert set(orig.keys()) == set(new.keys()), f"Keys mismatch at {path}"
        for k in orig:
            _assert_equal_recursive(orig[k], new[k], f"{path}['{k}']")
    elif isinstance(orig, list):
        assert isinstance(new, list), f"Type mismatch at {path}: expected list, got {type(new)}"
        assert len(orig) == len(new), f"Length mismatch at {path}: {len(orig)} vs {len(new)}"
        for i, (o, n) in enumerate(zip(orig, new)):
            _assert_equal_recursive(o, n, f"{path}[{i}]")
    elif isinstance(orig, tuple):
        assert isinstance(new, tuple), f"Type mismatch at {path}: expected tuple, got {type(new)}"
        assert len(orig) == len(new), f"Length mismatch at {path}"
        for i, (o, n) in enumerate(zip(orig, new)):
            _assert_equal_recursive(o, n, f"{path}[{i}]")
    elif isinstance(orig, np.ndarray):
        assert isinstance(new, np.ndarray), f"Type mismatch at {path}: expected ndarray, got {type(new)}"
        np.testing.assert_array_equal(orig, new, err_msg=f"Array mismatch at {path}")
    elif isinstance(orig, torch.Tensor):
        assert isinstance(new, torch.Tensor), f"Type mismatch at {path}: expected Tensor, got {type(new)}"
        torch.testing.assert_close(orig, new, msg=f"Tensor mismatch at {path}")
    else:
        assert type(orig) == type(new), f"Type mismatch at {path}: {type(orig)} vs {type(new)}"
        assert orig == new, f"Value mismatch at {path}: {orig} vs {new}"


def test_mooncake_hybrid_rollout_transfer_correctness():
    """Verify Mooncake put_rollout/get_rollout roundtrip preserves data."""
    os.environ.setdefault("SLIME_UNSAFE_PICKLE", "1")
    os.environ.setdefault("MOONCAKE_PROTOCOL", "tcp")
    os.environ.setdefault("MOONCAKE_MASTER", "127.0.0.1:50051")
    os.environ.setdefault("MOONCAKE_TE_META_DATA_SERVER", "http://127.0.0.1:8080/metadata")
    os.environ.setdefault("MC_STORE_MEMCPY", "1")

    data = make_mock_rollout_data(batch_size=8, seq_len=512, use_routing_replay=True)
    xfer = MooncakeHybridRolloutTransfer(tensor_min_bytes=1024 * 1024, enable_auto_cleanup=False)

    handle = xfer.put_rollout(data)
    received = xfer.get_rollout(handle, return_packed=False)

    _assert_equal_recursive(data, received)
    xfer.cleanup(handle)


if __name__ == "__main__":
    test_mooncake_hybrid_rollout_transfer_correctness()
    print("OK: Mooncake transfer correctness test passed.")
