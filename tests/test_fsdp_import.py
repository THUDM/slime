import pytest


def test_fsdp_import():
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        pytest.skip("FSDP not available in this environment")
    assert FSDP is not None
