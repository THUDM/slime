"""Unit tests for measured-host rollout engine address allocation and layout validation."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.sglang_utils import addr_allocator as allocator_mod  # noqa: E402
from slime.backends.sglang_utils.addr_allocator import (  # noqa: E402
    _allocate_rollout_engine_addr_and_ports_normal,
    assert_group_layout_valid,
)


class FakeEngine:
    """Mimics a Ray actor pinned on ``host``; port probing returns start_port verbatim."""

    def __init__(self, host):
        self._host = host

    @property
    def _get_current_node_ip_and_free_port(self):
        return self

    def remote(self, start_port=30000, consecutive=1):
        return self._host, start_port


@pytest.fixture(autouse=True)
def _patch_ray_get(monkeypatch):
    monkeypatch.setattr(allocator_mod.ray, "get", lambda x: x)


def _args(**overrides):
    ns = Namespace(num_gpus_per_node=8, rollout_num_gpus_per_engine=4, sglang_dp_size=1)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _alloc(hosts, **kwargs):
    engines = [(rank, FakeEngine(host)) for rank, host in enumerate(hosts)]
    return _allocate_rollout_engine_addr_and_ports_normal(args=_args(), rollout_engines=engines, **kwargs)


class _GroupConfig:
    def __init__(self, worker_type="regular", num_gpus=12):
        self.worker_type = worker_type
        self.num_gpus = num_gpus


def _validate(gpus_per_engine, group_abs_start, num_gpus=None, num_gpus_per_node=8):
    assert_group_layout_valid(
        group_config=_GroupConfig(num_gpus=num_gpus or 12),
        gpus_per_engine=gpus_per_engine,
        group_abs_start=group_abs_start,
        num_gpus_per_node=num_gpus_per_node,
    )


class TestMeasuredHostAllocation:
    def test_partial_node_prefix_uses_each_engines_actual_host(self):
        """actor=4 + rollout=12 on two 8-GPU nodes: engine1 sits on node1 even
        though ``local_rank // (8 // per_engine)`` would place it on node0."""
        addr_and_ports, _ = _alloc(["10.0.0.0", "10.0.0.1", "10.0.0.1"])

        assert addr_and_ports[0]["host"] == "10.0.0.0"
        assert addr_and_ports[1]["host"] == "10.0.0.1"
        assert addr_and_ports[2]["host"] == "10.0.0.1"
        for rank in range(3):
            assert addr_and_ports[rank]["node_rank"] == 0
            assert addr_and_ports[rank]["dist_init_addr"].startswith(addr_and_ports[rank]["host"])

    def test_ports_are_probed_on_the_engines_own_host(self):
        addr_and_ports, cursors = _alloc(["h0", "h0", "h1"], base_port=15000)

        # Engines sharing a host get distinct consecutive ports; engines on
        # different hosts may reuse the same port range.
        ports_h0 = [(addr_and_ports[0][k], addr_and_ports[1][k]) for k in ("port", "nccl_port")]
        assert all(a != b for a, b in ports_h0)
        assert set(cursors) == {"h0", "h1"}

    def test_prefill_group_allocates_bootstrap_port_per_engine(self):
        addr_and_ports, _ = _alloc(["h0", "h1"], worker_type="prefill")
        assert "disaggregation_bootstrap_port" in addr_and_ports[0]
        assert "disaggregation_bootstrap_port" in addr_and_ports[1]

    def test_multi_node_engine_shares_dist_init_and_group_relative_node_rank(self):
        """per_engine=16 on 8-GPU nodes: shards of one engine share the first
        shard's dist_init_addr, and node_rank is relative to this group (not
        the global rank) so it still aligns after another group."""
        engines = [(5 + i, FakeEngine(host)) for i, host in enumerate(["h0", "h1", "h2", "h3"])]
        addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
            args=_args(),
            rollout_engines=engines,
            num_gpus_per_engine=16,
            rank_offset=5,
        )

        assert addr_and_ports[5]["dist_init_addr"] == addr_and_ports[6]["dist_init_addr"]
        assert addr_and_ports[5]["dist_init_addr"].startswith("h0:")
        assert addr_and_ports[7]["dist_init_addr"] == addr_and_ports[8]["dist_init_addr"]
        assert addr_and_ports[7]["dist_init_addr"].startswith("h2:")
        assert [addr_and_ports[r]["node_rank"] for r in (5, 6, 7, 8)] == [0, 1, 0, 1]


class TestLayoutValidation:
    def test_aligned_partial_node_prefix_is_accepted(self):
        _validate(gpus_per_engine=4, group_abs_start=4)

    def test_layout_starting_at_node_boundary_is_accepted(self):
        _validate(gpus_per_engine=4, group_abs_start=0)
        _validate(gpus_per_engine=16, group_abs_start=8, num_gpus=16)

    def test_engine_straddling_node_boundary_is_rejected(self):
        with pytest.raises(AssertionError, match="straddle a node boundary"):
            _validate(gpus_per_engine=6, group_abs_start=4)

    def test_multi_node_shards_must_start_at_a_node_boundary(self):
        with pytest.raises(AssertionError, match="straddle a node boundary"):
            _validate(gpus_per_engine=16, group_abs_start=4, num_gpus=16)

    def test_engine_size_must_fit_or_tile_nodes(self):
        with pytest.raises(AssertionError, match="tiles whole nodes"):
            _validate(gpus_per_engine=12, group_abs_start=0)

    def test_group_gpus_must_divide_per_engine(self):
        with pytest.raises(AssertionError, match="not divisible"):
            _validate(gpus_per_engine=4, group_abs_start=0, num_gpus=10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
