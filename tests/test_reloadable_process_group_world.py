from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.utils import distributed_utils
from slime.utils import reloadable_process_group as rpg

NUM_GPUS = 0


def _run_pp_group_reload_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    timeout = timedelta(seconds=30)
    rpg.monkey_patch_torch_dist()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    distributed_utils.init_gloo_group()
    rpg.register_default_process_group(timeout=timeout)

    # Exercise the NCCL lifecycle with Gloo so this remains a CPU test.  The
    # relevant contract is the global ordering of WORLD and subgroup teardown,
    # not the backend implementation.
    rpg._uses_nccl = lambda _backend: True

    group_specs = [
        ([0], "TP_0"),
        ([1], "TP_1"),
        ([2], "TP_2"),
        ([3], "TP_3"),
        ([0, 1, 2, 3], "PP"),
        ([0, 3], "EMBEDDING"),
        ([0], "POSITION_EMBEDDING"),
        ([0, 2], "DP_0"),
        ([1, 3], "DP_1"),
    ]
    groups = [
        dist.new_group(ranks=ranks, backend="gloo", timeout=timeout, group_desc=desc) for ranks, desc in group_specs
    ]
    pp_group = groups[4]

    for generation in range(2):
        rpg.destroy_process_groups()
        assert all(group.group is None for group in groups)

        rpg.reload_process_groups()
        assert all(group.group is not None for group in groups)

        value = torch.tensor(rank + 1)
        dist.all_reduce(value, group=pp_group)
        assert value.item() == sum(range(1, world_size + 1))

        state = rpg.default_process_group_states[rpg.os.getpid()]
        assert state.generation == 2 * (generation + 1)

    dist.destroy_process_group()


@pytest.mark.unit
def test_register_default_process_group_captures_rendezvous_state(monkeypatch):
    timeout = timedelta(minutes=7)
    monkeypatch.setattr(rpg, "default_process_group_states", {})
    monkeypatch.setattr(rpg.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rpg.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(rpg.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(rpg.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(rpg, "_get_default_store", lambda: "rendezvous-store")

    rpg.register_default_process_group(timeout=timeout)

    state = rpg.default_process_group_states[rpg.os.getpid()]
    assert state.backend == "nccl"
    assert state.timeout == timeout
    assert state.store == "rendezvous-store"
    assert state.rank == 3
    assert state.world_size == 8
    assert not state.nccl_world_destroyed


@pytest.mark.unit
def test_world_and_subgroups_follow_destroy_reload_order(monkeypatch):
    timeout = timedelta(minutes=2)
    state = rpg._DefaultProcessGroupState(
        backend="nccl",
        timeout=timeout,
        store="base-store",
        rank=1,
        world_size=4,
    )
    monkeypatch.setattr(rpg, "default_process_group_states", {rpg.os.getpid(): state})

    events = []

    def barrier(group=None):
        events.append(("barrier", "WORLD" if group is None else group))

    def init_process_group(**kwargs):
        events.append(("init", kwargs))

    monkeypatch.setattr(rpg.dist, "barrier", barrier)
    monkeypatch.setattr(rpg.dist, "destroy_process_group", lambda: events.append(("destroy_world",)))
    monkeypatch.setattr(rpg.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(rpg, "PrefixStore", lambda prefix, store: (prefix, store))
    monkeypatch.setattr(rpg, "get_gloo_group", lambda: "canonical-gloo")
    monkeypatch.setattr(rpg, "set_gloo_group", lambda group: events.append(("set_gloo", group)))
    monkeypatch.setattr(rpg, "_get_default_group", lambda: "cpu-world")
    monkeypatch.setattr(rpg, "init_gloo_group", lambda: events.append(("init_canonical_gloo",)))
    monkeypatch.setattr(
        rpg,
        "_store_barrier",
        lambda state, phase: events.append(("store_barrier", state.generation, phase)),
    )
    monkeypatch.setattr(
        rpg.ReloadableProcessGroup,
        "invalidate_process_groups",
        staticmethod(lambda: events.append(("invalidate_subgroups",))),
    )
    monkeypatch.setattr(
        rpg.ReloadableProcessGroup,
        "reload_process_groups",
        staticmethod(lambda: events.append(("reload_subgroups",))),
    )

    rpg.destroy_process_groups()

    assert state.nccl_world_destroyed
    assert state.generation == 1
    assert events == [
        ("barrier", "canonical-gloo"),
        ("destroy_world",),
        ("invalidate_subgroups",),
        ("set_gloo", None),
        ("store_barrier", 0, "default-world-destroyed"),
        (
            "init",
            {
                "backend": "gloo",
                "store": ("slime-reloadable-world-1-gloo", "base-store"),
                "rank": 1,
                "world_size": 4,
                "timeout": timeout,
            },
        ),
        ("set_gloo", "cpu-world"),
        ("store_barrier", 1, "temporary-world-ready"),
    ]

    events.clear()
    rpg.reload_process_groups()

    assert not state.nccl_world_destroyed
    assert state.generation == 2
    assert events == [
        ("barrier", "WORLD"),
        ("destroy_world",),
        ("set_gloo", None),
        ("store_barrier", 1, "temporary-world-destroyed"),
        (
            "init",
            {
                "backend": "nccl",
                "store": ("slime-reloadable-world-2-nccl", "base-store"),
                "rank": 1,
                "world_size": 4,
                "timeout": timeout,
            },
        ),
        ("store_barrier", 2, "default-world-ready"),
        ("init_canonical_gloo",),
        ("store_barrier", 2, "canonical-gloo-ready"),
        ("reload_subgroups",),
        ("store_barrier", 2, "subgroups-ready"),
    ]


@pytest.mark.unit
def test_store_barrier_waits_for_every_rank():
    events = []

    class FakeStore:
        def set(self, key, value):
            events.append(("set", key, value))

        def wait(self, keys, timeout):
            events.append(("wait", keys, timeout))

    timeout = timedelta(seconds=30)
    state = rpg._DefaultProcessGroupState(
        backend="nccl",
        timeout=timeout,
        store=FakeStore(),
        rank=2,
        world_size=4,
        generation=3,
    )

    rpg._store_barrier(state, "world-ready")

    keys = [f"slime-reloadable-world-3-world-ready-{rank}" for rank in range(4)]
    assert events == [
        ("set", keys[2], b"1"),
        ("wait", keys, timeout),
    ]


@pytest.mark.unit
def test_invalidating_wrappers_drops_every_stale_group_handle(monkeypatch):
    groups = [SimpleNamespace(group=object()), SimpleNamespace(group=object())]
    monkeypatch.setattr(rpg.ReloadableProcessGroup, "GROUPS", {rpg.os.getpid(): groups})

    rpg.ReloadableProcessGroup.invalidate_process_groups()

    assert all(group.group is None for group in groups)


@pytest.mark.unit
def test_pp_topology_survives_repeated_world_and_subgroup_reload(tmp_path):
    world_size = 4
    mp.spawn(
        _run_pp_group_reload_worker,
        args=(world_size, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.unit
def test_unregistered_world_preserves_subgroup_only_behavior(monkeypatch):
    events = []
    monkeypatch.setattr(rpg, "default_process_group_states", {})
    monkeypatch.setattr(
        rpg.ReloadableProcessGroup,
        "destroy_process_groups",
        staticmethod(lambda: events.append("destroy_subgroups")),
    )
    monkeypatch.setattr(
        rpg.ReloadableProcessGroup,
        "reload_process_groups",
        staticmethod(lambda: events.append("reload_subgroups")),
    )
    monkeypatch.setattr(
        rpg.dist,
        "destroy_process_group",
        lambda: pytest.fail("unregistered WORLD must not be destroyed"),
    )

    rpg.destroy_process_groups()
    rpg.reload_process_groups()

    assert events == ["destroy_subgroups", "reload_subgroups"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
