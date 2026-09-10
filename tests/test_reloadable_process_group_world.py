from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
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

    # Exercise the accelerator lifecycle with Gloo so this remains a CPU test.
    # The contract under test is WORLD/subgroup teardown ordering, not a vendor
    # communication backend.
    rpg._uses_accelerator_backend = lambda _backend: True

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

        # Keep this NUM_GPUS=0 regression independent of CUDA-specific memory
        # checks while still exercising a real collective on the reloaded PP group.
        dist.barrier(group=pp_group)
        assert dist.get_world_size(pp_group) == world_size

        state = rpg.default_process_group_states[rpg.os.getpid()]
        assert state.generation == 2 * (generation + 1)

    dist.destroy_process_group()


def _run_backend_normalization_worker(_rank: int) -> None:
    calls = []
    mapped_backends = []

    def old_new_group(*args, **kwargs):
        calls.append((args, kwargs))
        return f"group-{len(calls)}"

    def process_group_backend(backend):
        mapped_backends.append(backend)
        return "mccl" if backend == "nccl" else backend

    rpg.old_new_group_dict.clear()
    rpg.default_process_group_states.clear()
    rpg.dist.new_group = old_new_group
    rpg.dist.get_backend = lambda: "gloo"
    rpg.accelerator.process_group_backend = process_group_backend
    rpg.monkey_patch_torch_dist()

    gloo_group = rpg.dist.new_group(ranks=[0], backend="gloo")
    assert gloo_group == "group-1"
    assert calls[-1][1]["backend"] == "gloo"
    assert mapped_backends == []

    mccl_group = rpg.dist.new_group([0], None, "nccl")
    assert mccl_group == "group-2"
    assert calls[-1][0][2] == "mccl"
    assert mapped_backends == ["nccl"]


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["nccl", "mccl", "cpu:gloo,musa:mccl"])
def test_accelerator_backend_detection(backend):
    assert rpg._uses_accelerator_backend(backend)


@pytest.mark.unit
def test_new_group_normalizes_only_logical_nccl_backend():
    # monkey_patch_torch_dist replaces process-wide torch.distributed symbols,
    # so isolate this behavior check in a spawned process.
    mp.spawn(_run_backend_normalization_worker, nprocs=1, join=True)


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
    assert not state.accelerator_world_destroyed


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

    assert state.accelerator_world_destroyed
    assert state.generation == 1
    assert events == [
        ("barrier", "canonical-gloo"),
        ("destroy_world",),
        ("invalidate_subgroups",),
        ("set_gloo", None),
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
    ]

    events.clear()
    rpg.reload_process_groups()

    assert not state.accelerator_world_destroyed
    assert state.generation == 2
    assert events == [
        ("barrier", "WORLD"),
        ("destroy_world",),
        ("set_gloo", None),
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
        ("init_canonical_gloo",),
        ("reload_subgroups",),
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


@pytest.mark.unit
def test_reload_replays_original_new_group_timeout_and_options(monkeypatch):
    """Reload must replay timeout/pg_options, not ``{ranks, backend=nccl}``.

    The multiprocess test in this file only checks that a reloaded group can
    still ``barrier``. That still passes if reload drops a 2-hour NCCL timeout
    and falls back to PyTorch's default — the offload/heal watchdog bug. Spy
    on the constructor instead.
    """
    pid = 4242
    calls: list[tuple[tuple, dict, object]] = []
    timeout = timedelta(minutes=120)
    pg_options = object()

    class FakeGroup:
        pass

    def fake_new_group(*args, **kwargs):
        group = FakeGroup()
        calls.append((args, dict(kwargs), group))
        return group

    monkeypatch.setattr(rpg, "old_new_group_dict", {})
    monkeypatch.setattr(rpg, "default_process_group_states", {})
    monkeypatch.setattr(rpg.ReloadableProcessGroup, "GROUPS", {})
    monkeypatch.setattr(rpg.os, "getpid", lambda: pid)
    monkeypatch.setattr(rpg.dist, "new_group", fake_new_group)
    monkeypatch.setattr(rpg.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(rpg.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(rpg.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(rpg.dist, "destroy_process_group", lambda group: None)

    rpg.monkey_patch_torch_dist()
    group = rpg.dist.new_group(
        [0, 1],
        backend="nccl",
        timeout=timeout,
        pg_options=pg_options,
        group_desc="TP",
    )
    assert isinstance(group, rpg.ReloadableProcessGroup)
    first_inner = calls[-1][2]
    assert group.group is first_inner

    rpg.ReloadableProcessGroup.destroy_process_groups()
    assert group.group is None

    rpg.ReloadableProcessGroup.reload_process_groups()
    reload_args, reload_kwargs, reloaded_inner = calls[-1]
    assert group.group is reloaded_inner
    assert reloaded_inner is not first_inner
    assert reload_args == ([0, 1],)
    assert reload_kwargs["backend"] == "nccl"
    assert reload_kwargs["timeout"] is timeout
    assert reload_kwargs["pg_options"] is pg_options
    assert reload_kwargs["group_desc"] == "TP"
    assert set(reload_kwargs) == {"backend", "timeout", "pg_options", "group_desc"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
