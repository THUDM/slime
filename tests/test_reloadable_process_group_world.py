from __future__ import annotations

from datetime import timedelta

import pytest

from slime.utils import reloadable_process_group as rpg

NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["nccl", "mccl", "cpu:gloo,musa:mccl"])
def test_accelerator_backend_detection(backend):
    assert rpg._uses_accelerator_backend(backend)


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
        "destroy_process_groups",
        staticmethod(lambda: events.append(("destroy_subgroups",))),
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
        ("destroy_subgroups",),
        ("barrier", "canonical-gloo"),
        ("destroy_world",),
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
