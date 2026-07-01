from __future__ import annotations

import sys
import types
from datetime import timedelta

import pytest

sys.modules.setdefault("psutil", types.ModuleType("psutil"))

from slime.utils import reloadable_process_group as rpg

NUM_GPUS = 0


@pytest.mark.unit
def test_selected_comm_ops_skip_memory_check():
    skipped_ops = {
        "all_gather_into_tensor",
        "allgather_into_tensor_coalesced",
        "barrier",
        "broadcast_object_list",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "isend",
        "irecv",
    }
    checked_ops = {
        "all_reduce",
        "all_gather",
        "broadcast",
        "reduce_scatter",
        "all_to_all",
        "send",
        "recv",
        "reduce_scatter_tensor_coalesced",
    }

    for op_name in skipped_ops:
        assert not rpg._should_check_memory_for_comm(op_name)

    for op_name in checked_ops:
        assert rpg._should_check_memory_for_comm(op_name)


@pytest.mark.unit
def test_wrap_low_level_call_can_skip_available_memory(monkeypatch):
    calls = []

    def fake_available_memory():
        calls.append("available_memory")
        return {"free_GB": 100}

    monkeypatch.setattr(rpg, "available_memory", fake_available_memory)

    with rpg._wrap_low_level_call(check_memory=False):
        pass

    assert calls == []


@pytest.mark.unit
def test_wrap_low_level_call_checks_available_memory_by_default(monkeypatch):
    calls = []

    def fake_available_memory():
        calls.append("available_memory")
        return {"free_GB": 100}

    monkeypatch.setattr(rpg, "available_memory", fake_available_memory)

    with rpg._wrap_low_level_call():
        pass

    assert calls == ["available_memory"]


@pytest.mark.unit
def test_reload_process_groups_preserves_new_group_timeout(monkeypatch):
    pid = 12345
    calls = []
    destroyed = []
    timeout = timedelta(minutes=120)
    pg_options = object()

    class FakeGroup:
        pass

    def fake_new_group(*args, **kwargs):
        group = FakeGroup()
        calls.append((args, dict(kwargs), group))
        return group

    original_new_group = rpg.dist.new_group
    had_old_new_group = hasattr(rpg.dist, "old_new_group")
    original_old_new_group = getattr(rpg.dist, "old_new_group", None)

    monkeypatch.setattr(rpg, "old_new_group_dict", {})
    monkeypatch.setattr(rpg.ReloadableProcessGroup, "GROUPS", {})
    monkeypatch.setattr(rpg.os, "getpid", lambda: pid)
    monkeypatch.setattr(rpg.dist, "new_group", fake_new_group)
    monkeypatch.setattr(rpg.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(rpg.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(rpg.dist, "destroy_process_group", lambda group: destroyed.append(group))

    try:
        rpg.monkey_patch_torch_dist()

        group = rpg.dist.new_group([0, 1], backend="nccl", timeout=timeout, pg_options=pg_options)
        first_inner_group = calls[-1][2]
        assert group.group is first_inner_group

        rpg.ReloadableProcessGroup.destroy_process_groups()
        assert group.group is None
        assert destroyed == [first_inner_group]

        rpg.ReloadableProcessGroup.reload_process_groups()
        assert group.group is calls[-1][2]
        assert calls[-1][0] == ([0, 1],)
        assert calls[-1][1]["backend"] == "nccl"
        assert calls[-1][1]["timeout"] is timeout
        assert calls[-1][1]["pg_options"] is pg_options
    finally:
        rpg.dist.new_group = original_new_group
        if had_old_new_group:
            rpg.dist.old_new_group = original_old_new_group
        elif hasattr(rpg.dist, "old_new_group"):
            delattr(rpg.dist, "old_new_group")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
