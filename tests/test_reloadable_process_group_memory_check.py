from __future__ import annotations

import pytest

from slime.utils import reloadable_process_group as rpg


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
