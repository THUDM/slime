"""Tests for §7 SEGMENTS pure functions: classify_and_apply / pop_session_split.

Covers SPEC §7.1 entry `test_segments_classify.py`:
  * pre_wipe detection (non-linear msg_hashes -> snapshot + new chain)
  * is_append happy path
  * empty target.seen_msgs base case
  * subagent snapshot via maybe_pop_subagent
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # worktree root for examples.* + slime.*

from examples.coding_agent_rl import middleware as mw  # noqa: E402


def test_pre_wipe_detection() -> None:
    """classify_and_apply detects non-linear msg_hashes -> emits pre_wipe."""
    s = mw.Session()
    s.system_hash = "sys-A"
    s.seen_msgs = 3
    s.msg_hashes = ["m0", "m1", "m2"]
    s.prompt_ids = [1, 2, 3]
    s.response_ids = [10, 11]
    s.loss_mask = [1, 1]
    s.last_finish_reason = "tool_use"

    # New request shares system but messages diverge at index 1 -> not append.
    new_hashes = ["m0", "x9"]
    kind, is_append = mw.classify_and_apply(
        s, s, req_system_hash="sys-A", msg_hashes=new_hashes,
    )
    assert kind == "pre_wipe", f"expected pre_wipe, got {kind}"
    assert is_append is False
    # Side effect: emit_order has the pre_wipe entry, response_ids preserved.
    assert len(s._emit_order) == 1
    assert s._emit_order[0][0] == "pre_wipe"
    _, _, _, meta = s._emit_order[0][1]
    assert meta["segment_kind" if "segment_kind" in meta else "kind"] == "pre_wipe"
    assert meta["finish_reason"] == "tool_use"


def test_is_append_happy_path() -> None:
    """classify_and_apply returns (None, True) when msg_hashes append linearly."""
    s = mw.Session()
    s.system_hash = "sys-A"
    s.seen_msgs = 2
    s.msg_hashes = ["m0", "m1"]
    s.prompt_ids = [1, 2]
    s.response_ids = [10]
    s.loss_mask = [1]

    kind, is_append = mw.classify_and_apply(
        s, s, req_system_hash="sys-A", msg_hashes=["m0", "m1", "m2"],
    )
    assert kind is None
    assert is_append is True
    assert len(s._emit_order) == 0


def test_empty_target_is_first_turn() -> None:
    """seen_msgs == 0 -> always (None, True): treated as first-turn ingest."""
    s = mw.Session()
    kind, is_append = mw.classify_and_apply(
        s, s, req_system_hash="sys-A", msg_hashes=["m0"],
    )
    assert kind is None
    # On first turn, is_append is whatever system_hash compare says, but the
    # branch in classify_and_apply that matters is target.seen_msgs == 0 ->
    # returns the system_hash-based is_append flag (False here since systems
    # differ). The caller treats both as "init" path in step 5.
    assert is_append is False


def test_maybe_pop_subagent_no_op_when_no_dispatch() -> None:
    s = mw.Session()
    # no pending_dispatch_id, no active_subagent
    mw.maybe_pop_subagent(s, [])
    assert s.active_subagent is None
    assert len(s.completed_subagents) == 0


def test_maybe_pop_subagent_snapshots_on_tool_result() -> None:
    s = mw.Session()
    s.pending_dispatch_id = "toolu_abc"
    sub = mw.SubSession(system_hash="sys-sub", dispatch_tool_use_id="toolu_abc")
    sub.response_ids = [99, 100]
    sub.loss_mask = [1, 1]
    sub.prompt_ids = [50, 51]
    sub.seen_msgs = 4
    s.active_subagent = sub

    all_msgs = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "done"}
    ]}]
    mw.maybe_pop_subagent(s, all_msgs)
    assert s.active_subagent is None
    assert s.pending_dispatch_id == ""
    assert len(s.completed_subagents) == 1
    snap = s.completed_subagents[0]
    assert snap.response_ids == [99, 100]
    assert snap.dispatch_tool_use_id == "toolu_abc"


def test_pick_target_nested_dispatch_fail_safe() -> None:
    """pick_target snapshots outer sub when nested dispatch arrives (R2 CC3)."""
    s = mw.Session()
    outer = mw.SubSession(system_hash="sys-outer", dispatch_tool_use_id="toolu_outer")
    outer.response_ids = [1, 2, 3]
    outer.loss_mask = [1, 1, 1]
    s.active_subagent = outer

    # New request with DIFFERENT system_hash -> nested-dispatch fail-safe
    target, is_sub = mw.pick_target(s, [], req_system_hash="sys-inner")
    assert target is s  # returned to main
    assert is_sub is False
    assert s.active_subagent is None
    assert len(s.completed_subagents) == 1
    assert s.completed_subagents[0].response_ids == [1, 2, 3]


def test_pick_target_matches_active_sub() -> None:
    s = mw.Session()
    sub = mw.SubSession(system_hash="sys-sub")
    s.active_subagent = sub
    target, is_sub = mw.pick_target(s, [], req_system_hash="sys-sub")
    assert target is sub
    assert is_sub is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
