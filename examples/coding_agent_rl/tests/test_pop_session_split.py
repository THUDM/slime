"""Tests for §7 SEGMENTS pop_session_split: emit_order chronological replay
+ 3 segment kinds + raw_dump shape.

Covers SPEC §7.1 entry `test_pop_session_split.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # worktree root for examples.* + slime.*

from examples.coding_agent_rl import middleware as mw  # noqa: E402


def test_emit_order_chronological() -> None:
    """Segments come out in the order they were appended to _emit_order."""
    s = mw.Session()
    # Two pre_wipes then one subagent then final.
    s.pre_wipes.append(([1], [10], [1], {"kind": "pre_wipe", "completed_turns": 1, "finish_reason": "tool_use",
                                          "num_aborts": 0, "tito_masked_turns": 0, "on_subagent": False}))
    s._emit_order.append(("pre_wipe", s.pre_wipes[-1]))

    sub_snap = mw.SubSnapshot(prompt_ids=[2], response_ids=[20, 21], loss_mask=[1, 1],
                              seen_msgs=4, dispatch_tool_use_id="toolu_a",
                              finish_reason="end_turn", num_aborts=0,
                              tito_masked_turn_count=0)
    s.completed_subagents.append(sub_snap)
    s._emit_order.append(("subagent", sub_snap))

    s.pre_wipes.append(([3], [30], [1], {"kind": "pre_wipe", "completed_turns": 5, "finish_reason": "tool_use",
                                          "num_aborts": 0, "tito_masked_turns": 0, "on_subagent": False}))
    s._emit_order.append(("pre_wipe", s.pre_wipes[-1]))

    # final
    s.response_ids = [40, 41, 42]
    s.prompt_ids = [4]
    s.loss_mask = [1, 1, 1]
    s.last_finish_reason = "stop"
    s.tito_total_turn_count = 7

    segs, dump = mw.pop_session_split(s)
    kinds = [m["segment_kind"] for (_, _, _, m) in segs]
    assert kinds == ["pre_wipe", "subagent", "pre_wipe", "final"], kinds
    assert dump["version"] == 4
    assert dump["num_turns"] == 0
    assert len(dump["pre_wipes"]) == 2
    assert len(dump["completed_subagents"]) == 1


def test_empty_response_segments_dropped() -> None:
    s = mw.Session()
    # pre_wipe with empty response should be dropped from output
    s.pre_wipes.append(([1], [], [], {"kind": "pre_wipe", "completed_turns": 1, "finish_reason": "",
                                       "num_aborts": 0, "tito_masked_turns": 0, "on_subagent": False}))
    s._emit_order.append(("pre_wipe", s.pre_wipes[-1]))
    s.response_ids = []  # final also empty
    segs, _ = mw.pop_session_split(s)
    assert segs == []


def test_drain_active_subagent_at_pop() -> None:
    """active_subagent at pop time is auto-snapshotted as a subagent segment."""
    s = mw.Session()
    sub = mw.SubSession(system_hash="sub-A", dispatch_tool_use_id="toolu_x")
    sub.prompt_ids = [5, 6]
    sub.response_ids = [50, 51, 52]
    sub.loss_mask = [1, 1, 1]
    sub.seen_msgs = 3
    sub.last_finish_reason = "end_turn"
    s.active_subagent = sub
    s.response_ids = [99]
    s.prompt_ids = [1]
    s.loss_mask = [1]
    s.last_finish_reason = "stop"

    segs, _ = mw.pop_session_split(s)
    kinds = [m["segment_kind"] for (_, _, _, m) in segs]
    assert kinds == ["subagent", "final"]
    assert s.active_subagent is None


def test_metadata_fields_present_each_segment() -> None:
    """U3: every segment carries finish_reason / num_aborts / tito_masked_turns."""
    s = mw.Session()
    sub_snap = mw.SubSnapshot(prompt_ids=[1], response_ids=[10], loss_mask=[1],
                              seen_msgs=1, dispatch_tool_use_id="toolu",
                              finish_reason="end_turn", num_aborts=2,
                              tito_masked_turn_count=1)
    s.completed_subagents.append(sub_snap)
    s._emit_order.append(("subagent", sub_snap))
    s.response_ids = [99]
    s.prompt_ids = [1]
    s.loss_mask = [1]
    s.last_finish_reason = "stop"
    s.num_aborts = 3
    s.tito_masked_turn_count = 1
    s.tito_total_turn_count = 5

    segs, _ = mw.pop_session_split(s)
    for _, _, _, meta in segs:
        assert "finish_reason" in meta
        assert "num_aborts" in meta
        assert "tito_masked_turns" in meta
        assert "segment_kind" in meta
        assert "completed_turns" in meta


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
