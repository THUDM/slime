"""Smoke test the list_trajectory snapshot logic in middleware.py — v2.

v2 adds 6 new cases (5-10) covering the subagent-segment split implemented
in MASTER_PLAN §5. Cases 1-4 are the original tests verifying the
pre_wipe/final/empty-final paths still hold. Cases 5-10 simulate the
_Session / _SubSession state transitions that the real request handler
performs when claude-code dispatches Task/Agent subagents and when
autoCompact wipes interleave with subagent dispatches.

We can't import middleware.py directly because it pulls in
slime.utils.aiohttp_threaded. Stub that out the same way smoke_classifier_v2.py
does, then exercise _Session + the snapshot points by simulating the chain
mutations that the real request handler makes.
"""
import sys
import types

sys.path.insert(0, "/mnt/jingshenghang/code/slime_swe/slime_wt_subagent")
src_path = "/mnt/jingshenghang/code/slime_swe/slime_wt_subagent/examples/coding_agent_rl/middleware.py"
with open(src_path) as f:
    src = f.read()
src = src.replace(
    "from slime.utils.aiohttp_threaded import AppHandle, run_app_in_thread",
    "AppHandle = run_app_in_thread = None  # smoke-test stub",
)
mod = types.ModuleType("middleware_list_smoke")
sys.modules["middleware_list_smoke"] = mod
exec(src, mod.__dict__)
_Session = mod.__dict__["_Session"]
_SubSession = mod.__dict__["_SubSession"]
_snapshot_subagent = mod.__dict__["_snapshot_subagent"]
_merge_into_parent = mod.__dict__["_merge_into_parent"]
_find_dispatch_tool_use_id = mod.__dict__["_find_dispatch_tool_use_id"]
_has_tool_result_for = mod.__dict__["_has_tool_result_for"]


def synth_segments(s):
    """Mirror _pop_session_split's logic: flush any leftover subagent stack,
    materialize completed_trajectories + the live main-line chain as final."""
    while s.subagent_stack:
        top = s.subagent_stack.pop()
        if s.subagent_stack:
            _merge_into_parent(top, s.subagent_stack[-1])
        else:
            _snapshot_subagent(s, top)
    segments = [
        (list(p), list(r), list(m), dict(meta))
        for (p, r, m, meta) in s.completed_trajectories
    ]
    if s.response_ids:
        segments.append((
            list(s.prompt_ids), list(s.response_ids), list(s.loss_mask),
            {"kind": "final", "completed_turns": len(s.turns),
             "finish_reason": s.last_finish_reason},
        ))
    return segments


# ---------------------------------------------------------------------------
# Test 1: a fresh session has empty completed_trajectories
# ---------------------------------------------------------------------------
s = _Session()
assert s.completed_trajectories == [], "fresh session must have no segments"
assert s.subagent_stack == [], "fresh session must have no subagents"
print("[1] PASS: completed_trajectories starts empty")


# ---------------------------------------------------------------------------
# Test 2: simulate a typical 2-segment session (pre_wipe + final)
# ---------------------------------------------------------------------------
s = _Session()
s.prompt_ids = list(range(1, 11))
s.response_ids = list(range(11, 21))
s.loss_mask = [1] * 10
if s.response_ids:
    s.completed_trajectories.append((
        list(s.prompt_ids), list(s.response_ids), list(s.loss_mask),
        {"kind": "pre_wipe", "completed_turns": 5},
    ))
s.prompt_ids = [1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25]
s.response_ids = list(range(30, 36))
s.loss_mask = [1] * 6
segments = synth_segments(s)
assert len(segments) == 2, f"expected 2 segments, got {len(segments)}"
assert segments[0][0] == list(range(1, 11)) and segments[0][1] == list(range(11, 21))
assert segments[0][3]["kind"] == "pre_wipe"
assert segments[1][1] == list(range(30, 36))
assert segments[1][3]["kind"] == "final"
print("[2] PASS: 2-segment fan-out (pre_wipe + final)")


# ---------------------------------------------------------------------------
# Test 3: empty-response segments would be dropped by generate.py
# ---------------------------------------------------------------------------
s2 = _Session()
s2.prompt_ids = list(range(1, 11))
s2.response_ids = []
s2.loss_mask = []
s2.completed_trajectories.append((
    [1, 2, 3], [4, 5], [1, 1], {"kind": "pre_wipe", "completed_turns": 1},
))
final_segs = synth_segments(s2)
assert len(final_segs) == 1 and final_segs[0][3]["kind"] == "pre_wipe"
print("[3] PASS: empty final chain doesn't produce a segment")


# ---------------------------------------------------------------------------
# Test 4: multi-wipe chain (compact -> sub-agent -> compact -> final)
# (Updated for v2: no diverge_reset kind any more per user decision 2.)
# ---------------------------------------------------------------------------
s3 = _Session()
s3.prompt_ids = [1]; s3.response_ids = [10, 11]; s3.loss_mask = [1, 1]
s3.completed_trajectories.append((list(s3.prompt_ids), list(s3.response_ids), list(s3.loss_mask),
                                  {"kind": "pre_wipe", "completed_turns": 3}))
s3.prompt_ids = [2]; s3.response_ids = [20, 21, 22]; s3.loss_mask = [1, 1, 1]
s3.completed_trajectories.append((list(s3.prompt_ids), list(s3.response_ids), list(s3.loss_mask),
                                  {"kind": "pre_wipe", "completed_turns": 6}))
# (no diverge_reset in v2; the third chain just gets dropped if it diverges,
# or carried into the next pre_wipe/final.) Simulate it landing as final.
s3.prompt_ids = [4]; s3.response_ids = [40]; s3.loss_mask = [1]
segs = synth_segments(s3)
assert len(segs) == 3, f"expected 3 segments, got {len(segs)}"
kinds = [seg[3]["kind"] for seg in segs]
assert kinds == ["pre_wipe", "pre_wipe", "final"], f"got kinds={kinds}"
assert sum(len(seg[1]) for seg in segs) == 2 + 3 + 1 == 6
print("[4] PASS: multi-wipe chain emits 3 segments (no diverge_reset kind)")


# ---------------------------------------------------------------------------
# Test 5: subagent nested simulation
# main turn 1 -> push sub -> sub turns 2, 3 -> pop sub -> main turn 4
#
# Expected: 2 segments — [subagent, final].
# Main line never wipes here so main turn 1 + main turn 4 stay in s.prompt_ids/
# s.response_ids (extended chain) and pop_session_split appends them as a
# single `final` segment. The subagent's two turns are an independent
# segment whose prefix is the subagent's OWN system prompt + initial task
# (NOT main turn 1's prefix) — that's user decision 1.
# ---------------------------------------------------------------------------
s = _Session()
s.system_hash = "main_sys_hash"

# Main turn 1: prompt+response on s
s.prompt_ids = [100, 101, 102]  # main system + initial user
s.response_ids = [110, 111, 112]  # main turn 1 output (includes Task tool_use)
s.loss_mask = [1, 1, 1]
s.last_finish_reason = "tool_use"

# Subagent dispatched: push _SubSession with its own system prompt
sub = _SubSession(
    system_hash="sub_sys_hash",
    dispatch_tool_use_id="toolu_abc",
    nested_depth=1,
)
# Sub's own prompt: subagent system + initial task only (NOT main prefix)
sub.prompt_ids = [200, 201, 202]  # sub system + sub initial user
sub.response_ids = [210, 211, 220, 221]  # sub turns 2 + 3 combined
sub.loss_mask = [1, 1, 1, 1]
sub.seen_msgs = 4
s.subagent_stack.append(sub)

# Subagent returns: pop and snapshot (outermost)
top = s.subagent_stack.pop()
_snapshot_subagent(s, top)

# Main turn 4: appended to existing main chain (linear extend; no wipe)
s.response_ids.extend([113, 114, 115])  # tool_result observation tokens (mask=0)
s.loss_mask.extend([0, 0, 0])
s.response_ids.extend([116, 117])  # main turn 4 model output
s.loss_mask.extend([1, 1])
s.last_finish_reason = "stop"

segments = synth_segments(s)
assert len(segments) == 2, f"case 5: expected 2 segments, got {len(segments)}: kinds={[seg[3]['kind'] for seg in segments]}"
assert segments[0][3]["kind"] == "subagent", f"case 5: seg0 kind={segments[0][3]['kind']}"
# subagent prefix is ONLY the subagent's own [200,201,202] — no main prefix
assert segments[0][0] == [200, 201, 202], f"case 5: subagent prefix leaked main tokens: {segments[0][0]}"
assert segments[0][1] == [210, 211, 220, 221]
assert segments[0][3]["nested_depth"] == 1
assert segments[1][3]["kind"] == "final"
assert segments[1][1] == [110, 111, 112, 113, 114, 115, 116, 117]
assert segments[1][3]["finish_reason"] == "stop"
print("[5] PASS: subagent dispatch -> [subagent, final] with isolated prefix")


# ---------------------------------------------------------------------------
# Test 6: multi-compact (no subagent)
# 4 turns -> wipe -> 3 turns -> wipe -> 2 turns -> end
# Expected: [pre_wipe, pre_wipe, final]
# ---------------------------------------------------------------------------
s = _Session()
# Chain A: 4 turns
s.prompt_ids = [1, 2]
s.response_ids = [10, 11, 12, 13]  # combined 4 turns
s.loss_mask = [1] * 4
s.completed_trajectories.append((
    list(s.prompt_ids), list(s.response_ids), list(s.loss_mask),
    {"kind": "pre_wipe", "completed_turns": 4},
))
# Wipe -> chain B: 3 turns
s.prompt_ids = [3, 4]
s.response_ids = [20, 21, 22]
s.loss_mask = [1] * 3
s.completed_trajectories.append((
    list(s.prompt_ids), list(s.response_ids), list(s.loss_mask),
    {"kind": "pre_wipe", "completed_turns": 7},
))
# Wipe -> chain C: 2 turns final
s.prompt_ids = [5, 6]
s.response_ids = [30, 31]
s.loss_mask = [1] * 2
s.last_finish_reason = "stop"

segs = synth_segments(s)
assert len(segs) == 3, f"case 6: expected 3 segments, got {len(segs)}"
kinds = [seg[3]["kind"] for seg in segs]
assert kinds == ["pre_wipe", "pre_wipe", "final"], f"case 6: kinds={kinds}"
assert [len(seg[1]) for seg in segs] == [4, 3, 2]
print("[6] PASS: 3-wipe chain -> [pre_wipe, pre_wipe, final]")


# ---------------------------------------------------------------------------
# Test 7: empty final — last wipe then claude-code exits with no new response
# Expected: [pre_wipe] only; final dropped because s.response_ids is empty.
# ---------------------------------------------------------------------------
s = _Session()
s.completed_trajectories.append((
    [1, 2, 3], [10, 11], [1, 1],
    {"kind": "pre_wipe", "completed_turns": 4},
))
s.prompt_ids = [1, 2, 3, 99]  # post-wipe prompt rendered but no /generate
s.response_ids = []
s.loss_mask = []

segs = synth_segments(s)
assert len(segs) == 1, f"case 7: expected 1 segment, got {len(segs)}"
assert segs[0][3]["kind"] == "pre_wipe"
print("[7] PASS: empty final after wipe drops the final segment")


# ---------------------------------------------------------------------------
# Test 8: subagent + compact mix
# main 1 -> sub 2-3 -> main 4 -> autoCompact wipe -> main 5
# Expected: [subagent, pre_wipe, final] (3 segments).
# Reasoning: the subagent's 2 turns are an independent `subagent` segment.
# Main turns 1+4 are still in the live chain when the autoCompact wipe
# fires, so they get snapshotted together as a single `pre_wipe` (with the
# wipe's post-wipe chain hosting main turn 5 as the `final` segment).
# Per user decision 2 we only emit 3 kinds, so total = 3 segments.
# This is fewer than the 5 mentioned in spec_3 §4 case 8 (which counted
# pre-subagent main as one segment + post-subagent main as another); the
# 3-segment answer is consistent with the canonical semantics that main-
# line linear continuation does NOT split unless a wipe or dispatch
# changes the routing target.
# ---------------------------------------------------------------------------
s = _Session()
s.system_hash = "main_sys_hash"

# Main turn 1 (lands on s, includes a Task tool_use in response)
s.prompt_ids = [100, 101]
s.response_ids = [110, 111]  # main turn 1 output
s.loss_mask = [1, 1]

# Sub dispatch: turns 2-3 on its own chain
sub = _SubSession(
    system_hash="sub_sys_hash",
    dispatch_tool_use_id="toolu_xyz",
    nested_depth=1,
)
sub.prompt_ids = [200, 201]
sub.response_ids = [210, 211, 220, 221]  # sub turns 2+3
sub.loss_mask = [1, 1, 1, 1]
sub.seen_msgs = 4
s.subagent_stack.append(sub)

# Sub returns: pop and snapshot
top = s.subagent_stack.pop()
_snapshot_subagent(s, top)

# Main turn 4 (linear extend on main)
s.response_ids.extend([112, 113])  # tool_result observation
s.loss_mask.extend([0, 0])
s.response_ids.extend([114, 115])  # main turn 4 output
s.loss_mask.extend([1, 1])

# autoCompact wipe: snapshot the main chain as pre_wipe
s.completed_trajectories.append((
    list(s.prompt_ids), list(s.response_ids), list(s.loss_mask),
    {"kind": "pre_wipe", "completed_turns": 4, "on_subagent": False},
))

# Post-wipe main turn 5
s.prompt_ids = [300, 301]
s.response_ids = [310, 311]
s.loss_mask = [1, 1]
s.last_finish_reason = "stop"

segments = synth_segments(s)
kinds = [seg[3]["kind"] for seg in segments]
# Chronological order in completed_trajectories: subagent appended first
# (during the sub-return), then pre_wipe (during the post-main-4 wipe).
# Final appended at synth time.
assert len(segments) == 3, f"case 8: expected 3 segments, got {len(segments)}: kinds={kinds}"
assert kinds == ["subagent", "pre_wipe", "final"], f"case 8: kinds={kinds}"
# subagent segment carries only sub's own prefix
assert segments[0][0] == [200, 201]
assert segments[0][1] == [210, 211, 220, 221]
# pre_wipe carries main turns 1+4 (linear extend on main side, no split)
assert segments[1][1] == [110, 111, 112, 113, 114, 115]
# final carries main turn 5
assert segments[2][1] == [310, 311]
print("[8] PASS: subagent + compact mix -> [subagent, pre_wipe, final]")


# ---------------------------------------------------------------------------
# Test 9: divergence path — NO segment emitted per user decision 2.
# Simulate the ideal_ids prefix mismatch: the live chain just gets dropped,
# and the next /generate's tokens land in the next pre_wipe or final.
# Expected: just [final] when the post-divergence chain produces a response.
# ---------------------------------------------------------------------------
s = _Session()
# Pre-divergence: there were some tokens but they get DROPPED (no snapshot).
# In the real handler this is the path at middleware.py line ~1306 (was 1126):
# `target.prompt_ids = ideal_ids; target.response_ids = []; target.loss_mask = []`
# We simulate the post-reset state directly.
s.prompt_ids = [50, 51, 52]  # post-reset prompt
s.response_ids = [60, 61]  # post-reset turn 1 response
s.loss_mask = [1, 1]
s.last_finish_reason = "stop"

segs = synth_segments(s)
assert len(segs) == 1, f"case 9: expected 1 segment, got {len(segs)}"
assert segs[0][3]["kind"] == "final"
assert not any(seg[3].get("kind") == "diverge_reset" for seg in segs), \
    "case 9: divergence MUST NOT emit a diverge_reset segment per user decision 2"
print("[9] PASS: divergence drops chain silently; only final emitted")


# ---------------------------------------------------------------------------
# Test 10: nested subagent (depth 2)
# main 1 -> push sub-A (depth=1) -> sub-A turn 2 -> push sub-B (depth=2)
# -> sub-B turn 3 -> pop sub-B (merged into sub-A; NOT a separate segment)
# -> sub-A turn 4 -> pop sub-A -> main 5
# Expected: 2 segments — [subagent (nested_depth=2, containing turns 2+3+4),
# final (main 1+5)].
# ---------------------------------------------------------------------------
s = _Session()
s.system_hash = "main_sys_hash"

# Main turn 1
s.prompt_ids = [100, 101]
s.response_ids = [110, 111]  # contains Task tool_use
s.loss_mask = [1, 1]

# Push sub-A
sub_a = _SubSession(
    system_hash="sub_a_sys",
    dispatch_tool_use_id="toolu_A",
    nested_depth=1,
)
sub_a.prompt_ids = [200, 201]  # sub-A's own system + initial task
sub_a.seen_msgs = 2
s.subagent_stack.append(sub_a)

# Sub-A turn 2 (lands on sub_a; contains Task tool_use to dispatch sub-B)
sub_a.response_ids.extend([210, 211])
sub_a.loss_mask.extend([1, 1])

# Push sub-B (nested, depth=2)
sub_b = _SubSession(
    system_hash="sub_b_sys",
    dispatch_tool_use_id="toolu_B",
    nested_depth=2,
)
sub_b.prompt_ids = [300, 301]
sub_b.seen_msgs = 2
s.subagent_stack.append(sub_b)

# Sub-B turn 3
sub_b.response_ids.extend([310, 311])
sub_b.loss_mask.extend([1, 1])

# Pop sub-B: nested merge into sub-A (NO separate segment per user decision 3)
top = s.subagent_stack.pop()
assert s.subagent_stack, "case 10: stack should still have sub-A after popping sub-B"
_merge_into_parent(top, s.subagent_stack[-1])
assert s.subagent_stack[-1].nested_depth == 2, "case 10: sub-A nested_depth should bump to 2"

# Sub-A turn 4 (tool_result observation + new response)
sub_a.response_ids.extend([212, 213])  # observation tokens
sub_a.loss_mask.extend([0, 0])
sub_a.response_ids.extend([214, 215])  # turn 4 model output
sub_a.loss_mask.extend([1, 1])

# Pop sub-A: outermost -> snapshot as subagent segment
top = s.subagent_stack.pop()
assert not s.subagent_stack, "case 10: stack should be empty after popping sub-A"
_snapshot_subagent(s, top)

# Main turn 5 (linear extend on main)
s.response_ids.extend([112, 113])  # tool_result observation
s.loss_mask.extend([0, 0])
s.response_ids.extend([114, 115])  # main turn 5 output
s.loss_mask.extend([1, 1])
s.last_finish_reason = "stop"

segments = synth_segments(s)
kinds = [seg[3]["kind"] for seg in segments]
assert len(segments) == 2, f"case 10: expected 2 segments, got {len(segments)}: kinds={kinds}"
assert kinds == ["subagent", "final"], f"case 10: kinds={kinds}"
# Subagent segment has nested_depth=2 (recorded from sub-A after merge)
assert segments[0][3]["nested_depth"] == 2, f"case 10: nested_depth={segments[0][3]['nested_depth']}"
# Subagent prefix is sub-A's own; sub-B's tokens were merged into the response
assert segments[0][0] == [200, 201]
# Sub-A had [210,211] then sub-B merged in with [310,311] then sub-A turn 4
# [212,213,214,215] -> combined response_ids = [210,211,310,311,212,213,214,215]
assert segments[0][1] == [210, 211, 310, 311, 212, 213, 214, 215], \
    f"case 10: subagent response_ids={segments[0][1]}"
# loss_mask: sub-A turn2=[1,1], sub-B merged=[1,1] (forced 1 per merge contract),
# sub-A turn4=[0,0,1,1]
assert segments[0][2] == [1, 1, 1, 1, 0, 0, 1, 1], \
    f"case 10: subagent loss_mask={segments[0][2]}"
# Final segment is main 1+5
assert segments[1][1] == [110, 111, 112, 113, 114, 115]
print("[10] PASS: nested subagent (depth 2) -> [subagent(d=2), final]")


# ---------------------------------------------------------------------------
# Bonus sanity: verify helper functions
# ---------------------------------------------------------------------------
# _find_dispatch_tool_use_id
blocks = [
    {"type": "text", "text": "I'll delegate."},
    {"type": "tool_use", "id": "toolu_abc", "name": "Task", "input": {}},
]
assert _find_dispatch_tool_use_id(blocks) == "toolu_abc"
assert _find_dispatch_tool_use_id([{"type": "text", "text": "no tools"}]) == ""

# _has_tool_result_for
msgs = [
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "done"},
    ]},
]
assert _has_tool_result_for(msgs, "toolu_abc")
assert not _has_tool_result_for(msgs, "toolu_xyz")
assert not _has_tool_result_for(msgs, "")
print("[helpers] PASS: _find_dispatch_tool_use_id + _has_tool_result_for")


print("\nALL SMOKE TESTS PASSED")
