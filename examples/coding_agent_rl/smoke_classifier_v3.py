"""Smoke test for the classifier v3 (linear-deficit + root-parent compact
+ nearest-parent selection). Covers cases 1-6 from v2 plus cases 7-15 from
spec_1 §4."""
import sys
import types

sys.path.insert(0, "/mnt/jingshenghang/code/slime_swe/slime_wt_classifier")
src_path = "/mnt/jingshenghang/code/slime_swe/slime_wt_classifier/examples/coding_agent_rl/middleware.py"
with open(src_path) as f:
    src = f.read()
src = src.replace(
    "from slime.utils.aiohttp_threaded import AppHandle, run_app_in_thread",
    "AppHandle = run_app_in_thread = None  # smoke-test stub",
)
mod = types.ModuleType("middleware_smoke3")
sys.modules["middleware_smoke3"] = mod
exec(src, mod.__dict__)
ns = mod.__dict__

_Turn = ns["_Turn"]
_Session = ns["_Session"]
_classify_branch = ns["_classify_branch"]
_new_turn = ns["_new_turn"]
_is_compact_resume_request = ns["_is_compact_resume_request"]
_is_summarization_request = ns["_is_summarization_request"]
LINEAR_DEFICIT_TOK = ns["LINEAR_DEFICIT_TOK"]

# ---------------------------------------------------------------------------
# Case 1: _is_compact_resume_request marker detection (preserved from v2)
# ---------------------------------------------------------------------------
body_resume = {"messages": [
    {"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>\nCalled the Read tool with..."},
        {"type": "text", "text": "This session is being continued from a previous conversation that ran out of context. Summary: ..."},
    ]},
    {"role": "user", "content": "next user"},
]}
assert _is_compact_resume_request(body_resume), "resume marker should be detected in list-style m[0]"
body_resume_str = {"messages": [
    {"role": "user", "content": "This session is being continued from a previous conversation..."},
]}
assert _is_compact_resume_request(body_resume_str), "marker should be detected in str m[0] too"
body_normal = {"messages": [
    {"role": "user", "content": "what is 2+2"},
]}
assert not _is_compact_resume_request(body_normal), "normal request shouldn't trigger"
print("[1] PASS: _is_compact_resume_request marker detection")

# ---------------------------------------------------------------------------
# Case 2: pfx in the 27K range with is_compact_resume=True -> compact
# ---------------------------------------------------------------------------
parent = _Turn(id=0, parent_id=None, parent_prefix_len=0,
               input_len=30000, output_len=200, full_ids=[0]*30200)
kind = _classify_branch(parent, parent_prefix_len=27000, parent_full_len=30200,
                        initial_prompt_len=21643, is_compact_resume=True)
assert kind == "compact", f"resume-marker should force compact, got {kind}"
print("[2] PASS: resume-marker overrides token-geometry rules")

# ---------------------------------------------------------------------------
# Case 3: pfx 27K, no marker, init_pl=21643 -> compact (16K cushion)
# ---------------------------------------------------------------------------
kind = _classify_branch(parent, parent_prefix_len=27000, parent_full_len=30200,
                        initial_prompt_len=21643, is_compact_resume=False)
assert kind == "compact", f"with 16K cushion this should be compact, got {kind}"
print("[3] PASS: 16K cushion catches the previously-misclassified sibling-6 cases")

# ---------------------------------------------------------------------------
# Case 4: pfx far above init_pl + 16K, no resume marker -> sibling
# ---------------------------------------------------------------------------
kind = _classify_branch(parent, parent_prefix_len=29000, parent_full_len=30200,
                        initial_prompt_len=10000, is_compact_resume=False)
assert kind == "sibling", f"deep-into-parent should still be sibling, got {kind}"
print("[4] PASS: real siblings still classified correctly")

# ---------------------------------------------------------------------------
# Case 5: pfx > parent_full_len -> linear (preserved)
# ---------------------------------------------------------------------------
kind = _classify_branch(parent, parent_prefix_len=30300, parent_full_len=30200,
                        initial_prompt_len=21643, is_compact_resume=False)
assert kind == "linear", f"pfx>=full means linear, got {kind}"
print("[5] PASS: linear classification preserved")

# ---------------------------------------------------------------------------
# Case 6: compact_summarization wins over compact-resume
# ---------------------------------------------------------------------------
body_summ_only = {"messages": [
    {"role": "user", "content": "ordinary"},
    {"role": "user", "content": "CRITICAL: Respond with TEXT ONLY. Your task is to create a detailed summary..."}
]}
_is_summarization_request(body_summ_only)  # sanity
kind = _classify_branch(parent, parent_prefix_len=29000, parent_full_len=30200,
                        initial_prompt_len=21643, is_summarization=True, is_compact_resume=False)
assert kind == "compact_summarization", f"summ should win, got {kind}"
print("[6] PASS: compact_summarization still takes precedence")

# ---------------------------------------------------------------------------
# Case 7: _is_summarization_request with markers in a non-last user msg
# ---------------------------------------------------------------------------
body_summ_non_last = {"messages": [
    {"role": "user", "content": [
        {"type": "text", "text": "Your task is to create a detailed summary of the conversation."},
        {"type": "text", "text": "Respond with TEXT ONLY, no tools."},
    ]},
    {"role": "user", "content": "tail noise"},
]}
assert _is_summarization_request(body_summ_non_last), \
    "summarization markers in non-last user msg should still trigger"
# And the negative: random text shouldn't fire
body_no_summ = {"messages": [
    {"role": "user", "content": "create a detailed summary please"},  # only one marker
]}
assert not _is_summarization_request(body_no_summ), \
    "single marker shouldn't trigger"
print("[7] PASS: _is_summarization_request scans all user msgs")

# ---------------------------------------------------------------------------
# Case 8: sib 14 geometry (par.in=46401, par.out=1475, pfx=46738) -> linear
# Cat A drift: deficit 47876-46738=1138 < LINEAR_DEFICIT_TOK, and pfx (46738)
# consumed 337 tokens of par.out (>0). Linear-deficit (C3) fires.
# ---------------------------------------------------------------------------
par8 = _Turn(id=13, parent_id=12, parent_prefix_len=0,
             input_len=46401, output_len=1475, full_ids=[0]*47876)
kind = _classify_branch(par8, parent_prefix_len=46738, parent_full_len=47876,
                        initial_prompt_len=21645, is_compact_resume=False)
assert kind == "linear", f"sib 14 geometry should be linear, got {kind}"
print("[8] PASS: sib 14 (deficit 1138) classified as linear")

# ---------------------------------------------------------------------------
# Case 9: par.in=21645, par.out=55, pfx=20873, is_compact_resume=True, parent.id=0
# Root-parent compact (C4) -- handles legacy dump where init_pl=0
# ---------------------------------------------------------------------------
par9 = _Turn(id=0, parent_id=None, parent_prefix_len=0,
             input_len=21645, output_len=55, full_ids=[0]*21700)
kind = _classify_branch(par9, parent_prefix_len=20873, parent_full_len=21700,
                        initial_prompt_len=0, is_compact_resume=True)
assert kind == "compact", f"root-parent compact (legacy dump) should be compact, got {kind}"
print("[9] PASS: root-parent compact (C4) handles init_pl=0")

# ---------------------------------------------------------------------------
# Case 10: sib 22 geometry (par.in=30059, par.out=230, pfx=30126) -> linear
# Cat A drift on a post-compact chain: deficit 30289-30126=163 < LINEAR_DEFICIT_TOK
# AND pfx (30126) > par.in (30059). Linear-deficit (C3) fires even though the
# resume marker is still floating in m[0..1] (linear-extend takes precedence
# over the marker check, by design).
# ---------------------------------------------------------------------------
par10 = _Turn(id=21, parent_id=0, parent_prefix_len=20873,
              input_len=30059, output_len=230, full_ids=[0]*30289)
kind = _classify_branch(par10, parent_prefix_len=30126, parent_full_len=30289,
                        initial_prompt_len=0, is_compact_resume=True)
assert kind == "linear", f"sib 22 geometry should be linear, got {kind}"
print("[10] PASS: sib 22 (deficit 163) classified as linear")

# ---------------------------------------------------------------------------
# Case 11: par.in=30059, par.out=230, pfx=20924, is_compact_resume=True, parent.id NOT 0
# -> compact (post-compact-init early turn re-wiped)
# ---------------------------------------------------------------------------
par11 = _Turn(id=21, parent_id=0, parent_prefix_len=20873,
              input_len=30059, output_len=230, full_ids=[0]*30289)
kind = _classify_branch(par11, parent_prefix_len=20924, parent_full_len=30289,
                        initial_prompt_len=0, is_compact_resume=True)
# pfx 20924 << par.full 30289, so deficit/C3 doesn't fire.
# C4 needs parent.id==0; parent.id=21 so C4 doesn't fire.
# Falls to `if is_compact_resume: return "compact"` → compact.
assert kind == "compact", f"resume on non-root parent w/ small pfx should be compact, got {kind}"
print("[11] PASS: post-compact-init re-wipe (parent != root) classified as compact")

# ---------------------------------------------------------------------------
# Case 12: par.in=21645, par.out=55, pfx=21643, is_summarization=True
# -> compact_summarization (preempts everything)
# ---------------------------------------------------------------------------
par12 = _Turn(id=0, parent_id=None, parent_prefix_len=0,
              input_len=21645, output_len=55, full_ids=[0]*21700)
kind = _classify_branch(par12, parent_prefix_len=21643, parent_full_len=21700,
                        initial_prompt_len=21645, is_summarization=True, is_compact_resume=False)
assert kind == "compact_summarization", f"summ flag should preempt, got {kind}"
print("[12] PASS: compact_summarization preempts geometry checks")

# ---------------------------------------------------------------------------
# Case 13: par.in=50000, par.out=10000, pfx=51000 -> sibling (true fork)
# ---------------------------------------------------------------------------
par13 = _Turn(id=5, parent_id=4, parent_prefix_len=0,
              input_len=50000, output_len=10000, full_ids=[0]*60000)
kind = _classify_branch(par13, parent_prefix_len=51000, parent_full_len=60000,
                        initial_prompt_len=8000, is_compact_resume=False)
# deficit = 60000 - 51000 = 9000 > 2048 → C3 doesn't fire.
# Not compact_resume. parent.id != 0 → C4 doesn't fire.
# init_pl=8000: pfx 51000 > 8000+16384=24384 → compact channel ① doesn't fire.
# pfx*2 = 102000 >= par.in 50000 → compact channel ② doesn't fire.
# → sibling.
assert kind == "sibling", f"true sub-agent fork should be sibling, got {kind}"
print("[13] PASS: real sub-agent fork still classified as sibling")

# ---------------------------------------------------------------------------
# Case 14: _new_turn picks NEAR-neighbor when its prefix ≈ full
# ---------------------------------------------------------------------------
s14 = _Session()
# Earlier turn with input 100, output 50 (full=150). Shares first 100 tokens with ideal.
far = _Turn(id=0, parent_id=None, parent_prefix_len=0,
            input_len=100, output_len=50, full_ids=list(range(100)) + [9999]*50)
# Nearer turn with input 200, output 50 (full=250). Shares first 250 tokens with ideal.
near = _Turn(id=1, parent_id=0, parent_prefix_len=100,
             input_len=200, output_len=50, full_ids=list(range(250)))
s14.turns = [far, near]
s14.record_tree = True
# ideal_ids must extend `near` (linear continuation): share full 250 + new bytes
ideal_ids = list(range(250)) + [-1]*40
turn14 = _new_turn(s14, ideal_ids, body={"messages": []})
assert turn14.parent_id == near.id, \
    f"pass-1 neighbor-first should pick id={near.id}, got {turn14.parent_id}"
assert turn14.parent_prefix_len == 250, \
    f"prefix len should be 250, got {turn14.parent_prefix_len}"
print("[14] PASS: _new_turn pass-1 picks near-neighbor when prefix ≈ full")

# ---------------------------------------------------------------------------
# Case 15: _new_turn falls back to longest-prefix when no pass-1 candidate
# Both stored turns have prefix_len <= their input_len, so pass-1
# (`prefix_len > prev.input_len`) rejects them. Pass-2 longest-prefix picks
# the one with the larger prefix_len. This guards real sub-agent forks where
# the near-neighbor is the forked sibling whose own input isn't fully shared.
# ---------------------------------------------------------------------------
s15 = _Session()
# older2: ideal shares only first 80 of older2.full_ids (< older2.input 200).
older2 = _Turn(id=0, parent_id=None, parent_prefix_len=0,
               input_len=200, output_len=100, full_ids=list(range(80)) + [7777]*220)
# fork2: ideal shares only first 30 of fork2.full_ids (< fork2.input 250).
fork2 = _Turn(id=1, parent_id=0, parent_prefix_len=30,
              input_len=250, output_len=50, full_ids=list(range(30)) + [8888]*270)
s15.turns = [older2, fork2]
s15.record_tree = True
# ideal_ids shares 80 with older2 and 30 with fork2. Pass-2 picks older2 (80 > 30).
ideal_ids = list(range(80)) + [-3]*40
turn15 = _new_turn(s15, ideal_ids, body={"messages": []})
assert turn15.parent_id == older2.id, \
    f"pass-2 longest-prefix should pick id={older2.id}, got {turn15.parent_id}"
assert turn15.parent_prefix_len == 80, \
    f"prefix len should be 80, got {turn15.parent_prefix_len}"
print("[15] PASS: _new_turn pass-2 falls back to longest-prefix on true fork")

print("\nALL SMOKE TESTS PASSED")
