"""Branching-matrix tests for TrajectoryManager via record_turn / get_trajectory.

This script drives the two public interfaces of
``slime.agent.trajectory.TrajectoryManager`` and exhaustively covers the
ways a trajectory can branch, organized as a two-axis matrix:

  * LAYER 1 — routing tree (record_turn). DFS merges on (role, message-equality)
    only, so MESSAGE IDENTITY determines tree shape; token ids are irrelevant here.
  * LAYER 2 — linearization (get_trajectory). TOKEN-ID prefix determines how each leaf
    chain becomes Samples (clean continuation / drift case A·B1·B2 / cross-leaf
    dedup / reward split).
  * COMBINED — both layers interacting (rewrite-merge, tree-fork + token-drift
    stacked, deep multi-leaf dedup, long mixed session).

Readability:
  Token ids are SEMANTIC small integers (see TOKEN_NAMES). Each message renders
  to ``[START, ...body, END]`` with a per-role band, so an id like 2001 reads as
  ``u:compute`` and 7001 reads as ``<DRIFT>``. Expected token sequences are built
  with the same render_* helpers used to feed record_turn, never hand-typed
  magic numbers.

Dual mode:
  Every case is a ``test_*`` function doing strict assertions, run under pytest
  by default. Setting ``TRAJ_DUMP=1`` instead runs them via ``main()``, which
  after each case prints the routing tree (token ids decoded to names) and every
  linearized Sample with token / loss_mask aligned, so a human can read exactly
  where each branch happened::

      python tests/test_agent/test_trajectory_manager_branching.py            # pytest
      TRAJ_DUMP=1 python tests/test_agent/test_trajectory_manager_branching.py  # human-readable dump
"""

from __future__ import annotations

import dataclasses  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

# Run as a plain script (CI does `python <file>`): make the repo root importable
# so `from tests.test_agent...` resolves without an installed `tests` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.test_agent._dump_helpers import dump_tree_txt  # noqa: E402

from slime.agent.adapters.common import TurnRecord  # noqa: E402
from slime.agent.trajectory import TrajectoryManager, _common_prefix_len  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

# ===========================================================================
# §1 Semantic token vocabulary + reverse table
# ===========================================================================
#
# Per-role band. A message renders to [START, ...body, END]; the generation
# prompt appends the assistant START as the open-turn marker.

_BANDS = {
    "system": 1000,
    "user": 2000,
    "assistant": 9000,
    "tool": 3000,
}
_GEN = _BANDS["assistant"]  # add_generation_prompt marker
_DRIFT_BAND = 7000

# Reverse table: token id -> human-readable name. Filled lazily as messages are
# registered so dumps translate ids back to labels.
TOKEN_NAMES: dict[int, str] = {}
_ABBR = {"system": "sys", "user": "usr", "assistant": "ast", "tool": "tul"}
for _role, _base in _BANDS.items():
    TOKEN_NAMES[_base] = f"<{_ABBR[_role]}>"
    TOKEN_NAMES[_base + 9] = f"</{_ABBR[_role]}>"
TOKEN_NAMES[_GEN] = "<gen>"


def name_of(tok: int) -> str:
    """Human-readable name for a token id (falls back to the raw int)."""
    return TOKEN_NAMES.get(tok, str(tok))


def _vis(label: str) -> str:
    """Make whitespace visible in a token label for the dump.

    Whitespace-only drift (e.g. a trailing space from a cc rewrite) is invisible
    in a terminal, which makes ``r:ok`` vs ``r:ok `` indistinguishable. Render
    spaces as ``␣`` so the difference is obvious in the readable output.
    """
    return label.replace(" ", "␣")


_ASST_BODY: dict[str, int] = {}


def _asst_body(label: str) -> int:
    """Stable assistant body token for a response/message label.

    An assistant message replayed in a later prompt must render to the SAME
    tokens the model generated for it, otherwise a clean continuation can never
    hold (the cumulative prompt+response would not prefix the next prompt). So
    both ``render_response`` and an assistant ``MsgTok`` derive their body token
    from this one function, keyed on the label. Bodies are assigned by a stable
    per-label counter (NOT a hash) so distinct labels never collide on one id —
    a collision would mislabel tokens in the dump and could spuriously match
    across turns.
    """
    if label not in _ASST_BODY:
        body = _BANDS["assistant"] + 100 + len(_ASST_BODY)
        _ASST_BODY[label] = body
        TOKEN_NAMES[body] = f"r:{_vis(label)}"
    return _ASST_BODY[label]


def render_ids(ids: list[int]) -> str:
    """Decode an id list into a space-joined readable string."""
    return " ".join(name_of(t) for t in ids)


class MsgTok:
    """A message bound to a fixed, deterministic token rendering.

    The same MsgTok always renders to the same token segment regardless of which
    turn replays it (a clean tokenizer). Token-id drift is injected explicitly by
    tests via ``drift`` — never by re-rendering.
    """

    _body_counter: dict[str, int] = {}

    def __init__(self, role: str, label: str) -> None:
        self.role = role
        self.label = label
        base = _BANDS[role]
        if role == "assistant":
            # An assistant message must render to the same body token as the
            # response it represents (label-keyed), so a replayed assistant in a
            # later prompt token-matches the original generation -> clean
            # continuation. See _asst_body.
            self.body = _asst_body(label)
        else:
            # Allocate one stable body token per (role, label). Offset past the
            # END marker (base+9): the counter is shared across cases, so bodies
            # must never climb into base+9 (END) or they'd collide with it.
            idx = MsgTok._body_counter.setdefault(role, 0) + 1
            MsgTok._body_counter[role] = idx
            self.body = base + 10 + idx
            TOKEN_NAMES[self.body] = f"{role}:{_vis(label)}"
        # message dict as the manager sees it (drives routing equality).
        self.message = {"role": role, "content": label}

    def render(self) -> list[int]:
        """[START, body, END] for this message."""
        base = _BANDS[self.role]
        return [base, self.body, base + 9]


def sys_msg(label: str) -> MsgTok:
    return MsgTok("system", label)


def usr_msg(label: str) -> MsgTok:
    return MsgTok("user", label)


def asst_msg(label: str) -> MsgTok:
    return MsgTok("assistant", label)


def tool_msg(label: str) -> MsgTok:
    return MsgTok("tool", label)


def render_prompt(msgs: list[MsgTok]) -> list[int]:
    """Render a prompt message list, appending the generation-prompt marker."""
    out: list[int] = []
    for m in msgs:
        out.extend(m.render())
    out.append(_GEN)
    return out


def render_response(label: str) -> list[int]:
    """Render an assistant response: [body, </asst>].

    The generation-prompt marker ``<gen>`` equals the assistant START token, so
    ``<gen> + render_response(x)`` == the assistant message ``[<asst>, body,
    </asst>]`` replayed in a later prompt. That identity is what makes a clean
    continuation hold across turns.
    """
    return [_asst_body(label), _BANDS["assistant"] + 9]


def messages(msgs: list[MsgTok]) -> list[dict]:
    """The plain message dicts record_turn wants for prompt_messages."""
    return [m.message for m in msgs]


def drift(ids: list[int], at: int, sentinel: int = _DRIFT_BAND + 1) -> list[int]:
    """Return a copy of ``ids`` with a sentinel spliced at index ``at``.

    The sentinel sits in the drift band (7000+), so a dump shows ``<DRIFT>`` at
    the exact divergence point. Splicing (insert) makes ``len`` grow by one,
    which is enough to make the lcp diverge at ``at``.
    """
    TOKEN_NAMES[sentinel] = "<DRIFT>"
    return ids[:at] + [sentinel] + ids[at:]


def drift_replace(ids: list[int], at: int, sentinel: int = _DRIFT_BAND + 2) -> list[int]:
    """Return a copy of ``ids`` with the token at ``at`` REPLACED by a sentinel.

    Unlike ``drift`` this keeps length constant — used when a test wants the
    divergence inside a response span without changing the cumulative length.
    """
    TOKEN_NAMES[sentinel] = "<DRIFT>"
    out = list(ids)
    out[at] = sentinel
    return out


def turn(prompt_ids, response_ids, *, finish_reason="stop", logprobs=None) -> TurnRecord:
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(response_ids),
        finish_reason=finish_reason,
        output_log_probs=list(logprobs) if logprobs is not None else [],
    )


# A scratch space for the dual-mode printer: each case appends (title, mgr, sid,
# samples) so main() can render after the assertions pass.
_PRINT_LOG: list[tuple[str, object, str, list]] = []

# Raw record_turn inputs, keyed by sid, captured at call time so the printer can
# show the SOURCE data (prompt_ids / response_ids / finish / logprobs) that fed
# the tree — before any tree-building or linearization happened.
_TURN_LOG: dict[str, list[dict]] = {}


def _record(title: str, mgr, sid: str, samples: list) -> None:
    _PRINT_LOG.append((title, mgr, sid, samples))


# Convenience: append a turn with semantic messages, auto-rendering prompt unless
# an explicit prompt_ids is supplied (for drift injection).
def append(
    mgr: TrajectoryManager,
    sid: str,
    prompt_msgs: list[MsgTok],
    response_label: str | None,
    *,
    prompt_ids=None,
    response_ids=None,
    finish_reason="stop",
    logprobs=None,
    response_message=None,
):
    p = list(prompt_ids) if prompt_ids is not None else render_prompt(prompt_msgs)
    if response_ids is not None:
        r = list(response_ids)
    elif response_label is not None:
        r = render_response(response_label)
    else:
        r = []
    rmsg = response_message
    if rmsg is None and response_label is not None:
        rmsg = {"role": "assistant", "content": response_label}
    lp = logprobs
    # Capture the raw turn inputs for the human-readable dump before the manager
    # consumes them.
    _TURN_LOG.setdefault(sid, []).append(
        {
            "prompt_msgs": [f"{m.role}:{_vis(m.label)}" for m in prompt_msgs],
            "prompt_ids": p,
            "response_ids": r,
            "finish": finish_reason,
            "has_lp": lp is not None,
        }
    )
    mgr.record_turn(
        sid,
        turn=turn(p, r, finish_reason=finish_reason, logprobs=lp),
        prompt_messages=messages(prompt_msgs),
        response_message=rmsg,
    )
    return p, r


def _leaves(mgr, sid):
    return [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]


def _iter_all(root):
    """Yield every non-root node in the tree (pre-order)."""
    stack = list(root.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


# Tree text snapshot captured the instant before get_trajectory drains the sid,
# so the human-readable dump can show [tree] AND [samples] side by side even
# though get_trajectory consumes the session.
_TREE_SNAP: dict[str, str] = {}

# Input reward passed to get_trajectory, keyed by sid, so the dump can show the
# split (input_reward / n_samples == per_sample_reward) explicitly.
_REWARD_IN: dict[str, float] = {}


def get_traj(mgr, sid, *args, **kwargs):
    """get_trajectory wrapper that snapshots the tree before draining.

    Linearization (get_trajectory) pops the sid, so a later dump would only see
    ``<drained>``. Capturing the tree text here keeps the routing tree visible
    next to the Samples it produced. The input ``reward`` is captured too so the
    dump can show how it splits across the emitted samples.
    """
    if mgr.has_session(sid):
        _TREE_SNAP[sid] = dump_tree_txt(mgr, sid)
    _REWARD_IN[sid] = kwargs.get("reward", 0.0)
    samples = mgr.get_trajectory(sid, *args, **kwargs)
    # Reward conservation: get_trajectory splits the input reward evenly across
    # every emitted sample, so the per-sample shares must sum back to the input
    # (modulo float error). This is the "averaged over sample count" invariant.
    if samples:
        total = sum(s.reward for s in samples)
        assert abs(total - _REWARD_IN[sid]) < 1e-9, (
            "reward not conserved across split",
            total,
            _REWARD_IN[sid],
        )
    return samples


def _check_invariants(samples):
    for s in samples:
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length, (
            "alignment broken",
            len(s.loss_mask),
            len(s.rollout_log_probs),
            s.response_length,
        )
        assert sum(s.loss_mask) > 0, "fully-masked sample emitted"


def golden(sample) -> str:
    """Render one Sample as a human-reviewable golden string.

    Every token is decoded to its readable name (``<sys>``, ``r:done``,
    ``<DRIFT>`` ...). The leading prompt prefix (no loss_mask entry) is shown as
    plain names; the response region is shown with each TRAINED token (loss=1)
    wrapped in ``[...]`` and each context token (loss=0) left bare. This makes the
    full linearized result — tokens, where the response region starts, and
    exactly which tokens carry training signal — a single literal a human can
    eyeball and assert against, instead of hand-derived index arithmetic.

    Example: ``<sys> system:S </sys> <usr> user:u </usr> <gen> [r:ok] [</ast>]``
    """
    toks = sample.tokens
    resp_start = len(toks) - sample.response_length
    parts: list[str] = []
    for i, t in enumerate(toks):
        nm = name_of(t)
        if i >= resp_start and sample.loss_mask[i - resp_start] == 1:
            parts.append(f"[{nm}]")
        else:
            parts.append(nm)
    return " ".join(parts)


def goldens(samples) -> list[str]:
    return [golden(s) for s in samples]


# ===========================================================================
# §2 Group 1 — routing tree layer (record_turn shapes the tree)
# ===========================================================================


def test_1_1_single_turn_chain():
    mgr = TrajectoryManager()
    sid = "1.1"
    s = sys_msg("S")
    u = usr_msg("compute")
    p, r = append(mgr, sid, [s, u], "ok")
    chain = _leaves(mgr, sid)[0].path_from_root()
    assert [n.role for n in chain] == ["system", "user", "assistant"]
    assert chain[-1].turn_index == 1
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:compute </usr> <gen> [r:ok] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.1 single turn -> linear chain", mgr, sid, samples)
    print("PASS 1.1")


def test_1_2_clean_multiturn_with_tool():
    mgr = TrajectoryManager()
    sid = "1.2"
    s, u = sys_msg("S"), usr_msg("compute")
    a1, t1 = asst_msg("call"), tool_msg("4")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls")
    p2, r2 = append(mgr, sid, [s, u, a1, t1], "done")
    chain = _leaves(mgr, sid)[0].path_from_root()
    assert [n.role for n in chain] == ["system", "user", "assistant", "tool", "assistant"]
    assert mgr.turn_count(sid) == 2
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:compute </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:4 </tul> <gen> [r:done] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.2 clean 2-turn with tool -> single chain", mgr, sid, samples)
    print("PASS 1.2")


def test_1_3_system_fork():
    mgr = TrajectoryManager()
    sid = "1.3"
    for sl in ["SA", "SB"]:
        append(mgr, sid, [sys_msg(sl), usr_msg("u")], "a")
    root = mgr._trees[sid]
    assert len(root.children) == 2, "different system -> two subtrees at root"
    assert len(_leaves(mgr, sid)) == 2
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert goldens(samples) == [
        "<sys> system:SA </sys> <usr> user:u </usr> <gen> [r:a] [</ast>]",
        "<sys> system:SB </sys> <usr> user:u </usr> <gen> [r:a] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.3 system fork -> two subtrees at root", mgr, sid, samples)
    print("PASS 1.3")


def test_1_4_user_fork_shared_system():
    mgr = TrajectoryManager()
    sid = "1.4"
    s = sys_msg("S")
    for ul in ["A", "B"]:
        append(mgr, sid, [s, usr_msg(ul)], ul.lower())
    root = mgr._trees[sid]
    assert len(root.children) == 1, "system shared"
    assert len(root.children[0].children) == 2, "user level forks"
    assert len(_leaves(mgr, sid)) == 2
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:A </usr> <gen> [r:a] [</ast>]",
        "<sys> system:S </sys> <usr> user:B </usr> <gen> [r:b] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.4 user fork (shared system)", mgr, sid, samples)
    print("PASS 1.4")


def test_1_5_assistant_message_fork():
    """Same (sys,user) prefix, two distinct assistant turns -> assistant fork."""
    mgr = TrajectoryManager()
    sid = "1.5"
    s, u = sys_msg("S"), usr_msg("u")
    append(mgr, sid, [s, u], "a1")
    append(mgr, sid, [s, u], "a2")
    user_node = mgr._trees[sid].children[0].children[0]
    assert len(user_node.children) == 2, "two assistant leaves hang off shared user"
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    # Two independent single-turn leaves sharing only the (sys,user) prefix.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a1] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a2] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.5 assistant fork under shared user", mgr, sid, samples)
    print("PASS 1.5")


def test_1_6_tool_fork_shared_assistant():
    """Same first assistant turn, two different tool results -> tool-level fork,
    making the first assistant a shared generated turn with 2 children."""
    mgr = TrajectoryManager()
    sid = "1.6"
    s, u, a1 = sys_msg("S"), usr_msg("u"), asst_msg("call")
    append(mgr, sid, [s, u], "call", finish_reason="tool_calls")
    append(mgr, sid, [s, u, a1, tool_msg("x")], "ax")
    append(mgr, sid, [s, u, a1, tool_msg("y")], "ay")
    asst1 = mgr._trees[sid].children[0].children[0].children[0]
    assert asst1.role == "assistant" and asst1.turn is not None
    assert len(asst1.children) == 2, "shared assistant forks at the tool level"
    assert len(_leaves(mgr, sid)) == 2
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    # Leaf X owns the shared turn 1 (r:call trained); leaf Y shares it -> r:call
    # demoted to loss=0 context, only r:ay trains (cross-leaf dedup).
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:x </tul> <gen> [r:ax] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:call </ast> " "<tul> tool:y </tul> <gen> [r:ay] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.6 tool fork (shared assistant snapshot)", mgr, sid, samples)
    print("PASS 1.6")


def test_1_7_token_only_drift_no_fork():
    """Identical messages, tampered prompt_ids -> NO tree fork (DFS ignores
    tokens), but the drift DOES surface in the linearized sample: it lands in
    leaf 2's prompt region (stripped / loss=0), proving token drift cannot
    corrupt a trained response yet is still carried in the sample tokens."""
    mgr = TrajectoryManager()
    sid = "1.7"
    s, u = sys_msg("S"), usr_msg("u")
    pa, _ = append(mgr, sid, [s, u], "a")
    tampered = drift(pa, 1)  # <DRIFT> spliced into the prompt at index 1
    append(mgr, sid, [s, u], "b", prompt_ids=tampered)
    # Tree: (sys,user) shared, two assistant turns hang off it -> two leaves; the
    # path above the assistant is single (NOT forked on tokens).
    user_node = mgr._trees[sid].children[0].children[0]
    assert len(user_node.children) == 2, "two assistant turns share the (sys,user) path"
    assert len(mgr._trees[sid].children) == 1

    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    # Leaf 1: clean. Leaf 2: the <DRIFT> token sits in the stripped prompt region
    # (bare, no brackets); the response r:b is fully trained ([...]).
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a] [</ast>]",
        "<sys> <DRIFT> system:S </sys> <usr> user:u </usr> <gen> [r:b] [</ast>]",
    ]
    # Belt-and-suspenders on the drift placement: token present, but never inside
    # the response region.
    s_b = samples[1]
    assert (_DRIFT_BAND + 1) in s_b.tokens, "drift token is still carried in the sample"
    assert (_DRIFT_BAND + 1) not in s_b.tokens[len(tampered) :], "drift not in the response region"
    _check_invariants(samples)
    _record("1.7 token-only drift -> no tree fork, drift lands in stripped prompt", mgr, sid, samples)
    print("PASS 1.7")


def test_1_8_multi_tool_per_turn():
    mgr = TrajectoryManager()
    sid = "1.8"
    s, u, a1 = sys_msg("S"), usr_msg("u"), asst_msg("call")
    ta, tb = tool_msg("A"), tool_msg("B")
    append(mgr, sid, [s, u], "call", finish_reason="tool_calls")
    append(mgr, sid, [s, u, a1, ta, tb], "done")
    chain = _leaves(mgr, sid)[0].path_from_root()
    assert [n.role for n in chain] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert chain[3].message == ta.message
    assert chain[4].message == tb.message
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:A </tul> <tul> tool:B </tul> <gen> [r:done] [</ast>]",
    ]
    _check_invariants(samples)
    _record("1.8 multi-tool turn -> one node per tool", mgr, sid, samples)
    print("PASS 1.8")


def test_1_9_cross_sid_isolation():
    mgr = TrajectoryManager()
    s = sys_msg("S")
    for sid, ul in [("sid-a", "A"), ("sid-b", "B")]:
        append(mgr, sid, [s, usr_msg(ul)], ul.lower())
    assert len(_leaves(mgr, "sid-a")) == 1
    assert len(_leaves(mgr, "sid-b")) == 1
    assert mgr._trees["sid-a"] is not mgr._trees["sid-b"]
    sa = get_traj(mgr, "sid-a", base_sample=Sample(index=0, prompt=""), reward=1.0)
    sb = get_traj(mgr, "sid-b", base_sample=Sample(index=1, prompt=""), reward=1.0)
    assert goldens(sa) == ["<sys> system:S </sys> <usr> user:A </usr> <gen> [r:a] [</ast>]"]
    assert goldens(sb) == ["<sys> system:S </sys> <usr> user:B </usr> <gen> [r:b] [</ast>]"]
    _check_invariants(sa)
    _check_invariants(sb)
    _record("1.9 cross-sid isolation (sid-a)", mgr, "sid-a", sa)
    _record("1.9 cross-sid isolation (sid-b)", mgr, "sid-b", sb)
    print("PASS 1.9")


def test_1_10_empty_response():
    mgr = TrajectoryManager()
    sid = "1.10"
    s, u = sys_msg("S"), usr_msg("u")
    append(mgr, sid, [s, u], None, response_ids=[], response_message=None, finish_reason="length")
    asst = _leaves(mgr, sid)[0]
    assert asst.role == "assistant"
    assert asst.turn.output_ids == []
    assert asst.message is None
    # Empty response -> the only turn has no trainable token, so its segment is
    # dropped at linearization (no fully-masked sample). Zero samples is correct.
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 0
    _record("1.10 empty response -> assistant leaf, no message (0 samples)", mgr, sid, samples)
    print("PASS 1.10")


# ===========================================================================
# §2 Group 2 — linearization layer (get_trajectory token routing)
# ===========================================================================


def test_2_1_single_turn_linearize():
    mgr = TrajectoryManager()
    sid = "2.1"
    s, u = sys_msg("S"), usr_msg("u")
    p, r = append(mgr, sid, [s, u], "a", logprobs=None)
    # attach explicit logprobs so we can check propagation
    leaf = _leaves(mgr, sid)[0]
    leaf.turn = dataclasses.replace(leaf.turn, output_log_probs=[-0.5] * len(r))
    samples = get_traj(mgr, sid, base_sample=Sample(index=7, prompt="hi"), reward=1.0)
    assert len(samples) == 1
    s0 = samples[0]
    assert goldens(samples) == ["<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a] [</ast>]"]
    assert s0.rollout_log_probs == [-0.5] * len(r)
    assert s0.reward == 1.0
    _check_invariants(samples)
    _record("2.1 single-turn linearize", mgr, sid, samples)
    print("PASS 2.1")


def test_2_2_clean_multiturn_linearize():
    mgr = TrajectoryManager()
    sid = "2.2"
    s, u, a1, t1 = sys_msg("S"), usr_msg("u"), asst_msg("call"), tool_msg("4")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    p2, r2 = append(mgr, sid, [s, u, a1, t1], "done", logprobs=[-0.4] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s0 = samples[0]
    L = _common_prefix_len(p1 + r1, p2)
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:4 </tul> <gen> [r:done] [</ast>]",
    ]
    assert s0.rollout_log_probs == [-0.5] * len(r1) + [0.0] * (len(p2) - L) + [-0.4] * len(r2)
    _check_invariants(samples)
    _record("2.2 clean 2-turn linearize", mgr, sid, samples)
    print("PASS 2.2")


def test_2_3_drift_case_A_forks():
    """Drift inside a PROMPT region -> case A -> fork, no token dropped."""
    mgr = TrajectoryManager()
    sid = "2.3"
    s, u, a1, t = sys_msg("S"), usr_msg("u"), asst_msg("call"), tool_msg("t")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    p2_honest = render_prompt([s, u, a1, t])
    p2 = drift(p2_honest, len(p1) - 1)  # inside p1's prompt region
    p2, r2 = append(mgr, sid, [s, u, a1, t], "done", prompt_ids=p2, logprobs=[-0.4] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    # case-A fork: two coherent single-turn segments; the <DRIFT> token stays in
    # segment 2's stripped prompt region (bare), no token dropped.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <DRIFT> <gen> r:call </ast> "
        "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    assert all(abs(s.reward - 1.0 / len(samples)) < 1e-9 for s in samples)
    _check_invariants(samples)
    _record("2.3 drift case A (prompt region) -> fork", mgr, sid, samples)
    print("PASS 2.3")


def test_2_4_drift_case_B1_short_replaces():
    """Small drift inside the most-recent response span -> replace."""
    mgr = TrajectoryManager()  # default threshold 1024
    sid = "2.4"
    s, u, a1, t = sys_msg("S"), usr_msg("u"), asst_msg("call"), tool_msg("t")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    p2_honest = render_prompt([s, u, a1, t])
    assert p2_honest[: len(p1) + len(r1)] == p1 + r1
    drift_idx = len(p1) + len(r1) - 1  # last token of r1's echo
    p2 = drift_replace(p2_honest, drift_idx)
    p2, r2 = append(mgr, sid, [s, u, a1, t], "done", prompt_ids=p2, logprobs=[-0.4] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s0 = samples[0]
    L = _common_prefix_len(p1 + r1, p2)
    assert L == drift_idx
    # replace: the drifted r:call response is no longer a faithful echo of what the
    # model generated (its tail diverged), so the WHOLE surviving span is masked and
    # re-supplied as loss=0 prompt context (the <DRIFT> token marks the divergence);
    # only the new r:done trains.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:call <DRIFT> "
        "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    assert s0.rollout_log_probs == [0.0] * (len(p2) - len(p1)) + [-0.4] * len(r2)
    _check_invariants(samples)
    _record("2.4 drift case B1 (small) -> replace", mgr, sid, samples)
    print("PASS 2.4")


def test_2_5_drift_case_B1_long_forks():
    mgr = TrajectoryManager(fork_threshold_tokens=1)  # d>=1 -> fork
    sid = "2.5"
    s, u, a1, t = sys_msg("S"), usr_msg("u"), asst_msg("call"), tool_msg("t")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls")
    p2_honest = render_prompt([s, u, a1, t])
    p2 = drift_replace(p2_honest, len(p1) + len(r1) - 1)
    p2, r2 = append(mgr, sid, [s, u, a1, t], "done", prompt_ids=p2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    # Both segments single-turn (the drift forked them apart): each trains its own
    # response. The <DRIFT> sits in segment 2's stripped prompt.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:call <DRIFT> "
        "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    assert all(abs(s.reward - 1.0 / len(samples)) < 1e-9 for s in samples)
    _check_invariants(samples)
    _record("2.5 drift case B1 (long) -> fork", mgr, sid, samples)
    print("PASS 2.5")


def test_2_6_drift_case_B1_threshold_zero_forks():
    mgr = TrajectoryManager(fork_threshold_tokens=0)
    sid = "2.6"
    s, u, a1, t = sys_msg("S"), usr_msg("u"), asst_msg("call"), tool_msg("t")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls")
    p2_honest = render_prompt([s, u, a1, t])
    p2 = drift_replace(p2_honest, len(p1) + len(r1) - 1)
    p2, r2 = append(mgr, sid, [s, u, a1, t], "done", prompt_ids=p2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:call <DRIFT> "
        "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    _check_invariants(samples)
    _record("2.6 drift case B1 threshold=0 -> fork", mgr, sid, samples)
    print("PASS 2.6")


def test_2_7_drift_case_B2_earlier_turn_forks():
    """Drift inside an EARLIER turn's response span -> always fork."""
    mgr = TrajectoryManager()
    sid = "2.7"
    s, u = sys_msg("S"), usr_msg("u")
    a1, t1 = asst_msg("a1"), tool_msg("t1")
    a2, t2 = asst_msg("a2"), tool_msg("t2")
    p1, r1 = append(mgr, sid, [s, u], "a1", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    p2, r2 = append(mgr, sid, [s, u, a1, t1], "a2", finish_reason="tool_calls", logprobs=[-0.4] * 2)
    p3_honest = render_prompt([s, u, a1, t1, a2, t2])
    p3 = drift_replace(p3_honest, len(p1) + len(r1) - 1)  # inside r1 (earlier span)
    p3, r3 = append(mgr, sid, [s, u, a1, t1, a2, t2], "a3", prompt_ids=p3, logprobs=[-0.3] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    # Segment 1 = clean turns 1+2; segment 2 = turn 3 alone (forked because the
    # drift hit an EARLIER turn's response span, which replace can't drop).
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a1] [</ast>] "
        "<tul> tool:t1 </tul> <gen> [r:a2] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:a1 <DRIFT> "
        "<tul> tool:t1 </tul> <gen> r:a2 </ast> <tul> tool:t2 </tul> <gen> [r:a3] [</ast>]",
    ]
    assert all(abs(s.reward - 1.0 / len(samples)) < 1e-9 for s in samples)
    _check_invariants(samples)
    _record("2.7 drift case B2 (earlier turn) -> fork", mgr, sid, samples)
    print("PASS 2.7")


def test_2_8_fork_reward_split():
    mgr = TrajectoryManager()
    sid = "2.8"
    s, u, a1, t = sys_msg("S"), usr_msg("u"), asst_msg("call"), tool_msg("t")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls")
    p2_honest = render_prompt([s, u, a1, t])
    p2 = drift(p2_honest, len(p1) - 1)  # prompt region -> case A fork
    p2, r2 = append(mgr, sid, [s, u, a1, t], "done", prompt_ids=p2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    # case-A fork: two single-turn segments, each trains its own response.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <DRIFT> <gen> r:call </ast> "
        "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    # reward 1.0 split evenly across the 2 forked samples -> 0.5 each.
    assert all(abs(s.reward - 0.5) < 1e-9 for s in samples)
    _check_invariants(samples)
    _record("2.8 fork reward split (1.0 / 2 = 0.5 each)", mgr, sid, samples)
    print("PASS 2.8")


def test_2_9_two_leaves_reward_split():
    mgr = TrajectoryManager()
    sid = "2.9"
    s = sys_msg("S")
    for ul in ["A", "B"]:
        append(mgr, sid, [s, usr_msg(ul)], ul.lower())
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:A </usr> <gen> [r:a] [</ast>]",
        "<sys> system:S </sys> <usr> user:B </usr> <gen> [r:b] [</ast>]",
    ]
    # reward 1.0 split evenly across the 2 leaves -> 0.5 each.
    assert all(abs(s.reward - 0.5) < 1e-9 for s in samples)
    _check_invariants(samples)
    _record("2.9 two leaves reward split (1.0 / 2 = 0.5 each)", mgr, sid, samples)
    print("PASS 2.9")


def test_2_10_cross_leaf_dedup():
    """Shared assistant trained on first leaf only; second leaf re-emits it
    as loss=0 context."""
    mgr = TrajectoryManager()
    sid = "2.10"
    s, u, a1 = sys_msg("S"), usr_msg("u"), asst_msg("call")
    tx, ty = tool_msg("x"), tool_msg("y")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    p2, r2 = append(mgr, sid, [s, u, a1, tx], "a2", logprobs=[-0.4] * 2)
    p3, r3 = append(mgr, sid, [s, u, a1, ty], "a3", logprobs=[-0.3] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    s_second = samples[1]
    # First leaf trains the shared r:call + its own r:a2; second leaf shares
    # r:call (demoted to loss=0) and trains only r:a3.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:x </tul> <gen> [r:a2] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:call </ast> " "<tul> tool:y </tul> <gen> [r:a3] [</ast>]",
    ]
    assert s_second.rollout_log_probs == [0.0] * (len(p3) - len(p1)) + [-0.3] * len(r3)
    _check_invariants(samples)
    _record("2.10 cross-leaf dedup (shared assistant trained once)", mgr, sid, samples)
    print("PASS 2.10")


def test_2_11_routing_only_assistant_filtered():
    """cc replays an assistant the manager never recorded -> mounts routing-only,
    must be filtered out of the strict-prefix walk (no raise)."""
    mgr = TrajectoryManager()
    sid = "2.11"
    s, u = sys_msg("S"), usr_msg("u")
    a1, t1 = asst_msg("a1"), tool_msg("t1")
    a2 = asst_msg("a2")
    append(mgr, sid, [s, u], "a1", finish_reason="tool_calls")
    append(mgr, sid, [s, u, a1, t1], "a2", finish_reason="tool_calls")
    foreign = asst_msg("foreign")
    t2 = tool_msg("t2")
    append(mgr, sid, [s, u, a1, t1, a2, foreign, t2], "a3")
    leaves = _leaves(mgr, sid)
    assert len(leaves) == 1
    chain = leaves[0].path_from_root()
    routing = [n for n in chain if n.role == "assistant" and n.turn is None]
    assert len(routing) == 1 and routing[0].message["content"] == "foreign"
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    # The foreign assistant (r:foreign) is routing-only -> appears as bare context
    # (no brackets); the three real turns r:a1/r:a2/r:a3 train.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a1] [</ast>] "
        "<tul> tool:t1 </tul> <gen> [r:a2] [</ast>] <gen> r:foreign </ast> "
        "<tul> tool:t2 </tul> <gen> [r:a3] [</ast>]",
    ]
    _record("2.11 routing-only assistant filtered (no raise)", mgr, sid, samples)
    print("PASS 2.11")


def test_2_12_drop_clears_sid():
    mgr = TrajectoryManager()
    sid = "2.12"
    s, u = sys_msg("S"), usr_msg("u")
    append(mgr, sid, [s, u], "a")
    assert mgr.has_session(sid)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert goldens(samples) == ["<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a] [</ast>]"]
    assert not mgr.has_session(sid)
    assert mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt="")) == []
    _check_invariants(samples)
    _record("2.12 drop clears sid (2nd get_trajectory -> [])", mgr, sid, samples)
    print("PASS 2.12")


# ===========================================================================
# §2 Group 3 — combined / stress (both layers interacting)
# ===========================================================================


def test_3_1_rewrite_merge_absorbs_short():
    mgr = TrajectoryManager()
    sid = "3.1"
    s, u = sys_msg("S"), usr_msg("u")
    a1_rw = asst_msg("ok ")  # cc-rewritten (different message identity)
    t1 = tool_msg("t")
    append(mgr, sid, [s, u], "ok", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    append(mgr, sid, [s, u, a1_rw, t1], "done", logprobs=[-0.4] * 2)
    leaves = _leaves(mgr, sid)
    assert len(leaves) == 1, "short rewrite absorbed, not forked"
    chain = leaves[0].path_from_root()
    merged = chain[2]
    assert merged.turn is None and merged.turn_index is None
    assert merged.message == a1_rw.message
    assert merged.metadata["merged_rewrite"]["abandoned_turn_index"] == 1
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    # The abandoned turn-1 response (r:ok␣) is demoted to routing-only -> appears
    # bare; only the surviving turn-2 r:done trains.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:ok␣ </ast> " "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    _check_invariants(samples)
    _record("3.1 rewrite-merge absorbs short assistant", mgr, sid, samples)
    print("PASS 3.1")


def test_3_2_rewrite_merge_long_forks():
    mgr = TrajectoryManager(fork_threshold_tokens=1)  # r1 len 2 >= 1
    sid = "3.2"
    s, u = sys_msg("S"), usr_msg("u")
    a1_rw, t1 = asst_msg("ok2 "), tool_msg("t")
    append(mgr, sid, [s, u], "ok", finish_reason="tool_calls")
    append(mgr, sid, [s, u, a1_rw, t1], "done")
    assert len(_leaves(mgr, sid)) == 2, "long rewrite forks"
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    # Leaf 1: the abandoned turn-1 standalone (r:ok). Leaf 2: turn-2 only (the
    # rewritten r:ok2␣ assistant mounts routing-only and is filtered out).
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:ok] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:ok2␣ </ast> " "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    _check_invariants(samples)
    _record("3.2 rewrite-merge long -> fork", mgr, sid, samples)
    print("PASS 3.2")


def test_3_3_rewrite_merge_threshold_zero_forks():
    mgr = TrajectoryManager(fork_threshold_tokens=0)
    sid = "3.3"
    s, u = sys_msg("S"), usr_msg("u")
    a1_rw, t1 = asst_msg("ok3 "), tool_msg("t")
    append(mgr, sid, [s, u], "ok", finish_reason="tool_calls")
    append(mgr, sid, [s, u, a1_rw, t1], "done")
    assert len(_leaves(mgr, sid)) == 2
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 2
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:ok] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:ok3␣ </ast> " "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    _check_invariants(samples)
    _record("3.3 rewrite-merge threshold=0 -> fork", mgr, sid, samples)
    print("PASS 3.3")


def test_3_4_rewrite_merge_ambiguous_forks():
    mgr = TrajectoryManager()
    sid = "3.4"
    s, u = sys_msg("S"), usr_msg("u")
    # two short assistant leaves under shared (sys,user)
    append(mgr, sid, [s, u], "a")
    append(mgr, sid, [s, u], "b")
    a_c, t1 = asst_msg("c"), tool_msg("t")
    append(mgr, sid, [s, u, a_c, t1], "d")
    assert len(_leaves(mgr, sid)) == 3, "ambiguous candidates fork"
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 3
    # Leaves "a" and "b" are standalone single turns; leaf "d" carries the
    # ambiguous-rewrite assistant (r:c) as routing-only (bare) -> trains only r:d.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:b] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:c </ast> " "<tul> tool:t </tul> <gen> [r:d] [</ast>]",
    ]
    _check_invariants(samples)
    _record("3.4 rewrite-merge ambiguous -> fork", mgr, sid, samples)
    print("PASS 3.4")


def test_3_5_rewrite_merge_match_key_updated():
    """After merge, a later turn replaying the rewritten message must descend
    through the merged node (match_key updated), not fork again."""
    mgr = TrajectoryManager()
    sid = "3.5"
    s, u = sys_msg("S"), usr_msg("u")
    a1_rw = asst_msg("ok5 ")
    t1, a2, t2 = tool_msg("t1"), asst_msg("second"), tool_msg("t2")
    append(mgr, sid, [s, u], "ok", finish_reason="tool_calls")
    append(mgr, sid, [s, u, a1_rw, t1], "second", finish_reason="tool_calls", logprobs=[-0.4] * 2)
    append(mgr, sid, [s, u, a1_rw, t1, a2, t2], "third", logprobs=[-0.3] * 2)
    leaves = _leaves(mgr, sid)
    assert len(leaves) == 1, "match_key updated -> no spurious fork"
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    # Turn 1 (r:ok) was absorbed as routing-only (rewrite merge), so it appears
    # bare; turns 2 and 3 (r:second / r:third) train in one clean chain.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:ok5␣ </ast> "
        "<tul> tool:t1 </tul> <gen> [r:second] [</ast>] <tul> tool:t2 </tul> <gen> [r:third] [</ast>]",
    ]
    _check_invariants(samples)
    _record("3.5 rewrite-merge match_key updated", mgr, sid, samples)
    print("PASS 3.5")


def test_3_6_tree_fork_plus_token_drift():
    """A tree fork (two leaves) where ONE leaf also drift-forks internally,
    yielding 3 Samples total. Combines layer-1 (message fork) with layer-2
    (token drift fork)."""
    mgr = TrajectoryManager()
    sid = "3.6"
    s, u = sys_msg("S"), usr_msg("u")
    a1, tx, ty = asst_msg("call"), tool_msg("x"), tool_msg("y")
    ax2 = asst_msg("ax2")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    # Leaf X: clean continuation, then a third turn with a case-A prompt drift.
    append(mgr, sid, [s, u, a1, tx], "ax2", finish_reason="tool_calls", logprobs=[-0.4] * 2)
    txx = tool_msg("xx")
    p3_honest = render_prompt([s, u, a1, tx, ax2, txx])
    p3 = drift(p3_honest, len(p1) - 1)  # case A drift -> fork inside leaf X
    append(mgr, sid, [s, u, a1, tx, ax2, txx], "ax3", prompt_ids=p3, logprobs=[-0.2] * 2)
    # Leaf Y: a separate tool result off the shared assistant.
    append(mgr, sid, [s, u, a1, ty], "ay2", logprobs=[-0.1] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 3, [s.tokens for s in samples]
    assert goldens(samples) == [
        # Sample 0: leaf X first segment, trains the shared r:call (first claim) + r:ax2.
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:x </tul> <gen> [r:ax2] [</ast>]",
        # Sample 1: leaf X second segment, a FRESH segment after the case-A fork ->
        # whole prompt stripped, only r:ax3 trains (the <DRIFT> sits in its prompt).
        "<sys> system:S </sys> <usr> user:u </usr> <DRIFT> <gen> r:call </ast> "
        "<tul> tool:x </tul> <gen> r:ax2 </ast> <tul> tool:xx </tul> <gen> [r:ax3] [</ast>]",
        # Sample 2: leaf Y, shares r:call (claimed by sample 0 -> bare), trains r:ay2.
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:call </ast> " "<tul> tool:y </tul> <gen> [r:ay2] [</ast>]",
    ]
    assert all(abs(s.reward - 1.0 / 3) < 1e-9 for s in samples)
    _check_invariants(samples)
    _record("3.6 tree fork + token drift -> 3 samples", mgr, sid, samples)
    print("PASS 3.6")


def test_3_7_deep_multi_leaf_dedup():
    """Three leaves sharing a 2-level assistant prefix; the shared turns are
    trained exactly once across all leaves."""
    mgr = TrajectoryManager()
    sid = "3.7"
    s, u = sys_msg("S"), usr_msg("u")
    a1, t1 = asst_msg("a1"), tool_msg("t1")
    a2 = asst_msg("a2")
    append(mgr, sid, [s, u], "a1", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    append(mgr, sid, [s, u, a1, t1], "a2", finish_reason="tool_calls", logprobs=[-0.4] * 2)
    # three different tool results off a2 -> three leaves sharing a1+a2
    for lbl in ["p", "q", "r"]:
        append(mgr, sid, [s, u, a1, t1, a2, tool_msg(lbl)], f"end-{lbl}", logprobs=[-0.3] * 2)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 3
    # Leaf 0 OWNS the shared r:a1 + r:a2 (both trained) and its own end-p; leaves
    # 1 and 2 SHARE r:a1 + r:a2 (bare, claimed by leaf 0) and train only their own
    # end response.
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a1] [</ast>] "
        "<tul> tool:t1 </tul> <gen> [r:a2] [</ast>] <tul> tool:p </tul> <gen> [r:end-p] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:a1 </ast> "
        "<tul> tool:t1 </tul> <gen> r:a2 </ast> <tul> tool:q </tul> <gen> [r:end-q] [</ast>]",
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:a1 </ast> "
        "<tul> tool:t1 </tul> <gen> r:a2 </ast> <tul> tool:r </tul> <gen> [r:end-r] [</ast>]",
    ]
    _check_invariants(samples)
    _record("3.7 deep multi-leaf dedup (3 leaves, shared trained once)", mgr, sid, samples)
    print("PASS 3.7")


def test_3_8_long_mixed_session():
    """A ~7-turn session combining clean continuation, a mid-session B1 replace,
    and a final case-A fork — verifying the mechanisms chain without interfering."""
    mgr = TrajectoryManager()
    sid = "3.8"
    s, u = sys_msg("S"), usr_msg("u")
    a = [asst_msg(f"a{i}") for i in range(6)]
    t = [tool_msg(f"t{i}") for i in range(6)]
    lp = [-0.5, -0.5]
    # turn 1
    p1, r1 = append(mgr, sid, [s, u], "a0", finish_reason="tool_calls", logprobs=lp)
    # turns 2..4 clean
    prefix = [s, u, a[0], t[0]]
    append(mgr, sid, prefix, "a1", finish_reason="tool_calls", logprobs=lp)
    prefix = prefix + [a[1], t[1]]
    append(mgr, sid, prefix, "a2", finish_reason="tool_calls", logprobs=lp)
    prefix = prefix + [a[2], t[2]]
    # turn 5: B1 small replace — drift the last token of the previous response.
    p5_honest = render_prompt(prefix)
    p5 = drift_replace(p5_honest, len(p5_honest) - 2)  # near tail, inside last resp echo region
    append(mgr, sid, prefix, "a3", prompt_ids=p5, finish_reason="tool_calls", logprobs=lp)
    prefix = prefix + [a[3], t[3]]
    # turn 6: clean
    append(mgr, sid, prefix, "a4", finish_reason="tool_calls", logprobs=lp)
    prefix = prefix + [a[4], t[4]]
    # turn 7: case-A fork (drift in early prompt region)
    p7_honest = render_prompt(prefix)
    p7 = drift(p7_honest, len(p1) - 1)
    append(mgr, sid, prefix, "a5", prompt_ids=p7, logprobs=lp)
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    # The final case-A fork splits the single leaf chain into 3 segments.
    assert goldens(samples) == [
        # Segment 1: turns 1-4 in one clean chain. The turn-5 B1 replace dropped a
        # drifted tail (the <DRIFT> is gone here) and realigned, so r:a0..r:a3 all
        # train.
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:a0] [</ast>] "
        "<tul> tool:t0 </tul> <gen> [r:a1] [</ast>] <tul> tool:t1 </tul> <gen> [r:a2] [</ast>] "
        "<tul> tool:t2 <DRIFT> <gen> [r:a3] [</ast>]",
        # Segment 2: cross-leaf-style dedup within the chain — the prior turns are
        # re-emitted as bare context and only r:a4 trains.
        "<sys> system:S </sys> <usr> user:u </usr> <gen> r:a0 </ast> "
        "<tul> tool:t0 </tul> <gen> r:a1 </ast> <tul> tool:t1 </tul> <gen> r:a2 </ast> "
        "<tul> tool:t2 </tul> <gen> r:a3 </ast> <tul> tool:t3 </tul> <gen> [r:a4] [</ast>]",
        # Segment 3: turn 7 after the case-A fork (the <DRIFT> in the early prompt
        # region); whole prefix bare, only r:a5 trains.
        "<sys> system:S </sys> <usr> user:u </usr> <DRIFT> <gen> r:a0 </ast> "
        "<tul> tool:t0 </tul> <gen> r:a1 </ast> <tul> tool:t1 </tul> <gen> r:a2 </ast> "
        "<tul> tool:t2 </tul> <gen> r:a3 </ast> <tul> tool:t3 </tul> <gen> r:a4 </ast> "
        "<tul> tool:t4 </tul> <gen> [r:a5] [</ast>]",
    ]
    assert abs(sum(s.reward for s in samples) - 1.0) < 1e-9
    _check_invariants(samples)
    _record(f"3.8 long mixed session -> {len(samples)} samples", mgr, sid, samples)
    print("PASS 3.8")


# ===========================================================================
# §2 Group 4 — boundary / defensive / feature-completion
#
# Fills coverage gaps the matrix above left open:
# input-validation contracts, mixed-logprobs trajectories, the case-B1 drift
# threshold boundary, and the default-base_sample path.
# ===========================================================================


def test_4_2_logprobs_length_mismatch_raises():
    """output_log_probs whose length != output_ids -> ValueError at record_turn."""
    mgr = TrajectoryManager()
    sid = "4.2"
    s, u = sys_msg("S"), usr_msg("u")
    bad = TurnRecord(
        prompt_ids=render_prompt([s, u]),
        output_ids=[9101, 9102, 9103],
        finish_reason="stop",
        output_log_probs=[-0.1, -0.2],  # length 2 != 3
    )
    raised = False
    try:
        mgr.record_turn(
            sid,
            turn=bad,
            prompt_messages=messages([s, u]),
            response_message={"role": "assistant", "content": "x"},
        )
    except AssertionError as e:
        raised = True
        assert "output_log_probs" in str(e)
    assert raised, "expected AssertionError on logprobs/ids length mismatch"
    print("PASS 4.2")


def test_4_3_empty_prompt_messages_skipped():
    """Empty prompt_messages -> record_turn is a no-op (warns, no node, no turn)."""
    mgr = TrajectoryManager()
    sid = "4.3"
    mgr.record_turn(
        sid,
        turn=turn([1], [2], finish_reason="stop"),
        prompt_messages=[],
        response_message=None,
    )
    assert mgr.turn_count(sid) == 0
    # The tree may be created empty (root only) or absent; either way no leaf.
    assert not mgr.has_session(sid) or list(_leaves(mgr, sid)) == []
    print("PASS 4.3")


def test_4_4_default_base_sample():
    """get_trajectory without base_sample is rejected (required keyword-only arg)."""
    mgr = TrajectoryManager()
    sid = "4.4"
    s, u = sys_msg("S"), usr_msg("u")
    append(mgr, sid, [s, u], "a")
    with pytest.raises(TypeError):
        mgr.get_trajectory(sid, reward=1.0)  # no base_sample
    print("PASS 4.4")


def test_4_5_mixed_logprobs_across_turns():
    """A trajectory where turn 1 carries logprobs and turn 2 does NOT: the
    sample's turn-1 response region has real logprobs, the turn-2 region is
    padded with 0.0 (the response is still trained, loss=1)."""
    mgr = TrajectoryManager()
    sid = "4.5"
    s, u = sys_msg("S"), usr_msg("u")
    a1, t1 = asst_msg("call"), tool_msg("t")
    p1, r1 = append(mgr, sid, [s, u], "call", finish_reason="tool_calls", logprobs=[-0.5] * 2)
    p2, r2 = append(mgr, sid, [s, u, a1, t1], "done", logprobs=None)  # no logprobs
    samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s0 = samples[0]
    L = _common_prefix_len(p1 + r1, p2)
    # both responses still trained (golden shows the loss layout)...
    assert goldens(samples) == [
        "<sys> system:S </sys> <usr> user:u </usr> <gen> [r:call] [</ast>] "
        "<tul> tool:t </tul> <gen> [r:done] [</ast>]",
    ]
    # ...but turn-2's region carries padded 0.0 logprobs (it had none), while
    # turn-1's region keeps its real logprobs.
    assert s0.rollout_log_probs == [-0.5] * len(r1) + [0.0] * (len(p2) - L) + [0.0] * len(r2)
    _check_invariants(samples)
    _record("4.5 mixed logprobs across turns (turn2 padded 0.0)", mgr, sid, samples)
    print("PASS 4.5")


def test_4_6_drift_B1_threshold_boundary():
    """case-B1 threshold compares the incoming turn's full ``output_ids`` length
    to ``fork_threshold`` (mirroring ``_try_merge_assistant_rewrite``): the gate
    is exclusive, so ``len(r2) == threshold`` forks and ``len(r2) < threshold``
    replaces. Drift-tail length is not part of the gate -- only its position
    (inside the most-recent response span) keeps REALIGN physically applicable."""

    def run(threshold, new_resp_len):
        mgr = TrajectoryManager(fork_threshold_tokens=threshold)
        sid = f"4.6-{threshold}-{new_resp_len}"
        s, u = sys_msg("S"), usr_msg("u")
        # 4-token response so the divergence can sit inside the response span.
        p1 = render_prompt([s, u])
        r1 = [9001, 9002, 9003, 9004]
        mgr.record_turn(
            sid,
            turn=turn(p1, r1, finish_reason="tool_calls"),
            prompt_messages=messages([s, u]),
            response_message={"role": "assistant", "content": "a1"},
        )
        a1m = {"role": "assistant", "content": "a1"}
        tm = tool_msg("t")
        # honest turn-2 prompt echoes p1 + r1 then the tool block + gen marker.
        p2_honest = p1 + r1 + tm.render() + [_GEN]
        # divergence one token before the end of r1's echo (well inside the response span).
        drift_idx = len(p1) + len(r1) - 1
        p2 = drift_replace(p2_honest, drift_idx)
        r2 = [9100 + i for i in range(new_resp_len)]
        mgr.record_turn(
            sid,
            turn=turn(p2, r2, finish_reason="stop"),
            prompt_messages=[*messages([s, u]), a1m, tm.message],
            response_message={"role": "assistant", "content": "done"},
        )
        samples = get_traj(mgr, sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
        return samples, p1, r1, p2, r2

    # len(r2) == threshold -> fork: two single-turn segments, each trains its own resp.
    forked, p1, r1, p2, r2 = run(threshold=2, new_resp_len=2)
    assert len(forked) == 2, f"len(r2)==threshold must fork, got {len(forked)}"
    assert forked[0].tokens == p1 + r1
    assert forked[0].loss_mask == [1] * len(r1)
    assert forked[1].tokens == p2 + r2
    assert forked[1].loss_mask == [1] * len(r2)
    # len(r2) < threshold -> replace: one coherent segment realigned to p2.
    replaced, p1b, r1b, p2b, r2b = run(threshold=3, new_resp_len=2)
    assert len(replaced) == 1, f"len(r2)<threshold must replace, got {len(replaced)}"
    assert replaced[0].tokens == p2b + r2b
    # the drifted r1 echo is not a faithful response anymore -> the WHOLE r1 span is
    # masked (loss=0 prompt context), r2 trains.
    assert replaced[0].loss_mask == [0] * (len(p2b) - len(p1b)) + [1] * len(r2b)
    _check_invariants(forked)
    _check_invariants(replaced)
    print("PASS 4.6")


# ===========================================================================
# §3 Dual-mode printer
# ===========================================================================


def _print_sample(idx: int, s: Sample) -> None:
    toks = s.tokens
    resp_start = len(toks) - s.response_length
    # build aligned token/loss rows over the response region (the trained part);
    # the leading prompt prefix has no loss_mask entry.
    names = [name_of(t) for t in toks]
    loss = ["-"] * resp_start + [str(x) for x in s.loss_mask]
    widths = [max(len(names[i]), len(loss[i])) for i in range(len(toks))]
    tok_row = " ".join(names[i].ljust(widths[i]) for i in range(len(toks)))
    loss_row = " ".join(loss[i].ljust(widths[i]) for i in range(len(toks)))
    print(f"  Sample#{idx} reward={s.reward:.3f} resp_len={s.response_length}")
    print(f"    tok : {tok_row}")
    print(f"    loss: {loss_row}")


def _print_raw_turns(sid: str) -> None:
    """Print the raw record_turn inputs (the SOURCE data) for a sid.

    Shows, per turn, the prompt message labels and the actual prompt_ids /
    response_ids decoded to readable names, plus finish_reason and whether
    logprobs were attached. This is what fed the tree, before any building or
    linearization.
    """
    turns = _TURN_LOG.get(sid, [])
    print(f"[raw turns] {len(turns)}")
    for k, t in enumerate(turns, start=1):
        msgs = " , ".join(t["prompt_msgs"])
        print(f"  turn#{k} finish={t['finish']} has_logprobs={t['has_lp']}")
        print(f"    msgs   : {msgs}")
        print(f"    prompt : {render_ids(t['prompt_ids'])}")
        print(f"    output : {render_ids(t['response_ids']) or '<empty>'}")


def _print_case(title: str, mgr, sid: str, samples: list) -> None:
    print(f"\n=== CASE {title} ===")
    _print_raw_turns(sid)
    if mgr.has_session(sid):
        txt = dump_tree_txt(mgr, sid)
    else:
        # Session already drained by get_trajectory; fall back to the snapshot
        # captured by get_traj just before draining.
        txt = _TREE_SNAP.get(sid, "<drained>")
    print("[tree]")
    for line in txt.splitlines():
        print("  " + line)
    n = len(samples)
    if n:
        r_in = _REWARD_IN.get(sid, 0.0)
        per = r_in / n
        print(f"[samples] {n}  (reward split: {r_in:.3f} / {n} = {per:.3f} per sample)")
    else:
        print(f"[samples] {n}")
    for i, s in enumerate(samples):
        _print_sample(i, s)


# ===========================================================================
# main
# ===========================================================================


_CASES = [
    test_1_1_single_turn_chain,
    test_1_2_clean_multiturn_with_tool,
    test_1_3_system_fork,
    test_1_4_user_fork_shared_system,
    test_1_5_assistant_message_fork,
    test_1_6_tool_fork_shared_assistant,
    test_1_7_token_only_drift_no_fork,
    test_1_8_multi_tool_per_turn,
    test_1_9_cross_sid_isolation,
    test_1_10_empty_response,
    test_2_1_single_turn_linearize,
    test_2_2_clean_multiturn_linearize,
    test_2_3_drift_case_A_forks,
    test_2_4_drift_case_B1_short_replaces,
    test_2_5_drift_case_B1_long_forks,
    test_2_6_drift_case_B1_threshold_zero_forks,
    test_2_7_drift_case_B2_earlier_turn_forks,
    test_2_8_fork_reward_split,
    test_2_9_two_leaves_reward_split,
    test_2_10_cross_leaf_dedup,
    test_2_11_routing_only_assistant_filtered,
    test_2_12_drop_clears_sid,
    test_3_1_rewrite_merge_absorbs_short,
    test_3_2_rewrite_merge_long_forks,
    test_3_3_rewrite_merge_threshold_zero_forks,
    test_3_4_rewrite_merge_ambiguous_forks,
    test_3_5_rewrite_merge_match_key_updated,
    test_3_6_tree_fork_plus_token_drift,
    test_3_7_deep_multi_leaf_dedup,
    test_3_8_long_mixed_session,
    test_4_2_logprobs_length_mismatch_raises,
    test_4_3_empty_prompt_messages_skipped,
    test_4_4_default_base_sample,
    test_4_5_mixed_logprobs_across_turns,
    test_4_6_drift_B1_threshold_boundary,
]


def main() -> None:
    for case in _CASES:
        case()
    # Replay the captured tree / sample snapshots as human-readable dumps.
    print("\n" + "=" * 70)
    print("HUMAN-READABLE DUMPS")
    print("=" * 70)
    for title, mgr, sid, samples in _PRINT_LOG:
        _print_case(title, mgr, sid, samples)
    print(f"\nALL CASES PASSED ({len(_CASES)} cases)")


if __name__ == "__main__":
    if os.environ.get("TRAJ_DUMP"):
        main()  # human-readable tree / sample dumps
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
