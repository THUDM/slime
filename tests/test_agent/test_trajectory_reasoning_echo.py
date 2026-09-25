"""reasoning-fork-policy tests for TrajectoryManager (env policy, default off).

Agent clients disagree on how they echo ``reasoning_content`` back: most echo it
verbatim or drop it. ``SLIME_AGENT_REASONING_FORK_POLICY`` selects the behaviour:

  * ``off`` (default / unset) -> no carve-out; the historic fork_threshold
                                 merge/fork behaviour is unchanged.
  * ``reasoning_dropped``     -> fork whenever the echo dropped the recorded
                                 reasoning, regardless of response length or other
                                 field changes (a rewrite is not a drop and falls
                                 through to the threshold logic).

This file drives ``record_turn`` / ``get_trajectory`` directly, simulating a
3-turn coding-agent session (two tool-call turns + final answer) in which the
model ALWAYS generates reasoning and the manager_message carries it; the axes are:

  * client echo behaviour -- verbatim echo vs stripped (rewrites get their own test);
  * policy                -- off / reasoning_dropped;
  * response length       -- below vs at/above ``fork_threshold`` (a policy fork
                             must ignore this threshold).

Token ids are semantic (per-label stable ids, same idea as the branching test):
an assistant message renders ``[<ast> (reasoning body...) content-body </ast>]``
and the model's generation renders ``[reasoning body... content-body </ast>]``, so
an echo WITH the same reasoning token-matches the generation (clean continuation)
while a stripped echo token-drifts inside the response span.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.test_agent._fakes import (  # noqa: E402
    THINK_OUTPUT,
    FakeSGLangServer,
    ScriptedTokenizer,
    drain_session,
    think_split,
)

from slime.agent.adapters import anthropic, common  # noqa: E402
from slime.agent.trajectory import ReasoningForkPolicy, TrajectoryManager, TurnRecord  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0

# ---------------------------------------------------------------------------
# semantic token vocabulary (stable per-label ids; no hand-typed magic numbers)
# ---------------------------------------------------------------------------

_SYS, _USR, _TUL, _AST = 1000, 2000, 3000, 9000
_END = 9  # role END marker == base + 9
_GEN = _AST  # add_generation_prompt marker == assistant START

_BODIES: dict[tuple[str, str], int] = {}


def _body(kind: str, label: str, base: int) -> int:
    """Allocate one stable token id per (kind, label), banded by ``base``."""
    key = (kind, label)
    if key not in _BODIES:
        _BODIES[key] = base + 10 + sum(1 for k, _ in _BODIES if k == kind)
    return _BODIES[key]


class M:
    """One message: the dict the manager routes on + its fixed token rendering."""

    def __init__(self, role: str, message: dict, tokens: list[int]) -> None:
        self.role = role
        self.message = message
        self.tokens = tokens


def _plain_msg(role: str, label: str) -> M:
    base = {"system": _SYS, "user": _USR, "tool": _TUL}[role]
    body = _body(role, label, base)
    return M(role, {"role": role, "content": label}, [base, body, base + _END])


def asst_echo(content_label: str, reasoning_labels: list[str], *, echo_mode: str, tool_calls=None) -> M:
    """A replayed assistant message; ``echo_mode`` selects the client behaviour:
    ``exact`` echoes the same reasoning, ``strip`` drops it."""
    message: dict = {"role": "assistant", "content": content_label}
    tokens = [_AST]
    if echo_mode == "exact":
        message["reasoning_content"] = "".join(reasoning_labels)
        tokens += [_body("reason", r, 9100) for r in reasoning_labels]
    elif echo_mode != "strip":
        raise ValueError(f"unknown echo_mode: {echo_mode}")
    tokens += [_body("content", content_label, 9200), _AST + _END]
    if tool_calls:
        message["tool_calls"] = tool_calls
    return M("assistant", message, tokens)


def response_ids(reasoning_labels: list[str], content_label: str) -> list[int]:
    """What the model generated for this turn: reasoning body + content body + END."""
    return [_body("reason", r, 9100) for r in reasoning_labels] + [_body("content", content_label, 9200), _AST + _END]


def render_prompt(msgs: list[M]) -> list[int]:
    out: list[int] = []
    for m in msgs:
        out += m.tokens
    out.append(_GEN)
    return out


# ---------------------------------------------------------------------------
# session driver: 3 turns (tool-call, tool-call, final answer); the model always
# generates reasoning and the manager_message carries it. Client echo behaviour,
# the fork policy (via env) and response length are the variables.
# ---------------------------------------------------------------------------

SID = "reasoning-echo"
TC1 = [{"type": "function", "function": {"name": "shell", "arguments": {"cmd": "ls"}}}]
TC2 = [{"type": "function", "function": {"name": "shell", "arguments": {"cmd": "pwd"}}}]


def run_session(*, echo_mode: str, long_reasoning: bool, threshold: int = 4):
    """Record a 3-turn session; returns (manager, response_id_lists per turn)."""
    mgr = TrajectoryManager(fork_threshold_tokens=threshold)
    rlabels = (lambda i: [f"r{i}a", f"r{i}b"]) if long_reasoning else (lambda i: [f"r{i}"])
    spec = [
        (rlabels(1), "c1", TC1, "tool_calls"),
        (rlabels(2), "c2", TC2, "tool_calls"),
        (rlabels(3), "c3", None, "stop"),
    ]

    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    prompt = [s, u]
    resps: list[list[int]] = []
    for i, (rl, cl, tc, finish) in enumerate(spec, 1):
        resp = response_ids(rl, cl)
        resps.append(resp)
        rmsg: dict = {"role": "assistant", "content": cl, "reasoning_content": "".join(rl)}
        if tc:
            rmsg["tool_calls"] = tc
        mgr.record_turn(
            SID,
            turn=TurnRecord(prompt_ids=render_prompt(prompt), output_ids=resp, finish_reason=finish),
            prompt_messages=[m.message for m in prompt],
            response_message=rmsg,
        )
        if i < 3:
            prompt.append(asst_echo(cl, rl, echo_mode=echo_mode, tool_calls=tc))
            prompt.append(_plain_msg("tool", f"t{i}"))
    return mgr, resps


# ---------------------------------------------------------------------------
# sample/tree inspection helpers
# ---------------------------------------------------------------------------


def drain(mgr) -> list[Sample]:
    samples = mgr.get_trajectory(SID, base_sample=Sample(index=0, prompt=""), reward=1.0)
    for s in samples:
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
        assert sum(s.loss_mask) > 0, "fully-masked sample emitted"
    return samples


def trained_spans(s: Sample) -> list[list[int]]:
    """Token sub-lists of each contiguous loss=1 run in the response region."""
    start = len(s.tokens) - s.response_length
    spans, i = [], 0
    while i < len(s.loss_mask):
        if s.loss_mask[i]:
            j = i
            while j < len(s.loss_mask) and s.loss_mask[j]:
                j += 1
            spans.append(s.tokens[start + i : start + j])
            i = j
        else:
            i += 1
    return spans


def asst_nodes(mgr) -> list:
    """Every assistant node in the tree (pre-order)."""
    out, stack = [], list(mgr._trees[SID].children)
    while stack:
        n = stack.pop()
        if n.role == "assistant":
            out.append(n)
        stack.extend(n.children)
    return out


def leaves(mgr) -> list:
    return [leaf for leaf in mgr._trees[SID].leaves() if not leaf.is_root]


# ---------------------------------------------------------------------------
# the matrix: client echo behaviour x fork policy x response length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("echo_mode", ["exact", "strip"], ids=["echo-exact", "echo-stripped"])
@pytest.mark.parametrize(
    "policy",
    [ReasoningForkPolicy.OFF, ReasoningForkPolicy.DROPPED],
    ids=["off", "dropped"],
)
@pytest.mark.parametrize("long_reasoning", [False, True], ids=["short-resp", "long-resp"])
def test_reasoning_fork_policy_matrix(monkeypatch, echo_mode, policy, long_reasoning):
    """Full matrix (manager_message carries reasoning). respK = the tokens the
    model generated on turn K. Per cell we assert the tree shape and, per emitted
    sample, which responses train."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, policy)

    mgr, resps = run_session(echo_mode=echo_mode, long_reasoning=long_reasoning)
    resp1, resp2, resp3 = resps

    nodes = asst_nodes(mgr)
    demoted = [n for n in nodes if "merged_rewrite" in n.metadata]
    generated = [n for n in nodes if n.turn is not None]

    if echo_mode == "exact":
        # The echo carries the SAME reasoning the manager stored -> exact history
        # match -> one chain, one sample, ALL turns trainable, for every policy.
        assert len(leaves(mgr)) == 1 and len(demoted) == 0 and len(generated) == 3
        samples = drain(mgr)
        assert len(samples) == 1
        assert trained_spans(samples[0]) == [resp1, resp2, resp3]
    elif policy == ReasoningForkPolicy.OFF:
        # No carve-out -> the historic fork_threshold logic. SHORT replayed
        # assistant rewrite-merges (old turns demoted, only the last trains); LONG
        # is blocked from merging by the threshold -> forks, all turns train.
        if long_reasoning:
            assert len(leaves(mgr)) == 3 and len(demoted) == 0 and len(generated) == 3
            # threshold forks record {reason, response_tokens} on the forked turns (1 and 2)
            forks = [n.metadata["fork"] for n in nodes if "fork" in n.metadata]
            assert len(forks) == 2 and all(f == {"reason": "threshold", "response_tokens": 4} for f in forks)
            samples = drain(mgr)
            assert trained_spans(samples[0]) == [resp1]
            assert trained_spans(samples[1]) == [resp2]
            assert trained_spans(samples[2]) == [resp3]
        else:
            assert len(leaves(mgr)) == 1 and len(generated) == 1
            assert len(demoted) == 2, "short rewrite-merge demotes both old turns"
            assert [n.metadata["merged_rewrite"]["abandoned_turn_index"] for n in demoted] == [1, 2]
            samples = drain(mgr)
            assert len(samples) == 1
            assert trained_spans(samples[0]) == [resp3]
    else:
        # reasoning_dropped: a stripped echo is a drop -> fork regardless of
        # fork_threshold; each generated turn keeps its own leaf and trains.
        assert len(leaves(mgr)) == 3 and len(demoted) == 0 and len(generated) == 3
        # the carve-out records {reason, policy} on the forked turns (1 and 2)
        forks = [n.metadata["fork"] for n in nodes if "fork" in n.metadata]
        assert len(forks) == 2 and all(f == {"reason": "reasoning", "policy": policy} for f in forks)
        samples = drain(mgr)
        assert len(samples) == 3
        assert trained_spans(samples[0]) == [resp1]
        assert trained_spans(samples[1]) == [resp2]
        assert trained_spans(samples[2]) == [resp3]


@pytest.mark.parametrize("policy", [ReasoningForkPolicy.OFF, ReasoningForkPolicy.DROPPED], ids=["off", "dropped"])
@pytest.mark.parametrize("long", [False, True], ids=["short-resp", "long-resp"])
def test_rewritten_reasoning_falls_back_to_threshold(monkeypatch, policy, long):
    """A REWRITE (both sides carry reasoning, different values) is not a drop, so
    it falls back to the threshold logic under either policy: a short response
    rewrite-merges (turn 1 demoted), a long one forks with ``reason: threshold``."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, policy)
    mgr = TrajectoryManager(fork_threshold_tokens=4)
    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    gen_reason = ["rw-r1", "rw-r2", "rw-r3"] if long else ["rw-r"]
    echo_reason = [f"{r}-rewritten" for r in gen_reason]
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u]), output_ids=response_ids(gen_reason, "ok"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u]],
        response_message={"role": "assistant", "content": "ok", "reasoning_content": "".join(gen_reason)},
    )
    # echo: content unchanged, reasoning_content rewritten -> not a drop
    echo = M(
        "assistant",
        {"role": "assistant", "content": "ok", "reasoning_content": "".join(echo_reason)},
        [_AST] + [_body("reason", r, 9100) for r in echo_reason] + [_body("content", "ok", 9200), _AST + _END],
    )
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo]), output_ids=response_ids(["rw-r2"], "done"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u, echo]],
        response_message={"role": "assistant", "content": "done", "reasoning_content": "rw-r2"},
    )

    nodes = asst_nodes(mgr)
    if long:
        # threshold fork, not a policy fork: tagged {"reason": "threshold"}
        assert len(leaves(mgr)) == 2
        assert all("merged_rewrite" not in n.metadata for n in nodes)
        assert sum(n.turn is not None for n in nodes) == 2
        forks = [n.metadata["fork"] for n in nodes if "fork" in n.metadata]
        assert forks == [{"reason": "threshold", "response_tokens": 5}]
        samples = drain(mgr)
        assert len(samples) == 2
        assert trained_spans(samples[0]) == [response_ids(gen_reason, "ok")]
        assert trained_spans(samples[1]) == [response_ids(["rw-r2"], "done")]
    else:
        # rewrite is not a drop -> threshold merge (short) -> turn 1 demoted
        assert len(leaves(mgr)) == 1
        assert sum("merged_rewrite" in n.metadata for n in nodes) == 1
        assert all("fork" not in n.metadata for n in nodes)
        samples = drain(mgr)
        assert len(samples) == 1
        assert trained_spans(samples[0]) == [response_ids(["rw-r2"], "done")]


def test_policy_fork_ignores_threshold_boundary(monkeypatch):
    """A policy fork triggers even at exactly fork_threshold (where the plain
    rewrite-merge already refuses to merge) and just below it (where the plain
    path WOULD merge); pin both edges here."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.DROPPED)
    for threshold, long in [(3, True), (4, True), (5, False)]:
        mgr, _ = run_session(echo_mode="strip", long_reasoning=long, threshold=threshold)
        assert len(leaves(mgr)) == 3, (threshold, long)
        nodes = asst_nodes(mgr)
        # all three generated turns keep their TurnRecord; nothing merge-demoted
        assert sum(n.turn is not None for n in nodes) == 3, (threshold, long)
        assert all("merged_rewrite" not in n.metadata for n in nodes), (threshold, long)
        samples = drain(mgr)
        assert len(samples) == 3 and all(len(trained_spans(s)) == 1 for s in samples), (threshold, long)


def test_non_reasoning_rewrite_still_merges(monkeypatch):
    """A rewrite whose difference is NOT a reasoning drop (a whitespace content
    rewrite, reasoning kept) keeps the original fork_threshold merge behaviour
    even with the policy active -- the carve-out only covers dropped reasoning."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.DROPPED)
    mgr = TrajectoryManager(fork_threshold_tokens=4)
    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    rl = ["rw-r"]
    mgr.record_turn(
        SID,
        turn=TurnRecord(prompt_ids=render_prompt([s, u]), output_ids=response_ids(rl, "ok"), finish_reason="stop"),
        prompt_messages=[m.message for m in [s, u]],
        response_message={"role": "assistant", "content": "ok", "reasoning_content": "".join(rl)},
    )
    # echo: SAME reasoning_content, content gains a trailing space -> not a drop
    echo = asst_echo("ok ", rl, echo_mode="exact")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo]), output_ids=response_ids(["rw-r2"], "done"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u, echo]],
        response_message={"role": "assistant", "content": "done", "reasoning_content": "rw-r2"},
    )
    nodes = asst_nodes(mgr)
    assert len(leaves(mgr)) == 1, "non-reasoning rewrite must still merge (short response)"
    assert [n.metadata["merged_rewrite"]["abandoned_turn_index"] for n in nodes if "merged_rewrite" in n.metadata] == [
        1
    ]
    samples = drain(mgr)
    assert len(samples) == 1
    assert trained_spans(samples[0]) == [response_ids(["rw-r2"], "done")]


def test_reasoning_fork_ignores_other_field_changes(monkeypatch):
    """A replay that stripped the reasoning forks EVEN THOUGH the content was also
    rewritten -- the drop check takes precedence over other-field changes."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.DROPPED)
    mgr = TrajectoryManager(fork_threshold_tokens=4)
    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u]), output_ids=response_ids(["cw-r"], "ok"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u]],
        response_message={"role": "assistant", "content": "ok", "reasoning_content": "cw-r"},
    )
    # content rewritten ("ok" -> "ok ") AND reasoning stripped -> a drop; other fields ignored
    echo = asst_echo("ok ", ["cw-r"], echo_mode="strip")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo]), output_ids=response_ids(["cw-r2"], "done"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u, echo]],
        response_message={"role": "assistant", "content": "done", "reasoning_content": "cw-r2"},
    )
    nodes = asst_nodes(mgr)
    # fork (not merge): turn 1 is preserved on its own leaf and carries fork metadata
    assert len(leaves(mgr)) == 2
    assert all("merged_rewrite" not in n.metadata for n in nodes)
    forked = [n for n in nodes if "fork" in n.metadata]
    assert len(forked) == 1 and forked[0].metadata["fork"] == {
        "reason": "reasoning",
        "policy": ReasoningForkPolicy.DROPPED,
    }
    samples = drain(mgr)
    assert len(samples) == 2
    assert trained_spans(samples[0]) == [response_ids(["cw-r"], "ok")]
    assert trained_spans(samples[1]) == [response_ids(["cw-r2"], "done")]


def test_policy_skips_already_demoted_node(monkeypatch):
    """An already-merge-demoted node (turn=None, message=last echo) takes the
    structural fork when a later echo drops the reasoning -- no {"reason":
    "reasoning"} metadata, since there is no generated turn left to protect."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.DROPPED)
    mgr = TrajectoryManager(fork_threshold_tokens=4)
    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u]), output_ids=response_ids(["d-r"], "ok"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u]],
        response_message={"role": "assistant", "content": "ok", "reasoning_content": "d-r"},
    )
    # turn 2: reasoning kept, content rewritten -> not a drop -> merge demotes turn 1
    echo1 = asst_echo("ok ", ["d-r"], echo_mode="exact")
    u2 = _plain_msg("user", "u2")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo1, u2]),
            output_ids=response_ids(["d-r2"], "ok2"),
            finish_reason="stop",
        ),
        prompt_messages=[m.message for m in [s, u, echo1, u2]],
        response_message={"role": "assistant", "content": "ok2", "reasoning_content": "d-r2"},
    )
    demoted = [n for n in asst_nodes(mgr) if "merged_rewrite" in n.metadata]
    assert len(demoted) == 1 and demoted[0].turn is None

    # turn 3: the new echo drops the reasoning; the demoted node's stored message
    # still has it, but the fork is structural -- no policy metadata
    echo2 = asst_echo("ok ", ["d-r"], echo_mode="strip")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo2, u2]),
            output_ids=response_ids(["d-r3"], "ok3"),
            finish_reason="stop",
        ),
        prompt_messages=[m.message for m in [s, u, echo2, u2]],
        response_message={"role": "assistant", "content": "ok3", "reasoning_content": "d-r3"},
    )

    assert "fork" not in demoted[0].metadata, "demoted node has no turn to protect; no policy fork metadata"
    assert all("fork" not in n.metadata for n in asst_nodes(mgr))
    # the structural fork still happens: turns 2 and 3 train on their own branches
    assert len(leaves(mgr)) == 2
    samples = drain(mgr)
    assert len(samples) == 2
    assert trained_spans(samples[0]) == [response_ids(["d-r2"], "ok2")]
    assert trained_spans(samples[1]) == [response_ids(["d-r3"], "ok3")]


def test_policy_tolerates_empty_response_message(monkeypatch):
    """A generated leaf with an empty response_message stores message=None; the
    policy check must not crash on it (None holds no reasoning -> not a drop ->
    threshold logic)."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.DROPPED)
    mgr = TrajectoryManager(fork_threshold_tokens=4)
    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u]), output_ids=response_ids(["e-r"], "ok"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u]],
        response_message=None,  # empty response -> leaf message is None
    )
    echo = asst_echo("ok", ["e-r"], echo_mode="exact")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo]), output_ids=response_ids(["e-r2"], "done"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u, echo]],
        response_message={"role": "assistant", "content": "done", "reasoning_content": "e-r2"},
    )
    nodes = asst_nodes(mgr)
    assert len(leaves(mgr)) == 1, "no crash; not a drop -> short rewrite-merge"
    assert sum("merged_rewrite" in n.metadata for n in nodes) == 1
    samples = drain(mgr)
    assert len(samples) == 1
    assert trained_spans(samples[0]) == [response_ids(["e-r2"], "done")]


# ---------------------------------------------------------------------------
# policy wiring: default off; named values; case/whitespace tolerant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ReasoningForkPolicy.OFF),
        ("off", ReasoningForkPolicy.OFF),
        ("reasoning_dropped", ReasoningForkPolicy.DROPPED),
        ("REASONING_DROPPED", ReasoningForkPolicy.DROPPED),
        (" reasoning_dropped ", ReasoningForkPolicy.DROPPED),
        ("reasoning_mismatch", ReasoningForkPolicy.OFF),  # removed value -> warns, treated as off
        ("some_future_policy", ReasoningForkPolicy.OFF),  # unknown -> warns, treated as off
    ],
)
def test_reasoning_fork_policy_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(ReasoningForkPolicy.ENV, value)
    assert ReasoningForkPolicy.from_env().name == expected


def test_reasoning_fork_policy_default_off(monkeypatch):
    monkeypatch.delenv(ReasoningForkPolicy.ENV, raising=False)
    assert ReasoningForkPolicy.from_env().name == ReasoningForkPolicy.OFF


def test_unknown_policy_never_forks(monkeypatch):
    """An unrecognized policy value is inert: a stripped replay still rewrite-merges
    (short response) exactly like ``off``."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, "some_future_policy")
    mgr, resps = run_session(echo_mode="strip", long_reasoning=False)
    nodes = asst_nodes(mgr)
    assert len(leaves(mgr)) == 1
    assert sum("merged_rewrite" in n.metadata for n in nodes) == 2
    samples = drain(mgr)
    assert len(samples) == 1
    assert trained_spans(samples[0]) == [resps[2]]


def test_unknown_policy_logs_warning(monkeypatch, caplog):
    """An unrecognized policy value (including the removed ``reasoning_mismatch``)
    logs a warning at construction; known values (and the default) do not."""
    with caplog.at_level(logging.WARNING, logger="slime.agent.trajectory"):
        for value in ("some_future_policy", "reasoning_mismatch"):
            monkeypatch.setenv(ReasoningForkPolicy.ENV, value)
            ReasoningForkPolicy.from_env()
            assert any(value in r.message for r in caplog.records), value
            caplog.clear()

        for value in ("off", "reasoning_dropped", ""):
            monkeypatch.setenv(ReasoningForkPolicy.ENV, value)
            ReasoningForkPolicy.from_env()
            assert not caplog.records, value


def test_policy_snapshotted_at_manager_construction(monkeypatch):
    """The manager reads the fork policy ONCE at construction (like
    fork_threshold_tokens). Flipping the env afterwards does not change an
    already-built manager, so a session's behaviour is fixed for its lifetime."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.DROPPED)
    mgr = TrajectoryManager(fork_threshold_tokens=4)  # snapshots "dropped"
    monkeypatch.setenv(ReasoningForkPolicy.ENV, ReasoningForkPolicy.OFF)  # flip AFTER construction

    s, u = _plain_msg("system", "S"), _plain_msg("user", "u")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u]), output_ids=response_ids(["snap-r"], "ok"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u]],
        response_message={"role": "assistant", "content": "ok", "reasoning_content": "snap-r"},
    )
    # echo strips reasoning -> a drop
    echo = asst_echo("ok", ["snap-r"], echo_mode="strip")
    mgr.record_turn(
        SID,
        turn=TurnRecord(
            prompt_ids=render_prompt([s, u, echo]), output_ids=response_ids(["snap-r2"], "done"), finish_reason="stop"
        ),
        prompt_messages=[m.message for m in [s, u, echo]],
        response_message={"role": "assistant", "content": "done", "reasoning_content": "snap-r2"},
    )

    # still the snapshotted "dropped" policy -> forks rather than merge-demoting
    assert len(leaves(mgr)) == 2
    assert all("merged_rewrite" not in n.metadata for n in asst_nodes(mgr))


# ---------------------------------------------------------------------------
# adapter-level E2E: the fork policy exercised end-to-end via the anthropic
# adapter (whose manager_message always carries reasoning_content)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy", [ReasoningForkPolicy.OFF, ReasoningForkPolicy.DROPPED], ids=["policy-off", "policy-dropped"]
)
def test_anthropic_multiturn_stripped_thinking_client(monkeypatch, policy):
    """Anthropic client that strips thinking blocks on replay. anthropic's
    manager_message always carries reasoning_content, so the stripped replay is a
    drop: policy off -> rewrite-merge (turn 1 demoted,
    1 sample); policy reasoning_dropped -> fork (both turns trainable, 2 samples)."""
    monkeypatch.setenv(ReasoningForkPolicy.ENV, policy)

    async def run_case():
        # r1 = [reasoning tok, content tok]; the stripped replay re-renders only the
        # content token, so p2 = p1 + [content tok] + next-user tokens.
        p1, r1 = [1, 2, 3], [10, 11]
        p2, r2 = p1 + [11] + [20, 21], [12, 13]
        tok = ScriptedTokenizer(prompts=[p1, p2], outputs={tuple(r1): THINK_OUTPUT, tuple(r2): "done"})
        async with FakeSGLangServer([[(-0.5, t) for t in r1], [(-0.4, t) for t in r2]]) as sglang:
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-ra")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            monkeypatch.setattr(common, "parse_model_output", think_split)
            h = {"Authorization": "Bearer sid-ra"}
            try:
                first = await client.post(
                    "/v1/messages",
                    headers=h,
                    json={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
                )
                fdata = await first.json()
                second = await client.post(
                    "/v1/messages",
                    headers=h,
                    json={
                        "model": "m",
                        "max_tokens": 8,
                        "messages": [
                            {"role": "user", "content": "hi"},
                            # client strips the thinking block on replay
                            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                            {"role": "user", "content": "next"},
                        ],
                    },
                )
                await second.json()
            finally:
                await client.close()
            samples = await drain_session(adapter, "sid-ra")

        # wire always carries the thinking block
        assert any(b.get("type") == "thinking" for b in fdata["content"])
        if policy == ReasoningForkPolicy.DROPPED:
            assert len(samples) == 2, "stripped-thinking replay forks under the dropped policy"
            assert samples[0].tokens == p1 + r1 and samples[0].loss_mask == [1, 1]
            assert samples[1].tokens == p2 + r2 and samples[1].loss_mask == [1, 1]
        else:
            assert len(samples) == 1, "policy off: rewrite-merge keeps one chain (turn 1 demoted)"
            assert samples[0].tokens == p2 + r2
            assert samples[0].loss_mask == [1, 1]

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
