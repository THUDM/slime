"""Unit tests for src_v2.trajectory_manager (Plan C: token-faithful).

What we test:
  (1) DFS merge only on (role, message-equality): same prefix in messages
      space always lands on the same path regardless of prompt_ids drift.
  (2) get_trajectory linearization: turn 1 = full prompt + response;
      turn k>=2 = strict exact-prefix append; tokens / loss_mask /
      logprobs all stay in sync.
  (3) TITO drift handling: when turn k+1.prompt diverges mid-stream from
      cumulative tokens, get_trajectory tolerates it by layered fork/replace
      keyed on where the divergence falls (case A / B1 / B2). It never raises
      and never lets logprobs misalign with tokens.
"""

from __future__ import annotations

import json

from slime.agent.adapters.common import TurnRecord  # noqa: E402
from slime.agent.trajectory_manager import (  # noqa: E402
    DriftKind,
    TrajectoryManager,
    _common_prefix_len,
    _SampleBuilder,
)
from slime.utils.types import Sample  # noqa: E402


def _classify(held, prompt, last_response_start_idx, fork_threshold):
    """Drive _SampleBuilder.classify_token_drift on a builder pinned to the given
    state.

    classify_token_drift now lives on the builder rather than as a free function,
    so we set its tokens / last_response_start_idx / threshold directly and feed
    the candidate prompt through a TurnRecord -- output_ids are irrelevant to the
    classification, so they stay empty.
    """
    b = _SampleBuilder(fork_threshold)
    b.tokens = list(held)
    b.last_response_start_idx = last_response_start_idx
    return b.classify_token_drift(_turn(prompt, [], finish_reason="stop"))


def _turn(prompt_ids, response_ids, *, finish_reason, logprobs=None):
    """Helper: build the TurnRecord the way call_sglang_generate would.

    ``logprobs=None`` maps to an empty ``output_log_probs`` (the dataclass
    default) so the manager treats this turn as carrying no logprob signal.
    Pass an explicit list to attach per-token logprobs.
    """
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(response_ids),
        finish_reason=finish_reason,
        output_log_probs=list(logprobs) if logprobs is not None else [],
    )


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_message_equality_is_dict_internal_sort_only():
    # Routing equality (what _find_mount_point uses) is plain dict ==:
    # key order is irrelevant, list order is significant.
    a = {"role": "u", "content": "x"}
    b = {"content": "x", "role": "u"}
    assert a == b

    c = [{"role": "u", "content": "x"}, {"role": "u", "content": "y"}]
    d = [{"role": "u", "content": "y"}, {"role": "u", "content": "x"}]
    assert c != d

    e = {"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]}
    f = {"role": "assistant", "tool_calls": [{"type": "function", "id": "1"}]}
    assert e == f
    print("PASS test_message_equality_is_dict_internal_sort_only")


def test_common_prefix_len():
    assert _common_prefix_len([], []) == 0
    assert _common_prefix_len([1, 2, 3], []) == 0
    assert _common_prefix_len([], [1, 2, 3]) == 0
    assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert _common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert _common_prefix_len([1, 2, 3, 4, 5], [1, 2, 3]) == 3
    print("PASS test_common_prefix_len")


def test_classify_drift_boundaries():
    """Truth table for _SampleBuilder.classify_token_drift: CLEAN / REALIGN / FORK.

    Pins each branch via a builder with hand-set state (no manager) so the drift
    decision is testable in isolation.
    """
    # Empty builder (last_response_start_idx is None): any prompt is a clean extend.
    d = _classify([], [1, 2, 3], None, 1024)
    assert d is DriftKind.CLEAN
    # drift == 0: held tokens are an exact prefix of prompt_ids -> CLEAN. The append
    # boundary is just len(held_tokens), so the kind alone suffices.
    d = _classify([1, 2, 3], [1, 2, 3, 4, 5], 1, 1024)
    assert d is DriftKind.CLEAN
    # Small drift inside the most-recent response span (realign_at >= start),
    # below threshold -> REALIGN.
    d = _classify([1, 2, 3, 4], [1, 2, 3, 9], 2, 1024)
    assert d is DriftKind.REALIGN
    # Same divergence but drift >= threshold -> FORK (case B1 long).
    d = _classify([1, 2, 3, 4], [1, 2, 3, 9], 2, 1)
    assert d is DriftKind.FORK
    # threshold == 0 disables REALIGN entirely -> FORK.
    d = _classify([1, 2, 3, 4], [1, 2, 3, 9], 2, 0)
    assert d is DriftKind.FORK
    # Divergence before the most-recent response span (realign_at < start):
    # prompt region (case A) or an earlier turn (case B2) -> FORK.
    d = _classify([1, 2, 3, 4], [1, 9, 3, 4], 3, 1024)
    assert d is DriftKind.FORK
    print("PASS test_classify_drift_boundaries")


def test_clean_iff_held_is_exact_prefix():
    """Guards the realign_at collapse: classify_token_drift no longer returns a boundary,
    relying on CLEAN  <=>  held is an exact prefix of prompt  <=>  realign_at ==
    len(held). If a divergence inside held could ever still classify CLEAN, the
    ``prompt_ids[len(self.tokens):]`` append in ``append_turn`` would corrupt the
    buffer. We attack that with chunk-boundary and fuzz inputs.
    """
    import random

    # _common_prefix_len must return the FIRST divergence index, never skip an
    # earlier one -- including at the 4096-token chunk boundary it scans in blocks.
    for n in [1, 5, 4095, 4096, 4097, 8192, 8193]:
        a = list(range(n))
        for i in [0, 1, n // 2, n - 1, n]:
            b = a[:]
            if i < n:
                b[i] = -999
            assert _common_prefix_len(a, b) == i, (n, i)

    # A divergence strictly before held's end (i.e. inside the held tokens) must
    # never be CLEAN, no matter how the prompt extends afterwards.
    for n in [3, 4096, 4097, 8200]:
        held = list(range(n))
        for i in [0, 1, n // 2, n - 2, n - 1]:
            prompt = held[:] + [10_000, 10_001]
            prompt[i] = -1  # diverge at i < len(held)
            # last_response_start_idx=0 is the most permissive (whole buffer is the
            # span), so if anything could wrongly classify CLEAN it shows here.
            assert _classify(held, prompt, 0, 10**9) is not DriftKind.CLEAN, (n, i)

    # Fuzz: whenever classify says CLEAN, held must be an exact prefix of prompt.
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(0, 9000)
        held = [rng.randint(0, 5) for _ in range(n)]
        if rng.random() < 0.5:
            prompt = held[:] + [rng.randint(0, 5) for _ in range(rng.randint(0, 5))]
        else:
            prompt = [rng.randint(0, 5) for _ in range(rng.randint(0, 9000))]
        if _classify(held, prompt, 0, 10**9) is DriftKind.CLEAN:
            assert prompt[: len(held)] == held
    print("PASS test_clean_iff_held_is_exact_prefix")


def test_manager_accepts_fork_threshold():
    # default 1024 when unspecified
    m_default = TrajectoryManager()
    assert m_default._fork_threshold == 1024
    # explicit value honored
    m_explicit = TrajectoryManager(fork_threshold_tokens=256)
    assert m_explicit._fork_threshold == 256
    # None -> default
    m_none = TrajectoryManager(fork_threshold_tokens=None)
    assert m_none._fork_threshold == 1024
    print("PASS test_manager_accepts_fork_threshold")


# ---------------------------------------------------------------------------
# Fake tokenizer (kept only as a shape-matching prompt/response generator;
# trajectory_manager doesn't invoke it under plan C).
# ---------------------------------------------------------------------------


class FakeTokenizer:
    ROLE_START = {"system": 9001, "user": 9002, "assistant": 9003, "tool": 9004}
    ROLE_END = {"system": 9101, "user": 9102, "assistant": 9103, "tool": 9104}

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        out: list[int] = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append(self.ROLE_START[role])
            out.extend(ord(c) for c in content)
            out.append(self.ROLE_END[role])
        if add_generation_prompt:
            out.append(self.ROLE_START["assistant"])
        return out


def _render_prompt(messages, tools=None, tokenizer=None):
    tok = tokenizer or FakeTokenizer()
    return tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True)


def _render_response(content_str, tokenizer=None):
    tok = tokenizer or FakeTokenizer()
    return [ord(c) for c in content_str] + [tok.ROLE_END["assistant"]]


# ---------------------------------------------------------------------------
# Plan-C semantics tests
# ---------------------------------------------------------------------------


SYSTEM_MSG = "You are a python coding agent."
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run python code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
        },
    },
]


def _three_turn_session(tok):
    """3-turn linear session via record_turn. Returns mgr, sid, per-turn (p,r)."""
    mgr = TrajectoryManager()
    sid = "three-turn"

    sys_msg = {"role": "system", "content": SYSTEM_MSG}
    user1 = {"role": "user", "content": "Compute 2+2."}
    asst1 = {"role": "assistant", "content": "Computing."}
    tool1 = {"role": "tool", "content": "4"}
    asst2 = {"role": "assistant", "content": "Answer is 4."}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("Computing.", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )

    p2 = _render_prompt([sys_msg, user1, asst1, tool1], tokenizer=tok)
    r2 = _render_response("Answer is 4.", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user1, asst1, tool1],
        response_message=asst2,
    )
    return mgr, sid, [(p1, r1), (p2, r2)]


def test_append_single_turn_shapes_tree():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "single"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    r = _render_response("a", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p, r, finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        response_message={"role": "assistant", "content": "a"},
    )
    chain = list(mgr._trees[sid].leaves())[0].path_from_root()
    roles = [n.role for n in chain]
    assert roles == ["system", "user", "assistant"], roles
    asst = chain[-1]
    assert asst.turn_index == 1
    assert asst.turn.prompt_ids == p
    assert asst.turn.output_ids == r
    assert asst.turn.finish_reason == "stop"
    print("PASS test_append_single_turn_shapes_tree")


def test_append_three_turn_chain_no_fork():
    """3-turn session with consistent prompts -> exactly 1 leaf."""
    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1
    chain = leaves[0].path_from_root()
    roles = [n.role for n in chain]
    assert roles == ["system", "user", "assistant", "tool", "assistant"], roles
    assert mgr.turn_count(sid) == 2
    print("PASS test_append_three_turn_chain_no_fork")


def test_fork_on_text_diff():
    """Different user content under shared sys -> 2 leaves, sys shared."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "fork-text"
    sys_msg = {"role": "system", "content": "S"}

    for content in ["uA", "uB"]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        r = _render_response(content[-1], tokenizer=tok)
        mgr.record_turn(
            sid,
            turn=_turn(p, r, finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            response_message={"role": "assistant", "content": content[-1]},
        )

    root = mgr._trees[sid]
    assert len(root.children) == 1, "sys node must be shared"
    sys_node = root.children[0]
    assert len(sys_node.children) == 2, "user level must fork"
    leaves = [leaf for leaf in root.leaves() if not leaf.is_root]
    assert len(leaves) == 2
    print("PASS test_fork_on_text_diff")


def test_no_fork_on_token_only_diff():
    """Plan C: same text but tampered prompt_ids -> NO fork (DFS ignores tokens).

    This is the load-bearing behavior change vs the old prefix-match design.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "tokens-diff-only"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    pa = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(pa, _render_response("a", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        response_message={"role": "assistant", "content": "a"},
    )
    tampered = list(pa)
    tampered[1] = tampered[1] ^ 1
    mgr.record_turn(
        sid,
        turn=_turn(tampered, _render_response("b", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        response_message={"role": "assistant", "content": "b"},
    )
    root = mgr._trees[sid]
    # Same (sys, user) path -> shared, but two different assistant turns
    # produce two assistant leaves under the same user node.
    assert len(root.children) == 1
    sys_node = root.children[0]
    assert len(sys_node.children) == 1
    user_node = sys_node.children[0]
    assert len(user_node.children) == 2, "two distinct assistant turns hang off shared user"
    leaves = [leaf for leaf in root.leaves() if not leaf.is_root]
    assert len(leaves) == 2
    print("PASS test_no_fork_on_token_only_diff")


def test_cross_sid_isolation():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sys_msg = {"role": "system", "content": "S"}
    for sid, content in [("sid-a", "uA"), ("sid-b", "uB")]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        r = _render_response(content[-1], tokenizer=tok)
        mgr.record_turn(
            sid,
            turn=_turn(p, r, finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            response_message={"role": "assistant", "content": content[-1]},
        )
    assert len(list(mgr._trees["sid-a"].leaves())) == 1
    assert len(list(mgr._trees["sid-b"].leaves())) == 1
    print("PASS test_cross_sid_isolation")


def test_role_tool_in_chain():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "tool-chain"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool_a = {"role": "tool", "content": "tA"}
    tool_b = {"role": "tool", "content": "tB"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    p2 = _render_prompt([sys_msg, user1, asst1, tool_a, tool_b], tokenizer=tok)
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1, tool_a, tool_b],
        response_message=asst2,
    )

    chain = list(mgr._trees[sid].leaves())[0].path_from_root()
    roles = [n.role for n in chain]
    # Per-message routing: the two tool_results mount as two separate nodes.
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"], roles
    assert chain[3].message == tool_a
    assert chain[4].message == tool_b
    print("PASS test_role_tool_in_chain")


def test_response_logprobs_length_mismatch_raises():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    bad_turn = TurnRecord(
        prompt_ids=p,
        output_ids=[1, 2, 3],
        finish_reason="stop",
        output_log_probs=[-0.1, -0.2],
    )
    try:
        mgr.record_turn(
            "x",
            turn=bad_turn,
            prompt_messages=[sys_msg, user1],
            response_message={"role": "assistant", "content": ""},
        )
    except ValueError as e:
        assert "output_log_probs" in str(e)
        print("PASS test_response_logprobs_length_mismatch_raises")
        return
    raise AssertionError("expected ValueError")


def test_missing_logprobs_turn_stays_length_aligned():
    """A turn carrying NO logprobs (empty output_log_probs) must still produce
    a Sample whose rollout_log_probs is length-aligned with the response region
    (zeros, not a short array). Guards the snapshot field that distinguishes
    "no logprob signal" (empty/None) from a real per-token list -- a truthiness
    vs ``is not None`` slip here would silently misalign logprobs with tokens.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "no-logprobs"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool1 = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    # turn 1 WITH logprobs; turn 2 WITHOUT (empty list).
    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1, tool1], tokenizer=tok)
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),  # logprobs=None -> empty list
        prompt_messages=[sys_msg, user1, asst1, tool1],
        response_message=asst2,
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]
    assert len(s.rollout_log_probs) == len(s.loss_mask) == s.response_length
    # turn 1 region keeps its real logprobs; turn 2 region is zeros (no signal).
    L = _common_prefix_len(p1 + r1, p2)
    assert s.rollout_log_probs == [-0.5] * len(r1) + [0.0] * (len(p2) - L) + [0.0] * len(r2)
    print("PASS test_missing_logprobs_turn_stays_length_aligned")


def test_response_ids_empty_ok():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.record_turn(
        "x",
        turn=_turn(p, [], finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        response_message=None,
    )
    chain = list(mgr._trees["x"].leaves())[0].path_from_root()
    asst = chain[-1]
    assert asst.role == "assistant"
    assert asst.turn.output_ids == []
    assert asst.turn.prompt_ids == p
    assert asst.message is None
    print("PASS test_response_ids_empty_ok")


# ---------------------------------------------------------------------------
# get_trajectory linearization (Plan C heart of the matter)
# ---------------------------------------------------------------------------


def test_get_trajectory_single_turn():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "g1"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user], tokenizer=tok)
    r = _render_response("a", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p, r, finish_reason="stop", logprobs=[-0.5] * len(r)),
        prompt_messages=[sys_msg, user],
        response_message={"role": "assistant", "content": "a"},
    )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=7, prompt="hi"), reward=1.0)
    assert len(samples) == 1
    s = samples[0]
    assert s.tokens == p + r
    # slime contract: loss_mask / rollout_log_probs cover only the response
    # region (tokens after the initial prompt) and len(loss_mask) ==
    # response_length. The first turn's prompt prefix is stripped.
    assert s.loss_mask == [1] * len(r)
    assert s.rollout_log_probs == [-0.5] * len(r)
    assert s.response_length == len(r)
    assert s.reward == 1.0
    # Leaf Sample metadata is intentionally empty: no dataset-row passthrough,
    # no per-turn finish_reason snapshot (that lives on the tree nodes).
    assert s.metadata == {}
    print("PASS test_get_trajectory_single_turn")


def test_get_trajectory_clean_multiturn():
    """Clean 2-turn session (no drift) linearizes as turn1 prompt+resp then
    turn2 (prompt - LCP) + resp, with full coherent loss_mask / logprobs."""
    tok = FakeTokenizer()
    mgr, sid, turns = _three_turn_session(tok)
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]

    (p1, r1), (p2, r2) = turns
    # LCP(p1+r1, p2) should equal len(p1)+len(r1) for our clean fake tokenizer
    # (p2 starts exactly with p1 contents + asst response + tool block + new gen prompt)
    L = _common_prefix_len(p1 + r1, p2)
    assert L == len(p1) + len(r1), f"clean session LCP should equal cumulative, got {L}"
    expected_tokens = p1 + r1 + p2[L:] + r2
    # loss_mask / rollout_log_probs are response-only (slime contract):
    # turn1 prompt is stripped; turn1 response keeps mask=1, then the
    # extra prompt slice (p2[L:]) is mask=0, then turn2 response is mask=1.
    expected_loss = [1] * len(r1) + [0] * (len(p2) - L) + [1] * len(r2)
    expected_logp = [-0.5] * len(r1) + [0.0] * (len(p2) - L) + [-0.4] * len(r2)
    assert s.tokens == expected_tokens
    assert s.loss_mask == expected_loss
    assert s.rollout_log_probs == expected_logp
    assert s.response_length == len(r1) + (len(p2) - L) + len(r2)
    assert s.metadata == {}
    print("PASS test_get_trajectory_clean_multiturn")


def test_drift_case_A_prompt_region_forks():
    """case A: turn 2 prompt diverges from cumulative INSIDE a prompt region
    (the divergence index L falls outside every recorded response span). This
    is a genuine prompt-level re-render -> always fork into two coherent
    Samples; no token is dropped.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "drift-A"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    # Splice a phantom token INSIDE turn 1's prompt region (before len(p1)) so
    # the divergence index lands in a prompt region, not a response span.
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_at = len(p1) - 1  # inside p1's prompt region
    p2 = p2_honest[:drift_at] + [55555] + p2_honest[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user, asst1, tool],
        response_message=asst2,
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    # Two coherent segments: turn 1 alone, turn 2 alone (each its own Sample).
    assert len(samples) == 2
    s1, s2 = samples
    # segment 1 = turn 1: tokens p1+r1, loss/logprobs cover only r1.
    assert s1.tokens == p1 + r1
    assert s1.loss_mask == [1] * len(r1)
    assert s1.rollout_log_probs == [-0.5] * len(r1)
    # segment 2 = turn 2: tokens p2+r2, loss/logprobs cover only r2.
    assert s2.tokens == p2 + r2
    assert s2.loss_mask == [1] * len(r2)
    assert s2.rollout_log_probs == [-0.4] * len(r2)
    # No token lost; reward split evenly across the two segments.
    assert all(s.reward == 1.0 for s in samples)
    # alignment invariants
    for s in samples:
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
    print("PASS test_drift_case_A_prompt_region_forks")


def test_drift_case_B1_short_replaces():
    """case B1, d < threshold: turn 2 prompt re-renders turn 1's response tail
    with a SMALL drift (the divergence index falls inside the most-recent
    response span). The drifted tail is dropped and the buffer realigned to
    turn 2's prompt, producing a single coherent Sample.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()  # default threshold 1024 -> short drift replaces
    sid = "drift-B1"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    # Honest turn-2 prompt begins with p1 + r1; mutate the LAST token of r1's
    # echo so the divergence falls inside turn 1's response span (small d=1).
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    assert p2_honest[: len(p1) + len(r1)] == p1 + r1  # sanity: clean echo
    drift_idx = len(p1) + len(r1) - 1  # last token of r1 inside the response span
    p2 = list(p2_honest)
    p2[drift_idx] = p2[drift_idx] ^ 1  # flip one bit -> d = 1 drifted tail token
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user, asst1, tool],
        response_message=asst2,
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    # Single coherent Sample (replace, not fork).
    assert len(samples) == 1
    s = samples[0]
    L = _common_prefix_len(p1 + r1, p2)
    assert L == drift_idx  # divergence at the flipped token
    # After replace the buffer realigns to p2 then appends r2; the drifted r1
    # tail (1 token) is dropped, re-supplied as prompt context (loss=0).
    assert s.tokens == p2 + r2
    # response region (after first-turn prompt strip): the drifted r1 echo is no
    # longer a faithful response, so the WHOLE r1 span is masked (loss=0 prompt
    # context) -- both the aligned head and the realigned tail -- and only r2 trains.
    expected_loss = [0] * (len(p2) - len(p1)) + [1] * len(r2)
    expected_logp = [0.0] * (len(p2) - len(p1)) + [-0.4] * len(r2)
    assert s.loss_mask == expected_loss
    assert s.rollout_log_probs == expected_logp
    assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
    assert s.reward == 1.0
    print("PASS test_drift_case_B1_short_replaces")


def test_drift_case_B1_long_forks():
    """case B1, d >= threshold: the drifted response tail is too long to drop
    silently -> fork into two coherent Samples instead of replace.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager(fork_threshold_tokens=1)  # d>=1 -> fork
    sid = "drift-B1-long"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_idx = len(p1) + len(r1) - 1  # inside r1's span; d will be 1 >= threshold
    p2 = list(p2_honest)
    p2[drift_idx] = p2[drift_idx] ^ 1
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user, asst1, tool],
        response_message=asst2,
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2  # forked, not replaced
    s1, s2 = samples
    assert s1.tokens == p1 + r1
    assert s2.tokens == p2 + r2
    assert all(s.reward == 1.0 for s in samples)
    print("PASS test_drift_case_B1_long_forks")


def test_drift_case_B1_disabled_by_zero_threshold():
    """fork_threshold_tokens=0 forces B1 to fork too (max fidelity:
    never drop a drifted response tail)."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(fork_threshold_tokens=0)
    sid = "drift-B1-off"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_idx = len(p1) + len(r1) - 1
    p2 = list(p2_honest)
    p2[drift_idx] = p2[drift_idx] ^ 1
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),
        prompt_messages=[sys_msg, user, asst1, tool],
        response_message=asst2,
    )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2  # threshold 0 -> B1 forks
    print("PASS test_drift_case_B1_disabled_by_zero_threshold")


def test_drift_case_B2_earlier_turn_forks():
    """case B2: the divergence falls inside an EARLIER turn's response span (not
    the most recent one). Dropping that would discard the later turn(s) too, so
    B2 always forks regardless of drift size.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "drift-B2"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool1 = {"role": "tool", "content": "t1"}
    asst2 = {"role": "assistant", "content": "a2"}
    tool2 = {"role": "tool", "content": "t2"}
    asst3 = {"role": "assistant", "content": "a3"}

    # turn 1 + turn 2 clean.
    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user, asst1, tool1], tokenizer=tok)
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="tool_calls", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user, asst1, tool1],
        response_message=asst2,
    )
    # turn 3 re-renders turn 1's RESPONSE region differently. The divergence
    # falls inside r1's span, which is NOT the most-recent span (r2 is) -> B2.
    p3_honest = _render_prompt([sys_msg, user, asst1, tool1, asst2, tool2], tokenizer=tok)
    drift_idx = len(p1) + len(r1) - 1  # inside r1's span (an earlier turn)
    p3 = list(p3_honest)
    p3[drift_idx] = p3[drift_idx] ^ 1
    r3 = _render_response("a3", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop", logprobs=[-0.3] * len(r3)),
        prompt_messages=[sys_msg, user, asst1, tool1, asst2, tool2],
        response_message=asst3,
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    # Fork at turn 3: segment 1 = turns 1+2 (clean), segment 2 = turn 3 alone.
    assert len(samples) == 2
    s1, s2 = samples
    L12 = _common_prefix_len(p1 + r1, p2)
    assert s1.tokens == p1 + r1 + p2[L12:] + r2
    assert s2.tokens == p3 + r3
    assert all(s.reward == 1.0 for s in samples)
    print("PASS test_drift_case_B2_earlier_turn_forks")


def test_drift_fork_splits_reward_across_segments():
    """A fork producing N segments splits reward evenly across all of them."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "drift-reward"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_at = len(p1) - 1  # prompt region -> case A fork
    p2 = p2_honest[:drift_at] + [55555] + p2_honest[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),
        prompt_messages=[sys_msg, user, asst1, tool],
        response_message=asst2,
    )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=3.0)
    assert len(samples) == 2
    assert all(abs(s.reward - 1.5) < 1e-9 for s in samples)
    print("PASS test_drift_fork_splits_reward_across_segments")


def test_get_trajectory_two_leaves_share_reward():
    """Forked tree (2 leaves) -> reward split evenly."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "split"
    sys_msg = {"role": "system", "content": "S"}
    for content in ["uA", "uB"]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        r = _render_response(content[-1], tokenizer=tok)
        mgr.record_turn(
            sid,
            turn=_turn(p, r, finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            response_message={"role": "assistant", "content": content[-1]},
        )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2
    assert all(s.reward == 1.0 for s in samples)
    print("PASS test_get_trajectory_two_leaves_share_reward")


def test_shared_assistant_prefix_dedup_masks_later_leaf():
    """A generated turn (assistant) shared by 2 sibling leaves is trained only on
    the FIRST (DFS / build-order) leaf; the later leaf re-emits that turn's
    response as loss=0 prompt context, so the shared prefix is trained exactly
    once instead of once per leaf.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "shared-prefix"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool_x = {"role": "tool", "content": "tx"}
    tool_y = {"role": "tool", "content": "ty"}
    asst2 = {"role": "assistant", "content": "a2"}
    asst3 = {"role": "assistant", "content": "a3"}

    # Turn 1: the shared first assistant turn.
    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user],
        response_message=asst1,
    )
    # Turn 2: continues asst1 with tool_x -> first leaf.
    p2 = _render_prompt([sys_msg, user, asst1, tool_x], tokenizer=tok)
    r2 = _render_response("a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user, asst1, tool_x],
        response_message=asst2,
    )
    # Turn 3: continues the SAME asst1 with a different tool result -> forks a
    # second leaf, making asst1 a generated turn with 2 children.
    p3 = _render_prompt([sys_msg, user, asst1, tool_y], tokenizer=tok)
    r3 = _render_response("a3", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop", logprobs=[-0.3] * len(r3)),
        prompt_messages=[sys_msg, user, asst1, tool_y],
        response_message=asst3,
    )

    # Sanity: asst1 is a generated turn with exactly 2 children (the fork point).
    root = mgr._trees[sid]
    asst1_node = root.children[0].children[0].children[0]  # sys -> user -> asst1
    assert asst1_node.role == "assistant"
    assert asst1_node.turn is not None
    assert len(asst1_node.children) == 2

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2
    s_first, s_second = samples

    L2 = _common_prefix_len(p1 + r1, p2)
    L3 = _common_prefix_len(p1 + r1, p3)
    assert L2 == len(p1) + len(r1)  # clean continuation: p1+r1 is an exact prefix
    assert L3 == len(p1) + len(r1)

    # First leaf OWNS asst1: both r1 and r2 are trained (loss=1).
    assert s_first.tokens == p2 + r2
    assert s_first.loss_mask == [1] * len(r1) + [0] * (len(p2) - L2) + [1] * len(r2)
    assert s_first.rollout_log_probs == [-0.5] * len(r1) + [0.0] * (len(p2) - L2) + [-0.4] * len(r2)

    # Second leaf SHARES asst1: r1 is now loss=0 context, only r3 is trained.
    assert s_second.tokens == p3 + r3
    assert s_second.loss_mask == [0] * (len(p3) - len(p1)) + [1] * len(r3)
    assert s_second.rollout_log_probs == [0.0] * (len(p3) - len(p1)) + [-0.3] * len(r3)

    # Reward still split evenly; no sample fully masked; alignment invariant holds.
    assert all(s.reward == 1.0 for s in samples)
    for s in samples:
        assert sum(s.loss_mask) > 0
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
    print("PASS test_shared_assistant_prefix_dedup_masks_later_leaf")


def test_drop_clears_sid():
    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    assert mgr.has_session(sid)
    mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    assert not mgr.has_session(sid)
    assert mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt="")) == []
    print("PASS test_drop_clears_sid")


def test_debug_dump_shape():
    from tests.test_agent.test_claude_code_agent._dump_helpers import dump_tree_json, dump_tree_txt

    tok = FakeTokenizer()
    mgr, sid, _ = _three_turn_session(tok)
    txt = dump_tree_txt(mgr, sid)
    assert isinstance(txt, str) and txt
    for needle in ("session=", "[system]", "[user]", "[assistant]", "[tool]", "turns=2"):
        assert needle in txt, f"missing {needle!r}"
    # Plan C: assistant rows show turn= / prompt_ids= / response_ids=
    assert "turn=1" in txt
    assert "turn=2" in txt
    assert "prompt_ids=" in txt
    assert "response_ids=" in txt

    j = dump_tree_json(mgr, sid)
    assert j["found"] is True and j["sid"] == sid and j["turns"] == 2
    assert j["nodes_total"] == 5

    miss = dump_tree_txt(mgr, "no-such")
    assert miss == "<no session: no-such>"
    miss_j = dump_tree_json(mgr, "no-such")
    assert miss_j == {"sid": "no-such", "found": False}
    print("PASS test_debug_dump_shape")


def test_get_trajectory_skips_routing_assistant_in_drift_loop():
    """cc replays a foreign assistant the manager never recorded as a leaf;
    per-message routing mounts it as a routing-only assistant that must be
    filtered out of the strict-prefix walk.

    With per-message routing each prompt message mounts its own node, so the
    previously-recorded ``asst2`` leaf (a single-message node) still matches
    by ``(role, message-equality)`` and the chain descends through it — turn 2
    stays on the main path and re-enters the strict-prefix check. Only the
    extra ``foreign`` assistant, which the manager never saw via record_turn,
    has no matching leaf and mounts as a routing-only assistant
    (``turn`` / ``turn_index`` both None).

    Such routing assistants must be filtered out of ``asst_chain`` (they
    carry no per-turn snapshot). If one leaked into the strict prefix walk it
    would look like a turn with an empty prompt and trip the exact-prefix
    check — raising spuriously. This guards that the filter keeps the
    otherwise-clean trajectory from raising, while turns 1/2/3 all stay in
    the single linearized chain.

    Regression for the 20260604-120030 batch where 6 instances surfaced
    routing assistants at depth 23-39, exact pattern: cc replays a prior
    assistant message that the manager never recorded as its own leaf.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "routing-asst"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u1"}
    asst1 = {"role": "assistant", "content": "real-a1"}
    tool1 = {"role": "tool", "content": "t1"}
    asst2 = {"role": "assistant", "content": "real-a2"}

    # turn 1
    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("real-a1", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    # turn 2 — clean continuation
    p2 = _render_prompt([sys_msg, user1, asst1, tool1], tokenizer=tok)
    r2 = _render_response("real-a2", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1, asst1, tool1],
        response_message=asst2,
    )
    # turn 3 — cc replays an extra prior asst that manager never saw via
    # add_turn. Per-message routing matches asst2's own leaf and descends, so
    # only the unmatched `foreign` message mounts as a routing-only assistant.
    foreign = {"role": "assistant", "content": "foreign-msg"}
    tool2 = {"role": "tool", "content": "t2"}
    asst3 = {"role": "assistant", "content": "real-a3"}
    p3 = _render_prompt([sys_msg, user1, asst1, tool1, asst2, foreign, tool2], tokenizer=tok)
    r3 = _render_response("real-a3", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1, tool1, asst2, foreign, tool2],
        response_message=asst3,
    )

    # Per-message routing keeps everything on one path: asst2 matched, so no
    # spurious fork.
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1, [n.message for n in leaves]
    leaf3 = leaves[0]
    assert leaf3.message.get("content") == "real-a3"
    chain = leaf3.path_from_root()
    asst_nodes = [n for n in chain if n.role == "assistant"]

    # Exactly one routing assistant (the foreign replay); turns 1/2/3 keep
    # their snapshots and stay in asst_chain.
    routing_asst = [n for n in asst_nodes if n.turn is None]
    assert len(routing_asst) == 1, (
        f"expected exactly one routing assistant; got " f"{[(n.turn_index, n.turn is not None) for n in asst_nodes]}"
    )
    assert routing_asst[0].turn_index is None
    assert routing_asst[0].message.get("content") == "foreign-msg"
    snapshot_turns = [n.turn_index for n in asst_nodes if n.turn is not None]
    assert snapshot_turns == [1, 2, 3], snapshot_turns

    # The routing assistant is filtered out, so the strict prefix walk sees a
    # clean chain (turns 1/2/3) and does NOT raise.
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    assert len(samples) == 1
    print("PASS test_get_trajectory_skips_routing_assistant_in_drift_loop")


# ---------------------------------------------------------------------------
# Assistant-rewrite merge (single tolerated exception to strict exact-prefix)
# ---------------------------------------------------------------------------


def test_rewrite_merge_absorbs_short_assistant():
    """cc re-renders a short prior assistant; the manager absorbs the rewrite
    onto the existing leaf (demoted to routing-only) instead of forking a
    reward-diluting stub. One leaf, one Sample, original response not trained.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()  # default threshold 1024 -> merge ON
    sid = "rw-merge"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "ok"}  # short raw output
    asst1_rw = {"role": "assistant", "content": "ok "}  # cc-rewritten (whitespace)
    tool1 = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "done"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("ok", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    r2 = _render_response("done", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        response_message=asst2,
    )

    # Single chain (no fork): the rewrite was absorbed.
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1, [n.message for n in leaves]
    chain = leaves[0].path_from_root()
    assert [n.role for n in chain] == ["system", "user", "assistant", "tool", "assistant"]

    merged = chain[2]
    # Demoted to routing-only: snapshot cleared, adopted the rewritten message.
    assert merged.turn is None
    assert merged.turn_index is None
    assert merged.message == asst1_rw
    assert merged.metadata["merged_rewrite"]["abandoned_turn_index"] == 1
    assert merged.metadata["merged_rewrite"]["abandoned_response_tokens"] == len(r1)

    # Linearization: only turn 2 participates; the abandoned turn-1 response is
    # NOT trained. tokens == p2 + r2; loss covers only r2.
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    assert len(samples) == 1
    s = samples[0]
    assert s.tokens == p2 + r2
    assert s.loss_mask == [1] * len(r2)
    assert s.rollout_log_probs == [-0.4] * len(r2)
    assert s.reward == 1.0
    print("PASS test_rewrite_merge_absorbs_short_assistant")


def test_rewrite_merge_skips_long_assistant():
    """A long abandoned response (>= threshold) is NOT absorbed: it forks into
    its own Sample (carrying enough real signal to train standalone). Forking
    does not raise.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager(fork_threshold_tokens=2)  # r1 (3 tok) >= 2
    sid = "rw-long"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "ok"}
    asst1_rw = {"role": "assistant", "content": "ok "}
    tool1 = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "done"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("ok", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    r2 = _render_response("done", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        response_message=asst2,
    )

    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 2, "long rewrite must fork, not merge"
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2  # no raise
    print("PASS test_rewrite_merge_skips_long_assistant")


def test_rewrite_merge_disabled_by_zero_threshold():
    """fork_threshold_tokens=0 disables merge: every rewrite forks."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(fork_threshold_tokens=0)
    sid = "rw-off"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "ok"}
    asst1_rw = {"role": "assistant", "content": "ok "}
    tool1 = {"role": "tool", "content": "t"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, _render_response("ok", tokenizer=tok), finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, _render_response("done", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        response_message={"role": "assistant", "content": "done"},
    )
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 2, "merge disabled -> rewrite forks"
    print("PASS test_rewrite_merge_disabled_by_zero_threshold")


def test_rewrite_merge_ambiguous_candidates_fork():
    """Two eligible short-leaf assistant siblings -> ambiguous -> fork (no
    arbitrary merge).
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "rw-ambig"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}

    # Two turns sharing the (sys, user) prefix produce two assistant leaves.
    for content in ["a", "b"]:
        p = _render_prompt([sys_msg, user1], tokenizer=tok)
        mgr.record_turn(
            sid,
            turn=_turn(p, _render_response(content, tokenizer=tok), finish_reason="stop"),
            prompt_messages=[sys_msg, user1],
            response_message={"role": "assistant", "content": content},
        )
    user_node = mgr._trees[sid].children[0].children[0]
    assert len(user_node.children) == 2

    # A third turn rewrites at the assistant slot -> two merge candidates.
    asst_c = {"role": "assistant", "content": "c"}
    tool1 = {"role": "tool", "content": "t"}
    p3 = _render_prompt([sys_msg, user1, asst_c, tool1], tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p3, _render_response("d", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst_c, tool1],
        response_message={"role": "assistant", "content": "d"},
    )
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 3, "ambiguous candidates must fork, not merge"
    print("PASS test_rewrite_merge_ambiguous_candidates_fork")


def test_rewrite_merge_mixed_short_long_forks_without_destroying_short():
    """Mount point holds two generated leaves -- one short (eligible), one long
    (>= threshold, ineligible). The rewrite's true target is undecidable from
    content, so it forks. Crucially the short leaf's TurnRecord is NOT destroyed:
    the old guard counted only eligible candidates, found exactly one, and would
    have absorbed (irreversibly) onto a node the rewrite may not even target.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager(fork_threshold_tokens=4)  # "long" (5 tok) >= 4 > "a" (2 tok)
    sid = "rw-mixed"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}

    # Two turns share the (sys, user) prefix: one short response, one long.
    for content in ["a", "long"]:  # _render_response -> len(content)+1 tokens
        p = _render_prompt([sys_msg, user1], tokenizer=tok)
        mgr.record_turn(
            sid,
            turn=_turn(p, _render_response(content, tokenizer=tok), finish_reason="stop"),
            prompt_messages=[sys_msg, user1],
            response_message={"role": "assistant", "content": content},
        )
    user_node = mgr._trees[sid].children[0].children[0]
    assert len(user_node.children) == 2
    short_leaf = next(c for c in user_node.children if c.turn is not None and len(c.turn.output_ids) == 2)

    # A rewrite arrives at the assistant slot. Two assistant children -> ambiguous.
    asst_c = {"role": "assistant", "content": "c"}
    tool1 = {"role": "tool", "content": "t"}
    p3 = _render_prompt([sys_msg, user1, asst_c, tool1], tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p3, _render_response("d", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst_c, tool1],
        response_message={"role": "assistant", "content": "d"},
    )

    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 3, "mixed short/long must fork, not merge"
    # The short leaf survived intact -- not demoted to routing-only.
    assert short_leaf.turn is not None, "short leaf's TurnRecord must not be destroyed"
    assert "merged_rewrite" not in short_leaf.metadata
    print("PASS test_rewrite_merge_mixed_short_long_forks_without_destroying_short")


def test_rewrite_merge_non_assistant_mismatch_forks():
    """A non-assistant divergence (different user) is left to fork; the merge
    hook does not touch it even with merge enabled.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "rw-nonasst"
    sys_msg = {"role": "system", "content": "S"}
    for content in ["uA", "uB"]:
        user = {"role": "user", "content": content}
        p = _render_prompt([sys_msg, user], tokenizer=tok)
        mgr.record_turn(
            sid,
            turn=_turn(p, _render_response(content[-1], tokenizer=tok), finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            response_message={"role": "assistant", "content": content[-1]},
        )
    sys_node = mgr._trees[sid].children[0]
    assert len(sys_node.children) == 2, "user-level divergence still forks"
    print("PASS test_rewrite_merge_non_assistant_mismatch_forks")


def test_rewrite_merge_match_key_updated_so_next_turn_descends():
    """Regression: after merge, the node's cached match_key must follow the
    adopted (rewritten) message so a LATER turn's DFS descends through it
    instead of forking again. Also exercises clean strict-prefix continuation
    across the merged node.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "rw-matchkey"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "ok"}
    asst1_rw = {"role": "assistant", "content": "ok "}
    tool1 = {"role": "tool", "content": "t1"}
    asst2 = {"role": "assistant", "content": "second"}
    tool2 = {"role": "tool", "content": "t2"}
    asst3 = {"role": "assistant", "content": "third"}

    # turn 1: short assistant.
    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p1, _render_response("ok", tokenizer=tok), finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        response_message=asst1,
    )
    # turn 2: rewrite asst1 -> merge.
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    r2 = _render_response("second", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="tool_calls", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        response_message=asst2,
    )
    # turn 3: prompt carries the rewritten asst1_rw again; DFS must descend
    # through the merged node (match_key updated) and not fork.
    p3 = _render_prompt([sys_msg, user1, asst1_rw, tool1, asst2, tool2], tokenizer=tok)
    r3 = _render_response("third", tokenizer=tok)
    mgr.record_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop", logprobs=[-0.3] * len(r3)),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1, asst2, tool2],
        response_message=asst3,
    )

    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1, ("match_key not updated -> spurious fork", [n.message for n in leaves])

    # Strict prefix holds across the merged node: turns 2 and 3 linearize into
    # one clean Sample (no raise). The demoted turn-1 node is filtered out.
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    assert len(samples) == 1
    s = samples[0]
    L = _common_prefix_len(p2 + r2, p3)
    assert s.tokens == p2 + r2 + p3[L:] + r3
    print("PASS test_rewrite_merge_match_key_updated_so_next_turn_descends")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    test_message_equality_is_dict_internal_sort_only()
    test_common_prefix_len()
    test_classify_drift_boundaries()
    test_clean_iff_held_is_exact_prefix()
    test_append_single_turn_shapes_tree()
    test_append_three_turn_chain_no_fork()
    test_fork_on_text_diff()
    test_no_fork_on_token_only_diff()
    test_cross_sid_isolation()
    test_role_tool_in_chain()
    test_response_logprobs_length_mismatch_raises()
    test_missing_logprobs_turn_stays_length_aligned()
    test_response_ids_empty_ok()
    test_get_trajectory_single_turn()
    test_get_trajectory_clean_multiturn()
    test_drift_case_A_prompt_region_forks()
    test_drift_case_B1_short_replaces()
    test_drift_case_B1_long_forks()
    test_drift_case_B1_disabled_by_zero_threshold()
    test_drift_case_B2_earlier_turn_forks()
    test_drift_fork_splits_reward_across_segments()
    test_get_trajectory_two_leaves_share_reward()
    test_drop_clears_sid()
    test_debug_dump_shape()
    test_get_trajectory_skips_routing_assistant_in_drift_loop()
    test_rewrite_merge_absorbs_short_assistant()
    test_rewrite_merge_skips_long_assistant()
    test_rewrite_merge_disabled_by_zero_threshold()
    test_rewrite_merge_ambiguous_candidates_fork()
    test_rewrite_merge_mixed_short_long_forks_without_destroying_short()
    test_rewrite_merge_non_assistant_mismatch_forks()
    test_rewrite_merge_match_key_updated_so_next_turn_descends()
    print("\nALL PLAN-C TESTS PASSED.")


if __name__ == "__main__":
    main()
