"""Unit tests for src_v2.trajectory_manager (Plan C: token-faithful).

What we test:
  (1) DFS merge only on (role, node_match_key): same prefix in messages
      space always lands on the same path regardless of prompt_ids drift.
  (2) get_trajectory linearization: turn 1 = full prompt + response;
      turn k>=2 = strict exact-prefix append; tokens / loss_mask /
      logprobs all stay in sync.
  (3) TITO drift handling: when turn k+1.prompt diverges mid-stream from
      cumulative tokens, get_trajectory RAISES (no silent drop/realign).
"""

from __future__ import annotations

import json

from slime.agent.adapters.common import TurnRecord  # noqa: E402
from slime.agent.trajectory_manager import TrajectoryManager, _lcp_len, node_match_key  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


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


def test_node_match_key_is_dict_internal_sort_only():
    a = [{"role": "u", "content": "x"}]
    b = [{"content": "x", "role": "u"}]
    assert node_match_key(a) == node_match_key(b)

    c = [{"role": "u", "content": "x"}, {"role": "u", "content": "y"}]
    d = [{"role": "u", "content": "y"}, {"role": "u", "content": "x"}]
    assert node_match_key(c) != node_match_key(d)

    e = [{"role": "assistant", "tool_calls": [{"id": "1", "type": "function"}]}]
    f = [{"role": "assistant", "tool_calls": [{"type": "function", "id": "1"}]}]
    assert node_match_key(e) == node_match_key(f)
    print("PASS test_node_match_key_is_dict_internal_sort_only")


def test_lcp_len():
    assert _lcp_len([], []) == 0
    assert _lcp_len([1, 2, 3], []) == 0
    assert _lcp_len([], [1, 2, 3]) == 0
    assert _lcp_len([1, 2, 3], [1, 2, 3]) == 3
    assert _lcp_len([1, 2, 3], [1, 2, 4]) == 2
    assert _lcp_len([1, 2, 3, 4, 5], [1, 2, 3]) == 3
    print("PASS test_lcp_len")


def test_manager_accepts_fork_threshold():
    # default 1024 when unspecified
    m_default = TrajectoryManager()
    assert m_default._fork_threshold == 1024
    # explicit value honored
    m_explicit = TrajectoryManager(fork_merge_max_response_tokens=256)
    assert m_explicit._fork_threshold == 256
    # None -> default
    m_none = TrajectoryManager(fork_merge_max_response_tokens=None)
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
    """3-turn linear session via append_turn. Returns mgr, sid, per-turn (p,r)."""
    mgr = TrajectoryManager()
    sid = "three-turn"

    sys_msg = {"role": "system", "content": SYSTEM_MSG}
    user1 = {"role": "user", "content": "Compute 2+2."}
    asst1 = {"role": "assistant", "content": "Computing."}
    tool1 = {"role": "tool", "content": "4"}
    asst2 = {"role": "assistant", "content": "Answer is 4."}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("Computing.", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user1],
        tools=TOOLS_OPENAI,
        response_message=asst1,
    )

    p2 = _render_prompt([sys_msg, user1, asst1, tool1], tokenizer=tok)
    r2 = _render_response("Answer is 4.", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user1, asst1, tool1],
        tools=TOOLS_OPENAI,
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
    mgr.append_turn(
        sid,
        turn=_turn(p, r, finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message={"role": "assistant", "content": "a"},
    )
    chain = list(mgr._trees[sid].leaves())[0].path_from_root()
    roles = [n.role for n in chain]
    assert roles == ["system", "user", "assistant"], roles
    asst = chain[-1]
    assert asst.turn_index == 1
    assert asst.turn_prompt_ids == p
    assert asst.turn_response_ids == r
    assert asst.turn_finish_reason == "stop"
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
        mgr.append_turn(
            sid,
            turn=_turn(p, r, finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            tools=None,
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
    mgr.append_turn(
        sid,
        turn=_turn(pa, _render_response("a", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message={"role": "assistant", "content": "a"},
    )
    tampered = list(pa)
    tampered[1] = tampered[1] ^ 1
    mgr.append_turn(
        sid,
        turn=_turn(tampered, _render_response("b", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        tools=None,
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
        mgr.append_turn(
            sid,
            turn=_turn(p, r, finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            tools=None,
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
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=asst1,
    )
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1, tool_a, tool_b],
        tools=None,
        response_message=asst2,
    )

    chain = list(mgr._trees[sid].leaves())[0].path_from_root()
    roles = [n.role for n in chain]
    # Per-message routing: the two tool_results mount as two separate nodes.
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"], roles
    assert chain[3].messages == [tool_a]
    assert chain[4].messages == [tool_b]
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
        mgr.append_turn(
            "x",
            turn=bad_turn,
            prompt_messages=[sys_msg, user1],
            tools=None,
            response_message={"role": "assistant", "content": ""},
        )
    except ValueError as e:
        assert "output_log_probs" in str(e)
        print("PASS test_response_logprobs_length_mismatch_raises")
        return
    raise AssertionError("expected ValueError")


def test_response_ids_empty_ok():
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    p = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.append_turn(
        "x",
        turn=_turn(p, [], finish_reason="stop"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=None,
    )
    chain = list(mgr._trees["x"].leaves())[0].path_from_root()
    asst = chain[-1]
    assert asst.role == "assistant"
    assert asst.turn_response_ids == []
    assert asst.turn_prompt_ids == p
    assert asst.messages == []
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
    mgr.append_turn(
        sid,
        turn=_turn(p, r, finish_reason="stop", logprobs=[-0.5] * len(r)),
        prompt_messages=[sys_msg, user],
        tools=TOOLS_OPENAI,
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
    # no per-turn tools / finish_reason snapshot (those live on the tree nodes).
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
    L = _lcp_len(p1 + r1, p2)
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


def test_get_trajectory_tito_drift_raises():
    """Strict exact-prefix: turn 2 prompt diverges mid-stream from cumulative
    tokens. The accumulated turn-1 (prompt + response) is no longer a prefix of
    turn 2's prompt, so get_trajectory must RAISE instead of dropping/realigning.
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "tito"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "a2"}

    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user],
        tools=None,
        response_message=asst1,
    )
    # Build turn 2 prompt the "honest" way, then INJECT a synthetic divergence
    # inside the assistant response region — simulating chat-template drift. We
    # splice 3 fake tokens past the LCP so cumulative (p1+r1) is no longer a
    # prefix of p2.
    p2_honest = _render_prompt([sys_msg, user, asst1, tool], tokenizer=tok)
    drift_at = len(p1) + 1  # inside r1
    p2 = list(p2_honest)
    p2 = p2[:drift_at] + [77777, 77778, 77779] + p2[drift_at:]
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user, asst1, tool],
        tools=None,
        response_message=asst2,
    )

    try:
        mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
    except ValueError as e:
        msg = str(e)
        assert "TITO drift" in msg, msg
        assert f"turn={2}" in msg, msg  # the turn whose prompt failed the check
        assert "common_prefix_len" in msg, msg
        print("PASS test_get_trajectory_tito_drift_raises")
        return
    raise AssertionError("expected ValueError on TITO drift")


def test_get_trajectory_tito_drift_late_surfacing_attributes_early_turn():
    """A drift introduced in an early turn's region but only re-rendered at a
    later turn must still raise, and the message should attribute the divergence
    to the early turn's prompt region (not just the failing turn).
    """
    tok = FakeTokenizer()
    mgr = TrajectoryManager()
    sid = "tito-late"
    sys_msg = {"role": "system", "content": "S"}
    user = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "a1"}
    tool1 = {"role": "tool", "content": "t1"}
    asst2 = {"role": "assistant", "content": "a2"}
    tool2 = {"role": "tool", "content": "t2"}
    asst3 = {"role": "assistant", "content": "a3"}

    # turn 1 + turn 2: clean continuation.
    p1 = _render_prompt([sys_msg, user], tokenizer=tok)
    r1 = _render_response("a1", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user],
        tools=None,
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user, asst1, tool1], tokenizer=tok)
    r2 = _render_response("a2", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user, asst1, tool1],
        tools=None,
        response_message=asst2,
    )
    # turn 3: re-render the turn-1 region differently (splice a phantom token
    # inside p1's range). Cumulative now diverges from p3 deep inside turn 1.
    p3_honest = _render_prompt([sys_msg, user, asst1, tool1, asst2, tool2], tokenizer=tok)
    drift_at = len(p1) - 1  # inside turn 1's prompt region
    p3 = list(p3_honest)
    p3 = p3[:drift_at] + [99999] + p3[drift_at:]
    r3 = _render_response("a3", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop"),
        prompt_messages=[sys_msg, user, asst1, tool1, asst2, tool2],
        tools=None,
        response_message=asst3,
    )

    try:
        mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    except ValueError as e:
        msg = str(e)
        assert "TITO drift" in msg, msg
        # The failing turn is turn 3, but the divergence falls in turn 1's region.
        assert "turn=3" in msg, msg
        assert "turn 1's prompt region" in msg, msg
        print("PASS test_get_trajectory_tito_drift_late_surfacing_attributes_early_turn")
        return
    raise AssertionError("expected ValueError on late-surfacing TITO drift")


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
        mgr.append_turn(
            sid,
            turn=_turn(p, r, finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            tools=None,
            response_message={"role": "assistant", "content": content[-1]},
        )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2
    assert all(s.reward == 1.0 for s in samples)
    print("PASS test_get_trajectory_two_leaves_share_reward")


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
    by ``(role, node_match_key)`` and the chain descends through it — turn 2
    stays on the main path and re-enters the strict-prefix check. Only the
    extra ``foreign`` assistant, which the manager never saw via append_turn,
    has no matching leaf and mounts as a routing-only assistant
    (``turn_prompt_ids`` / ``turn_index`` both None).

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
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=asst1,
    )
    # turn 2 — clean continuation
    p2 = _render_prompt([sys_msg, user1, asst1, tool1], tokenizer=tok)
    r2 = _render_response("real-a2", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1, asst1, tool1],
        tools=None,
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
    mgr.append_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1, tool1, asst2, foreign, tool2],
        tools=None,
        response_message=asst3,
    )

    # Per-message routing keeps everything on one path: asst2 matched, so no
    # spurious fork.
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1, [n.messages for n in leaves]
    leaf3 = leaves[0]
    assert leaf3.messages[0].get("content") == "real-a3"
    chain = leaf3.path_from_root()
    asst_nodes = [n for n in chain if n.role == "assistant"]

    # Exactly one routing assistant (the foreign replay); turns 1/2/3 keep
    # their snapshots and stay in asst_chain.
    routing_asst = [n for n in asst_nodes if n.turn_prompt_ids is None]
    assert len(routing_asst) == 1, (
        f"expected exactly one routing assistant; got "
        f"{[(n.turn_index, n.turn_prompt_ids is not None) for n in asst_nodes]}"
    )
    assert routing_asst[0].turn_index is None
    assert routing_asst[0].messages[0].get("content") == "foreign-msg"
    snapshot_turns = [n.turn_index for n in asst_nodes if n.turn_prompt_ids is not None]
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
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls", logprobs=[-0.5] * len(r1)),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    r2 = _render_response("done", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        tools=None,
        response_message=asst2,
    )

    # Single chain (no fork): the rewrite was absorbed.
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1, [n.messages for n in leaves]
    chain = leaves[0].path_from_root()
    assert [n.role for n in chain] == ["system", "user", "assistant", "tool", "assistant"]

    merged = chain[2]
    # Demoted to routing-only: snapshot cleared, adopted the rewritten message.
    assert merged.turn_prompt_ids is None
    assert merged.turn_index is None
    assert merged.messages == [asst1_rw]
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
    mgr = TrajectoryManager(fork_merge_max_response_tokens=2)  # r1 (3 tok) >= 2
    sid = "rw-long"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "ok"}
    asst1_rw = {"role": "assistant", "content": "ok "}
    tool1 = {"role": "tool", "content": "t"}
    asst2 = {"role": "assistant", "content": "done"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    r1 = _render_response("ok", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p1, r1, finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    r2 = _render_response("done", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        tools=None,
        response_message=asst2,
    )

    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 2, "long rewrite must fork, not merge"
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=2.0)
    assert len(samples) == 2  # no raise
    print("PASS test_rewrite_merge_skips_long_assistant")


def test_rewrite_merge_disabled_by_zero_threshold():
    """fork_merge_max_response_tokens=0 disables merge: every rewrite forks."""
    tok = FakeTokenizer()
    mgr = TrajectoryManager(fork_merge_max_response_tokens=0)
    sid = "rw-off"
    sys_msg = {"role": "system", "content": "S"}
    user1 = {"role": "user", "content": "u"}
    asst1 = {"role": "assistant", "content": "ok"}
    asst1_rw = {"role": "assistant", "content": "ok "}
    tool1 = {"role": "tool", "content": "t"}

    p1 = _render_prompt([sys_msg, user1], tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p1, _render_response("ok", tokenizer=tok), finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=asst1,
    )
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, _render_response("done", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        tools=None,
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
        mgr.append_turn(
            sid,
            turn=_turn(p, _render_response(content, tokenizer=tok), finish_reason="stop"),
            prompt_messages=[sys_msg, user1],
            tools=None,
            response_message={"role": "assistant", "content": content},
        )
    user_node = mgr._trees[sid].children[0].children[0]
    assert len(user_node.children) == 2

    # A third turn rewrites at the assistant slot -> two merge candidates.
    asst_c = {"role": "assistant", "content": "c"}
    tool1 = {"role": "tool", "content": "t"}
    p3 = _render_prompt([sys_msg, user1, asst_c, tool1], tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p3, _render_response("d", tokenizer=tok), finish_reason="stop"),
        prompt_messages=[sys_msg, user1, asst_c, tool1],
        tools=None,
        response_message={"role": "assistant", "content": "d"},
    )
    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 3, "ambiguous candidates must fork, not merge"
    print("PASS test_rewrite_merge_ambiguous_candidates_fork")


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
        mgr.append_turn(
            sid,
            turn=_turn(p, _render_response(content[-1], tokenizer=tok), finish_reason="stop"),
            prompt_messages=[sys_msg, user],
            tools=None,
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
    mgr.append_turn(
        sid,
        turn=_turn(p1, _render_response("ok", tokenizer=tok), finish_reason="tool_calls"),
        prompt_messages=[sys_msg, user1],
        tools=None,
        response_message=asst1,
    )
    # turn 2: rewrite asst1 -> merge.
    p2 = _render_prompt([sys_msg, user1, asst1_rw, tool1], tokenizer=tok)
    r2 = _render_response("second", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p2, r2, finish_reason="tool_calls", logprobs=[-0.4] * len(r2)),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1],
        tools=None,
        response_message=asst2,
    )
    # turn 3: prompt carries the rewritten asst1_rw again; DFS must descend
    # through the merged node (match_key updated) and not fork.
    p3 = _render_prompt([sys_msg, user1, asst1_rw, tool1, asst2, tool2], tokenizer=tok)
    r3 = _render_response("third", tokenizer=tok)
    mgr.append_turn(
        sid,
        turn=_turn(p3, r3, finish_reason="stop", logprobs=[-0.3] * len(r3)),
        prompt_messages=[sys_msg, user1, asst1_rw, tool1, asst2, tool2],
        tools=None,
        response_message=asst3,
    )

    leaves = [leaf for leaf in mgr._trees[sid].leaves() if not leaf.is_root]
    assert len(leaves) == 1, ("match_key not updated -> spurious fork", [n.messages for n in leaves])

    # Strict prefix holds across the merged node: turns 2 and 3 linearize into
    # one clean Sample (no raise). The demoted turn-1 node is filtered out.
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0, prompt=""))
    assert len(samples) == 1
    s = samples[0]
    L = _lcp_len(p2 + r2, p3)
    assert s.tokens == p2 + r2 + p3[L:] + r3
    print("PASS test_rewrite_merge_match_key_updated_so_next_turn_descends")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    test_node_match_key_is_dict_internal_sort_only()
    test_lcp_len()
    test_append_single_turn_shapes_tree()
    test_append_three_turn_chain_no_fork()
    test_fork_on_text_diff()
    test_no_fork_on_token_only_diff()
    test_cross_sid_isolation()
    test_role_tool_in_chain()
    test_response_logprobs_length_mismatch_raises()
    test_response_ids_empty_ok()
    test_get_trajectory_single_turn()
    test_get_trajectory_clean_multiturn()
    test_get_trajectory_tito_drift_raises()
    test_get_trajectory_tito_drift_late_surfacing_attributes_early_turn()
    test_get_trajectory_two_leaves_share_reward()
    test_drop_clears_sid()
    test_debug_dump_shape()
    test_get_trajectory_skips_routing_assistant_in_drift_loop()
    test_rewrite_merge_absorbs_short_assistant()
    test_rewrite_merge_skips_long_assistant()
    test_rewrite_merge_disabled_by_zero_threshold()
    test_rewrite_merge_ambiguous_candidates_fork()
    test_rewrite_merge_non_assistant_mismatch_forks()
    test_rewrite_merge_match_key_updated_so_next_turn_descends()
    print("\nALL PLAN-C TESTS PASSED.")


if __name__ == "__main__":
    main()
