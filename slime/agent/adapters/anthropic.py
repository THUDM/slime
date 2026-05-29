"""Anthropic Messages adapter for agent rollouts.

The adapter exposes ``/v1/messages`` and ``/v1/messages/count_tokens``. It
renders each Anthropic message history with the served model's chat template,
calls SGLang ``/generate`` with ``input_ids``, and records the exact sampled
token ids/logprobs as ``TurnRecord`` objects. ``pop_session_split()`` converts
those records into trainable ``TokenSegment`` objects.

It also handles Claude Code sub-agent and compaction patterns by splitting one
session into ``subagent``, ``wipe``, and ``final`` segments.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import AdapterChain as Chain
from slime.agent.adapters.common import (
    call_sglang_generate,
    ok_response,
    register_session,
    render_token_ids,
    request_session_id,
    shutdown_session_tasks,
)
from slime.agent.adapters.common import stable_hash as _hash
from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TokenSegment, TurnRecord, TurnSegment, make_turn_segment, merge_turn_segments

logger = logging.getLogger(__name__)


# Tool names claude-code uses to dispatch a sub-agent.
_SUBAGENT_TOOLS = {"Task", "Agent"}


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    active_sub: Chain | None = None  # at most one sub-agent at a time
    pending_dispatch_id: str = ""  # tool_use_id we're waiting to close
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)  # frozen output


_Store = dict[str, Session]


# Drain state for shutdown_session. Module-level so it survives
# pop_session_split; _closed is a permanent tombstone (late requests 503).
_inflight: dict[str, set[asyncio.Task]] = {}
_closed: set[str] = set()


# =============================================================================
# 2. Per-turn stages
# =============================================================================


def _select_chain(s: Session, body: dict) -> tuple[Chain, bool, str]:
    """Decide which chain this turn operates on.

    1. fingerprint body.messages and body.system into hashes
    2. if main now contains the tool_result for a pending sub dispatch,
       snapshot the sub chain into s.segments and clear s.active_sub
    3. pick main vs s.active_sub based on whether request continues main's prefix
    4. classify as 'new' | 'append' | 'wipe' against the chosen target;
       a wipe also snapshots the target's current state into s.segments

    Returns (target_chain, is_sub, kind).
    """
    all_msgs = body.get("messages") or []
    msg_hashes = [_hash(m) for m in all_msgs]
    req_system_hash = _hash(body.get("system")) if "system" in body else s.main.system_hash

    # Close active sub-agent if its dispatch tool_result has landed on main.
    if s.pending_dispatch_id and s.active_sub is not None:
        tu_id = s.pending_dispatch_id
        for m in all_msgs:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            done = any(
                isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tu_id
                for b in content
            )
            if done:
                if s.active_sub.turns:
                    s.segments.append(make_turn_segment(s.active_sub.turns, kind="subagent"))
                s.active_sub = None
                s.pending_dispatch_id = ""
                break

    # Route: main iff request continues main's prefix. Sub system_hash can be
    # "" (armed before sub dialled in), so never route by sub equality alone.
    if s.active_sub is None:
        target, is_sub = s.main, False
    else:
        main_continues = (
            req_system_hash == s.main.system_hash
            and len(msg_hashes) >= s.main.seen_msgs
            and msg_hashes[: s.main.seen_msgs] == s.main.msg_hashes[: s.main.seen_msgs]
        )
        target, is_sub = (s.main, False) if main_continues else (s.active_sub, True)

    # Classify; snapshot a "wipe" segment first if we're discarding work.
    if target.seen_msgs == 0:
        kind = "new"
    else:
        is_append = (
            req_system_hash == target.system_hash
            and len(msg_hashes) >= target.seen_msgs
            and msg_hashes[: target.seen_msgs] == target.msg_hashes[: target.seen_msgs]
        )
        if is_append:
            kind = "append"
        else:
            if target.turns:
                s.segments.append(make_turn_segment(target.turns, kind="wipe"))
            kind = "wipe"

    return target, is_sub, kind


def _flatten(c: Any) -> str:
    """Recursive Anthropic content flattener: text/tool_result(content) joined
    by newline, images replaced with a placeholder."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                parts.append(_flatten(b.get("content")))
            elif t == "image":
                parts.append("[image omitted]")
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def _translate_anthropic(msgs: list[dict], system: Any) -> list[dict]:
    """Anthropic messages + system -> chat-template messages. Pure function."""
    translated: list[dict] = []
    if system:
        translated.append({"role": "system", "content": _flatten(system)})
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    translated.append({"role": "tool", "content": _flatten(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": _flatten(b)})
        elif role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    tcs.append({"function": {"name": b.get("name", "tool"), "arguments": b.get("input") or {}}})
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": _flatten(content)})
    return translated


def _anthropic_tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tools to tokenizer chat-template tool schema."""
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None


def _replace_chat_messages(target: Chain, body: dict) -> None:
    """new/wipe: full reset of chat state and turn log."""
    all_msgs = body.get("messages") or []
    target.chat_messages = _translate_anthropic(all_msgs, body.get("system"))
    if "system" in body:
        target.system_hash = _hash(body.get("system"))
    target.turns.clear()
    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))


def _extend_chat_messages(target: Chain, body: dict) -> None:
    """append: translate only the new tail."""
    all_msgs = body.get("messages") or []
    translated = _translate_anthropic(all_msgs[target.seen_msgs :], None)
    target.chat_messages.extend(translated)

    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))


def _build_prompt(target: Chain, body: dict, kind: str, tok) -> list[int]:
    """Replace/extend chat_messages and render input ids for sglang."""
    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, body)
    return render_token_ids(target, tok)


async def _generate(
    prompt_ids: list[int], s: Session, body: dict, app, *, session_id: str | None = None
) -> TurnRecord:
    """Call sglang and return a TurnRecord.

    1. build sampling_params (session defaults overlaid with body overrides)
    2. POST sglang /generate; on cancel/error fire /abort_request
    3. keep the exact prompt/output token ids; trajectory merge later compares
       later prompt tokens with earlier outputs to build the loss mask
    """
    return await call_sglang_generate(
        prompt_ids,
        s,
        body,
        app,
        max_token_keys=("max_tokens",),
        stop_keys=("stop_sequences",),
        log_prefix="anthropic_adapter",
        logger=logger,
        session_id=session_id,
    )


def _build_reply(target: Chain, output_ids: list[int], finish: str, app) -> tuple[list[dict], str, str]:
    """Turn the model's raw output ids into the reply we send back to claude-code.

    1. parse decoded text -> (thinking, visible, tool_uses) via sglang parsers
    2. pack into Anthropic content blocks; tag dispatch_id when a tool_use
       names Task/Agent (sub-agent trigger)
    3. derive stop_reason: 'tool_use' | 'max_tokens' | 'end_turn'

    Returns (blocks, stop_reason, dispatch_id).
    """
    tok = app["tokenizer"]

    raw_output = tok.decode(output_ids, skip_special_tokens=False) if output_ids else ""
    parsed = parse_model_output(
        raw_output,
        tools_schema=target.tools_schema,
        tool_parser_name=app["tool_parser"],
        reasoning_parser_name=app["reasoning_parser"],
    )
    blocks, dispatch_id = _anthropic_blocks(parsed.reasoning, parsed.text, parsed.tool_uses)
    return blocks, _stop_reason(parsed.tool_uses, finish), dispatch_id


def _anthropic_blocks(thinking: str, visible: str, tool_uses: list[dict]) -> tuple[list[dict], str]:
    """Pack parsed model output into Anthropic content blocks."""
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if visible:
        blocks.append({"type": "text", "text": visible})
    dispatch_id = ""
    for tu in tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        if tu["name"] in _SUBAGENT_TOOLS:
            dispatch_id = tu_id
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks, dispatch_id


def _stop_reason(tool_uses: list[dict], finish: str) -> str:
    if tool_uses:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "end_turn"


def _start_sub_chain(s: Session, dispatch_id: str) -> None:
    """Start a fresh sub chain on this session and remember the tool_use_id
    we'll watch for on main to know when this sub is done. The matching
    'sub done' step lives inside _select_chain."""
    s.pending_dispatch_id = dispatch_id
    if s.active_sub is None:
        s.active_sub = Chain()


# =============================================================================
# 3. Request handling -- one full turn + SSE wrap
# =============================================================================


def _request_session_id(request: web.Request) -> str:
    return request_session_id(request, include_x_api_key=True)


async def _handle_request(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = _request_session_id(request)
    if sid in _closed:  # session drained; refuse stragglers
        return web.Response(status=503, text="session closed")
    app = request.app
    s = app["store"].setdefault(sid, Session())
    task = asyncio.current_task()
    _inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:  # same sid -> serialized
            target, is_sub, kind = _select_chain(s, body)
            ideal_ids = _build_prompt(target, body, kind, app["tokenizer"])
            turn = await _generate(ideal_ids, s, body, app, session_id=sid)
            blocks, stop, did = _build_reply(target, turn.output_ids, turn.finish_reason, app)
            target.turns.append(turn)
            if did and not is_sub:  # sub doesn't nest
                _start_sub_chain(s, did)
            in_tok, out_tok = len(ideal_ids), len(turn.output_ids)
        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await _stream_response(request, blocks, stop, in_tok, out_tok)
        return web.json_response(_message_response(body, blocks, stop, in_tok, out_tok))
    finally:
        _inflight.get(sid, set()).discard(task)


def _message_response(body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "slime-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


async def _stream_response(request, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    """Stream blocks back to claude-code as an Anthropic Messages SSE
    response: message_start, (content_block_start, content_block_delta,
    content_block_stop)*N, message_delta, message_stop."""
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    # message_start
    ms_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": "slime-actor",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms_data, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:  # tool_use
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }

        cbs_data = {"type": "content_block_start", "index": idx, "content_block": start}
        await out.write(f"event: content_block_start\ndata: {json.dumps(cbs_data, ensure_ascii=False)}\n\n".encode())

        cbd_data = {"type": "content_block_delta", "index": idx, "delta": delta}
        await out.write(f"event: content_block_delta\ndata: {json.dumps(cbd_data, ensure_ascii=False)}\n\n".encode())

        cbe_data = {"type": "content_block_stop", "index": idx}
        await out.write(f"event: content_block_stop\ndata: {json.dumps(cbe_data, ensure_ascii=False)}\n\n".encode())

    md_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md_data, ensure_ascii=False)}\n\n".encode())

    mst_data = {"type": "message_stop"}
    await out.write(f"event: message_stop\ndata: {json.dumps(mst_data, ensure_ascii=False)}\n\n".encode())

    return out


# =============================================================================
# 4. Public API
# =============================================================================


def open_session(
    store: _Store,
    sid: str,
    *,
    sampling_defaults: dict | None = None,
    max_context_tokens: int = 0,
) -> None:
    """Register a new session. Fail-fast on duplicate sid: silently sharing
    state would interleave two independent rollouts into one chain and corrupt
    chain bookkeeping. `sampling_defaults` seeds the session's default sglang
    sampling_params (overlaid by per-request body in `_generate`).
    `max_context_tokens` caps each turn's prompt+response budget and drops
    oversized final segments; 0 disables this guard."""
    register_session(
        store,
        sid,
        Session,
        sampling_defaults=sampling_defaults,
        max_context_tokens=max_context_tokens,
    )


def pop_session_split(store: _Store, sid: str) -> list[TokenSegment]:
    """Snapshot whatever chains are still alive (active_sub + main) into
    segments, drop empty and oversized ones. Called by the train loop at
    trajectory end."""
    s = store.pop(sid, None)
    if s is None:
        return []
    if s.active_sub is not None and s.active_sub.turns:
        s.segments.append(make_turn_segment(s.active_sub.turns, kind="subagent"))
    if s.main.turns:
        s.segments.append(make_turn_segment(s.main.turns, kind="final"))

    return merge_turn_segments(s.segments, max_context_tokens=s.max_context_tokens)


async def shutdown_session(sid: str, *, wait_timeout: float = 5.0) -> None:
    """Tombstone sid (late requests 503) and drain in-flight local handlers
    (cancel fires /abort_request to sglang). Does NOT wait for sglang idle --
    sglang_engine.release_memory_occupation already calls flush_cache() with
    60×1s polling. Idempotent."""
    await shutdown_session_tasks(sid, _closed, _inflight, wait_timeout=wait_timeout)


# Trivial endpoints claude-code probes during a session: count_tokens runs
# every turn (return 0 -- client uses it as a hint, not a hard budget),
# healthz/v1/models are startup readiness checks.
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return await ok_response(request)


def start(*, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None):
    """Build the aiohttp app + store. Caller is responsible for running the
    server (e.g. `aiohttp.web.run_app` or a daemon-thread wrapper).

    The runner MUST set ``handler_cancellation=True`` so a client disconnect
    actually cancels the handler coroutine, arming the fire-and-forget
    /abort_request inside `_generate`. Without it a cancelled client leaves an
    inflight sglang /generate that races with the next release_memory_occupation
    and trips sglang's "server is idle" assertion -- crashing the scheduler.

    Use `open_session(store, sid, ...)` to register a session before
    claude-code dials in (fail-fast on duplicate sid, seeds sampling defaults).
    Use `pop_session_split(store, sid)` to drain its trajectory at rollout end.
    """
    store: _Store = {}
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["tokenizer"] = tokenizer
    app["sglang_url"] = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
    app["tool_parser"] = tool_parser
    app["reasoning_parser"] = reasoning_parser
    app["store"] = store
    app.router.add_post("/v1/messages", _handle_request)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    return app, store
