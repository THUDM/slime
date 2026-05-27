"""Anthropic Messages API shim that translates claude-code requests into
sglang ``/generate`` calls while tracking the token-level training target
for slime RL rollouts.

Each turn:
* fingerprints the incoming Anthropic conversation against per-session state
  to decide whether it continues the main chain or an active sub-agent chain,
  and whether the request appends to / wipes / restarts that chain;
* renders the chosen message chain through the model's chat template;
* posts to sglang ``/generate`` and records prompt_ids/output_ids as a
  TurnRecord;
* parses the decoded output back into Anthropic blocks and streams them to
  claude-code as a Messages SSE response.

Public surface:
    start(...)                       build the aiohttp app + store
    open_session(store, sid, ...)    register a session with sampling defaults
    pop_session_split(store, sid)    drain a session's trajectory for training
    shutdown_session(sid, ...)       tombstone sid + drain in-flight handlers
    Chain, Session                   dataclasses (exposed for type hints)

Read `_handle_request` (§3) top-to-bottom -- that's the online turn:
    _select_chain      pick main vs active sub; snapshot wipe/sub-done into segments
    _build_prompt      replace/extend chat_messages -> render token ids
    _generate          POST sglang /generate -> TurnRecord
    _build_reply       output_ids -> Anthropic blocks
    append turn        remember the prompt/output boundary for later merge

`pop_session_split()` drains frozen TurnRecords and merges them into training
segments. The online path serves and records; the pop path linearizes.

`_build_prompt` is itself a thin orchestrator over three helpers:
    _replace_chat_messages / _extend_chat_messages    translate Anthropic blocks -> chat_messages
    _render_token_ids                                 chat template -> input ids

Design notes:
* `kind` (new/wipe/append) is consumed at the dispatch site:
  `_replace_chat_messages` vs `_extend_chat_messages` is picked at call time.
* `_render_token_ids` only reads target; the caller owns all state mutation.
* Loss-mask repair happens in `trajectory.merge_turns`: later prompt tokens are
  compared with earlier generated tokens, and unmatched historical outputs stop
  being trainable.
* `_hash` strips Anthropic `cache_control` keys before hashing so the same
  logical message hashes identically across turns even as cache_control moves.
* Server lifecycle (binding a port, running the loop, `handler_cancellation`)
  lives in the caller, not here -- see `start()` for the required runner
  contract.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import secrets
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TokenSegment, TurnRecord, TurnSegment, make_turn_segment, merge_turn_segments

logger = logging.getLogger(__name__)


# Tool names claude-code uses to dispatch a sub-agent.
_SUBAGENT_TOOLS = {"Task", "Agent"}


def _strip_cache_control(obj: Any) -> Any:
    """Drop Anthropic prompt-caching ``cache_control`` keys before hashing -
    cache_control moves across turns so the same logical message would
    otherwise hash differently each request."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _hash(obj: Any) -> str:
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


# =============================================================================
# 1. Data structures
# =============================================================================


@dataclasses.dataclass
class Chain:
    """One conversation chain (main, or an active sub-agent)."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)

    # Online turn log. pop_session_split() merges these into training tensors.
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)


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


def _render_token_ids(target: Chain, tok) -> list[int]:
    """Render the current Claude Code message history into model input ids.

    We intentionally tokenize the messages as-is on every request. If a later
    prompt no longer token-matches an earlier raw model output, trajectory
    merging will mask/drop the unmatched history instead of rewriting the
    online prompt.
    """
    # Qwen3.x fast tokenizers return a BatchEncoding here, not a list[int];
    # list(BatchEncoding) yields dict keys (["input_ids", ...]) and poisons
    # sglang /generate. Unwrap input_ids defensively.
    enc = tok.apply_chat_template(
        target.chat_messages,
        tools=target.tools_schema,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def _build_prompt(target: Chain, body: dict, kind: str, tok) -> list[int]:
    """Replace/extend chat_messages and render input ids for sglang."""
    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, body)
    return _render_token_ids(target, tok)


async def _generate(prompt_ids: list[int], s: Session, body: dict, app) -> TurnRecord:
    """Call sglang and return a TurnRecord.

    1. build sampling_params (session defaults overlaid with body overrides)
    2. POST sglang /generate; on cancel/error fire /abort_request
    3. keep the exact prompt/output token ids; trajectory merge later compares
       later prompt tokens with earlier outputs to build the loss mask
    """
    # ---- (a) Build sampling_params --------------------------------------
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(s.sampling_defaults or {}),
    }
    if "max_tokens" in body:
        # Claude's request cap may be lower, but rollout_max_response_len is
        # the per-turn ceiling from slime. Keep the stricter of the two.
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body["max_tokens"])), int(body["max_tokens"]))
    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]
    if body.get("stop_sequences"):
        sp["stop"] = body["stop_sequences"]

    if s.max_context_tokens > 0:
        remaining_context = s.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[middleware] prompt exceeds max_context_tokens (%d >= %d); returning length stop",
                len(prompt_ids),
                s.max_context_tokens,
            )
            return TurnRecord(
                prompt_ids=list(prompt_ids),
                output_ids=[],
                finish_reason="length",
            )
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    # ---- (b) POST sglang /generate (with abort on cancel/error) --------
    # Without abort, a cancelled client + inflight request can race with the
    # next release_memory_occupation and trip sglang's "server is idle" assert.
    sglang_url = app["sglang_url"]
    rid = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": prompt_ids,
                "sampling_params": sp,
                "return_logprob": True,
            },
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            # SGLang's /generate can return JSON with application/octet-stream
            # as the Content-Type; aiohttp rejects that unless we disable the
            # header check.
            data = await r.json(content_type=None)
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError):
        # Best-effort abort with fresh short-timeout session; swallow errors.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
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


async def _handle_request(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = request.headers["Authorization"].removeprefix("Bearer ").strip()
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
            turn = await _generate(ideal_ids, s, body, app)
            blocks, stop, did = _build_reply(target, turn.output_ids, turn.finish_reason, app)
            target.turns.append(turn)
            if did and not is_sub:  # sub doesn't nest
                _start_sub_chain(s, did)
            in_tok, out_tok = len(ideal_ids), len(turn.output_ids)
        return await _stream_response(request, blocks, stop, in_tok, out_tok)
    finally:
        _inflight.get(sid, set()).discard(task)


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
    if sid in store:
        raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
    s = store[sid] = Session()
    s.sampling_defaults = dict(sampling_defaults or {})
    s.max_context_tokens = int(max_context_tokens or 0)


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
    _closed.add(sid)
    tasks = [t for t in _inflight.pop(sid, ()) if not t.done()]
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# Trivial endpoints claude-code probes during a session: count_tokens runs
# every turn (return 0 -- client uses it as a hint, not a hard budget),
# healthz/v1/models are startup readiness checks.
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


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
