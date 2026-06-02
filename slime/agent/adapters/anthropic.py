"""Anthropic Messages adapter for agent rollouts.

The adapter exposes ``/v1/messages`` and ``/v1/messages/count_tokens``. It
renders each Anthropic message history with the served model's chat template,
calls SGLang ``/generate`` with ``input_ids``, and folds the turn into a
per-session turn-node :class:`~slime.agent.trajectory_manager.TrajectoryTree`.

The tree routes everything by text prefix, so Claude Code sub-agent and
compaction patterns split into independent leaves automatically -- no manual
``active_sub`` / ``wipe`` bookkeeping. Call ``finish_session()`` at trajectory
end to drain trainable ``TokenSegment`` objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import (
    ADAPTER_KEY,
    REASONING_PARSER_KEY,
    TOKENIZER_KEY,
    TOOL_PARSER_KEY,
    BaseAdapter,
    Session,
    assemble_turns,
    call_sglang_generate,
    ok_response,
    render_prompt,
    request_session_id,
)
from slime.agent.parsing import parse_model_output
from slime.agent.trajectory_manager import record_turn

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter with session lifecycle helpers."""

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        super().__init__(
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        self.app.router.add_post("/v1/messages", _handle_request)
        self.app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)


# =============================================================================
# Translation (Anthropic wire <-> chat-template messages) -- unchanged
# =============================================================================


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


# =============================================================================
# Reply building (raw output text -> Anthropic content blocks) -- unchanged shape
# =============================================================================


async def _generate(prompt_ids: list[int], s: Session, body: dict, app, *, session_id: str | None = None):
    """Call sglang and return a GenResult (output ids/logprobs/text)."""
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


def _build_reply(output_text: str, finish: str, tools_schema: list[dict] | None, app) -> tuple[list[dict], str]:
    """Turn the model's raw output text into the reply we send back to claude-code.

    1. parse decoded text -> (thinking, visible, tool_uses) via sglang parsers
    2. pack into Anthropic content blocks
    3. derive stop_reason: 'tool_use' | 'max_tokens' | 'end_turn'

    Returns (blocks, stop_reason).
    """
    parsed = parse_model_output(
        output_text or "",
        tools_schema=tools_schema,
        tool_parser_name=app[TOOL_PARSER_KEY],
        reasoning_parser_name=app[REASONING_PARSER_KEY],
    )
    blocks = _anthropic_blocks(parsed.reasoning, parsed.text, parsed.tool_uses)
    return blocks, _stop_reason(parsed.tool_uses, finish)


def _anthropic_blocks(thinking: str, visible: str, tool_uses: list[dict]) -> list[dict]:
    """Pack parsed model output into Anthropic content blocks."""
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if visible:
        blocks.append({"type": "text", "text": visible})
    for tu in tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _stop_reason(tool_uses: list[dict], finish: str) -> str:
    if tool_uses:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "end_turn"


# =============================================================================
# Request handling -- one full turn + SSE wrap
# =============================================================================


def _request_session_id(request: web.Request) -> str:
    return request_session_id(request, include_x_api_key=True)


async def _handle_request(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = _request_session_id(request)
    adapter = request.app[ADAPTER_KEY]
    if sid in adapter.closed:  # session drained; refuse stragglers
        return web.Response(status=503, text="session closed")
    app = request.app
    tok = app[TOKENIZER_KEY]
    s = adapter.store.setdefault(sid, Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:  # same sid -> serialized
            translated = _translate_anthropic(body.get("messages") or [], body.get("system"))
            tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))
            full_prompt_ids = render_prompt(s.traj, translated, tok, tools_schema)
            gen = await _generate(full_prompt_ids, s, body, app, session_id=sid)
            turns, pending_key = assemble_turns(s.traj, translated, tok, tools_schema, gen, full_prompt_ids)
            node = record_turn(s.traj.tree, turns)
            if node is not None:
                s.traj.resp_truth[pending_key] = (
                    list(gen.output_ids),
                    list(gen.output_log_probs),
                    gen.output_text,
                )
            blocks, stop = _build_reply(gen.output_text, gen.finish_reason, tools_schema, app)
            in_tok, out_tok = len(full_prompt_ids), len(gen.output_ids)
        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await _stream_response(request, blocks, stop, in_tok, out_tok)
        return web.json_response(_message_response(body, blocks, stop, in_tok, out_tok))
    finally:
        adapter.inflight.get(sid, set()).discard(task)


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


# Trivial endpoints claude-code probes during a session: count_tokens runs
# every turn (return 0 -- client uses it as a hint, not a hard budget),
# healthz/v1/models are startup readiness checks.
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return await ok_response(request)
