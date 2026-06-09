"""Anthropic Messages adapter for agent rollouts.

The adapter exposes ``/v1/messages`` and ``/v1/messages/count_tokens``. It
renders each Anthropic message history with the served model's chat template,
calls SGLang ``/generate`` with ``input_ids``, and feeds the turn into a
shared :class:`~slime.agent.trajectory_manager.TrajectoryManager` keyed by
session id. ``finish_session(sid)`` drains a session's trajectory into a list
of :class:`~slime.utils.types.Sample`.

The per-sid tree inside TrajectoryManager handles sub-agent and compaction
patterns automatically (any divergence in the prompt prefix forks into a new
leaf), so we no longer track ``main`` / ``active_sub`` chains here.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import BaseAdapter, Reply, request_session_id
from slime.agent.parsing import ParsedModelOutput

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter.

    Exposes ``/v1/messages`` and ``/v1/messages/count_tokens``. All the
    shared turn machinery lives in :class:`~slime.agent.adapters.common.BaseAdapter`;
    this class only supplies the Anthropic wire translation and reply framing.
    """

    logger = logger
    log_prefix = "anthropic_adapter"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request)

    def _preprocess_body(self, body: dict) -> None:
        _fold_mid_list_system_into_user(body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        translated = _translate_anthropic(body.get("messages") or [], body.get("system"))
        tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        blocks, stop_reason, response_message, finish_reason = _build_blocks_and_response_message(parsed, raw_finish)
        return Reply(
            manager_message=response_message,
            finish_reason=finish_reason,
            wire=(blocks, stop_reason),
            skip_append=_is_cc_title_generation_request(translated, tools_schema),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        blocks, stop_reason = reply.wire
        if stream:
            return await _stream_response(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_message_response(body, blocks, stop_reason, in_tok, out_tok))


# =============================================================================
# Translation (Anthropic wire <-> chat-template messages)
# =============================================================================


_MID_SYSTEM_WRAP_PREFIX = "<system-reminder>\n"
_MID_SYSTEM_WRAP_SUFFIX = "\n</system-reminder>\n"


def _fold_mid_list_system_into_user(body_obj: dict) -> bool:
    """Fold non-leading ``role: system`` messages into a neighbouring user
    message as a ``<system-reminder>`` text block. Mutates ``body_obj`` in
    place; returns True iff any fold happened.

    Claude Code CLI >= 2.1.161 inserts ``{"role":"system","content":"<skills
    list>"}`` in the middle of ``messages``. Qwen3-style chat templates reject
    any system message past index 0 with ``System message must be at the
    beginning.`` This wrap mirrors the older claude-code (<= 2.1.143)
    behaviour by attaching the wrapped reminder to the preceding user message
    (or the next one if no prior user message exists).
    """
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    system_idx = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and i > 0]
    if not system_idx:
        return False

    def _promote_to_list(msg: dict) -> list:
        c = msg.get("content")
        if isinstance(c, list):
            return c
        msg["content"] = [{"type": "text", "text": c if isinstance(c, str) else ""}]
        return msg["content"]

    def _wrap(text: str) -> dict:
        return {
            "type": "text",
            "text": _MID_SYSTEM_WRAP_PREFIX + text + _MID_SYSTEM_WRAP_SUFFIX,
        }

    changed = False
    TOMBSTONE: dict = {"__folded__": True}
    for i in system_idx:
        sys_msg = msgs[i]
        wrapped = _wrap(_flatten(sys_msg.get("content")))
        target = None
        for j in range(i - 1, -1, -1):
            cand = msgs[j]
            if isinstance(cand, dict) and cand.get("role") == "user":
                target = cand
                _promote_to_list(target).append(wrapped)
                break
        if target is None:
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if isinstance(cand, dict) and cand.get("role") == "user":
                    target = cand
                    _promote_to_list(target).insert(0, wrapped)
                    break
        if target is None:
            msgs[i] = {"role": "user", "content": [wrapped]}
            changed = True
            continue
        msgs[i] = TOMBSTONE
        changed = True

    if changed:
        body_obj["messages"] = [m for m in msgs if m is not TOMBSTONE]
    return changed


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


# Marker string Claude Code embeds in the system prompt of its per-session
# title-generation request (a meta request that asks the LLM to produce a
# short conversation title). Title-gen requests should NOT enter the RL
# trajectory — they aren't agent work. See spec
# docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md.
_CC_TITLE_GEN_MARKER = "Generate a concise, sentence-case title"


def _is_cc_title_generation_request(
    translated: list[dict],
    tools_schema: list[dict] | None,
) -> bool:
    """Return True iff this is a Claude Code per-session title-generation request.

    Detection is AND-conjunction:
      (1) ``tools_schema`` is falsy (cc sends tools=[]; converter returns None).
      (2) one of the leading ``role=system`` messages' content contains
          ``_CC_TITLE_GEN_MARKER``.

    Scanning stops at the first non-system message — title-gen system blocks
    always sit at the head of the request.
    """
    if tools_schema:
        return False
    for msg in translated:
        if msg.get("role") != "system":
            break
        content = msg.get("content")
        if isinstance(content, str):
            if _CC_TITLE_GEN_MARKER in content:
                return True
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and _CC_TITLE_GEN_MARKER in block["text"]
                ):
                    return True
    return False


def _tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call used both for echoed history and sampled
    leaves so TrajectoryManager.node_match_key hashes them identically.

    ``arguments`` stays a dict (NOT a JSON string): the same list is fed to the
    chat template (Qwen3's template does ``arguments | items``, which needs a
    mapping -- a string raises "Can only get item pairs from a mapping"), and
    node_match_key's ``json.dumps(sort_keys=True)`` hashes equivalent dicts
    identically regardless of key order. The wire-only ``tool_use`` id is
    intentionally dropped so a sampled leaf and its replayed echo match.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


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
                    # Drop the wire-only "id"; see _tool_call_dict for why
                    # arguments stays a dict and ids are dropped.
                    tcs.append(_tool_call_dict(b.get("name", "tool"), b.get("input")))
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
# Reply building: parsed output -> Anthropic blocks + OpenAI-shape response_message
# =============================================================================


def _build_blocks_and_response_message(
    parsed: ParsedModelOutput,
    finish: str,
) -> tuple[list[dict], str, dict[str, Any], str]:
    """Pack parsed model output into:
      - Anthropic content blocks (sent over the wire),
      - wire ``stop_reason`` (tool_use / max_tokens / end_turn),
      - response_message (OpenAI shape) for TrajectoryManager.append_turn,
      - manager ``finish_reason`` (tool_calls / raw finish) for the TurnRecord.

    The tool_calls inside response_message use canonical args (see
    ``_tool_call_dict``) so the node_match_key the manager computes for this
    assistant turn matches the same turn replayed as history on the next
    /v1/messages request.
    """
    blocks: list[dict] = []
    if parsed.reasoning:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning})
    if parsed.text:
        blocks.append({"type": "text", "text": parsed.text})

    response_tcs: list[dict] = []
    for tu in parsed.tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        # The id above is wire-only; _tool_call_dict drops it (and keeps
        # arguments a dict) so the leaf matches its replayed echo.
        response_tcs.append(_tool_call_dict(tu["name"], tu.get("input")))

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if parsed.tool_uses:
        stop_reason, finish_reason = "tool_use", "tool_calls"
    elif finish == "length":
        stop_reason, finish_reason = "max_tokens", finish
    else:
        stop_reason, finish_reason = "end_turn", finish or "stop"

    response_message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        response_message["reasoning_content"] = parsed.reasoning
    if response_tcs:
        response_message["tool_calls"] = response_tcs

    return blocks, stop_reason, response_message, finish_reason


# =============================================================================
# Request handling -- session id + wire response framing
# =============================================================================


def _request_session_id(request: web.Request) -> str:
    return request_session_id(request, include_x_api_key=True)


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


# count_tokens runs every turn; claude-code uses it as a hint, not a hard
# budget, so returning 0 is fine. (healthz / v1/models readiness probes are
# served by BaseAdapter via common._ok.)
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})
