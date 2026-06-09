"""OpenAI Chat-Completions adapter for agent rollouts.

Mirrors :mod:`slime.agent.adapters.anthropic` but speaks the OpenAI
``/v1/chat/completions`` wire protocol so that the Codex CLI (and any other
OpenAI-compatible client) can drive the slime SGLang server. Each incoming
request is rendered with the served model's chat template, sent to SGLang
``/generate`` as ``input_ids``, parsed, and folded into a shared
:class:`~slime.agent.trajectory_manager.TrajectoryManager` keyed by session id.
``finish_session(sid)`` drains a session's trajectory into a list of
:class:`~slime.utils.types.Sample`.

The per-sid tree inside TrajectoryManager handles sub-agent and compaction
patterns automatically (any divergence in the prompt prefix forks into a new
leaf), so we do not track explicit chains here.

Only ``/v1/chat/completions`` is implemented; the older Responses API
(``/v1/responses``) is intentionally out of scope -- Codex 0.30.0 uses
``wire_api = "chat"``.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import BaseAdapter, Reply, request_session_id
from slime.agent.parsing import ParsedModelOutput

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter):
    """OpenAI Chat-Completions-compatible HTTP adapter.

    Exposes ``/v1/chat/completions``. The older Responses API
    (``/v1/responses``) is intentionally out of scope -- Codex 0.30.0 uses
    ``wire_api = "chat"``. All shared turn machinery lives in
    :class:`~slime.agent.adapters.common.BaseAdapter`; this class only supplies
    the OpenAI wire translation and reply framing.
    """

    logger = logger
    log_prefix = "openai_adapter"
    max_token_keys = ("max_completion_tokens", "max_tokens", "max_output_tokens")
    stop_keys = ("stop",)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/chat/completions", self._run_turn)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request, body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            raise web.HTTPBadRequest(text="messages must be a list")
        translated = _translate_openai_chat(messages)
        tools_schema = _openai_tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        wire_message, manager_message, wire_finish = _build_oai_response(parsed, raw_finish)
        return Reply(
            manager_message=manager_message,
            finish_reason=_finish_reason_for_manager(raw_finish, parsed.tool_uses),
            wire=(wire_message, wire_finish),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        wire_message, wire_finish = reply.wire
        if stream:
            return await _stream_chat_completion(request, body, wire_message, wire_finish, in_tok, out_tok)
        return web.json_response(_chat_completion_response(body, wire_message, wire_finish, in_tok, out_tok))


# =============================================================================
# Translation (OpenAI wire <-> chat-template messages)
# =============================================================================


def _flatten_content(content: Any) -> str:
    """Flatten OpenAI text/content parts into a chat-template string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        typ = item.get("type")
        if typ in {"text", "input_text", "output_text"}:
            parts.append(item.get("text", ""))
        elif typ in {"image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in item:
            parts.append(_flatten_content(item.get("content")))
        elif "text" in item:
            parts.append(str(item.get("text") or ""))
    return "\n".join(p for p in parts if p)


def _arguments_as_dict(arguments: Any) -> dict[str, Any]:
    """Coerce wire-shape ``tool_calls[].function.arguments`` into a dict.

    OpenAI sends ``arguments`` as a JSON-encoded string; the chat template and
    ``trajectory_manager.node_match_key`` both expect a mapping. ``json.loads``
    is tried first; malformed payloads fall back to ``{"_raw_arguments": s}``
    (mirrors :func:`slime.agent.parsing.parse_tool_uses`).
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return {"_raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"_raw_arguments": arguments}
    return {"_raw_arguments": str(arguments)}


def _translate_openai_chat(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> tokenizer chat-template messages.

    Mirrors :func:`slime.agent.adapters.anthropic._translate_anthropic` so that
    a replayed assistant turn hashes identically (via
    ``trajectory_manager.node_match_key``) to the leaf the manager appended on
    the previous request. Two invariants must hold:

      * ``tool_calls[i].function.arguments`` is a ``dict`` (NOT a JSON string).
        Qwen3-style chat templates call ``arguments | items`` which requires
        a mapping; ``node_match_key`` uses ``json.dumps(sort_keys=True)`` so
        equivalent dicts hash the same regardless of key order.
      * Wire-only correlation ids are DROPPED: ``tool_call_id`` on ``role:
        "tool"`` history messages and ``tool_calls[i].id`` on echoed assistant
        messages. The adapter mints fresh ids on each response, so keeping the
        wire ids would diverge the replay hash from the original leaf.
    """
    translated: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "developer":  # OpenAI Responses API alias
            role = "system"

        if role in {"system", "user"}:
            translated.append({"role": role, "content": _flatten_content(content)})
        elif role == "tool":
            # DROP tool_call_id -- wire-only correlation field; see docstring.
            translated.append({"role": "tool", "content": _flatten_content(content)})
        elif role == "assistant":
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": _flatten_content(content),
            }
            reasoning = msg.get("reasoning_content")
            if reasoning:
                assistant["reasoning_content"] = reasoning
            tool_calls = msg.get("tool_calls") or []
            normalized: list[dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = function.get("name") or call.get("name") or "tool"
                arguments = function.get("arguments")
                if arguments is None:
                    arguments = call.get("arguments", {})
                # NB: arguments stays a dict (NOT a JSON string), and we DROP
                # the wire-only ``id``. See docstring above.
                normalized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _arguments_as_dict(arguments),
                        },
                    }
                )
            if normalized:
                assistant["tool_calls"] = normalized
            translated.append(assistant)
        # Unknown roles are silently dropped.
    return translated


def _openai_tools_to_chat_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI tools list to tokenizer chat-template tool schema."""
    if not tools:
        return None
    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") and tool.get("type") != "function":
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is not None:
            name = function.get("name")
            if not name:
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        else:
            name = tool.get("name")
            if not name:
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
    return normalized or None


# =============================================================================
# Reply building: parsed output -> OpenAI wire message + manager response_message
# =============================================================================


def _build_oai_response(parsed: ParsedModelOutput, finish: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return ``(wire_message, manager_message, finish_reason)``.

    ``wire_message`` follows OpenAI Chat-Completions spec: ``tool_calls[].id``
    is a unique correlation id, and ``tool_calls[].function.arguments`` is a
    JSON-encoded **string** (clients depend on this).

    ``manager_message`` is the shape fed to ``TrajectoryManager.append_turn``:
    ``tool_calls[].function.arguments`` is a **dict** so chat-template replay
    (Qwen3 etc.) succeeds and ``node_match_key`` hashes match the echo on the
    next turn. The wire-only ``id`` is omitted (the next turn's echo will not
    include it; matching anthropic.py:444-470).
    """
    wire_tool_calls: list[dict[str, Any]] = []
    manager_tool_calls: list[dict[str, Any]] = []
    for tu in parsed.tool_uses:
        name = tu.get("name", "tool")
        args_dict = tu.get("input") or {}
        if not isinstance(args_dict, dict):
            args_dict = {"_raw_arguments": str(args_dict)}
        call_id = f"call_{secrets.token_hex(12)}"
        wire_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False, sort_keys=True),
                },
            }
        )
        manager_tool_calls.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_dict,
                },
            }
        )

    wire_message: dict[str, Any] = {
        "role": "assistant",
        # OpenAI spec allows null content when tool_calls are present. The
        # Codex CLI 0.30.0 splits a single assistant turn containing both
        # text and tool_calls into TWO echoed messages on the next request
        # (a tool_calls-only one followed by a text-only one), which breaks
        # node_match_key against our leaf -- so when we have tool_calls we
        # send content=null so the echo is a single tool_calls-only message.
        "content": None if wire_tool_calls else (parsed.text or None),
    }
    # ``manager_message`` must match what the OAI client will echo on the
    # next request, otherwise ``node_match_key`` diverges and every turn
    # forks. Three empirically necessary differences vs ``wire_message``:
    #
    #   * NO ``reasoning_content`` -- the Codex CLI strips it on echo, so we
    #     must not store it in the leaf either. The reasoning token ids are
    #     preserved in ``response_ids`` (used for loss), only the rendered
    #     text is dropped.
    #   * Only the FIRST ``tool_call`` -- the Codex CLI silently drops any
    #     additional parallel tool_calls on echo (validates them serially and
    #     aborts after the first response), so the leaf can only hold one.
    #   * When ``tool_calls`` is present, ``content`` is empty -- the wire
    #     side also sends ``content=null`` (see above) so the echo stays
    #     a single tool_calls-only message that matches the leaf.
    manager_message: dict[str, Any] = {
        "role": "assistant",
        "content": "" if wire_tool_calls else (parsed.text or ""),
    }
    if parsed.reasoning:
        wire_message["reasoning_content"] = parsed.reasoning
    if wire_tool_calls:
        wire_message["tool_calls"] = wire_tool_calls[:1]
        manager_message["tool_calls"] = manager_tool_calls[:1]

    if parsed.tool_uses:
        wire_finish = "tool_calls"
    elif finish == "length":
        wire_finish = "length"
    else:
        wire_finish = "stop"

    return wire_message, manager_message, wire_finish


def _finish_reason_for_manager(finish: str, tool_uses: list[dict]) -> str:
    if tool_uses:
        return "tool_calls"
    return finish or "stop"


def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


# =============================================================================
# Request handling -- session id + wire response framing
# =============================================================================


def _request_session_id(request: web.Request, body: dict) -> str:
    """Resolve sid from request.

    Tries ``Authorization: Bearer <sid>`` first (the canonical OpenAI auth
    header; Codex CLI propagates ``OPENAI_API_KEY`` here), then falls back
    to body-level hints (``metadata.session_id`` / ``user``).
    """
    return request_session_id(request, body=body)


def _chat_completion_response(
    body: dict,
    wire_message: dict[str, Any],
    wire_finish: str,
    in_tok: int,
    out_tok: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl_{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "slime-actor"),
        "choices": [
            {
                "index": 0,
                "message": wire_message,
                "finish_reason": wire_finish,
            }
        ],
        "usage": _usage(in_tok, out_tok),
    }


async def _stream_chat_completion(
    request: web.Request,
    body: dict,
    wire_message: dict[str, Any],
    wire_finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    """Emit the OpenAI Chat-Completions SSE stream.

    Each chunk shape: ``data: {chatcmpl ...}\n\n`` ending with ``data: [DONE]``.
    The whole turn is fully realised on the server before we start streaming
    (we don't have token-level deltas from SGLang here), so we emit one role
    chunk, then content / reasoning / tool_calls in single delta chunks each.
    """
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    completion_id = f"chatcmpl_{secrets.token_hex(12)}"
    created = int(time.time())

    async def emit(delta: dict[str, Any], finish_reason: str | None = None, usage: dict | None = None) -> None:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.get("model", "slime-actor"),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        await out.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())

    await emit({"role": "assistant"})
    reasoning = wire_message.get("reasoning_content")
    if reasoning:
        await emit({"reasoning_content": reasoning})
    content = wire_message.get("content")
    if content:
        await emit({"content": content})
    # NB: emit ALL tool_calls in a single chunk. Codex CLI 0.30.0 incorrectly
    # accumulates per-index ``arguments`` fragments across chunks, causing N
    # parallel tool_calls to collapse into a single call with a concatenated
    # arguments string (``{"command": "..."}{"command": "..."}``) -- this then
    # fails Codex's own arguments parser and the tool result becomes a parse
    # error, breaking node_match_key alignment on the next turn.
    tool_calls = wire_message.get("tool_calls") or []
    if tool_calls:
        await emit({"tool_calls": [{**call, "index": idx} for idx, call in enumerate(tool_calls)]})
    await emit({}, finish_reason=wire_finish, usage=_usage(in_tok, out_tok))
    await out.write(b"data: [DONE]\n\n")
    return out
