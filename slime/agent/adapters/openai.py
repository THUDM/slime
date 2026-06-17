"""OpenAI Chat-Completions adapter for agent rollouts.

Mirrors slime.agent.adapters.anthropic but speaks the OpenAI
/v1/chat/completions protocol, so an OpenAI-compatible client (e.g. the Codex
CLI) can drive the slime sglang server. Each request is rendered with the served
model's chat template, sent to sglang /generate as input_ids, parsed, and folded
into a shared TrajectoryManager keyed by session id. finish_session(sid) drains
a session's trajectory into a list of Sample.

Only /v1/chat/completions is implemented; the Responses API (/v1/responses) is
out of scope. The section layout (adapter class -> translation -> reply building
-> request framing) mirrors slime.agent.adapters.anthropic.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import (
    BaseAdapter,
    Reply,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    sid_from_body,
)
from slime.agent.parsing import ParsedModelOutput

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter):
    """OpenAI Chat-Completions-compatible HTTP adapter: wire translation and
    reply framing only; the turn machinery is inherited from BaseAdapter."""

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
        translated = _translate_messages(messages)
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        wire_message, manager_message, wire_finish = _build_reply_parts(parsed, raw_finish)
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(wire_message, wire_finish),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        wire_message, wire_finish = reply.wire
        if stream:
            return await _render_stream(request, body, wire_message, wire_finish, in_tok, out_tok)
        return web.json_response(_render_response(body, wire_message, wire_finish, in_tok, out_tok))


# --- Translation (OpenAI wire -> chat-template messages) ---


def _arguments_as_dict(arguments: Any) -> dict[str, Any]:
    """Coerce wire-shape tool_calls[].function.arguments into a dict.

    OpenAI sends arguments as a JSON-encoded string; the chat template and the
    trajectory manager's history matching both expect a mapping. Malformed
    payloads fall back to {"_raw_arguments": s}.
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


def _translate_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> tokenizer chat-template messages.

    Mirrors anthropic._translate_messages so a replayed assistant turn compares
    equal (dict equality) to the leaf the manager appended on the previous
    request. Two invariants must hold:

      * tool_calls[i].function.arguments is a dict (not a JSON string): the chat
        template needs a mapping, and the manager matches history by dict
        equality regardless of key order.
      * Wire-only correlation ids are dropped (tool_call_id on tool messages,
        tool_calls[i].id on echoed assistant messages). Fresh ids are minted on
        each response, so keeping the wire ids would diverge the replay match.
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
            translated.append({"role": role, "content": flatten_content(content)})
        elif role == "tool":
            # drop tool_call_id -- wire-only correlation field; see docstring
            translated.append({"role": "tool", "content": flatten_content(content)})
        elif role == "assistant":
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": flatten_content(content),
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
                # NB: arguments stays a dict (not a JSON string), and the
                # wire-only id is dropped. See docstring above.
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
        # unknown roles are silently dropped
    return translated


def _tools_to_chat_tools(tools: list[dict] | None) -> list[dict] | None:
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


# --- Reply building: parsed output -> OpenAI wire message + manager_message ---


def _build_reply_parts(parsed: ParsedModelOutput, finish: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return (wire_message, manager_message, wire_finish).

    wire_message follows the OpenAI Chat-Completions spec: tool_calls[].id is a
    unique correlation id and tool_calls[].function.arguments is a JSON-encoded
    string (clients depend on this).

    manager_message is the shape fed to record_turn: arguments is a dict so
    chat-template replay succeeds and the manager's history match (dict equality)
    holds against the echo on the next turn, and the wire-only id is omitted.
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
        # send content=null when there are tool_calls: some OpenAI clients split
        # a mixed text+tool_calls turn into two echoed messages otherwise, which
        # diverges the history match against our leaf
        "content": None if wire_tool_calls else (parsed.text or None),
    }
    # manager_message must match what the client echoes on the next request, or
    # the manager's history match (dict equality) diverges and every turn forks.
    # Differences from wire_message, each needed to match the echo:
    #   * no reasoning_content -- some clients strip it on echo (the reasoning
    #     token ids are still kept in the trained tokens, only the text drops)
    #   * only the first tool_call -- some clients drop extra parallel tool_calls
    #   * empty content when tool_calls are present -- mirrors content=null above
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


# --- Request framing: session id + wire response/stream rendering ---


def _request_session_id(request: web.Request, body: dict) -> str:
    """Resolve sid: Authorization: Bearer <sid> first (where an OpenAI client
    propagates its API key), then body-level hints (metadata.session_id / user)."""
    return sid_from_bearer(request) or sid_from_body(body) or "default"


def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def _render_response(
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


async def _render_stream(
    request: web.Request,
    body: dict,
    wire_message: dict[str, Any],
    wire_finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    """Emit the OpenAI Chat-Completions SSE stream.

    Each chunk has the shape `data: {chatcmpl ...}\n\n`, ending with
    `data: [DONE]`. The whole turn is realised on the server before streaming
    (we have no token-level deltas from sglang here), so we emit one role chunk,
    then content / reasoning / tool_calls in single delta chunks each.
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
    # emit all tool_calls in a single chunk: some clients accumulate per-index
    # arguments fragments across chunks, collapsing N parallel tool_calls into
    # one call with a concatenated (and unparseable) arguments string
    tool_calls = wire_message.get("tool_calls") or []
    if tool_calls:
        await emit({"tool_calls": [{**call, "index": idx} for idx, call in enumerate(tool_calls)]})
    await emit({}, finish_reason=wire_finish, usage=_usage(in_tok, out_tok))
    await out.write(b"data: [DONE]\n\n")
    return out
