"""OpenAI-compatible adapters for agent rollouts.

The adapter exposes ``/v1/chat/completions`` and ``/v1/responses``. Both
endpoints render incoming messages with the served model's chat template, call
SGLang ``/generate`` with ``input_ids``, and record the exact sampled token
ids/logprobs as ``TurnRecord`` objects. New code should use ``OpenAIAdapter``
and call ``finish_session()`` at trajectory end to drain trainable
``TokenSegment`` objects.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
import time
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import ADAPTER_KEY, REASONING_PARSER_KEY, TOKENIZER_KEY, TOOL_PARSER_KEY
from slime.agent.adapters.common import AdapterChain as Chain
from slime.agent.adapters.common import BaseAdapter, call_sglang_generate
from slime.agent.adapters.common import json_arguments as _json_arguments
from slime.agent.adapters.common import ok_response, render_token_ids, request_session_id
from slime.agent.adapters.common import stable_hash as _hash
from slime.agent.parsing import ParsedModelOutput, parse_model_output
from slime.agent.trajectory import TokenSegment, TurnRecord, TurnSegment, make_turn_segment, merge_turn_segments

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[TurnSegment] = dataclasses.field(default_factory=list)


class OpenAIAdapter(BaseAdapter):
    """OpenAI-compatible HTTP adapter with session lifecycle helpers."""

    session_cls = Session

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        super().__init__(
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        self.app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        self.app.router.add_post("/v1/responses", _handle_responses)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)

    async def finish_session(self, sid: str, *, wait_timeout: float = 5.0) -> list[TokenSegment]:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        s = self.store.pop(sid, None)
        if s is None:
            return []
        if s.main.turns:
            s.segments.append(make_turn_segment(s.main.turns, kind="final"))
        return merge_turn_segments(s.segments)


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


def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    name = function.get("name") or call.get("name") or "tool"
    arguments = function.get("arguments", call.get("arguments", {}))
    out = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_arguments(arguments),
        },
    }
    if call.get("id"):
        out["id"] = call["id"]
    return out


def _translate_chat_messages(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages -> tokenizer chat-template messages."""
    translated: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "developer":
            role = "system"

        if role in {"system", "user"}:
            translated.append({"role": role, "content": _flatten_content(content)})
        elif role == "tool":
            tool_msg = {"role": "tool", "content": _flatten_content(content)}
            if msg.get("tool_call_id"):
                tool_msg["tool_call_id"] = msg["tool_call_id"]
            translated.append(tool_msg)
        elif role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant", "content": _flatten_content(content)}
            if msg.get("reasoning_content"):
                assistant["reasoning_content"] = msg["reasoning_content"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = [_normalize_tool_call(c) for c in tool_calls if isinstance(c, dict)]
            translated.append(assistant)
    return translated


def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") != "function":
        return None
    if isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = function.get("name")
        if not name:
            return None
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    name = tool.get("name")
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def _normalize_tools(tools: list[dict] | None) -> list[dict] | None:
    normalized = [_normalize_tool(t) for t in tools or []]
    return [t for t in normalized if t is not None] or None


def _responses_input_to_messages(input_value: Any, instructions: Any = None) -> list[dict]:
    """Responses API input -> OpenAI chat message list.

    This intentionally covers the common message/function-call shapes used by
    agent SDKs. Unknown input items are preserved as user text where possible.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": _flatten_content(instructions)})

    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if not isinstance(input_value, list):
        messages.append({"role": "user", "content": _flatten_content(input_value)})
        return messages

    for item in input_value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue

        typ = item.get("type")
        if typ == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": item.get("output", ""),
                }
            )
        elif typ == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or f"call_{secrets.token_hex(8)}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", "tool"),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif item.get("role"):
            messages.append({"role": item.get("role"), "content": item.get("content", "")})
        elif typ == "message":
            messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
        else:
            messages.append({"role": "user", "content": _flatten_content(item)})
    return messages


def _select_kind(s: Session, messages: list[dict]) -> str:
    target = s.main
    msg_hashes = [_hash(m) for m in messages]
    if target.seen_msgs == 0:
        kind = "new"
    else:
        is_append = len(msg_hashes) >= target.seen_msgs and msg_hashes[: target.seen_msgs] == target.msg_hashes
        if is_append:
            kind = "append"
        else:
            if target.turns:
                s.segments.append(make_turn_segment(target.turns, kind="wipe"))
            kind = "wipe"
    return kind


def _replace_chat_messages(target: Chain, messages: list[dict], tools_schema: list[dict] | None) -> None:
    target.chat_messages = _translate_chat_messages(messages)
    target.turns.clear()
    target.seen_msgs = len(messages)
    target.msg_hashes = [_hash(m) for m in messages]
    if tools_schema is not None:
        target.tools_schema = tools_schema


def _extend_chat_messages(target: Chain, messages: list[dict], tools_schema: list[dict] | None) -> None:
    translated = _translate_chat_messages(messages[target.seen_msgs :])
    target.chat_messages.extend(translated)
    target.seen_msgs = len(messages)
    target.msg_hashes = [_hash(m) for m in messages]
    if tools_schema is not None:
        target.tools_schema = tools_schema


def _build_prompt(target: Chain, messages: list[dict], tools_schema: list[dict] | None, kind: str, tok) -> list[int]:
    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, messages, tools_schema)
    return render_token_ids(target, tok)


async def _generate(
    prompt_ids: list[int], s: Session, body: dict, app, *, session_id: str | None = None
) -> TurnRecord:
    return await call_sglang_generate(
        prompt_ids,
        s,
        body,
        app,
        max_token_keys=("max_output_tokens", "max_completion_tokens", "max_tokens"),
        stop_keys=("stop",),
        log_prefix="openai_adapter",
        logger=logger,
        session_id=session_id,
    )


def _parse_turn(target: Chain, turn: TurnRecord, app) -> ParsedModelOutput:
    tok = app[TOKENIZER_KEY]
    raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
    return parse_model_output(
        raw_output,
        tools_schema=target.tools_schema,
        tool_parser_name=app[TOOL_PARSER_KEY],
        reasoning_parser_name=app[REASONING_PARSER_KEY],
    )


def _openai_tool_calls(tool_uses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tool_use in tool_uses:
        call_id = f"call_{secrets.token_hex(12)}"
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_use.get("name", "tool"),
                    "arguments": _json_arguments(tool_use.get("input") or {}),
                },
            }
        )
    return calls


def _finish_reason(parsed: ParsedModelOutput, finish: str) -> str:
    if parsed.tool_uses:
        return "tool_calls"
    if finish == "length":
        return "length"
    return "stop"


def _chat_message(parsed: ParsedModelOutput) -> dict[str, Any]:
    tool_calls = _openai_tool_calls(parsed.tool_uses)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.text if parsed.text else None,
    }
    if parsed.reasoning:
        message["reasoning_content"] = parsed.reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def _responses_usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def _request_session_id(request: web.Request, body: dict) -> str:
    return request_session_id(request, body=body)


async def _run_turn(
    request: web.Request, body: dict, messages: list[dict]
) -> tuple[TurnRecord, ParsedModelOutput, int, int]:
    sid = _request_session_id(request, body)
    adapter = request.app[ADAPTER_KEY]
    if sid in adapter.closed:
        raise web.HTTPServiceUnavailable(text="session closed")
    app = request.app
    s = adapter.store.setdefault(sid, Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:
            target = s.main
            tools_schema = _normalize_tools(body.get("tools"))
            kind = _select_kind(s, messages)
            prompt_ids = _build_prompt(target, messages, tools_schema, kind, app[TOKENIZER_KEY])
            turn = await _generate(prompt_ids, s, body, app, session_id=sid)
            parsed = _parse_turn(target, turn, app)
            target.turns.append(turn)
            return turn, parsed, len(prompt_ids), len(turn.output_ids)
    finally:
        adapter.inflight.get(sid, set()).discard(task)


async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise web.HTTPBadRequest(text="messages must be a list")
    turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
    if body.get("stream"):
        return await _stream_chat_completion(request, body, parsed, turn.finish_reason, in_tok, out_tok)
    return web.json_response(_chat_completion_response(body, parsed, turn.finish_reason, in_tok, out_tok))


def _chat_completion_response(
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
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
                "message": _chat_message(parsed),
                "finish_reason": _finish_reason(parsed, finish),
            }
        ],
        "usage": _usage(in_tok, out_tok),
    }


async def _stream_chat_completion(
    request: web.Request,
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
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

    async def emit(choice_delta: dict[str, Any], finish_reason: str | None = None, usage: dict | None = None) -> None:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.get("model", "slime-actor"),
            "choices": [{"index": 0, "delta": choice_delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        await out.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())

    await emit({"role": "assistant"})
    if parsed.reasoning:
        await emit({"reasoning_content": parsed.reasoning})
    if parsed.text:
        await emit({"content": parsed.text})
    for idx, call in enumerate(_openai_tool_calls(parsed.tool_uses)):
        await emit({"tool_calls": [{**call, "index": idx}]})
    await emit({}, finish_reason=_finish_reason(parsed, finish), usage=_usage(in_tok, out_tok))
    await out.write(b"data: [DONE]\n\n")
    return out


async def _handle_responses(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    messages = _responses_input_to_messages(body.get("input", ""), body.get("instructions"))
    turn, parsed, in_tok, out_tok = await _run_turn(request, body, messages)
    if body.get("stream"):
        return await _stream_response(request, body, parsed, turn.finish_reason, in_tok, out_tok)
    return web.json_response(_response_response(body, parsed, turn.finish_reason, in_tok, out_tok))


def _response_output(parsed: ParsedModelOutput) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if parsed.reasoning:
        output.append(
            {
                "id": f"rs_{secrets.token_hex(12)}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": parsed.reasoning}],
            }
        )
    if parsed.text:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": parsed.text, "annotations": []}],
            }
        )
    for call in _openai_tool_calls(parsed.tool_uses):
        output.append(
            {
                "id": f"fc_{secrets.token_hex(12)}",
                "type": "function_call",
                "status": "completed",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
        )
    if not output:
        output.append(
            {
                "id": f"msg_{secrets.token_hex(12)}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "", "annotations": []}],
            }
        )
    return output


def _response_response(
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> dict[str, Any]:
    status = "incomplete" if finish == "length" else "completed"
    return {
        "id": f"resp_{secrets.token_hex(12)}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": body.get("model", "slime-actor"),
        "output": _response_output(parsed),
        "usage": _responses_usage(in_tok, out_tok),
    }


async def _stream_response(
    request: web.Request,
    body: dict,
    parsed: ParsedModelOutput,
    finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    response = _response_response(body, parsed, finish, in_tok, out_tok)
    created = {"type": "response.created", "response": response}
    await out.write(f"event: response.created\ndata: {json.dumps(created, ensure_ascii=False)}\n\n".encode())
    if parsed.text:
        delta = {"type": "response.output_text.delta", "delta": parsed.text}
        await out.write(
            f"event: response.output_text.delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n".encode()
        )
    completed = {"type": "response.completed", "response": response}
    await out.write(f"event: response.completed\ndata: {json.dumps(completed, ensure_ascii=False)}\n\n".encode())
    return out


async def _ok(request: web.Request) -> web.Response:
    return await ok_response(request)
