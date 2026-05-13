from __future__ import annotations

import asyncio
import copy
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


# ---------------------------------------------------------------------------
# Plain helpers
# ---------------------------------------------------------------------------

_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_MAX_TURNS_TEXT = "Reached max turns; stopping without another tool call."


def _stringify_content(content: Any) -> str:
    """Flatten OpenAI / Anthropic / Responses content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in _TEXT_BLOCK_TYPES:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(_stringify_content(item.get("content")))
        return "".join(parts)
    return "" if content is None else str(content)


def _route_path(raw_path: str) -> str:
    """Strip query string and trailing slash from a request path."""
    return urlsplit(raw_path).path.rstrip("/") or "/"


def _normalize_assistant_for_match(content: Any) -> str:
    """Reduce assistant content to its natural-language portion for matching.

    Used to recognise our own prior generations in echoed history. The
    chat-completion response we emit lifts <tool_call>...</tool_call> blocks
    into the structured `tool_calls` field, so the round-tripped `content`
    no longer carries them — strip those blocks on both sides for equality.
    """
    return _TOOL_CALL_RE.sub("", _stringify_content(content)).strip()


def _coerce_param_value(value: str) -> Any:
    """Best-effort literal coercion for openhands XML parameter values."""
    value = value.strip()
    if not value:
        return value
    if value in {"true", "false", "null"}:
        return json.loads(value)
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if (value[0], value[-1]) in {("[", "]"), ("{", "}")}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _proxy_warn(message: str) -> None:
    print(f"[sglang_openai_proxy] WARNING: {message}", flush=True)


# ---------------------------------------------------------------------------
# Generation output: parse `<tool_call>` / `<function=...>` XML
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, arguments: Any, index: int) -> dict[str, Any]:
    # Raw XML tool calls do not carry ids; clients echo this id back with tool results.
    return {
        "type": "function",
        "id": f"call_{uuid.uuid4().hex}_{index}",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract OpenAI-shaped tool_calls from a generated assistant string.

    Two on-the-wire conventions:
      - xml `<function=name><parameter=k>v</parameter></function>`
      - json `<tool_call>{...JSON...}</tool_call>`
    """
    if "<function" in text:
        out: list[dict[str, Any]] = []
        for i, (name, body) in enumerate(
            re.findall(r"<function\s*=\s*([^>]+)>(.*?)</function>", text, flags=re.DOTALL)
        ):
            params = {
                k.strip(): _coerce_param_value(v)
                for k, v in re.findall(
                    r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", body, flags=re.DOTALL
                )
            }
            out.append(_make_tool_call(name.strip(), params, i))
        return out

    if "<tool_call>" in text:
        if "</tool_call>" not in text:
            text = text + "</tool_call>"
        out = []
        for i, blob in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)):
            try:
                parsed = json.loads(blob.strip())
            except json.JSONDecodeError:
                continue
            out.append(_make_tool_call(parsed.get("name") or "", parsed.get("arguments") or {}, i))
        return out

    return []


def _format_assistant_message(text: str) -> tuple[dict[str, Any], str, int]:
    """Build a chat-completion `message` from a generated string.

    Returns (message, finish_reason, tool_call_count). When tool calls are
    present, `content` is truncated to the prefix before the first XML
    marker so it carries only the natural-language preamble.
    """
    tool_calls = _parse_tool_calls(text)
    content = re.split(r"<tool_call>|<function", text, maxsplit=1)[0] if tool_calls else text
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, "tool_calls" if tool_calls else "stop", len(tool_calls)


# ---------------------------------------------------------------------------
# Chat-completion message helpers (the canonical internal form)
# ---------------------------------------------------------------------------


def _normalize_arguments_for_template(message: dict[str, Any]) -> dict[str, Any]:
    """Convert tool_calls[*].function.arguments from JSON string to dict.

    The Qwen chat template iterates over `arguments` as a mapping; the raw
    OpenAI wire form (a JSON string) renders as a literal `{"k": "v"}`
    inside the prompt.
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    converted: list[Any] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            converted.append(tc)
            continue
        new_tc = copy.deepcopy(tc)
        fn = new_tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            try:
                fn["arguments"] = json.loads(fn["arguments"])
            except json.JSONDecodeError:
                pass
        converted.append(new_tc)
    return {**message, "tool_calls": converted}


def _normalize_tools_for_template(tools: Any) -> list[dict[str, Any]]:
    """Convert known tool schemas to OpenAI-compatible function tools."""
    if not isinstance(tools, list):
        return []

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        fn = tool.get("function")
        if isinstance(fn, dict):
            new_fn = copy.deepcopy(fn)
        elif tool.get("type") == "function" and tool.get("name"):
            new_fn = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {},
            }
        elif tool.get("name") and "input_schema" in tool:
            new_fn = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {},
            }
        else:
            _proxy_warn(f"ignoring unsupported non-function tool schema: {tool.get('type')!r}")
            continue

        if "strict" in tool and "strict" not in new_fn:
            new_fn["strict"] = tool["strict"]
        normalized.append({"type": "function", "function": new_fn})

    return normalized


def _split_assistant(message: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Pull (text, tool_calls) from a chat-completion assistant message."""
    return str(message.get("content") or ""), list(message.get("tool_calls") or [])


def _tool_call_parts(tc: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    """Decompose a tool_call into (id, name, arguments_json, parsed_args)."""
    fn = tc.get("function") or {}
    name = str(fn.get("name") or "")
    raw = fn.get("arguments") or "{}"
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    args_json = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return str(tc.get("id") or ""), name, args_json, parsed


def _tool_call_id_set(message: dict[str, Any]) -> frozenset[str]:
    """Extract the set of tool_call ids from an assistant message."""
    return frozenset(
        str(tc.get("id"))
        for tc in (message.get("tool_calls") or [])
        if isinstance(tc, dict) and tc.get("id")
    )


def _is_title_generation_request(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> bool:
    if tools or not messages:
        return False
    text = "\n".join(_stringify_content(m.get("content")) for m in messages[:2])
    return "Generate a concise, sentence-case title" in text or "You are a title generator" in text


# ---------------------------------------------------------------------------
# Anthropic /v1/messages adapter
# ---------------------------------------------------------------------------


def _anthropic_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Anthropic /v1/messages body → OpenAI chat-completion messages.

    `tool_use` blocks fold into the assistant's `tool_calls`; each
    `tool_result` block emits its own `role="tool"` message keyed by
    `tool_call_id`. Stringifying all of this (the previous behaviour)
    silently dropped tool_use and conflated tool_result with user text,
    which broke multi-turn agents after the first call.
    """
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _stringify_content(body["system"])})

    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = "assistant" if raw.get("role") == "assistant" else "user"
        content = raw.get("content")

        if not isinstance(content, list):
            messages.append({"role": role, "content": _stringify_content(content)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                tool_calls.append(
                    {
                        "type": "function",
                        "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex}"),
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            elif btype == "tool_result":
                if text_parts:
                    messages.append({"role": role, "content": "".join(text_parts)})
                    text_parts = []
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": _stringify_content(block.get("content")),
                    }
                )
            elif btype in _TEXT_BLOCK_TYPES:
                text_parts.append(str(block.get("text") or ""))
            elif "content" in block:
                text_parts.append(_stringify_content(block.get("content")))

        text = "".join(text_parts)
        if role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
        elif text:
            messages.append({"role": role, "content": text})

    return [_normalize_arguments_for_template(m) for m in messages]


def _anthropic_response(model: str, message: dict[str, Any]) -> dict[str, Any]:
    text, tool_calls = _split_assistant(message)
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        cid, name, _args_json, parsed = _tool_call_parts(tc)
        content.append(
            {
                "type": "tool_use",
                "id": cid or f"toolu_{uuid.uuid4().hex}",
                "name": name,
                "input": parsed,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ---------------------------------------------------------------------------
# OpenAI Responses adapter
# ---------------------------------------------------------------------------


def _responses_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Responses API body → OpenAI chat-completion messages.

    `function_call` items merge into the running assistant message's
    `tool_calls`; each `function_call_output` becomes a separate role=tool
    message keyed by `tool_call_id`.
    """
    messages: list[dict[str, Any]] = []
    instructions = str(body.get("instructions") or "").strip()
    if instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages
    if not isinstance(raw_input, list):
        return messages

    seen_call_ids: set[str] = set()
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function_call":
            call_id = str(item.get("call_id") or "")
            if not call_id:
                call_id = f"call_{uuid.uuid4().hex}"
                _proxy_warn("Responses function_call missing call_id; generated a local call_id.")
            seen_call_ids.add(call_id)
            tc = {
                "type": "function",
                "id": call_id,
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": item.get("arguments") or "{}",
                },
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tc)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
        elif itype == "function_call_output":
            call_id = str(item.get("call_id") or "")
            if not call_id:
                _proxy_warn("Responses function_call_output missing call_id.")
            elif call_id not in seen_call_ids:
                _proxy_warn(f"Responses function_call_output has no matching function_call: {call_id!r}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(
                        item.get("output") or _stringify_content(item.get("content"))
                    ),
                }
            )
        elif itype in {None, "message"} or item.get("role") in {"system", "user", "assistant", "tool"}:
            role = item.get("role") or "user"
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            messages.append({"role": role, "content": _stringify_content(item.get("content"))})
        else:
            _proxy_warn(f"unsupported Responses input item type {itype!r}; ignoring item.")

    return [_normalize_arguments_for_template(m) for m in messages]


def _responses_response(model: str, message: dict[str, Any]) -> dict[str, Any]:
    text, tool_calls = _split_assistant(message)
    output: list[dict[str, Any]] = []
    if text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for tc in tool_calls:
        cid, name, args_json, _parsed = _tool_call_parts(tc)
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": cid or f"call_{uuid.uuid4().hex}",
                "name": name,
                "arguments": args_json,
            }
        )
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# SSE encoders
# ---------------------------------------------------------------------------


def _chat_completion_chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    choice = payload["choices"][0]
    base = {
        "id": payload["id"],
        "object": "chat.completion.chunk",
        "created": payload["created"],
        "model": payload["model"],
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    ]
    delta = {k: v for k, v in choice["message"].items() if k != "role"}
    if delta:
        chunks.append({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    chunks.append(
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}]}
    )
    return chunks


def _anthropic_events(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    start = {**payload, "content": [], "stop_reason": None, "stop_sequence": None}
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": start})
    ]
    for idx, block in enumerate(payload.get("content") or []):
        if isinstance(block, dict):
            events.extend(_anthropic_content_block_events(block, idx))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": payload.get("stop_reason"),
                    "stop_sequence": payload.get("stop_sequence"),
                },
                "usage": {
                    "output_tokens": payload.get("usage", {}).get("output_tokens", 0)
                },
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return events


def _anthropic_content_block_events(
    block: dict[str, Any], idx: int
) -> list[tuple[str, dict[str, Any]]]:
    btype = block.get("type")
    if btype == "text":
        text = str(block.get("text") or "")
        events: list[tuple[str, dict[str, Any]]] = [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]
        if text:
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": idx})
        )
        return events

    if btype == "tool_use":
        input_json = json.dumps(block.get("input") or {}, ensure_ascii=False)
        return [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": input_json,
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": idx}),
        ]

    return []


def _responses_events(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Stream a final Responses payload as the canonical event sequence.

    codex CLI reconstructs function_call state from the
    `function_call_arguments.delta` events, so even though we have the
    finished payload in hand we must replay the per-item event sequence.
    """
    in_progress = {**copy.deepcopy(payload), "status": "in_progress", "output": [], "output_text": ""}
    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {"type": "response.created", "response": in_progress})
    ]
    for idx, item in enumerate(payload.get("output") or []):
        if isinstance(item, dict):
            events.extend(_response_item_events(item, idx))
    events.append(("response.completed", {"type": "response.completed", "response": payload}))
    return events


def _response_item_events(item: dict[str, Any], idx: int) -> list[tuple[str, dict[str, Any]]]:
    item_id = str(item.get("id") or "")
    skeleton = copy.deepcopy(item)
    skeleton["status"] = "in_progress"
    if skeleton.get("type") == "function_call":
        skeleton["arguments"] = ""
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "response.output_item.added",
            {"type": "response.output_item.added", "output_index": idx, "item": skeleton},
        ),
    ]
    if item.get("type") == "message":
        for cidx, part in enumerate(item.get("content") or []):
            if isinstance(part, dict):
                events.extend(_response_message_part_events(part, item_id, idx, cidx))
    elif item.get("type") == "function_call":
        args = str(item.get("arguments") or "{}")
        if args:
            events.append(
                (
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item_id,
                        "output_index": idx,
                        "delta": args,
                    },
                )
            )
        events.append(
            (
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "output_index": idx,
                    "arguments": args,
                },
            )
        )
    events.append(
        (
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": idx, "item": item},
        )
    )
    return events


def _response_message_part_events(
    part: dict[str, Any], item_id: str, idx: int, cidx: int
) -> list[tuple[str, dict[str, Any]]]:
    skeleton = (
        {**part, "text": ""} if part.get("type") == "output_text" else dict(part)
    )
    base_keys = {"item_id": item_id, "output_index": idx, "content_index": cidx}
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "response.content_part.added",
            {"type": "response.content_part.added", **base_keys, "part": skeleton},
        )
    ]
    if part.get("type") == "output_text":
        text = str(part.get("text") or "")
        if text:
            events.append(
                (
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", **base_keys, "delta": text},
                )
            )
        events.append(
            (
                "response.output_text.done",
                {"type": "response.output_text.done", **base_keys, "text": text},
            )
        )
    events.append(
        (
            "response.content_part.done",
            {"type": "response.content_part.done", **base_keys, "part": part},
        )
    )
    return events


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class _ProxyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    proxy: SGLangOpenAIProxy


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if _route_path(self.path) == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": self._proxy.model_name, "object": "model", "created": 0}
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            stream = bool(body.get("stream"))
            path = _route_path(self.path)

            if path == "/v1/chat/completions":
                payload = self._proxy.chat_completions(body)
                self._send_chat_completion(payload, stream=stream)
            elif path in {"/v1/messages", "/messages"}:
                payload = self._proxy.anthropic_messages(body)
                self._send_anthropic_messages(payload, stream=stream)
            elif path == "/v1/responses":
                payload = self._proxy.responses(body)
                self._send_responses(payload, stream=stream)
            else:
                self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})
        except Exception as exc:
            self._send_json(500, {"error": {"message": str(exc), "type": exc.__class__.__name__}})

    @property
    def _proxy(self) -> SGLangOpenAIProxy:
        return self.server.proxy  # type: ignore[attr-defined,return-value]

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_chat_completion(self, payload: dict[str, Any], *, stream: bool) -> None:
        if not stream:
            self._send_json(200, payload)
            return
        self._begin_sse()
        for chunk in _chat_completion_chunks(payload):
            self._write_sse_data(chunk)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _send_anthropic_messages(self, payload: dict[str, Any], *, stream: bool) -> None:
        if not stream:
            self._send_json(200, payload)
            return
        self._begin_sse()
        for event, data in _anthropic_events(payload):
            self._write_sse_event(event, data)
        self.close_connection = True

    def _send_responses(self, payload: dict[str, Any], *, stream: bool) -> None:
        if not stream:
            self._send_json(200, payload)
            return
        self._begin_sse()
        for event, data in _responses_events(payload):
            self._write_sse_event(event, data)
        self.close_connection = True

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

    def _write_sse_data(self, data: dict[str, Any]) -> None:
        """Write one OpenAI Chat Completions SSE data frame."""
        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _write_sse_event(self, event: str, data: dict[str, Any]) -> None:
        """Write one named SSE event used by Anthropic Messages and Responses."""
        self.wfile.write(
            f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
        )
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Proxy core
# ---------------------------------------------------------------------------


@dataclass
class SGLangOpenAIProxy:
    args: Any
    rollout_state: dict[str, Any]
    loop: asyncio.AbstractEventLoop
    model_name: str
    evaluation: bool = False
    max_assistant_turns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _server: _ProxyHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _training_messages: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _training_tools: list[dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _turns: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _turn_responses: list[str] = field(default_factory=list, init=False, repr=False)
    _generated_assistants: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("proxy is not started")
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> SGLangOpenAIProxy:
        if self._server is not None:
            return self
        server = _ProxyHTTPServer(("127.0.0.1", 0), _Handler)
        server.proxy = self
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server, self._thread = server, thread
        return self

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None

    # --- public entry points ---------------------------------------------

    def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        raw = body.get("messages")
        if not isinstance(raw, list):
            raise ValueError("chat completion request must include a messages list")
        messages = [_normalize_arguments_for_template(m) for m in raw if isinstance(m, dict)]
        return self._run_turn(body=body, messages=messages)

    def anthropic_messages(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = _anthropic_to_messages(body)
        model = str(body.get("model") or self.model_name)
        chat = self._run_turn(
            body={"model": model, "tools": body.get("tools") or []},
            messages=messages,
        )
        return _anthropic_response(model, chat["choices"][0]["message"])

    def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = _responses_to_messages(body)
        model = str(body.get("model") or self.model_name)
        chat = self._run_turn(
            body={"model": model, "tools": body.get("tools") or []},
            messages=messages,
        )
        return _responses_response(model, chat["choices"][0]["message"])

    def snapshot(
        self,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]], list[str]
    ]:
        with self._lock:
            return (
                copy.deepcopy(self._training_messages),
                copy.deepcopy(self._training_tools),
                copy.deepcopy(self._turns),
                list(self._turn_responses),
            )

    # --- internal: one rollout turn --------------------------------------

    def _run_turn(
        self, *, body: dict[str, Any], messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        tools = _normalize_tools_for_template(body.get("tools"))
        body = {**body, "tools": tools}
        if self._should_stop_for_max_turns(messages, tools):
            assistant_text = _MAX_TURNS_TEXT
            generation_meta = {"finish_reason": {"type": "max_turns"}}
        else:
            future = asyncio.run_coroutine_threadsafe(
                self._generate_assistant_text(messages, tools), self.loop
            )
            assistant_text, generation_meta = future.result()
        message, finish_reason, tool_call_count = _format_assistant_message(assistant_text)
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or self.model_name),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
        with self._lock:
            self._record_turn(
                body=body,
                messages=messages,
                tools=tools,
                assistant_text=assistant_text,
                payload=payload,
                finish_reason=finish_reason,
                tool_call_count=tool_call_count,
                generation_meta=generation_meta,
            )
        return payload

    def _should_stop_for_max_turns(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> bool:
        if not self.max_assistant_turns or self.max_assistant_turns <= 0:
            return False
        if _is_title_generation_request(messages, tools):
            return False
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        return assistant_turns >= self.max_assistant_turns

    def _match_prior_generation(self, echoed: dict[str, Any], cursor: int) -> int:
        echoed_ids = _tool_call_id_set(echoed)
        if echoed_ids:
            for j in range(cursor, len(self._generated_assistants)):
                if self._generated_assistants[j]["tool_call_ids"] == echoed_ids:
                    return j
            return -1
        # Pure-text echo: only match against pure-text generations to
        # avoid matching a stripped-of-tool-calls echo against a
        # generation whose natural-language preamble happens to equal it.
        norm = _normalize_assistant_for_match(echoed.get("content"))
        for j in range(cursor, len(self._generated_assistants)):
            gen = self._generated_assistants[j]
            if gen["tool_call_ids"]:
                continue
            if _normalize_assistant_for_match(gen["text"]) == norm:
                return j
        return -1

    def _record_turn(
        self,
        *,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        assistant_text: str,
        payload: dict[str, Any],
        finish_reason: str,
        tool_call_count: int,
        generation_meta: dict[str, Any],
    ) -> None:
        # Training messages snapshot the *current* turn's view: whatever
        # the agent sent us this turn, plus the assistant we just produced.
        # Anything the agent dropped from history won't appear, keeping the
        # trace bounded by the agent's live context window.
        #
        # Prior assistants are matched against proxy-generated history by
        # tool_call_id first, then by normalized text for pure-text turns.
        # The forward cursor skips hidden requests (title/subagent), while
        # harness-injected assistant messages remain unmatched and mask=0.
        raw_finish = ""
        if isinstance(generation_meta, dict):
            raw_finish = (generation_meta.get("finish_reason") or {}).get("type", "")
        current_loss_mask = 0 if raw_finish in {"length", "max_turns"} else 1
        update_training = not _is_title_generation_request(messages, tools)

        copied = copy.deepcopy(messages)
        cursor = 0
        for m in copied:
            if m.get("role") != "assistant":
                continue
            found = self._match_prior_generation(m, cursor)
            if found >= 0:
                m["step_loss_mask"] = self._generated_assistants[found]["loss_mask"]
                cursor = found + 1
            else:
                m["step_loss_mask"] = 0

        if update_training:
            self._training_messages = copied + [
                {"role": "assistant", "content": assistant_text, "step_loss_mask": current_loss_mask}
            ]
            new_message = payload["choices"][0]["message"]
            self._generated_assistants.append(
                {
                    "tool_call_ids": _tool_call_id_set(new_message),
                    "text": assistant_text,
                    "loss_mask": current_loss_mask,
                }
            )
            self._training_tools = copy.deepcopy(tools) if tools else self._training_tools

        last = messages[-1] if messages else {}
        request_obj = {**copy.deepcopy(body), "messages": copy.deepcopy(messages)}
        self._turns.append(
            {
                "turn_id": len(self._turns),
                "observation": {
                    "role": str(last.get("role") or ""),
                    "content": _stringify_content(last.get("content")),
                },
                "model_request": request_obj,
                "raw_model_response": assistant_text,
                "formatted_model_response": payload,
                "generation_finish_reason": finish_reason,
                "raw_generation_finish_reason": raw_finish,
                "tool_call_count": tool_call_count,
            }
        )
        self._turn_responses.append(assistant_text)

    async def _generate_assistant_text(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        from slime.rollout.sglang_rollout import get_model_url
        from slime.utils.http_utils import post

        tokenizer = self.rollout_state["tokenizer"]
        model_url = get_model_url(self.args, getattr(self.args, "swe_model_name", "default"))
        sampling_params = dict(self.rollout_state["generate_state"].sampling_params)
        # Each assistant_text is fed back through apply_chat_template on the
        # next turn, which re-adds its own `<|im_end|>` separator. slime's
        # default skip_special_tokens=False would otherwise duplicate the
        # EOS at training time.
        sampling_params["skip_special_tokens"] = True
        if self.evaluation:
            sampling_params.update(temperature=0.0, top_p=1.0, top_k=1)

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools or None,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        output = await post(
            model_url,
            {"text": prompt_text, "sampling_params": sampling_params, "return_logprob": False},
        )
        if not isinstance(output, dict):
            return str(output), {}
        return str(output.get("text") or ""), dict(output.get("meta_info") or {})
