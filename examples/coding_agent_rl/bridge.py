"""Anthropic Messages API <-> SGLang /generate bridge.

Translates Claude Code's Anthropic Messages API into slime's SGLang ``/generate``
(token-native + logprobs) and captures the actual tokens per session_id so the
trainer never re-tokenizes (no TITO mismatch).

Model-agnostic: chat formatting via ``tokenizer.apply_chat_template``; tool-call
parsing via ``sglang.srt.function_call.FunctionCallParser``; reasoning parsing
via ``sglang.srt.parser.reasoning_parser.ReasoningParser``.

---

Forking this file (kept dict-based on purpose so the dataflow is easy to fork):

* Swap the agent's API protocol (e.g. Codex speaks OpenAI Chat Completions):
  replace ``_handle_messages``. The session token store + prefix-diff over
  ``apply_chat_template`` is reusable as-is; only the inbound/outbound shape
  changes.

* Swap the inference engine: change the ``POST {sglang_url}/generate`` block
  to whatever token-native endpoint your engine exposes. The session needs
  ``output_token_ids`` and a finish reason; everything else is optional.

* Swap the model family: pass different ``tool_parser`` / ``reasoning_parser``
  names to ``start()``. No code change here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
from typing import Any

import aiohttp
from aiohttp import web

from slime.utils.aiohttp_threaded import AppHandle, run_app_in_thread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session token store
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _Session:
    # Canonical chat log. Each assistant turn we append after /generate carries
    # reasoning_content so the next round's apply_chat_template re-render matches
    # the tokens the model actually emitted (preserving prefix match).
    glm_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)
    sampling_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class _Store:
    def __init__(self) -> None:
        self._s: dict[str, _Session] = {}
        self._guard = asyncio.Lock()

    async def get(self, sid: str) -> _Session:
        async with self._guard:
            return self._s.setdefault(sid, _Session())

    async def pop(self, sid: str) -> _Session | None:
        async with self._guard:
            return self._s.pop(sid, None)

    def set_defaults(self, sid: str, defaults: dict[str, Any]) -> None:
        self._s.setdefault(sid, _Session()).sampling_defaults = dict(defaults or {})


# ---------------------------------------------------------------------------
# Anthropic body -> chat-template format
# ---------------------------------------------------------------------------
def _flatten(c: Any) -> str:
    """Anthropic content blocks -> a single text blob (drop images, etc.)."""
    if c is None: return ""
    if isinstance(c, str): return c
    if not isinstance(c, list): return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, dict):
            t = b.get("type")
            if t == "text": parts.append(b.get("text", ""))
            elif t == "tool_result": parts.append(_flatten(b.get("content")))
            elif t == "image": parts.append("[image omitted]")
        elif isinstance(b, str):
            parts.append(b)
    return "\n".join(p for p in parts if p)


def _translate_messages(messages: list[dict], system: Any) -> list[dict]:
    """Anthropic blocks -> chat-template messages (system/user/assistant/tool).

    Thinking blocks are dropped from input; the bridge re-injects them via
    reasoning_content after parsing /generate output (so the next round's
    template re-render produces tokens matching what the model actually emitted)."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": _flatten(system)})
    for m in messages or []:
        if not isinstance(m, dict): continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    out.append({"role": "tool", "content": _flatten(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    out.append({"role": "user", "content": b.get("text", "")})
                else:
                    out.append({"role": "user", "content": _flatten(b)})
        elif role == "assistant":
            texts, tcs = [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if not isinstance(b, dict): continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    tcs.append({"function": {"name": b.get("name", "tool"),
                                              "arguments": b.get("input") or {}}})
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if tcs: mo["tool_calls"] = tcs
            out.append(mo)
        elif role == "system":
            out.append({"role": "system", "content": _flatten(content)})
    return out


def _tools_schema(anthropic_tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tool defs -> the dict shape HF chat templates expect under ``tools=``."""
    if not anthropic_tools: return None
    out = []
    for t in anthropic_tools:
        if not isinstance(t, dict) or "name" not in t: continue
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
        }})
    return out or None


# ---------------------------------------------------------------------------
# Output parsing (reuse SGLang)
# ---------------------------------------------------------------------------
def _parse_output(
    text: str, *, tool_parser_name: str | None, reasoning_parser_name: str | None,
    tools_schema: list[dict] | None,
) -> tuple[str, str, list[dict]]:
    """Raw model output -> (thinking, visible_text, tool_uses)."""
    thinking, body = "", text
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        r, b = ReasoningParser(model_type=reasoning_parser_name, stream_reasoning=False).parse_non_stream(text)
        thinking, body = r or "", b or ""

    tool_uses: list[dict] = []
    if tool_parser_name and tools_schema:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        body, calls = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name).parse_non_stream(body)
        for c in calls:
            try: args = json.loads(c.parameters or "{}")
            except json.JSONDecodeError: args = {"_raw_arguments": c.parameters}
            tool_uses.append({"name": c.name or "tool", "input": args})
    return thinking, (body or "").strip(), tool_uses


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
async def _handle_messages(request: web.Request) -> web.StreamResponse:
    A = request.app
    tok = A["tokenizer"]
    sglang_url = A["sglang_url"]
    tool_parser = A["tool_parser"]
    reasoning_parser = A["reasoning_parser"]
    store: _Store = A["store"]

    body = await request.json()
    streaming = bool(body.get("stream", False))
    session_id = (request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                  or request.headers.get("x-session-id", ""))
    if not session_id:
        return web.json_response({"type": "error", "error": {
            "type": "missing_session",
            "message": "Authorization Bearer <session_id> required",
        }}, status=400)

    s = await store.get(session_id)
    async with s.lock:
        # --- 1) ingest new messages (since last call) into the canonical log
        all_msgs = body.get("messages") or []
        new = _translate_messages(all_msgs[s.seen_msgs:],
                                   body.get("system") if s.seen_msgs == 0 else None)
        s.glm_messages.extend(new)
        s.seen_msgs = len(all_msgs)
        if s.tools_schema is None:
            s.tools_schema = _tools_schema(body.get("tools"))

        # --- 2) re-render full template; diff vs cumulative -> obs (loss_mask=0)
        ideal_text = tok.apply_chat_template(
            s.glm_messages, tools=s.tools_schema, tokenize=False, add_generation_prompt=True,
        )
        ideal_ids = tok.encode(ideal_text, add_special_tokens=False)

        if not s.prompt_ids:
            s.prompt_ids = ideal_ids
        else:
            cumulative = s.prompt_ids + s.response_ids
            if ideal_ids[:len(cumulative)] == cumulative:
                obs = ideal_ids[len(cumulative):]
                s.response_ids.extend(obs)
                s.loss_mask.extend([0] * len(obs))
            else:
                logger.warning("[bridge] %s template-rerender mismatch; rebaselining", session_id)
                s.response_ids = ideal_ids[len(s.prompt_ids):]
                s.loss_mask = [0] * len(s.response_ids)

        # --- 3) sampling params (request overrides session defaults)
        sp = dict(s.sampling_defaults or {})
        for k_a, k_s in [("max_tokens", "max_new_tokens"), ("temperature", "temperature"),
                          ("top_p", "top_p"), ("top_k", "top_k")]:
            if k_a in body: sp[k_s] = body[k_a]
        if body.get("stop_sequences"): sp["stop"] = body["stop_sequences"]
        sp.setdefault("max_new_tokens", 4096)
        sp.setdefault("skip_special_tokens", False)
        sp.setdefault("spaces_between_special_tokens", False)
        sp.setdefault("no_stop_trim", True)

        # --- 4) /generate (non-streaming upstream; we may stream downstream)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess, \
                       sess.post(f"{sglang_url}/generate", json={
                           "input_ids": ideal_ids, "sampling_params": sp, "return_logprob": True,
                       }) as r:
                if r.status >= 400:
                    return web.json_response({"type": "error", "error": {
                        "type": "upstream_error", "message": await r.text(),
                    }}, status=r.status)
                upstream = await r.json()
        except aiohttp.ClientError as e:
            return web.json_response({"type": "error", "error": {
                "type": "upstream_unreachable", "message": str(e),
            }}, status=502)

        # --- 5) record output tokens; parse; stash assistant turn for re-render
        meta = upstream.get("meta_info") or {}
        output_ids = [x[1] for x in (meta.get("output_token_logprobs") or [])]
        s.response_ids.extend(output_ids)
        s.loss_mask.extend([1] * len(output_ids))

        thinking, visible, tool_uses = _parse_output(
            tok.decode(output_ids, skip_special_tokens=False),
            tool_parser_name=tool_parser, reasoning_parser_name=reasoning_parser,
            tools_schema=s.tools_schema,
        )

        asst: dict[str, Any] = {"role": "assistant", "content": visible}
        # If the model produced any thinking, stash it for re-render.
        # The empty-thinking case requires care: the chat template's "no
        # reasoning_content" branch emits a bare `</think>` (no opener), but
        # the prompt's add_generation_prompt suffix already provided
        # `<|assistant|><think>` -- so the actual emitted tokens start with a
        # `</think>` after a prompt-side `<think>`. To make the next round's
        # re-render match, we feed back reasoning_content=" " (a whitespace
        # that strips to empty) which renders as `<think></think>` -- byte-
        # equivalent to "prompt opens `<think>` + model emits `</think>`".
        if reasoning_parser:
            asst["reasoning_content"] = thinking if thinking else " "
        elif thinking:
            asst["reasoning_content"] = thinking
        if tool_uses:
            asst["tool_calls"] = [{"function": {"name": tu["name"], "arguments": tu["input"]}}
                                   for tu in tool_uses]
        s.glm_messages.append(asst)

        # --- 6) build Anthropic blocks (shared by JSON + SSE paths)
        blocks: list[dict] = []
        if thinking: blocks.append({"type": "thinking", "thinking": thinking})
        if visible: blocks.append({"type": "text", "text": visible})
        for tu in tool_uses:
            blocks.append({"type": "tool_use", "id": f"toolu_{secrets.token_hex(8)}",
                            "name": tu["name"], "input": tu["input"]})
        if not blocks: blocks.append({"type": "text", "text": ""})

        finish = (meta.get("finish_reason") or {}).get("type", "stop")
        stop_reason = "tool_use" if tool_uses else ("max_tokens" if finish == "length" else "end_turn")
        in_tokens, out_tokens = len(ideal_ids), len(output_ids)
        model = body.get("model") or "slime-actor"

    # --- 7) emit Anthropic response (JSON or SSE) ---------------------------
    if not streaming:
        return web.json_response({
            "id": f"msg_{secrets.token_hex(12)}", "type": "message", "role": "assistant",
            "model": model, "content": blocks,
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
        })

    msg_id = f"msg_{secrets.token_hex(12)}"
    out = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive",
    })
    await out.prepare(request)
    await out.write(_sse("message_start", {
        "type": "message_start",
        "message": {"id": msg_id, "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": in_tokens, "output_tokens": 0}},
    }))
    for idx, b in enumerate(blocks):
        # block start: zero-content placeholder of the right type
        bt = b["type"]
        start = ({"type": "thinking", "thinking": ""} if bt == "thinking"
                 else {"type": "text", "text": ""} if bt == "text"
                 else {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}})
        delta = ({"type": "thinking_delta", "thinking": b["thinking"]} if bt == "thinking"
                 else {"type": "text_delta", "text": b["text"]} if bt == "text"
                 else {"type": "input_json_delta", "partial_json": json.dumps(b["input"], ensure_ascii=False)})
        await out.write(_sse("content_block_start", {
            "type": "content_block_start", "index": idx, "content_block": start,
        }))
        await out.write(_sse("content_block_delta", {
            "type": "content_block_delta", "index": idx, "delta": delta,
        }))
        await out.write(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))
    await out.write(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    }))
    await out.write(_sse("message_stop", {"type": "message_stop"}))
    return out


# ---------------------------------------------------------------------------
# Stub handlers (claude_code probes these; minimal responses are enough)
# ---------------------------------------------------------------------------
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Public handle + start()
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class BridgeHandle:
    app_handle: AppHandle
    store: _Store
    public_host: str

    @property
    def public_url(self) -> str:
        return f"http://{self.public_host}:{self.app_handle.port}"

    def open_session(self, session_id: str, *, sampling_defaults: dict[str, Any] | None = None) -> None:
        self.store.set_defaults(session_id, sampling_defaults or {})

    def pop_session(self, session_id: str) -> tuple[list[int], list[int], list[int]]:
        fut = asyncio.run_coroutine_threadsafe(self.store.pop(session_id), self.app_handle.loop)
        s = fut.result(timeout=10)
        if s is None: return [], [], []
        return s.prompt_ids, s.response_ids, s.loss_mask

    def stop(self) -> None:
        self.app_handle.stop()


def start(
    *,
    tokenizer,
    sglang_url: str,
    tool_parser: str | None = "glm47",
    reasoning_parser: str | None = "glm45",
    host: str = "0.0.0.0",
    port: int = 0,
    public_host: str | None = None,
) -> BridgeHandle:
    """Spin up the bridge on a daemon thread; return a handle.

    Args:
        tokenizer:        HF tokenizer that supports apply_chat_template(tools=)
        sglang_url:       slime SGLang router base URL (e.g. ``http://10.0.0.1:30000``)
        tool_parser:      Name in ``FunctionCallParser.ToolCallParserEnum``
                          ('glm47' / 'qwen25' / 'deepseekv3' / ...) or None to disable.
        reasoning_parser: Name in ``ReasoningParser.DetectorMap``
                          ('glm45' / 'qwen3' / 'deepseek-r1' / ...) or None to disable.
    """
    store = _Store()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["tokenizer"] = tokenizer
    app["sglang_url"] = sglang_url.rstrip("/")
    app["tool_parser"] = tool_parser
    app["reasoning_parser"] = reasoning_parser
    app["store"] = store
    app.router.add_post("/v1/messages", _handle_messages)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    handle = run_app_in_thread(app, host=host, port=port, thread_name="anthropic-bridge")
    logger.info("[coding_agent_rl.bridge] %s -> %s (tool=%s reasoning=%s)",
                handle.url, sglang_url, tool_parser, reasoning_parser)
    return BridgeHandle(app_handle=handle, store=store, public_host=public_host or host)
