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

import asyncio
import dataclasses
import json
import logging
import re
import secrets
from collections.abc import Callable
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import (
    ADAPTER_KEY,
    REASONING_PARSER_KEY,
    TOKENIZER_KEY,
    TOOL_PARSER_KEY,
    BaseAdapter,
    call_sglang_generate,
    ok_response,
    request_session_id,
)
from slime.agent.parsing import ParsedModelOutput, parse_model_output
from slime.agent.trajectory_manager import TrajectoryManager
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults, context budget, request lock.

    Trajectory state lives in ``AnthropicAdapter.manager`` (one shared
    TrajectoryManager keyed by sid across all sessions).
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter with session lifecycle helpers."""

    session_cls = Session

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        tito_snapshot_min_loss_tokens: int | None = None,
        max_turns_per_sid: int | None = None,
        on_turn_appended: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
        )
        # ONE manager shared across all sids; per-sid trees live inside.
        # ``None`` here means "caller did not specify" → let TrajectoryManager's
        # own default take over. Pass an int (incl. 0 to disable) to override.
        mgr_kwargs: dict[str, int] = {}
        if tito_snapshot_min_loss_tokens is not None:
            mgr_kwargs["tito_snapshot_min_loss_tokens"] = tito_snapshot_min_loss_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)
        # Optional debug hook invoked after each successful append_turn.
        # Signature: (sid, prompt_messages, tools, response_message,
        #             prompt_ids, response_ids, finish_reason) -> None.
        # Exceptions are swallowed; never block the SSE response.
        self.on_turn_appended: Callable[..., None] | None = on_turn_appended
        # Per-sid turn cap; None disables. When set, /v1/messages returns 429
        # once a sid has made this many turns. Prevents runaway agents from
        # burning the whole budget.
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}
        self.app.router.add_post("/v1/messages", _handle_request)
        self.app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample: Sample | None = None,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        wait_timeout: float = 5.0,
    ) -> list[Sample]:
        """Drain a session's trajectory into Sample objects.

        Waits out in-flight requests for ``sid``, then linearises the
        per-sid tree via ``TrajectoryManager.get_trajectory``. Idempotent --
        a second call for an already-popped sid returns ``[]``.
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        # Drop the per-sid adapter Session; the trajectory itself is in
        # manager._trees and will be popped by get_trajectory(drop=True).
        self.store.pop(sid, None)
        return self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
        )


# =============================================================================
# Translation (Anthropic wire <-> chat-template messages)
# =============================================================================


# Claude Code CLI leaks ``x-anthropic-billing-header: ...cch=<hash>;`` as a text
# block at the top of the system prompt. The cch hash changes per request, so
# without stripping it the rendered system tokens differ every turn and the
# manager tree can't chain consecutive turns together.
_CLAUDE_CODE_BILLING_HEADER_RE = re.compile(
    r"^\s*x-anthropic-billing-header:[^\n]*\n?",
    re.IGNORECASE,
)


def _scrub_claude_code_billing_header_in_body(body_obj: dict) -> bool:
    """Strip Claude Code's billing-header sidechannel from ``body['system']``.

    Handles both Anthropic shapes (``system: str`` and
    ``system: list[{type:"text",text:"..."}]``). Mutates ``body_obj`` in
    place; returns True iff anything changed.
    """
    sysm = body_obj.get("system")
    changed = False
    if isinstance(sysm, str):
        cleaned = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", sysm)
        if cleaned != sysm:
            body_obj["system"] = cleaned if cleaned.strip() else ""
            changed = True
    elif isinstance(sysm, list):
        new_blocks: list = []
        for block in sysm:
            if not isinstance(block, dict) or block.get("type") != "text":
                new_blocks.append(block)
                continue
            txt = block.get("text") or ""
            cleaned = _CLAUDE_CODE_BILLING_HEADER_RE.sub("", txt)
            if not cleaned.strip():
                # Whole block was the sidechannel — drop it.
                changed = True
                continue
            if cleaned != txt:
                new_block = dict(block)
                new_block["text"] = cleaned
                new_blocks.append(new_block)
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            body_obj["system"] = new_blocks
    return changed


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
                    # Match the canonical shape produced by
                    # _build_blocks_and_response_message for sampled leaves so
                    # that node_match_key (json.dumps sort_keys) hashes a
                    # replayed assistant identically to its leaf. We drop the
                    # wire-only "id" — see the matching note on the leaf side.
                    # NB: arguments stays a dict here (NOT a JSON string).
                    # The translated list is also fed to the chat template
                    # via apply_chat_template; Qwen3's template calls
                    # `arguments | items` which requires a mapping. A JSON
                    # string would raise "Can only get item pairs from a
                    # mapping." mid-render. node_match_key's json.dumps
                    # sort_keys=True recursively sorts dict keys so two
                    # equivalent dicts still hash identically.
                    tcs.append(
                        {
                            "type": "function",
                            "function": {
                                "name": b.get("name", "tool"),
                                "arguments": b.get("input") or {},
                            },
                        }
                    )
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
# Local chat-template render helper.
#
# The Anthropic adapter renders directly from a message list -- no chain
# bookkeeping needed because TrajectoryManager is the routing authority.
# =============================================================================


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
) -> list[int]:
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


# =============================================================================
# Reply building: parsed output -> Anthropic blocks + OpenAI-shape response_message
# =============================================================================


def _build_blocks_and_response_message(
    parsed: ParsedModelOutput,
    finish: str,
) -> tuple[list[dict], str, dict[str, Any]]:
    """Pack parsed model output into:
      - Anthropic content blocks (sent over the wire),
      - stop_reason,
      - response_message (OpenAI shape) for TrajectoryManager.append_turn.

    The tool_calls inside response_message use canonical JSON args (sorted
    keys, str-encoded) so the node_match_key the manager computes for this
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
        # NB: do NOT include tu_id here. The id is wire-only (clients use it
        # to correlate tool_result blocks). When cc echoes this assistant on
        # the next /v1/messages, it sends the original id; slime regenerates
        # a fresh id each call. Including the id in response_message would
        # make node_match_key differ between leaf and echo, breaking the
        # leaf-vs-replay merge — see trajectory_manager DFS Step 1.
        #
        # arguments stays a dict to mirror _translate_anthropic. The
        # trajectory_manager node_match_key uses json.dumps(sort_keys=True)
        # which is invariant to dict key order, so two equivalent dicts
        # hash identically.
        response_tcs.append(
            {
                "type": "function",
                "function": {
                    "name": tu["name"],
                    "arguments": tu.get("input") or {},
                },
            }
        )

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if parsed.tool_uses:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    response_message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        response_message["reasoning_content"] = parsed.reasoning
    if response_tcs:
        response_message["tool_calls"] = response_tcs

    return blocks, stop_reason, response_message


def _finish_reason_for_manager(finish: str, tool_uses: list[dict]) -> str:
    if tool_uses:
        return "tool_calls"
    return finish or "stop"


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

    # Per-sid turn cap (HTTP 429). When the adapter was constructed with a
    # ``max_turns_per_sid`` ceiling, refuse further /v1/messages calls past
    # that count so a runaway agent in a sandbox exits cleanly instead of
    # burning the whole token budget. ``None`` (default) disables.
    cap = adapter.max_turns_per_sid
    if cap is not None:
        prior = adapter._sid_turn_count.get(sid, 0)
        if prior >= cap:
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (f"adapter: sid {sid!r} exceeded max_turns_per_sid={cap}; killing run"),
                    }
                },
                status=429,
            )
        adapter._sid_turn_count[sid] = prior + 1

    # Strip Claude Code's per-request billing-header sidechannel BEFORE the
    # adapter renders prompt_ids. Also fold mid-list ``role: system`` messages
    # into a neighbouring user message so Qwen3 chat templates accept them.
    # Both are no-ops when the relevant patterns aren't present.
    _scrub_claude_code_billing_header_in_body(body)
    _fold_mid_list_system_into_user(body)

    app = request.app
    tok = app[TOKENIZER_KEY]
    s = adapter.store.setdefault(sid, Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:  # same sid -> serialized
            translated = _translate_anthropic(body.get("messages") or [], body.get("system"))
            tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))
            prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

            turn = await call_sglang_generate(
                prompt_ids,
                s,
                body,
                app,
                max_token_keys=("max_tokens",),
                stop_keys=("stop_sequences",),
                log_prefix="anthropic_adapter",
                logger=logger,
                session_id=sid,
            )

            raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=app[TOOL_PARSER_KEY],
                reasoning_parser_name=app[REASONING_PARSER_KEY],
            )
            blocks, stop_reason, response_message = _build_blocks_and_response_message(parsed, turn.finish_reason)

            output_ids = list(turn.output_ids)
            finish_reason = _finish_reason_for_manager(turn.finish_reason, parsed.tool_uses)

            if _is_cc_title_generation_request(translated, tools_schema):
                # Claude Code meta request (per-session title generation).
                # Skip the trajectory so it doesn't pollute the tree / become
                # an RL sample. The on_turn_appended hook below still fires,
                # so per-turn dumps (request, sse, openai.json) keep landing
                # on disk for debugging. See spec
                # docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md.
                logger.info(
                    "skipping append_turn for cc title-generation request (sid=%s)",
                    sid,
                )
            else:
                try:
                    adapter.manager.append_turn(
                        sid,
                        prompt_messages=translated,
                        tools=tools_schema,
                        prompt_ids=prompt_ids,
                        response_ids=output_ids,
                        response_logprobs=(
                            list(turn.output_log_probs)
                            if turn.output_log_probs and len(turn.output_log_probs) == len(turn.output_ids)
                            else None
                        ),
                        response_message=response_message,
                        finish_reason=finish_reason,
                        metadata={"sid": sid},
                    )
                except Exception:
                    logger.exception("append_turn(sid=%s) failed", sid)

            hook = adapter.on_turn_appended
            if hook is not None:
                try:
                    hook(
                        sid,
                        translated,
                        tools_schema,
                        response_message,
                        prompt_ids,
                        output_ids,
                        finish_reason,
                    )
                except Exception:
                    logger.exception("on_turn_appended hook failed (sid=%s)", sid)

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)

        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await _stream_response(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_message_response(body, blocks, stop_reason, in_tok, out_tok))
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
