"""Anthropic Messages API <-> SGLang /generate middleware (sync RL, fan-out).

claude-code calls /v1/messages here as if it were Anthropic. Per session_id
(Bearer token), the middleware:
  * keeps an append-only translated chat history per chain (main + active sub)
  * renders with raw-token splice so prefix tokens stay byte-identical
  * masks model-generated tokens (loss_mask=1) vs template/observation (0)
  * verifies TITO (Tokenizer In / Tokenizer Out) per turn; zeros loss_mask on mismatch
  * emits 3 kinds of segments at the end (fan-out):
        - subagent    completed Task/Agent dispatch
        - pre_wipe    chain frozen by auto-compact / wipe
        - final       tail of the main chain

Public API used by generate.py:
    start(...)                    -> MiddlewareHandle
    handle.public_url             public URL for sandbox -> shim
    handle.open_session(sid, ...) reset sampling defaults
    handle.pop_session_split(sid) -> list[segment]
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import secrets
from typing import Any

import aiohttp
from aiohttp import web

from .aiohttp_threaded import AppHandle, run_app_in_thread

logger = logging.getLogger(__name__)

# Per-segment hard cap on prompt+response token count. Drops any segment
# (subagent / pre_wipe / final) over this — claude-code auto-compact estimates
# in its own tokenizer space, so a 100k autoCompactWindow can produce 130k+
# Qwen-tokenized segments after sub-agent dispatch reads large files. Such
# segments OOM fused CE on actor_train (single sample dynamic-batch can't
# split). 0 disables the cap.
_MAX_SEGMENT_TOKENS = int(os.environ.get("SWE_MAX_SEGMENT_TOKENS", "96000") or 0)

# Qwen3 reasoning chat template auto-injects this before any completed
# assistant `content` that has no `reasoning_content` entry. The raw-splice
# renderer must swallow it so spliced output isn't doubled.
_EMPTY_THINK_STUB_TEXT = "<think>\n\n</think>\n\n"

# Used to find the chat template's generation-prompt tokens (e.g. `<think>\n`)
# so we can stitch them onto the front of raw output_ids.
_ASSISTANT_MARKER_TEXT = "<|im_start|>assistant\n"

# Raw-splice placeholder bracket. \x07 (BEL) keeps BPE boundaries clean.
_RAW_PH_PREFIX = "\x07RAWSPLICE_"
_RAW_PH_SUFFIX = "_END\x07"

# Tool names claude-code uses to dispatch a sub-agent.
_SUBAGENT_TOOL_NAMES = frozenset({"Task", "Agent"})


# =============================================================================
# Dataclasses
# =============================================================================


@dataclasses.dataclass
class Chain:
    """One conversation chain. Either the main chain or an active sub-agent.

    Owns its own splice/pending state because sub system_prompt differs from
    main, so their cumulative prefixes are different."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)
    asst_raw_tokens: dict[int, tuple[list[int], int]] = dataclasses.field(default_factory=dict)
    pending_raw_tokens: list[tuple[list[int], int]] = dataclasses.field(default_factory=list)
    dispatch_tool_use_id: str = ""   # empty for main; set for sub-agent
    last_finish_reason: str = ""


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    active_sub: Chain | None = None
    pending_dispatch_id: str = ""

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    last_finish_reason: str = ""

    # Chronological completed segments. Each entry is
    # (kind, (prompt_ids, response_ids, loss_mask, meta)) where kind is one of
    # {"subagent", "pre_wipe"}.
    _emit_order: list[tuple[str, tuple[list[int], list[int], list[int], dict]]] = \
        dataclasses.field(default_factory=list)


# =============================================================================
# Primitives + Store
# =============================================================================


def _strip_cache_control(obj: Any) -> Any:
    """Drop Anthropic prompt-caching ``cache_control`` keys before hashing -
    cache_control moves across turns so the same logical message would
    otherwise hash differently each request."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True,
                        ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class Store:
    def __init__(self) -> None:
        self._d: dict[str, Session] = {}
        self._guard = asyncio.Lock()

    async def get(self, sid: str) -> Session:
        async with self._guard:
            return self._d.setdefault(sid, Session())

    async def pop(self, sid: str) -> "Session | None":
        async with self._guard:
            return self._d.pop(sid, None)

    def open_session(self, sid: str, *, defaults: dict[str, Any]) -> None:
        s = self._d.setdefault(sid, Session())
        s.sampling_defaults = dict(defaults or {})


def _attach_pending_raw_tokens(
    chain: Chain, base_index: int, new_translated_msgs: list[dict],
) -> None:
    """Pop pending_raw_tokens onto asst_raw_tokens for each new assistant
    message. Pairs with the one-push-per-/generate site in _handle_messages."""
    for offset, m in enumerate(new_translated_msgs):
        if m.get("role") != "assistant":
            continue
        if not chain.pending_raw_tokens:
            continue
        chain.asst_raw_tokens[base_index + offset] = chain.pending_raw_tokens.pop(0)


# =============================================================================
# Anthropic <-> chat-template translation
# =============================================================================


def _flatten(c: Any) -> str:
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


def _translate_messages(messages: list[dict], system: Any) -> list[dict]:
    """Anthropic blocks -> chat-template messages."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": _flatten(system)})
    for m in messages or []:
        if not isinstance(m, dict):
            continue
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
                    tcs.append({"function": {"name": b.get("name", "tool"),
                                             "arguments": b.get("input") or {}}})
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            out.append(mo)
        elif role == "system":
            out.append({"role": "system", "content": _flatten(content)})
    return out


def _tools_schema(anthropic_tools: list[dict] | None) -> list[dict] | None:
    if not anthropic_tools:
        return None
    out = []
    for t in anthropic_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
        }})
    return out or None


# =============================================================================
# Raw-token splice
# =============================================================================


def _detect_gen_prefix(ideal_ids: list[int], marker_ids: list[int]) -> list[int]:
    """Return tokens after the LAST marker_ids occurrence in ideal_ids
    (e.g. `<think>\\n` for Qwen3 reasoning)."""
    if not marker_ids:
        return []
    n = len(marker_ids)
    for start in range(len(ideal_ids) - n, -1, -1):
        if ideal_ids[start:start + n] == marker_ids:
            return list(ideal_ids[start + n:])
    return []


def _render_with_raw_splice(
    tok: Any,
    chat_messages: list[dict],
    tools_schema: list[dict] | None,
    asst_raw_tokens: dict[int, tuple[list[int], int]],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Render chat_messages but substitute raw tokens for any assistant idx in
    asst_raw_tokens. Returns (ideal_ids, raw_ranges) where each range is
    (splice_start, gen_start, splice_end) in ideal_ids coords:
      [splice_start:gen_start] = template-injected gen prefix (loss_mask=0)
      [gen_start:splice_end]   = model-generated (loss_mask=1)

    Requires the tokenizer to return offset_mapping (true for HF Fast
    tokenizers used by Qwen3 / GLM-4.x). No fallback path."""
    valid = {i: tup for i, tup in asst_raw_tokens.items() if 0 <= i < len(chat_messages)}
    if not valid:
        text = tok.apply_chat_template(
            chat_messages, tools=tools_schema, tokenize=False, add_generation_prompt=True,
        )
        return tok.encode(text, add_special_tokens=False), []

    placeholders: dict[int, str] = {}
    render_msgs: list[dict] = []
    for i, m in enumerate(chat_messages):
        if i in valid:
            ph = f"{_RAW_PH_PREFIX}{i}_{secrets.token_hex(6)}{_RAW_PH_SUFFIX}"
            placeholders[i] = ph
            render_msgs.append({"role": "assistant", "content": ph})
        else:
            render_msgs.append(m)

    text = tok.apply_chat_template(
        render_msgs, tools=tools_schema, tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    template_ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])

    stub_ids = tok.encode(_EMPTY_THINK_STUB_TEXT, add_special_tokens=False)
    stub_ids = list(stub_ids.ids) if hasattr(stub_ids, "ids") else list(stub_ids)

    placeholder_ranges: list[tuple[int, int, int]] = []
    for asst_idx, ph in placeholders.items():
        char_start = text.find(ph)
        if char_start < 0:
            logger.warning("[middleware] raw-splice: placeholder for asst %d not found", asst_idx)
            continue
        char_end = char_start + len(ph)
        tok_start = tok_end = None
        for j, (cs, ce) in enumerate(offsets):
            if tok_start is None and ce > char_start:
                tok_start = j
            if cs < char_end:
                tok_end = j + 1
            elif cs >= char_end:
                break
        if tok_start is None or tok_end is None:
            logger.warning("[middleware] raw-splice: no tokens overlap placeholder for asst %d", asst_idx)
            continue
        # Roll back over the empty-think stub if the template injected one.
        n_stub = len(stub_ids)
        if n_stub and tok_start >= n_stub \
                and template_ids[tok_start - n_stub:tok_start] == stub_ids:
            tok_start -= n_stub
        placeholder_ranges.append((tok_start, tok_end, asst_idx))
    placeholder_ranges.sort()

    ideal_ids: list[int] = []
    raw_ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for tok_start, tok_end, asst_idx in placeholder_ranges:
        ideal_ids.extend(template_ids[cursor:tok_start])
        rs = len(ideal_ids)
        full_raw, gen_off = valid[asst_idx]
        ideal_ids.extend(full_raw)
        re_ = len(ideal_ids)
        raw_ranges.append((rs, rs + gen_off, re_))
        cursor = tok_end
    ideal_ids.extend(template_ids[cursor:])
    return ideal_ids, raw_ranges


def verify_tito_for_turn(tok: Any, raw_text: str, output_ids: list[int]) -> bool:
    """TITO check: re-encode the decoded text and compare to output_ids.
    Caller must guard `len(output_ids) > 0`."""
    retok = tok.encode(raw_text, add_special_tokens=False)
    if hasattr(retok, "ids"):
        retok = list(retok.ids)
    return list(retok) == list(output_ids)


# =============================================================================
# Output parsing
# =============================================================================


def _parse_output(
    text: str, *, tool_parser_name: str | None, reasoning_parser_name: str | None,
    tools_schema: list[dict] | None,
) -> tuple[str, str, list[dict]]:
    thinking, body = "", text
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser
        r, b = ReasoningParser(model_type=reasoning_parser_name, stream_reasoning=False).parse_non_stream(text)
        thinking, body = r or "", b or ""
        if not thinking and "</think>" in body:
            thinking, body = body.split("</think>", 1)

    tool_uses: list[dict] = []
    if tool_parser_name and tools_schema:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        body, calls = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name).parse_non_stream(body)
        for c in calls:
            try:
                args = json.loads(c.parameters or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": c.parameters}
            tool_uses.append({"name": c.name or "tool", "input": args})
    if not tool_uses and tools_schema:
        body, tool_uses = _parse_xml_tool_calls(body, tools_schema)
    return thinking, (body or "").strip(), tool_uses


def _parse_xml_tool_calls(text: str, tools_schema: list[dict]) -> tuple[str, list[dict]]:
    """Fallback for Qwen's occasional Anthropic-style XML tool-call format."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    calls: list[dict[str, Any]] = []

    def repl(match: re.Match[str]) -> str:
        name, inner = match.group(1), match.group(2)
        if name not in valid_tools:
            return match.group(0)
        args = {
            p.group(1): p.group(2).strip()
            for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
        }
        calls.append({"name": name, "input": args})
        return ""

    cleaned = re.sub(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        repl, text, flags=re.DOTALL,
    )
    cleaned = cleaned.replace("<|im_end|>", "")
    return cleaned, calls


# =============================================================================
# Engine: single /generate call (sync RL)
# =============================================================================


async def _post_generate(
    sglang_url: str, input_ids: list[int], sampling_params: dict[str, Any],
) -> tuple[list[int], str]:
    """Single non-streaming POST to sglang. Returns (output_ids, finish_reason)."""
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout) as sess, \
               sess.post(f"{sglang_url}/generate", json={
                   "input_ids": input_ids, "sampling_params": sampling_params,
                   "return_logprob": True,
               }) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
        data = await r.json()
    meta = data.get("meta_info") or {}
    output_ids = [x[1] for x in (meta.get("output_token_logprobs") or [])]
    finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    return output_ids, finish


# =============================================================================
# Segments: sub-agent routing + pre_wipe snapshot + final emit
# =============================================================================


def _snapshot_chain(session: Session, chain: Chain, kind: str) -> None:
    """Freeze chain into a 4-tuple and append to _emit_order under `kind`."""
    session._emit_order.append((kind, (
        list(chain.prompt_ids), list(chain.response_ids), list(chain.loss_mask),
        {"segment_kind": kind, "finish_reason": chain.last_finish_reason},
    )))


def maybe_pop_subagent(session: Session, all_msgs: list[dict]) -> None:
    """Close active sub-agent when its dispatch tool_result appears on main."""
    if not session.pending_dispatch_id or session.active_sub is None:
        return
    tool_use_id = session.pending_dispatch_id
    for m in all_msgs:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result" \
                    and b.get("tool_use_id") == tool_use_id:
                _snapshot_chain(session, session.active_sub, "subagent")
                session.active_sub = None
                session.pending_dispatch_id = ""
                return


def pick_target(session: Session, req_system_hash: str) -> tuple[Chain, bool]:
    """Route to main or active sub-agent.

    If a sub is active and its system_hash matches, route to sub. Otherwise
    route to main (sub stays open; it'll close when its tool_result arrives
    via maybe_pop_subagent)."""
    if session.active_sub is not None \
            and req_system_hash == session.active_sub.system_hash:
        return session.active_sub, True
    return session.main, False


def classify_and_apply(
    target: Chain, session: Session, *,
    req_system_hash: str, msg_hashes: list[str],
) -> tuple[str | None, bool]:
    """Returns (kind, is_append). kind in {None, "pre_wipe"}.
    Side effect: snapshots pre_wipe on non-linear update with non-empty response."""
    is_append = (
        req_system_hash == target.system_hash
        and len(msg_hashes) >= target.seen_msgs
        and msg_hashes[:target.seen_msgs] == target.msg_hashes[:target.seen_msgs]
    )
    if target.seen_msgs == 0 or is_append:
        return None, is_append
    if target.response_ids:
        _snapshot_chain(session, target, "pre_wipe")
    return "pre_wipe", False


def pop_session_split(session: Session) -> list[
    tuple[list[int], list[int], list[int], dict],
]:
    """Drain any in-flight sub-agent, then replay _emit_order chronologically
    and append the final main-line segment. Empty-response segments dropped;
    oversized (prompt+response > SWE_MAX_SEGMENT_TOKENS) segments also dropped."""
    if session.active_sub is not None:
        _snapshot_chain(session, session.active_sub, "subagent")
        session.active_sub = None

    segments: list[tuple[list[int], list[int], list[int], dict]] = []
    dropped: list[tuple[str, int]] = []
    for _kind, (p, r, m, meta) in session._emit_order:
        if not r:
            continue
        total = len(p) + len(r)
        if _MAX_SEGMENT_TOKENS > 0 and total > _MAX_SEGMENT_TOKENS:
            dropped.append((meta.get("segment_kind", "?"), total))
            continue
        segments.append((p, r, m, dict(meta)))

    if session.main.response_ids:
        total = len(session.main.prompt_ids) + len(session.main.response_ids)
        if _MAX_SEGMENT_TOKENS > 0 and total > _MAX_SEGMENT_TOKENS:
            dropped.append(("final", total))
        else:
            segments.append((
                list(session.main.prompt_ids), list(session.main.response_ids),
                list(session.main.loss_mask),
                {"segment_kind": "final", "finish_reason": session.last_finish_reason},
            ))

    if dropped:
        logger.warning(
            "[middleware] dropped %d oversized segments (cap=%d): %s",
            len(dropped), _MAX_SEGMENT_TOKENS,
            ", ".join(f"{k}={n}" for k, n in dropped),
        )
    return segments


# =============================================================================
# Handler
# =============================================================================


def _update_prompt_and_response(
    target: Chain, ideal_ids: list[int],
    raw_ranges: list[tuple[int, int, int]], kind: str | None,
) -> None:
    # First turn or post-wipe: reset.
    if not target.prompt_ids or kind == "pre_wipe":
        target.prompt_ids = ideal_ids
        target.response_ids = []
        target.loss_mask = []
        return

    # Divergence: keep prompt anchor, treat everything after as obs (loss=0).
    if ideal_ids[:len(target.prompt_ids)] != target.prompt_ids:
        logger.warning("[middleware] template re-render mismatch; rebaselining")
        target.response_ids = ideal_ids[len(target.prompt_ids):]
        target.loss_mask = [0] * len(target.response_ids)
        return

    # Linear append.
    response = ideal_ids[len(target.prompt_ids):]
    mask = [0] * len(response)
    prompt_len = len(target.prompt_ids)
    response_len = len(response)
    for _splice_start, gen_start, splice_end in raw_ranges:
        a = max(0, gen_start - prompt_len)
        b = min(response_len, max(0, splice_end - prompt_len))
        for k in range(a, b):
            mask[k] = 1
    target.response_ids = response
    target.loss_mask = mask


async def _handle_messages(request: web.Request) -> web.StreamResponse:
    A = request.app
    tok = A["tokenizer"]
    store: Store = A["store"]

    body = await request.json()
    sid = (request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
           or request.headers.get("x-session-id", ""))
    if not sid:
        return web.json_response({"type": "error", "error": {
            "type": "missing_session",
            "message": "Authorization Bearer <session_id> required",
        }}, status=400)
    s = await store.get(sid)

    async with s.lock:
        # 1. Fingerprints.
        all_msgs = body.get("messages") or []
        msg_hashes = [_hash_obj(m) for m in all_msgs]
        system_value = body.get("system") if "system" in body else None
        req_system_hash = _hash_obj(system_value) if "system" in body else s.main.system_hash

        # 2. Close any returning sub-agent BEFORE routing.
        maybe_pop_subagent(s, all_msgs)

        # 3. Route + classify (may snapshot pre_wipe on target).
        target, target_is_sub = pick_target(s, req_system_hash)
        kind, is_append = classify_and_apply(
            target, s, req_system_hash=req_system_hash, msg_hashes=msg_hashes,
        )

        # 4. Ingest new messages + attach pending raw tokens.
        if target.seen_msgs == 0 or kind == "pre_wipe":
            # Reset: pending was just cleared, no raw tokens to attach.
            target.chat_messages = _translate_messages(all_msgs, body.get("system"))
            target.system_hash = req_system_hash
            target.asst_raw_tokens.clear()
            target.pending_raw_tokens.clear()
        else:  # is_append (only remaining case)
            new = _translate_messages(all_msgs[target.seen_msgs:], None)
            base_idx = len(target.chat_messages)
            target.chat_messages.extend(new)
            _attach_pending_raw_tokens(target, base_idx, new)

        target.seen_msgs = len(all_msgs)
        target.msg_hashes = msg_hashes
        if target.tools_schema is None:
            target.tools_schema = _tools_schema(body.get("tools"))

        # 5. Render with raw splice + update prompt/response.
        ideal_ids, raw_ranges = _render_with_raw_splice(
            tok, target.chat_messages, target.tools_schema, target.asst_raw_tokens,
        )
        _update_prompt_and_response(target, ideal_ids, raw_ranges, kind)

        # 6. Sampling params (request overrides session defaults).
        sp = {"skip_special_tokens": False, "spaces_between_special_tokens": False,
              "no_stop_trim": True, "max_new_tokens": 4096,
              **(s.sampling_defaults or {})}
        for src_k, dst_k in (("max_tokens", "max_new_tokens"), ("temperature", "temperature"),
                             ("top_p", "top_p"), ("top_k", "top_k")):
            if src_k in body:
                sp[dst_k] = body[src_k]
        if body.get("stop_sequences"):
            sp["stop"] = body["stop_sequences"]

        # 7. Single /generate call.
        try:
            output_ids, finish = await _post_generate(A["sglang_url"], ideal_ids, sp)
        except Exception as e:
            return web.json_response({"type": "error", "error": {
                "type": "upstream_error", "message": str(e),
            }}, status=502)

        # 8. Extend response_ids + loss_mask.
        target.response_ids.extend(output_ids)
        target.loss_mask.extend([1] * len(output_ids))
        target.last_finish_reason = finish
        if not target_is_sub:
            s.last_finish_reason = finish

        # 9. Decode once, used by both TITO check and parser.
        n = len(output_ids)
        raw_output = tok.decode(output_ids, skip_special_tokens=False) if n > 0 else ""
        if n > 0 and not verify_tito_for_turn(tok, raw_output, output_ids):
            target.loss_mask[-n:] = [0] * n
            logger.warning("[middleware] TITO mismatch; loss_mask zeroed (n=%d)", n)

        # 10. Parse + queue raw tokens for next-round attach.
        thinking, visible, tool_uses = _parse_output(
            raw_output,
            tool_parser_name=A["tool_parser"], reasoning_parser_name=A["reasoning_parser"],
            tools_schema=target.tools_schema,
        )
        gen_prefix = _detect_gen_prefix(ideal_ids, A["assistant_marker_ids"])
        target.pending_raw_tokens.append((gen_prefix + list(output_ids), len(gen_prefix)))

        # 11. Build Anthropic blocks + arm sub-agent dispatch in one pass.
        blocks: list[dict] = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": thinking})
        if visible:
            blocks.append({"type": "text", "text": visible})
        dispatch_id = ""
        for tu in tool_uses:
            tu_id = f"toolu_{secrets.token_hex(8)}"
            blocks.append({"type": "tool_use", "id": tu_id,
                           "name": tu["name"], "input": tu["input"]})
            if tu["name"] in _SUBAGENT_TOOL_NAMES:
                dispatch_id = tu_id
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        if dispatch_id and not target_is_sub:
            s.pending_dispatch_id = dispatch_id
            if s.active_sub is None:
                s.active_sub = Chain(dispatch_tool_use_id=dispatch_id)

        stop_reason = ("tool_use" if tool_uses
                       else "max_tokens" if finish == "length" else "end_turn")

    # 12. Emit Anthropic SSE (outside lock). claude-code always streams.
    in_tok, out_tok = len(ideal_ids), len(output_ids)
    out = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive",
    })
    await out.prepare(request)
    await out.write(_sse("message_start", {
        "type": "message_start",
        "message": {"id": f"msg_{secrets.token_hex(12)}", "type": "message",
                    "role": "assistant", "model": body.get("model") or "slime-actor",
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": in_tok, "output_tokens": 0}},
    }))
    for idx, b in enumerate(blocks):
        bt = b["type"]
        start = ({"type": "thinking", "thinking": ""} if bt == "thinking"
                 else {"type": "text", "text": ""} if bt == "text"
                 else {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}})
        delta = ({"type": "thinking_delta", "thinking": b["thinking"]} if bt == "thinking"
                 else {"type": "text_delta", "text": b["text"]} if bt == "text"
                 else {"type": "input_json_delta",
                       "partial_json": json.dumps(b["input"], ensure_ascii=False)})
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
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }))
    await out.write(_sse("message_stop", {"type": "message_stop"}))
    return out


# =============================================================================
# Shell + entry
# =============================================================================


async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


@dataclasses.dataclass
class MiddlewareHandle:
    app_handle: AppHandle
    store: Store
    public_host: str

    @property
    def public_url(self) -> str:
        return f"http://{self.public_host}:{self.app_handle.port}"

    def open_session(
        self, session_id: str, *,
        sampling_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.store.open_session(session_id, defaults=sampling_defaults or {})

    async def _pop_session_split(self, session_id: str) -> list[
        tuple[list[int], list[int], list[int], dict],
    ]:
        s = await self.store.pop(session_id)
        if s is None:
            return []
        async with s.lock:
            return pop_session_split(s)

    def pop_session_split(self, session_id: str) -> list[
        tuple[list[int], list[int], list[int], dict],
    ]:
        """Return chronological segments (segment_kind in
        {"subagent", "pre_wipe", "final"}). Session popped."""
        fut = asyncio.run_coroutine_threadsafe(
            self._pop_session_split(session_id), self.app_handle.loop,
        )
        return fut.result(timeout=10)

    def stop(self) -> None:
        self.app_handle.stop()


def start(
    *, tokenizer, sglang_url: str,
    tool_parser: str | None = None, reasoning_parser: str | None = None,
    host: str = "0.0.0.0", port: int = 0, public_host: str | None = None,
) -> MiddlewareHandle:
    """Spin up the middleware on a daemon thread; return a handle."""
    store = Store()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["tokenizer"] = tokenizer
    app["sglang_url"] = sglang_url.rstrip("/")
    app["tool_parser"] = tool_parser
    app["reasoning_parser"] = reasoning_parser
    app["store"] = store
    app["assistant_marker_ids"] = tokenizer.encode(
        _ASSISTANT_MARKER_TEXT, add_special_tokens=False,
    )
    app.router.add_post("/v1/messages", _handle_messages)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    handle = run_app_in_thread(app, host=host, port=port, thread_name="anthropic-middleware")
    logger.info("[coding_agent_rl.middleware] %s -> %s (tool=%s reasoning=%s)",
                handle.url, sglang_url, tool_parser, reasoning_parser)
    return MiddlewareHandle(app_handle=handle, store=store, public_host=public_host or host)
