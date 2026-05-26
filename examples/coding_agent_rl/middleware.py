"""Anthropic Messages API <-> SGLang /generate middleware (sync RL, fan-out).

claude-code calls /v1/messages here as if it were Anthropic. Per session_id
(Bearer token), the middleware:
  * keeps an append-only translated chat history per chain (main + active sub)
  * renders with raw-token splice so prefix tokens stay byte-identical
  * masks model-generated tokens (loss_mask=1) vs template/observation (0)
  * verifies TITO (Tokenizer In / Tokenizer Out) per turn; zeros loss_mask on mismatch
  * emits 3 kinds of segments at the end (fan-out):
        - subagent    completed Task/Agent dispatch
        - wipe        chain frozen by auto-compact / re-baselined
        - final       tail of the main chain

Public API used by generate.py:
    start(...)                    -> MiddlewareHandle
    handle.public_url             public URL for sandbox -> shim
    handle.open_session(sid, ...) reset sampling defaults
    handle.pop_session_split(sid) -> list[segment]

File layout (top to bottom):
    1. Constants
    2. Pure helpers:  hashing, SSE encoding
    3. Pure:          Anthropic <-> chat-template translation
    4. Pure:          raw-token splice rendering + TITO check
    5. Pure:          assistant-output parsing (reasoning / tools / xml fallback)
    6. State:         Chain / Session dataclasses (with their own methods)
    7. State:         Store (session registry)
    8. I/O:           upstream sglang /generate call
    9. I/O:           per-request handler, split into one function per pipeline stage
   10. Entry:         MiddlewareHandle + start()
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
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from .aiohttp_threaded import AppHandle, run_app_in_thread

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Constants
# =============================================================================

# Per-segment hard cap on prompt+response token count. Drops any segment
# (subagent / wipe / final) over this — claude-code auto-compact estimates
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
# 2. Pure helpers: hashing & SSE
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
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


# =============================================================================
# 3. Pure: Anthropic <-> chat-template translation
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
                    tcs.append({"function": {"name": b.get("name", "tool"), "arguments": b.get("input") or {}}})
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
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


# =============================================================================
# 4. Pure: raw-token splice rendering + TITO
# =============================================================================


def _detect_gen_prefix(ideal_ids: list[int], marker_ids: list[int]) -> list[int]:
    """Return tokens after the LAST marker_ids occurrence in ideal_ids
    (e.g. `<think>\\n` for Qwen3 reasoning)."""
    if not marker_ids:
        return []
    n = len(marker_ids)
    for start in range(len(ideal_ids) - n, -1, -1):
        if ideal_ids[start : start + n] == marker_ids:
            return list(ideal_ids[start + n :])
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
        # Qwen3.x fast tokenizers return a BatchEncoding here, not a list[int];
        # list(BatchEncoding) yields its dict keys (["input_ids", ...]), which
        # then poisons sglang /generate as non-integer input_ids -> 502 storm.
        enc = tok.apply_chat_template(
            chat_messages,
            tools=tools_schema,
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
        return list(ids), []

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
        render_msgs,
        tools=tools_schema,
        tokenize=False,
        add_generation_prompt=True,
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
        if n_stub and tok_start >= n_stub and template_ids[tok_start - n_stub : tok_start] == stub_ids:
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
# 5. Pure: output parsing
# =============================================================================


def _parse_output(
    text: str,
    *,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
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
        repl,
        text,
        flags=re.DOTALL,
    )
    cleaned = cleaned.replace("<|im_end|>", "")
    return cleaned, calls


# =============================================================================
# 6. State: Chain / Session dataclasses
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
    dispatch_tool_use_id: str = ""  # empty for main; set for sub-agent
    last_finish_reason: str = ""

    def attach_pending_raw_tokens(
        self,
        base_index: int,
        new_translated_msgs: list[dict],
    ) -> None:
        """Pop pending_raw_tokens onto asst_raw_tokens for each new assistant
        message. Pairs with the one-push-per-/generate site in handler stage
        ``_run_turn``."""
        for offset, m in enumerate(new_translated_msgs):
            if m.get("role") != "assistant":
                continue
            if not self.pending_raw_tokens:
                continue
            self.asst_raw_tokens[base_index + offset] = self.pending_raw_tokens.pop(0)

    def update_prompt_and_response(
        self,
        ideal_ids: list[int],
        raw_ranges: list[tuple[int, int, int]],
        kind: str,
    ) -> None:
        # New chain or wipe: reset.
        if kind != "append":
            self.prompt_ids = ideal_ids
            self.response_ids = []
            self.loss_mask = []
            return

        # Cross-turn analog of per-turn TITO (verify_tito_for_turn): the
        # re-rendered prefix must be byte-identical to the anchor we set on
        # turn 0. If chat_template / tokenizer drifts between renders (tools
        # list change, raw-splice side-effect on historical assistant turns,
        # template non-determinism), raw-splice token accounting becomes
        # unsound for this chain — demote the tail to observation (loss=0)
        # rather than train on a token sequence the model never emitted.
        if ideal_ids[: len(self.prompt_ids)] != self.prompt_ids:
            logger.warning("[middleware] template re-render mismatch; rebaselining")
            self.response_ids = ideal_ids[len(self.prompt_ids) :]
            self.loss_mask = [0] * len(self.response_ids)
            return

        # Linear append.
        response = ideal_ids[len(self.prompt_ids) :]
        mask = [0] * len(response)
        prompt_len = len(self.prompt_ids)
        response_len = len(response)
        for _splice_start, gen_start, splice_end in raw_ranges:
            a = max(0, gen_start - prompt_len)
            b = min(response_len, max(0, splice_end - prompt_len))
            for k in range(a, b):
                mask[k] = 1
        self.response_ids = response
        self.loss_mask = mask


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    active_sub: Chain | None = None
    pending_dispatch_id: str = ""

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)

    # Chronological completed segments. Each entry is
    # (kind, (prompt_ids, response_ids, loss_mask, meta)) where kind is one of
    # {"subagent", "wipe", "final"}.
    _emit_order: list[tuple[str, tuple[list[int], list[int], list[int], dict]]] = dataclasses.field(
        default_factory=list
    )

    # -- segment bookkeeping --------------------------------------------------

    def snapshot(self, chain: Chain, kind: str) -> None:
        """Freeze chain into a 4-tuple and append to _emit_order under `kind`."""
        self._emit_order.append(
            (
                kind,
                (
                    list(chain.prompt_ids),
                    list(chain.response_ids),
                    list(chain.loss_mask),
                    {"segment_kind": kind, "finish_reason": chain.last_finish_reason},
                ),
            )
        )

    # -- sub-agent lifecycle --------------------------------------------------

    def maybe_pop_subagent(self, all_msgs: list[dict]) -> None:
        """Close active sub-agent when its dispatch tool_result appears on main."""
        if not self.pending_dispatch_id or self.active_sub is None:
            return
        tool_use_id = self.pending_dispatch_id
        for m in all_msgs:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tool_use_id:
                    self.snapshot(self.active_sub, "subagent")
                    self.active_sub = None
                    self.pending_dispatch_id = ""
                    return

    def arm_subagent_dispatch(self, dispatch_id: str) -> None:
        """Mark a tool_use_id as the next sub-agent dispatch; open a fresh sub
        chain if none is active."""
        self.pending_dispatch_id = dispatch_id
        if self.active_sub is None:
            self.active_sub = Chain(dispatch_tool_use_id=dispatch_id)

    # -- routing + classification --------------------------------------------

    def pick_target(self, req_system_hash: str) -> tuple[Chain, bool]:
        """Route to main or active sub-agent.

        If a sub is active and its system_hash matches, route to sub. Otherwise
        route to main (sub stays open; it'll close when its tool_result arrives
        via maybe_pop_subagent)."""
        if self.active_sub is not None and req_system_hash == self.active_sub.system_hash:
            return self.active_sub, True
        return self.main, False

    def classify(
        self,
        target: Chain,
        *,
        req_system_hash: str,
        msg_hashes: list[str],
    ) -> str:
        """Returns "new" | "append" | "wipe". Side effect: snapshots the
        existing chain as a "wipe" segment when overwriting a chain with
        non-empty response."""
        if target.seen_msgs == 0:
            return "new"
        is_append = (
            req_system_hash == target.system_hash
            and len(msg_hashes) >= target.seen_msgs
            and msg_hashes[: target.seen_msgs] == target.msg_hashes[: target.seen_msgs]
        )
        if is_append:
            return "append"
        if target.response_ids:
            self.snapshot(target, "wipe")
        return "wipe"

    # -- drain ---------------------------------------------------------------

    def pop_split(self) -> list[tuple[list[int], list[int], list[int], dict],]:
        """Snapshot any in-flight sub-agent and the main chain, then replay
        _emit_order chronologically. Empty-response segments dropped;
        oversized (prompt+response > SWE_MAX_SEGMENT_TOKENS) ones also dropped.

        One-shot: caller must not invoke twice on the same session (would
        re-snapshot main). Current call site pops session from Store first."""
        if self.active_sub is not None:
            self.snapshot(self.active_sub, "subagent")
            self.active_sub = None
        if self.main.response_ids:
            self.snapshot(self.main, "final")

        segments: list[tuple[list[int], list[int], list[int], dict]] = []
        dropped: list[tuple[str, int]] = []
        for kind, (p, r, m, meta) in self._emit_order:
            if not r:
                continue
            total = len(p) + len(r)
            if _MAX_SEGMENT_TOKENS > 0 and total > _MAX_SEGMENT_TOKENS:
                dropped.append((kind, total))
                continue
            segments.append((p, r, m, dict(meta)))

        if dropped:
            logger.warning(
                "[middleware] dropped %d oversized segments (cap=%d): %s",
                len(dropped),
                _MAX_SEGMENT_TOKENS,
                ", ".join(f"{k}={n}" for k, n in dropped),
            )
        return segments


# =============================================================================
# 7. State: Store (session registry)
# =============================================================================


class Store:
    def __init__(self) -> None:
        self._d: dict[str, Session] = {}
        self._guard = asyncio.Lock()

    async def get(self, sid: str) -> Session:
        async with self._guard:
            return self._d.setdefault(sid, Session())

    async def pop(self, sid: str) -> Session | None:
        async with self._guard:
            return self._d.pop(sid, None)

    def open_session(self, sid: str, *, defaults: dict[str, Any]) -> None:
        # Fail-fast on duplicate sid: silently sharing state would interleave
        # two independent rollouts into one chain and corrupt TITO bookkeeping.
        if sid in self._d:
            raise ValueError(
                f"session_id {sid!r} already exists; sids must be unique per agent run",
            )
        s = self._d[sid] = Session()
        s.sampling_defaults = dict(defaults or {})


# =============================================================================
# 8. I/O: upstream sglang /generate call
# =============================================================================


async def _post_generate(
    sglang_url: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
) -> tuple[list[int], str]:
    """Single non-streaming POST to sglang. Returns (output_ids, finish_reason).

    On client-side cancel or transport error we fire-and-forget /abort_request
    so the inflight req drains immediately. Without this, a subsequent
    release_memory_occupation can hit sglang's "server is idle" assertion and
    crash the scheduler (race between cancelled client and pending generate)."""
    rid = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            data = await r.json()
        meta = data.get("meta_info") or {}
        output_ids = [x[1] for x in (meta.get("output_token_logprobs") or [])]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
        return output_ids, finish
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError):
        # Best-effort abort. Use a fresh short-timeout session because the
        # outer one may be tearing down. Swallow errors -- if abort itself
        # fails (eg. sglang already dead) we don't make things worse.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise


# =============================================================================
# 9. I/O: per-request handler, one function per pipeline stage
# =============================================================================


def _extract_session_id(request: web.Request) -> str:
    return request.headers.get("Authorization", "").removeprefix("Bearer ").strip() or request.headers.get(
        "x-session-id", ""
    )


def _err_response(err_type: str, message: str, status: int) -> web.Response:
    return web.json_response(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status=status,
    )


def _ingest_request(
    target: Chain,
    *,
    all_msgs: list[dict],
    body: dict,
    msg_hashes: list[str],
    req_system_hash: str,
    kind: str,
) -> None:
    """Apply incoming messages to target chain.
    kind="append": extend chat history and pop pending raw tokens onto the
    new assistant turns. kind="new"|"wipe": replace chain state."""
    if kind == "append":
        new = _translate_messages(all_msgs[target.seen_msgs :], None)
        base_idx = len(target.chat_messages)
        target.chat_messages.extend(new)
        target.attach_pending_raw_tokens(base_idx, new)
    else:  # "new" or "wipe": full reset
        target.chat_messages = _translate_messages(all_msgs, body.get("system"))
        target.system_hash = req_system_hash
        target.asst_raw_tokens.clear()
        target.pending_raw_tokens.clear()

    target.seen_msgs = len(all_msgs)
    target.msg_hashes = msg_hashes
    if target.tools_schema is None:
        target.tools_schema = _tools_schema(body.get("tools"))


def _build_sampling_params(s: Session, body: dict) -> dict[str, Any]:
    """Session defaults overlaid with per-request overrides."""
    sp = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(s.sampling_defaults or {}),
    }
    for src_k, dst_k in (
        ("max_tokens", "max_new_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
    ):
        if src_k in body:
            sp[dst_k] = body[src_k]
    if body.get("stop_sequences"):
        sp["stop"] = body["stop_sequences"]
    return sp


def _decode_and_verify(
    tok: Any,
    target: Chain,
    output_ids: list[int],
) -> str:
    """Decode output_ids; if the round-trip doesn't match, zero the loss_mask
    over this turn (cause: tokenizer ambiguity). Returns raw decoded text."""
    n = len(output_ids)
    if n == 0:
        return ""
    raw_output = tok.decode(output_ids, skip_special_tokens=False)
    if not verify_tito_for_turn(tok, raw_output, output_ids):
        target.loss_mask[-n:] = [0] * n
        logger.warning("[middleware] TITO mismatch; loss_mask zeroed (n=%d)", n)
    return raw_output


def _build_anthropic_blocks(
    thinking: str,
    visible: str,
    tool_uses: list[dict],
) -> tuple[list[dict], str]:
    """Pack parsed output into Anthropic content blocks; return
    (blocks, dispatch_id). dispatch_id is non-empty iff a sub-agent tool was
    called."""
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if visible:
        blocks.append({"type": "text", "text": visible})
    dispatch_id = ""
    for tu in tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        if tu["name"] in _SUBAGENT_TOOL_NAMES:
            dispatch_id = tu_id
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks, dispatch_id


class _UpstreamError(RuntimeError):
    """sglang /generate failed; surfaces as a 502 to claude-code."""


async def _run_turn(
    s: Session,
    body: dict,
    app: web.Application,
) -> tuple[list[dict], str, int, int]:
    """One protected RL turn against sglang. Holds the session lock for the
    whole turn (state mutation + upstream call). Raises _UpstreamError on
    sglang failure."""
    tok = app["tokenizer"]

    async with s.lock:
        # Fingerprints + routing.
        all_msgs = body.get("messages") or []
        msg_hashes = [_hash_obj(m) for m in all_msgs]
        req_system_hash = _hash_obj(body.get("system")) if "system" in body else s.main.system_hash

        s.maybe_pop_subagent(all_msgs)
        target, target_is_sub = s.pick_target(req_system_hash)
        kind = s.classify(target, req_system_hash=req_system_hash, msg_hashes=msg_hashes)

        # Ingest + render.
        _ingest_request(
            target, all_msgs=all_msgs, body=body, msg_hashes=msg_hashes, req_system_hash=req_system_hash, kind=kind
        )
        ideal_ids, raw_ranges = _render_with_raw_splice(
            tok,
            target.chat_messages,
            target.tools_schema,
            target.asst_raw_tokens,
        )
        target.update_prompt_and_response(ideal_ids, raw_ranges, kind)

        # Upstream call.
        sp = _build_sampling_params(s, body)
        try:
            output_ids, finish = await _post_generate(app["sglang_url"], ideal_ids, sp)
        except Exception as e:
            raise _UpstreamError(str(e)) from e

        # Commit output + record loss.
        target.response_ids.extend(output_ids)
        target.loss_mask.extend([1] * len(output_ids))
        target.last_finish_reason = finish

        raw_output = _decode_and_verify(tok, target, output_ids)
        thinking, visible, tool_uses = _parse_output(
            raw_output,
            tool_parser_name=app["tool_parser"],
            reasoning_parser_name=app["reasoning_parser"],
            tools_schema=target.tools_schema,
        )

        # Queue raw tokens for the next /generate to attach.
        gen_prefix = _detect_gen_prefix(ideal_ids, app["assistant_marker_ids"])
        target.pending_raw_tokens.append((gen_prefix + list(output_ids), len(gen_prefix)))

        blocks, dispatch_id = _build_anthropic_blocks(thinking, visible, tool_uses)
        if dispatch_id and not target_is_sub:
            s.arm_subagent_dispatch(dispatch_id)

        stop_reason = "tool_use" if tool_uses else "max_tokens" if finish == "length" else "end_turn"

    return blocks, stop_reason, len(ideal_ids), len(output_ids)


async def _stream_anthropic_sse(
    request: web.Request,
    body: dict,
    blocks: list[dict],
    in_tok: int,
    out_tok: int,
    stop_reason: str,
) -> web.StreamResponse:
    """Emit the Anthropic Messages SSE stream. claude-code always streams."""
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    await out.write(
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{secrets.token_hex(12)}",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model") or "slime-actor",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": in_tok, "output_tokens": 0},
                },
            },
        )
    )
    for idx, b in enumerate(blocks):
        bt = b["type"]
        start = (
            {"type": "thinking", "thinking": ""}
            if bt == "thinking"
            else (
                {"type": "text", "text": ""}
                if bt == "text"
                else {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}
            )
        )
        delta = (
            {"type": "thinking_delta", "thinking": b["thinking"]}
            if bt == "thinking"
            else (
                {"type": "text_delta", "text": b["text"]}
                if bt == "text"
                else {"type": "input_json_delta", "partial_json": json.dumps(b["input"], ensure_ascii=False)}
            )
        )
        await out.write(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": start,
                },
            )
        )
        await out.write(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": delta,
                },
            )
        )
        await out.write(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))
    await out.write(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            },
        )
    )
    await out.write(_sse("message_stop", {"type": "message_stop"}))
    return out


async def _handle_messages(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = _extract_session_id(request)
    if not sid:
        return _err_response(
            "missing_session",
            "Authorization Bearer <session_id> required",
            400,
        )

    store: Store = request.app["store"]
    s = await store.get(sid)

    try:
        blocks, stop_reason, in_tok, out_tok = await _run_turn(s, body, request.app)
    except _UpstreamError as e:
        return _err_response("upstream_error", str(e), 502)

    return await _stream_anthropic_sse(
        request,
        body,
        blocks,
        in_tok,
        out_tok,
        stop_reason,
    )


async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


# =============================================================================
# 10. Entry: MiddlewareHandle + start()
# =============================================================================


@dataclasses.dataclass
class MiddlewareHandle:
    app_handle: AppHandle
    store: Store
    public_host: str

    @property
    def public_url(self) -> str:
        return f"http://{self.public_host}:{self.app_handle.port}"

    def open_session(
        self,
        session_id: str,
        *,
        sampling_defaults: dict[str, Any] | None = None,
    ) -> None:
        """Register a new session. session_id must be globally unique for the
        middleware's lifetime; raises ValueError on duplicate."""
        self.store.open_session(session_id, defaults=sampling_defaults or {})

    async def _pop_session_split(self, session_id: str) -> list[tuple[list[int], list[int], list[int], dict],]:
        s = await self.store.pop(session_id)
        if s is None:
            return []
        async with s.lock:
            return s.pop_split()

    def pop_session_split(self, session_id: str) -> list[tuple[list[int], list[int], list[int], dict],]:
        """Return chronological segments (segment_kind in
        {"subagent", "wipe", "final"}). Session popped."""
        fut = asyncio.run_coroutine_threadsafe(
            self._pop_session_split(session_id),
            self.app_handle.loop,
        )
        return fut.result(timeout=10)

    def stop(self) -> None:
        self.app_handle.stop()


def start(
    *,
    tokenizer,
    sglang_url: str,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    host: str = "0.0.0.0",
    port: int = 0,
    public_host: str | None = None,
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
        _ASSISTANT_MARKER_TEXT,
        add_special_tokens=False,
    )
    app.router.add_post("/v1/messages", _handle_messages)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    handle = run_app_in_thread(app, host=host, port=port, thread_name="anthropic-middleware")
    logger.info(
        "[coding_agent_rl.middleware] %s -> %s (tool=%s reasoning=%s)",
        handle.url,
        sglang_url,
        tool_parser,
        reasoning_parser,
    )
    return MiddlewareHandle(app_handle=handle, store=store, public_host=public_host or host)
