"""middleware.py - anthropic shim, sglang engine, TITO mask, 3-kind segment split.

Single-file by design (per 0522 user decision M1: no sub-module split).
~1330 lines, 9 sections, each <= 250 lines. Run `grep -n "^# =====" middleware.py`
to list section banners as TOC.

Table of Contents
=================
  §0  MODULE DOC + TOC                  (top of file)
  §1  IMPORTS + CONSTANTS
  §2  DATACLASSES
       Turn, SubSession, SubSnapshot, Session, Store
  §3  PRIMITIVES
       _hash_obj, _strip_cache_control, _common_prefix_len, _sse, _err
  §4  STORE
       Store, _attach_pending_raw_tokens (P3)
  §5  TRANSLATE
       _translate_messages, _tools_schema, _render_with_raw_splice (P1),
       _parse_output, _detect_gen_prefix, verify_tito_for_turn (NEW, D2)
  §6  ENGINE
       _AbortCoordinator (P2), _post_generate, _generate_with_abort_resume (P2)
  §7  SEGMENTS
       pick_target, classify_and_apply, snapshot_subagent, pop_session_split
  §8  HANDLER
       _handle_messages (15-step orchestration), block builder, partial recorder
  §9  SHELL + ENTRY
       MiddlewareHandle, start(...), HTTP health handlers

Preserved Protocols (any PR touching these requires high-priority review):
  P1  _render_with_raw_splice (§5)            - chat-template drift killer
  P2  _AbortCoordinator + abort/resume (§6)   - async pause/resume semantics
  P3  pending_raw_tokens FIFO (§4)            - anti-double-append
                                                (lose it -> 75% loss_mask=0)
  P4  _hash_obj + _strip_cache_control (§3)   - system_hash stability
  P5  _EMPTY_THINK_STUB_TEXT + marker (§1+§5) - empty-think absorption
  P6  Session.lock single-acquisition (§8)    - handler is ONLY lock site

Invariants (see SPEC §2.3):
  I1  len(target.response_ids) == len(target.loss_mask) always.
  I2  target.prompt_ids is set on first turn, only reset on pre_wipe snapshot.
  I3  Session.lock is the ONLY sync primitive; §4-§7 helpers never take it.
  I4  pending_raw_tokens FIFO: 1 push per /generate, 1 pop per next-turn ingest.
  I5  pick_target -> classify_and_apply order is fixed; never swap.
  I6  Session._emit_order is the single source of truth for segment order.
  I7  TITO mask only mutates target.loss_mask slice; nothing else.
  I8  TITO check requires n = len(output_ids) > 0 (or [-0:] = [:] clears all).
  I9  TITO skipped while turn.aborted_prefix_len > 0 or num_aborts_this_turn > 0.
"""

from __future__ import annotations

# =============================================================================
# §1  IMPORTS + CONSTANTS
# =============================================================================

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import secrets
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web

# U5 (SPEC §10.3): aiohttp_threaded lives next to middleware.py (not in
# slime/utils/). Relative import works in both ray-worker mode
# (examples.coding_agent_rl.middleware) and smoke-test mode (pytest run from
# the worktree root with examples/ as a real package).
from .aiohttp_threaded import AppHandle, run_app_in_thread

logger = logging.getLogger(__name__)

# P5 - empty-think stub absorption text. Qwen3 reasoning chat templates auto-
# inject this BEFORE the assistant `content` field for completed assistants
# that lack a `reasoning_content` entry. The raw-splice render in §5 must
# swallow it; otherwise the spliced output would be doubled into
# `<think>\n\n</think>\n\n<think>\n...` and prefix-match across turns breaks.
_EMPTY_THINK_STUB_TEXT = "<think>\n\n</think>\n\n"

# P5 - assistant generation-prompt marker. Used by _detect_gen_prefix to find
# the chat template's gen-prefix tokens (e.g. `<think>\n` for reasoning models)
# so we can stitch them onto the front of raw output ids in pending_raw_tokens.
_ASSISTANT_MARKER_TEXT = "<|im_start|>assistant\n"

# P1 - raw-splice placeholder bracket. Surrounding \x07 (BEL) chars keep BPE
# boundaries clean (BEL is rare in normal text so the tokenizer encodes it
# identically regardless of surrounding context).
_RAW_PLACEHOLDER_PREFIX = "\x07RAWSPLICE_"
_RAW_PLACEHOLDER_SUFFIX = "_END\x07"

# Tool names claude-code uses to dispatch into a subagent. A matching
# tool_result on the main line marks the subagent's return (see §7).
_SUBAGENT_TOOL_NAMES = frozenset({"Task", "Agent"})

# Max number of /generate retries within one logical turn during abort/resume.
MAX_RESUME_ATTEMPTS = int(os.environ.get("SWE_ABORT_RESUME_MAX_ATTEMPTS", "8"))
# Minimum max_new_tokens we will request on a retry (avoid 0-budget POSTs).
_BUDGET_FLOOR = int(os.environ.get("SWE_ABORT_RESUME_MIN_TOKENS", "16"))


# =============================================================================
# §2  DATACLASSES - Turn, SubSession, SubSnapshot, Session
# =============================================================================


@dataclasses.dataclass
class Turn:
    """Per-call viz record only. 0521 legacy fields removed (branch_kind,
    parent_id, parent_prefix_len, full_ids); see SPEC §1.4."""

    id: int
    input_len: int = 0
    output_len: int = 0
    finish_reason: str = "unknown"
    stop_reason: str = "unknown"
    aborted_prefix_len: int = 0  # bumped on each abort partial within this turn
    tito_masked: bool = False    # U3: True iff TITO check failed this turn
    # Only populated when record_raw_dump=True; verbose JSON of request/response
    # for offline viz / debug.
    request: dict = dataclasses.field(default_factory=dict, repr=False)
    response: dict = dataclasses.field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "tito_masked": self.tito_masked,
        }
        if self.aborted_prefix_len:
            d["aborted_prefix_len"] = self.aborted_prefix_len
        if self.request:
            d["request"] = self.request
        if self.response:
            d["response"] = self.response
        return d


@dataclasses.dataclass
class SubSession:
    """Live sub-agent state. Owns its own splice/pending state because the
    sub system_prompt != main; their cumulative prefixes are different."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)  # I1
    initial_prompt_len: int = 0
    asst_raw_tokens: dict[int, tuple[list[int], int]] = dataclasses.field(default_factory=dict)
    pending_raw_tokens: list[tuple[list[int], int]] = dataclasses.field(default_factory=list)  # P3
    dispatch_tool_use_id: str = ""  # main-line Task tool_use_id that armed this sub
    # per-sub accounting (U3)
    num_aborts: int = 0
    last_finish_reason: str = ""
    tito_masked_turn_count: int = 0


@dataclasses.dataclass
class SubSnapshot:
    """Frozen completed dispatch - flat append-only entry on
    Session.completed_subagents (and Session._emit_order)."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    seen_msgs: int
    dispatch_tool_use_id: str
    finish_reason: str = ""           # U3
    num_aborts: int = 0               # U3
    tito_masked_turn_count: int = 0   # U3


@dataclasses.dataclass
class Session:
    # === main-line chat state ===
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    system_hash: str = ""
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)        # I1
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    record_raw_dump: bool = False
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)  # P6 (I3)
    initial_prompt_len: int = 0
    asst_raw_tokens: dict[int, tuple[list[int], int]] = dataclasses.field(default_factory=dict)
    pending_raw_tokens: list[tuple[list[int], int]] = dataclasses.field(default_factory=list)  # P3, I4
    last_finish_reason: str = ""

    # === viz turn log ===
    turns: list[Turn] = dataclasses.field(default_factory=list)

    # === segment containers (flat: no nested stack) ===
    active_subagent: "SubSession | None" = None
    pending_dispatch_id: str = ""
    # Projections of _emit_order; kept for raw_dump convenience. Single source
    # of truth is _emit_order (I6); helpers must append to BOTH.
    completed_subagents: list[SubSnapshot] = dataclasses.field(default_factory=list)
    pre_wipes: list[tuple[list[int], list[int], list[int], dict]] = dataclasses.field(default_factory=list)
    _emit_order: list[tuple[str, object]] = dataclasses.field(default_factory=list)  # I6

    # === TITO + abort accounting (U2, U3) ===
    num_aborts: int = 0                  # cross-turn count for this session
    num_aborts_this_turn: int = 0        # U2: TITO skip signal; reset at start of each turn
    tito_masked_turn_count: int = 0
    tito_total_turn_count: int = 0


# =============================================================================
# §3  PRIMITIVES - hash, prefix, sse, error helpers
# =============================================================================


def _strip_cache_control(obj: Any) -> Any:
    """P4 - drop Anthropic prompt-caching ``cache_control`` keys.

    The cache_control breakpoint migrates across turns (claude-code re-pins it
    to the most recent few blocks), so the *same* logical message has different
    hashes between consecutive requests. Stripping cache_control before hashing
    keeps msg_hashes stable across turns."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _hash_obj(obj: Any) -> str:
    """P4 - stable short hash of any JSON-compatible object."""
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True,
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _request_snapshot(body: dict[str, Any]) -> dict[str, Any]:
    keep = ("model", "system", "messages", "tools", "max_tokens", "stop_sequences", "stream")
    return {k: body[k] for k in keep if k in body}


def _extract_session_id(request: web.Request) -> str:
    return (request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            or request.headers.get("x-session-id", ""))


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _err(code: str, msg: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": code, "message": msg}}


# =============================================================================
# §4  STORE - Session container + pending-raw-tokens FIFO attach (P3)
# =============================================================================


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

    def open_session(self, sid: str, *, defaults: dict[str, Any], record_raw_dump: bool) -> None:
        """Reentrant: re-opening an existing sid resets sampling_defaults +
        record_raw_dump but preserves no other state. To start fresh, call
        pop() first."""
        s = self._d.setdefault(sid, Session())
        s.sampling_defaults = dict(defaults or {})
        s.record_raw_dump = record_raw_dump


def _attach_pending_raw_tokens(
    target: "Session | SubSession",
    base_index: int,
    new_translated_msgs: list[dict],
) -> None:
    """P3 - pop entries from target.pending_raw_tokens onto asst_raw_tokens.

    Pairs with the §8 step-13 push (1 push per /generate, 1 pop per next-turn
    ingest; see I4 invariant). If pending FIFO is empty for an echoed
    assistant (resumed mid-stream session), the splice falls back to
    chat_template for THAT message only; other assistants still benefit."""
    for offset, m in enumerate(new_translated_msgs):
        if m.get("role") != "assistant":
            continue
        if not target.pending_raw_tokens:
            continue
        target.asst_raw_tokens[base_index + offset] = target.pending_raw_tokens.pop(0)


# =============================================================================
# §5  TRANSLATE - Anthropic <-> chat-template, raw-splice (P1), TITO verify
# =============================================================================


def _flatten(c: Any) -> str:
    """Anthropic content blocks -> a single text blob (drop images, etc.)."""
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
    """Anthropic blocks -> chat-template messages (system/user/assistant/tool).

    Thinking blocks are dropped from input; the middleware re-injects them via
    reasoning_content after parsing /generate output (so the next round's
    template re-render produces tokens matching what the model actually
    emitted)."""
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
    """Anthropic tool defs -> the dict shape HF chat templates expect under
    ``tools=``."""
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


def _detect_gen_prefix(ideal_ids: list[int], marker_ids: list[int]) -> list[int]:
    """P5 - return tokens AFTER the LAST occurrence of ``marker_ids`` in
    ``ideal_ids``. For Qwen3-reasoning this is typically ``<think>\\n``
    (2 tokens). For non-reasoning models it may be empty."""
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
    *,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """P1 - render chat_messages but splice in stored raw token sequences for
    any assistant entry whose index is in ``asst_raw_tokens``.

    Returns ``(ideal_ids, raw_ranges)``. Each entry of raw_ranges is
    ``(splice_start, gen_start, splice_end)`` in ideal_ids coords;
    [splice_start:gen_start] is template-injected gen prefix (loss_mask=0),
    [gen_start:splice_end] is model-generated (loss_mask=1).

    Trick: for each spliced assistant we substitute its content with a unique
    placeholder string before rendering. Tokenize once, locate each
    placeholder's token range (preferring offset_mapping; fallback to segment
    encoding), and substitute the stored raw tokens. Template prefix/suffix
    (``<|im_start|>assistant\\n`` and ``<|im_end|>``) stays intact -- only the
    assistant body comes from raw tokens."""
    valid = {i: tup for i, tup in asst_raw_tokens.items() if 0 <= i < len(chat_messages)}
    if not valid:
        text = tok.apply_chat_template(
            chat_messages, tools=tools_schema, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tok.encode(text, add_special_tokens=False), []

    placeholders: dict[int, str] = {}
    render_msgs: list[dict] = []
    for i, m in enumerate(chat_messages):
        if i in valid:
            ph = f"{_RAW_PLACEHOLDER_PREFIX}{i}_{secrets.token_hex(6)}{_RAW_PLACEHOLDER_SUFFIX}"
            placeholders[i] = ph
            render_msgs.append({"role": "assistant", "content": ph})
        else:
            render_msgs.append(m)

    text = tok.apply_chat_template(
        render_msgs, tools=tools_schema, tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        template_ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        use_offsets = True
    except (TypeError, ValueError, NotImplementedError):
        use_offsets = False
        template_ids = tok.encode(text, add_special_tokens=False)
        offsets = []

    if use_offsets:
        stub_ids = tok.encode(_EMPTY_THINK_STUB_TEXT, add_special_tokens=False)
        if hasattr(stub_ids, "ids"):
            stub_ids = list(stub_ids.ids)
        else:
            stub_ids = list(stub_ids)

        placeholder_ranges: list[tuple[int, int, int]] = []
        for asst_idx, ph in placeholders.items():
            char_start = text.find(ph)
            if char_start < 0:
                logger.warning("[middleware] raw-splice: placeholder for asst %d not found", asst_idx)
                continue
            char_end = char_start + len(ph)
            tok_start: int | None = None
            tok_end: int | None = None
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
            n_stub = len(stub_ids)
            if n_stub > 0 and tok_start >= n_stub \
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

    # ---- fallback: encode the text in segments around placeholders ----
    if _EMPTY_THINK_STUB_TEXT:
        for ph in placeholders.values():
            text = text.replace(_EMPTY_THINK_STUB_TEXT + ph, ph)
    char_ranges: list[tuple[int, int, int]] = []
    for asst_idx, ph in placeholders.items():
        cs = text.find(ph)
        if cs < 0:
            continue
        char_ranges.append((cs, cs + len(ph), asst_idx))
    char_ranges.sort()

    ideal_ids = []
    raw_ranges = []
    cursor_char = 0
    for cs, ce, asst_idx in char_ranges:
        if cs > cursor_char:
            ideal_ids.extend(tok.encode(text[cursor_char:cs], add_special_tokens=False))
        rs = len(ideal_ids)
        full_raw, gen_off = valid[asst_idx]
        ideal_ids.extend(full_raw)
        re_ = len(ideal_ids)
        raw_ranges.append((rs, rs + gen_off, re_))
        cursor_char = ce
    if cursor_char < len(text):
        ideal_ids.extend(tok.encode(text[cursor_char:], add_special_tokens=False))
    return ideal_ids, raw_ranges


def _build_loss_mask_from_raw_ranges(
    response_len: int,
    prompt_len: int,
    raw_ranges: list[tuple[int, int, int]],
) -> list[int]:
    """Build loss_mask aligned to response_ids (= ideal_ids[prompt_len:]).
    For each (splice_start, gen_start, splice_end), tokens in
    [gen_start:splice_end] get mask=1 (model-generated). Tokens in
    [splice_start:gen_start] (template-injected gen prefix) and tokens outside
    any range get mask=0 (observation)."""
    mask = [0] * response_len
    for splice_start, gen_start, splice_end in raw_ranges:
        a = max(0, gen_start - prompt_len)
        b = max(0, splice_end - prompt_len)
        b = min(b, response_len)
        for k in range(a, b):
            mask[k] = 1
    return mask


def verify_tito_for_turn(tok: Any, raw_text: str, output_ids: list[int]) -> bool:
    """D2/U2 - pure function. Retokenize raw_text; compare to output_ids.

    Returns False -> drift detected. **Caller MUST check `len(output_ids) > 0`
    before calling** (I8) and SHOULD skip when abort happened this turn (I9)."""
    retok = tok.encode(raw_text, add_special_tokens=False)
    # tokenizers Fast may return an Encoding object; ensure list-equality
    if hasattr(retok, "ids"):
        retok = list(retok.ids)
    return list(retok) == list(output_ids)


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
    """Parse Qwen's occasional Anthropic-style XML tool-call fallback."""
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
# §6  ENGINE - sglang call + abort/resume coordinator (P2)
# =============================================================================


class AbortCoordinator:
    """P2 - shared "weight update in progress" state for the middleware.

    Two wiring paths:
      * explicit  - caller invokes aborted_now / resumed_now at known points.
      * polled    - register a 0-arg should_abort_fn and we flip on changes.

    Handler logic only ever consults ``is_aborted`` / ``await
    wait_for_resume()``; both paths converge here."""

    def __init__(self) -> None:
        self._aborted: bool = False
        self._cleared = asyncio.Event()
        self._cleared.set()
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task | None = None
        self._should_abort_fn: Callable[[], bool] | None = None
        self._max_wait_sec: float = 0.0  # 0 = wait forever
        self.on_aborted: Callable[[], None] | None = None
        self.on_resumed: Callable[[], None] | None = None

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    async def aborted_now(self) -> None:
        async with self._lock:
            if self._aborted:
                return
            self._aborted = True
            self._cleared.clear()
        if self.on_aborted is not None:
            try:
                self.on_aborted()
            except Exception as e:
                logger.warning("[middleware] on_aborted callback failed: %s", e)

    async def resumed_now(self) -> None:
        async with self._lock:
            if not self._aborted:
                return
            self._aborted = False
            self._cleared.set()
        if self.on_resumed is not None:
            try:
                self.on_resumed()
            except Exception as e:
                logger.warning("[middleware] on_resumed callback failed: %s", e)

    async def wait_for_resume(self, max_wait_sec: float | None = None) -> bool:
        """Block until aborted state clears. Returns True if cleared in time."""
        if not self._aborted:
            return True
        budget = max_wait_sec if max_wait_sec is not None else self._max_wait_sec
        if budget <= 0:
            await self._cleared.wait()
            return True
        try:
            await asyncio.wait_for(self._cleared.wait(), timeout=budget)
            return True
        except asyncio.TimeoutError:
            return False

    def install_poll(
        self,
        should_abort_fn: Callable[[], bool],
        *,
        interval_sec: float = 0.5,
        max_wait_sec: float = 1800.0,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._should_abort_fn = should_abort_fn
        self._max_wait_sec = max_wait_sec

        async def _poll() -> None:
            prev = False
            while True:
                try:
                    cur = bool(self._should_abort_fn())
                except Exception as e:
                    logger.warning("[middleware] should_abort_fn raised: %s; aborting poll", e)
                    return
                if cur and not prev:
                    await self.aborted_now()
                elif not cur and prev:
                    await self.resumed_now()
                prev = cur
                await asyncio.sleep(interval_sec)

        async def _spawn() -> None:
            self._poll_task = asyncio.create_task(_poll())

        asyncio.run_coroutine_threadsafe(_spawn(), loop)


async def _post_generate(
    sglang_url: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    """Single non-streaming POST. Returns parsed JSON body."""
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout) as sess, \
               sess.post(f"{sglang_url}/generate", json={
                   "input_ids": input_ids, "sampling_params": sampling_params,
                   "return_logprob": True,
               }) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
        return await r.json()


async def generate_with_abort_resume(
    *,
    sglang_url: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
    abort: AbortCoordinator,
    on_partial: Callable[[list[int], str], Awaitable[None]] | None = None,
) -> tuple[list[int], str, dict[str, Any]]:
    """P2 - run /generate and transparently retry across abort/resume cycles.

    Returns ``(full_output_ids, finish_reason, meta_info_of_last_call)``.

    Loops at most ``MAX_RESUME_ATTEMPTS`` times. On an aborted upstream call:
      1. record the partial output via ``on_partial`` if provided
      2. wait for the abort coordinator to clear
      3. re-issue with input + accumulated, max_new_tokens -= accumulated

    The returned output_ids is the **full concatenation** across all
    abort/resume retries within this turn (TITO masks the merged segment as
    one logical turn per D2)."""
    accumulated: list[int] = []
    last_finish = "unknown"
    last_meta: dict[str, Any] = {}

    attempt = 0
    while attempt < MAX_RESUME_ATTEMPTS:
        if abort.is_aborted:
            await abort.wait_for_resume()

        cur_input = input_ids + accumulated
        budget = int(sampling_params.get("max_new_tokens", 4096)) - len(accumulated)
        if budget <= 0:
            last_finish = "length"
            break
        sp = dict(sampling_params)
        sp["max_new_tokens"] = max(budget, _BUDGET_FLOOR)

        try:
            data = await _post_generate(sglang_url, cur_input, sp)
        except Exception as e:
            if abort.is_aborted:
                logger.info("[middleware] /generate raised during abort: %s; will retry post-resume", e)
                await abort.wait_for_resume()
                attempt += 1
                continue
            raise

        meta = data.get("meta_info") or {}
        ids = [x[1] for x in (meta.get("output_token_logprobs") or [])]
        accumulated.extend(ids)
        last_meta = meta
        last_finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"

        if last_finish == "abort":
            if on_partial is not None:
                try:
                    await on_partial(ids, last_finish)
                except Exception as e:
                    logger.warning("[middleware] on_partial failed: %s", e)
            await abort.wait_for_resume()
            attempt += 1
            continue
        break
    else:
        logger.warning("[middleware] hit MAX_RESUME_ATTEMPTS=%d; returning accumulated",
                       MAX_RESUME_ATTEMPTS)

    return accumulated, last_finish, last_meta


# =============================================================================
# §7  SEGMENTS - pick_target, classify_and_apply, snapshot, pop_session_split
# =============================================================================


def find_dispatch_tool_use_id(blocks: list[dict]) -> str:
    """Return the tool_use id of the most recent Task/Agent block in
    ``blocks``, or "" if none. Used on the response to mark the next request's
    expected subagent dispatch."""
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use" and b.get("name") in _SUBAGENT_TOOL_NAMES:
            return b.get("id", "") or ""
    return ""


def has_tool_result_for(messages: list[dict], tool_use_id: str) -> bool:
    """True when any user message contains a tool_result block whose
    tool_use_id matches ``tool_use_id``. Used on the main line to detect that
    an outstanding Task/Agent dispatch has returned."""
    if not tool_use_id:
        return False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result" \
                    and b.get("tool_use_id") == tool_use_id:
                return True
    return False


def maybe_pop_subagent(session: Session, all_msgs: list[dict]) -> None:
    """If session has a pending_dispatch_id AND an active subagent AND a
    tool_result for that id is in all_msgs -> snapshot the active subagent
    and pop. Called BEFORE pick_target in handler step 2."""
    if not session.pending_dispatch_id or session.active_subagent is None:
        return
    if has_tool_result_for(all_msgs, session.pending_dispatch_id):
        snapshot_subagent(session, session.active_subagent)
        session.active_subagent = None
        session.pending_dispatch_id = ""


def pick_target(session: Session, msgs: list[dict], req_system_hash: str
                ) -> tuple["Session | SubSession", bool]:
    """Routing only (no append to chat_messages).

    Returns ``(target, target_is_sub)``.

    Logic:
      * no active sub -> (session, False)
      * req_system_hash matches active sub's system_hash -> (active, True)
      * else NESTED-DISPATCH FAIL-SAFE (R2 CC3):
          snapshot the outer sub, drop active_subagent, return main.
          The new "inner" sub will be initialized on the next request's
          pick_target/classify_and_apply pair (treated as a fresh sub)."""
    if session.active_subagent is None:
        return session, False
    if req_system_hash == session.active_subagent.system_hash:
        return session.active_subagent, True
    logger.warning(
        "[middleware] nested sub-agent dispatch detected; snapshotting outer + opening inner "
        "(outer system_hash=%s, inner=%s)",
        session.active_subagent.system_hash[:8] if session.active_subagent.system_hash else "",
        req_system_hash[:8] if req_system_hash else "",
    )
    snapshot_subagent(session, session.active_subagent)
    session.active_subagent = None
    return session, False


def classify_and_apply(
    target: "Session | SubSession",
    session: Session,
    *,
    req_system_hash: str,
    msg_hashes: list[str],
) -> tuple[str | None, bool]:
    """Returns ``(kind, is_append)``. kind in {None, "pre_wipe"}.

    Side effect: if non-linear AND target had response_ids, snapshots the
    current chain to session.pre_wipes (projection) AND session._emit_order
    (I6 source of truth)."""
    is_append = (
        req_system_hash == target.system_hash
        and len(msg_hashes) >= target.seen_msgs
        and msg_hashes[:target.seen_msgs] == target.msg_hashes[:target.seen_msgs]
    )
    if target.seen_msgs == 0 or is_append:
        return None, is_append
    if target.response_ids:
        meta = {
            "kind": "pre_wipe",
            "on_subagent": isinstance(target, SubSession),
            "completed_turns": len(session.turns),
            # U3 per-segment fields
            "finish_reason": getattr(target, "last_finish_reason", ""),
            "num_aborts": getattr(target, "num_aborts", 0),
            "tito_masked_turns": getattr(target, "tito_masked_turn_count", 0),
        }
        snap = (list(target.prompt_ids), list(target.response_ids),
                list(target.loss_mask), meta)
        session.pre_wipes.append(snap)             # projection
        session._emit_order.append(("pre_wipe", snap))  # I6 source of truth
    return "pre_wipe", False


def snapshot_subagent(session: Session, sub: SubSession) -> None:
    """Freeze sub into a SubSnapshot. Append to completed_subagents
    (projection) AND _emit_order (I6 source of truth). Includes U3
    per-segment fields."""
    snap = SubSnapshot(
        prompt_ids=list(sub.prompt_ids),
        response_ids=list(sub.response_ids),
        loss_mask=list(sub.loss_mask),
        seen_msgs=sub.seen_msgs,
        dispatch_tool_use_id=sub.dispatch_tool_use_id,
        finish_reason=sub.last_finish_reason,
        num_aborts=sub.num_aborts,
        tito_masked_turn_count=sub.tito_masked_turn_count,
    )
    session.completed_subagents.append(snap)        # projection
    session._emit_order.append(("subagent", snap))  # I6


def pop_session_split(session: Session) -> tuple[
    list[tuple[list[int], list[int], list[int], dict]],
    dict,
]:
    """Drain any in-flight active_subagent (claude_code exited mid-dispatch),
    replay _emit_order in chronological order, append final segment.

    Per Q6: if any TITO turn was masked, emit ONE summary logger.warning
    with the per-sample mask rate.

    Returns ``(segments, raw_dump)``."""
    if session.tito_masked_turn_count > 0:
        pct = 100.0 * session.tito_masked_turn_count / max(1, session.tito_total_turn_count)
        logger.warning(
            "[middleware] TITO mask rate %.2f%% (%d/%d turns masked in this sample)",
            pct, session.tito_masked_turn_count, session.tito_total_turn_count,
        )

    if session.active_subagent is not None:
        snapshot_subagent(session, session.active_subagent)
        session.active_subagent = None

    segments: list[tuple[list[int], list[int], list[int], dict]] = []
    for kind, payload in session._emit_order:  # I6 chronological replay
        if kind == "subagent":
            snap: SubSnapshot = payload  # type: ignore[assignment]
            if snap.response_ids:
                segments.append((
                    list(snap.prompt_ids), list(snap.response_ids), list(snap.loss_mask),
                    {"segment_kind": "subagent",           # U3 rename from "kind"
                     "completed_turns": snap.seen_msgs,
                     "finish_reason": snap.finish_reason,
                     "num_aborts": snap.num_aborts,
                     "tito_masked_turns": snap.tito_masked_turn_count},
                ))
        else:  # "pre_wipe"
            p, r, m, meta = payload  # type: ignore[assignment]
            if r:
                m2 = dict(meta)
                m2["segment_kind"] = m2.pop("kind", "pre_wipe")  # U3 rename
                segments.append((list(p), list(r), list(m), m2))

    if session.response_ids:
        segments.append((
            list(session.prompt_ids), list(session.response_ids), list(session.loss_mask),
            {"segment_kind": "final",
             "completed_turns": len(session.turns),
             "finish_reason": session.last_finish_reason,
             "num_aborts": session.num_aborts,
             "tito_masked_turns": session.tito_masked_turn_count,
             "tito_total_turns": session.tito_total_turn_count},
        ))
    return segments, _export_raw_dump(session)


def _export_raw_dump(session: Session) -> dict[str, Any]:
    """Build the trajectory_raw_dump (SPEC §6.2 v4 schema). Only populated
    fields are written; record_raw_dump=False sessions still get a slim dump
    (no per-turn request/response payloads since Turn.request/response are
    empty in that case)."""
    return {
        "version": 4,
        "num_turns": len(session.turns),
        "num_aborts": session.num_aborts,
        "tito_masked_turns": session.tito_masked_turn_count,
        "tito_total_turns": session.tito_total_turn_count,
        "initial_prompt_len": session.initial_prompt_len,
        "turns": [t.to_dict() for t in session.turns],
        "completed_subagents": [
            {"seen_msgs": s.seen_msgs,
             "response_len_total": len(s.response_ids),
             "dispatch_tool_use_id": s.dispatch_tool_use_id,
             "finish_reason": s.finish_reason,
             "num_aborts": s.num_aborts,
             "tito_masked_turn_count": s.tito_masked_turn_count}
            for s in session.completed_subagents
        ],
        "pre_wipes": [
            {"prompt_len": len(p), "response_len": len(r),
             "on_subagent": meta.get("on_subagent", False),
             "finish_reason": meta.get("finish_reason", ""),
             "num_aborts": meta.get("num_aborts", 0),
             "tito_masked_turns": meta.get("tito_masked_turns", 0)}
            for (p, r, _m, meta) in session.pre_wipes
        ],
    }


# =============================================================================
# §8  HANDLER - _handle_messages (15-step orchestration)
# =============================================================================


def _resolve_sampling_params(s: Session, body: dict[str, Any],
                             target: "Session | SubSession") -> dict[str, Any]:
    sp = dict(s.sampling_defaults or {})
    for k_a, k_s in [("max_tokens", "max_new_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("top_k", "top_k")]:
        if k_a in body:
            sp[k_s] = body[k_a]
    if body.get("stop_sequences"):
        sp["stop"] = body["stop_sequences"]
    sp.setdefault("max_new_tokens", 4096)
    sp.setdefault("skip_special_tokens", False)
    sp.setdefault("spaces_between_special_tokens", False)
    sp.setdefault("no_stop_trim", True)

    max_response_tokens = int(os.environ.get("SWE_MAX_RESPONSE_TOKENS", "0") or 0)
    # Cap counts only model-generated tokens (loss_mask==1).
    remaining = max_response_tokens - sum(target.loss_mask) if max_response_tokens > 0 else None
    if remaining is not None and remaining > 0:
        sp["max_new_tokens"] = min(int(sp["max_new_tokens"]), remaining)
    sp["_remaining"] = remaining  # caller uses this to short-circuit when <= 0
    return sp


def _update_prompt_and_response(
    target: "Session | SubSession",
    ideal_ids: list[int],
    raw_ranges: list[tuple[int, int, int]],
    is_append: bool,
    kind: str | None,
) -> None:
    """Step 7 - 4 paths:
      (a) first turn (initial_prompt_len == 0)
      (b) linear append (is_append=True)
      (c) post-wipe first turn (kind == "pre_wipe")
      (d) divergence (ideal prefix mismatches) -> log + treat as (a)
    """
    # Path (a)/(c): first turn or post-wipe -> set prompt_ids = ideal_ids
    if not target.prompt_ids or kind == "pre_wipe":
        target.prompt_ids = ideal_ids
        if target.initial_prompt_len == 0:
            target.initial_prompt_len = len(ideal_ids)
        target.response_ids = []
        target.loss_mask = []
        return

    # Path (d): divergence detection. ideal_ids should start with target.prompt_ids.
    if len(ideal_ids) < len(target.prompt_ids) \
            or ideal_ids[:len(target.prompt_ids)] != target.prompt_ids:
        logger.info("[middleware] divergence reset; dropping previous segment "
                    "(ideal=%d prompt=%d)", len(ideal_ids), len(target.prompt_ids))
        target.prompt_ids = ideal_ids
        target.response_ids = []
        target.loss_mask = []
        return

    # Path (b): linear append - extend cumulative response_ids matching ideal prefix.
    target.response_ids = ideal_ids[len(target.prompt_ids):]
    target.loss_mask = _build_loss_mask_from_raw_ranges(
        len(target.response_ids), len(target.prompt_ids), raw_ranges,
    )


def _maybe_new_turn(s: Session, ideal_ids: list[int], body: dict[str, Any],
                    target_is_sub: bool) -> "Turn | None":
    """Append a fresh Turn record to s.turns. Returns it so caller can fill
    output_len/finish_reason/etc later. Always appends (viz log is cheap);
    `request`/`response` payloads only filled when record_raw_dump=True."""
    turn = Turn(id=len(s.turns), input_len=len(ideal_ids))
    if s.record_raw_dump:
        turn.request = _request_snapshot(body)
    s.turns.append(turn)
    return turn


def _build_anthropic_blocks(thinking: str, visible: str, tool_uses: list[dict]) -> list[dict]:
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if visible:
        blocks.append({"type": "text", "text": visible})
    for tu in tool_uses:
        blocks.append({"type": "tool_use", "id": f"toolu_{secrets.token_hex(8)}",
                       "name": tu["name"], "input": tu["input"]})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


async def _emit_anthropic_response(
    request: web.Request, body: dict[str, Any], blocks: list[dict],
    in_tokens: int, out_tokens: int, stop_reason: str,
) -> web.StreamResponse:
    streaming = bool(body.get("stream", False))
    model = body.get("model") or "slime-actor"
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
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    }))
    await out.write(_sse("message_stop", {"type": "message_stop"}))
    return out


async def _handle_messages(request: web.Request) -> web.StreamResponse:
    A = request.app
    tok = A["tokenizer"]
    store: Store = A["store"]
    abort: AbortCoordinator = A["abort"]

    body = await request.json()
    sid = _extract_session_id(request)
    if not sid:
        return web.json_response(_err("missing_session",
                                      "Authorization Bearer <session_id> required"), status=400)
    s = await store.get(sid)

    async with s.lock:  # P6 / I3: ONLY lock acquisition site
        # ---- 1. extract request fingerprints ----
        all_msgs = body.get("messages") or []
        msg_hashes = [_hash_obj(m) for m in all_msgs]
        system_value = body.get("system") if "system" in body else None
        req_system_hash = _hash_obj(system_value) if "system" in body else s.system_hash

        # ---- 2. close any returning sub-agent BEFORE routing ----
        maybe_pop_subagent(s, all_msgs)

        # ---- 3. pick_target (handles nested dispatch fail-safe) ----
        target, target_is_sub = pick_target(s, all_msgs, req_system_hash)

        # ---- 4. classify_and_apply (may snapshot pre_wipe) ----
        kind, is_append = classify_and_apply(
            target, s, req_system_hash=req_system_hash, msg_hashes=msg_hashes,
        )

        # ---- 5. ingest new messages + attach pending raw tokens (P3 / I4) ----
        if target.seen_msgs == 0 or kind == "pre_wipe":
            target.chat_messages = _translate_messages(all_msgs, body.get("system"))
            target.system_hash = req_system_hash
            target.asst_raw_tokens.clear()
            target.pending_raw_tokens.clear()
            _attach_pending_raw_tokens(target, 0, target.chat_messages)
        elif is_append:
            new = _translate_messages(all_msgs[target.seen_msgs:], None)
            base_idx = len(target.chat_messages)
            target.chat_messages.extend(new)
            _attach_pending_raw_tokens(target, base_idx, new)
        else:
            # Should not reach: classify_and_apply returns kind="pre_wipe"
            # on non-linear non-empty -> branch above. Defensive fallthrough.
            target.chat_messages = _translate_messages(all_msgs, body.get("system"))
            target.system_hash = req_system_hash
            target.asst_raw_tokens.clear()
            target.pending_raw_tokens.clear()
            _attach_pending_raw_tokens(target, 0, target.chat_messages)

        target.seen_msgs = len(all_msgs)
        target.msg_hashes = msg_hashes
        if target.tools_schema is None:
            target.tools_schema = _tools_schema(body.get("tools"))

        # ---- 6. render with raw splice ----
        ideal_ids, raw_ranges = _render_with_raw_splice(
            tok, target.chat_messages, target.tools_schema, target.asst_raw_tokens,
            add_generation_prompt=True,
        )

        # ---- 7. update prompt vs response (4 paths) ----
        _update_prompt_and_response(target, ideal_ids, raw_ranges, is_append, kind)

        # ---- 8. record turn (viz log) + reset per-turn abort counter ----
        s.num_aborts_this_turn = 0  # U2: reset before this turn's abort accounting
        turn = _maybe_new_turn(s, ideal_ids, body, target_is_sub)

        # ---- 9. sampling params ----
        sp = _resolve_sampling_params(s, body, target)
        remaining = sp.pop("_remaining", None)

        # ---- 10. /generate with abort/resume (P2) ----
        async def _record_partial(ids: list[int], _why: str) -> None:
            s.num_aborts += 1
            s.num_aborts_this_turn += 1
            if isinstance(target, SubSession):
                target.num_aborts += 1
            if turn is not None:
                turn.aborted_prefix_len += len(ids)

        if remaining is not None and remaining <= 0:
            output_ids: list[int] = []
            finish = "length"
            meta: dict[str, Any] = {}
        else:
            try:
                output_ids, finish, meta = await generate_with_abort_resume(
                    sglang_url=A["sglang_url"], input_ids=ideal_ids,
                    sampling_params=sp, abort=abort,
                    on_partial=_record_partial,
                )
            except aiohttp.ClientError as e:
                return web.json_response(_err("upstream_unreachable", str(e)), status=502)
            except Exception as e:
                return web.json_response(_err("upstream_error", str(e)), status=502)

        # ---- 11. extend response_ids + loss_mask (I1) ----
        target.response_ids.extend(output_ids)
        target.loss_mask.extend([1] * len(output_ids))
        target.last_finish_reason = finish
        if isinstance(target, Session):
            s.last_finish_reason = finish

        # ---- 12. TITO verify per-turn (D2/U2) ----
        # I8 guard (n>0) + I9 abort skip + per-turn mask (only this turn).
        n = len(output_ids)
        if n > 0 and s.num_aborts_this_turn == 0:
            raw_text_for_tito = tok.decode(output_ids, skip_special_tokens=False)
            s.tito_total_turn_count += 1
            if not verify_tito_for_turn(tok, raw_text_for_tito, output_ids):
                # I7: only mutate target.loss_mask slice; nothing else.
                target.loss_mask[-n:] = [0] * n
                s.tito_masked_turn_count += 1
                if isinstance(target, SubSession):
                    target.tito_masked_turn_count += 1
                if turn is not None:
                    turn.tito_masked = True
        # else: empty turn (I8) or aborted turn (I9) -> skip TITO

        # ---- 13. parse + queue raw tokens for next-round attach (P3 push, I4) ----
        raw_output = tok.decode(output_ids, skip_special_tokens=False) if n > 0 else ""
        thinking, visible, tool_uses = _parse_output(
            raw_output,
            tool_parser_name=A["tool_parser"], reasoning_parser_name=A["reasoning_parser"],
            tools_schema=target.tools_schema,
        )
        marker_ids = A.get("_assistant_marker_ids")
        if marker_ids is None:
            marker_ids = tok.encode(_ASSISTANT_MARKER_TEXT, add_special_tokens=False)
            A["_assistant_marker_ids"] = marker_ids
        gen_prefix = _detect_gen_prefix(ideal_ids, marker_ids)
        target.pending_raw_tokens.append((gen_prefix + list(output_ids), len(gen_prefix)))

        # ---- 14. build blocks + arm dispatch marker ----
        blocks = _build_anthropic_blocks(thinking, visible, tool_uses)
        dispatch_id = find_dispatch_tool_use_id(blocks)
        if dispatch_id:
            if target_is_sub:
                target.dispatch_tool_use_id = dispatch_id  # rare nested case
            else:
                # If main was about to dispatch a sub and no SubSession is
                # active yet, this Task tool_use arms the next request to
                # create one (classify_and_apply will route on system_hash).
                s.pending_dispatch_id = dispatch_id
                # active_subagent stays None; the NEXT request whose
                # req_system_hash differs from main triggers SubSession init
                # (see _handle_messages step 5 + pick_target nested-fallback).
                # We pre-create the SubSession HERE so pick_target can route
                # the next request based on system_hash equality.
                if s.active_subagent is None:
                    s.active_subagent = SubSession(dispatch_tool_use_id=dispatch_id)

        stop_reason = "tool_use" if tool_uses else (
            "max_tokens" if finish == "length" else "end_turn"
        )
        in_tokens = len(ideal_ids)
        out_tokens = len(output_ids)

        if turn is not None:
            turn.output_len = out_tokens
            turn.finish_reason = finish
            turn.stop_reason = stop_reason
            if s.record_raw_dump:
                turn.response = {
                    "raw_output": raw_output,
                    "thinking": thinking,
                    "visible": visible,
                    "tool_uses": tool_uses,
                }

    # ---- 15. emit Anthropic response (OUTSIDE lock for SSE streaming) ----
    return await _emit_anthropic_response(request, body, blocks, in_tokens, out_tokens, stop_reason)


# =============================================================================
# §9  SHELL + ENTRY - MiddlewareHandle, start(...), HTTP health handlers
# =============================================================================


async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


@dataclasses.dataclass
class MiddlewareHandle:
    app_handle: AppHandle
    store: Store
    abort: AbortCoordinator
    public_host: str

    @property
    def public_url(self) -> str:
        return f"http://{self.public_host}:{self.app_handle.port}"

    def open_session(
        self,
        session_id: str,
        *,
        sampling_defaults: dict[str, Any] | None = None,
        record_raw_dump: bool = False,
    ) -> None:
        """Reentrant: re-opening an existing sid resets sampling_defaults +
        record_raw_dump but preserves no other state. To start fresh, call
        pop_session_split() first."""
        self.store.open_session(
            session_id,
            defaults=sampling_defaults or {},
            record_raw_dump=record_raw_dump,
        )

    async def _pop_session_split(self, session_id: str) -> tuple[
        list[tuple[list[int], list[int], list[int], dict]],
        dict,
    ]:
        s = await self.store.pop(session_id)
        if s is None:
            return [], {}
        async with s.lock:
            return pop_session_split(s)

    def pop_session_split(self, session_id: str) -> tuple[
        list[tuple[list[int], list[int], list[int], dict]],
        dict,
    ]:
        """list_trajectory mode (D4 default): returns one (prompt_ids,
        response_ids, loss_mask, meta) tuple per chain segment.

        Side effects:
          1. pops session from store (subsequent gets create a fresh Session);
          2. drains any in-flight active_subagent (claude_code exited
             mid-dispatch -> emit subagent segment);
          3. emits ONE logger.warning per sample if any TITO mask occurred
             (Q6 decision).

        Segments are emitted in chronological order with `segment_kind` in
        {"pre_wipe", "subagent", "final"}. Empty-response segments are
        dropped. The raw_dump (SPEC §6.2) is returned alongside so the caller
        can attach it to ONE of the fanned-out Samples."""
        fut = asyncio.run_coroutine_threadsafe(
            self._pop_session_split(session_id), self.app_handle.loop,
        )
        return fut.result(timeout=10)

    # --- Weight-update abort/resume control ---------------------------------
    def aborted_now(self) -> None:
        """Tell the middleware: stop accepting new generate calls; mark
        in-flight ones as ABORTED. Idempotent; safe to call from any thread."""
        asyncio.run_coroutine_threadsafe(
            self.abort.aborted_now(), self.app_handle.loop,
        ).result(timeout=5)

    def resumed_now(self) -> None:
        """Tell the middleware: weights are reloaded; re-issue aborted
        /generate calls with reduced max_new_tokens and let new requests through."""
        asyncio.run_coroutine_threadsafe(
            self.abort.resumed_now(), self.app_handle.loop,
        ).result(timeout=5)

    def install_abort_poll(
        self,
        should_abort_fn: Callable[[], bool],
        *,
        interval_sec: float = 0.5,
        max_wait_sec: float = 1800.0,
    ) -> None:
        """Wire ``should_abort_fn`` to be polled inside the middleware loop."""
        self.abort.install_poll(
            should_abort_fn,
            interval_sec=interval_sec,
            max_wait_sec=max_wait_sec,
            loop=self.app_handle.loop,
        )

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
    abort = AbortCoordinator()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["tokenizer"] = tokenizer
    app["sglang_url"] = sglang_url.rstrip("/")
    app["tool_parser"] = tool_parser
    app["reasoning_parser"] = reasoning_parser
    app["store"] = store
    app["abort"] = abort
    app.router.add_post("/v1/messages", _handle_messages)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    handle = run_app_in_thread(app, host=host, port=port, thread_name="anthropic-middleware")
    logger.info("[coding_agent_rl.middleware] %s -> %s (tool=%s reasoning=%s)",
                handle.url, sglang_url, tool_parser, reasoning_parser)
    return MiddlewareHandle(app_handle=handle, store=store, abort=abort,
                            public_host=public_host or host)
