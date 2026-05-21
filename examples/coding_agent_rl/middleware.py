"""Anthropic Messages API <-> SGLang /generate middleware.

Translates Claude Code's Anthropic Messages API into slime's SGLang ``/generate``
(token-native + logprobs) and captures the actual tokens per session_id so the
trainer never re-tokenizes (no TITO mismatch).

Model-agnostic: chat formatting via ``tokenizer.apply_chat_template``; tool-call
parsing via ``sglang.srt.function_call.FunctionCallParser``; reasoning parsing
via ``sglang.srt.parser.reasoning_parser.ReasoningParser``.

---

What this file owns:

* **session token store** — keeps the canonical chat log per session_id
  (Anthropic Bearer token), re-renders the chat template each turn, diffs
  against cumulative ids so observation tokens get loss_mask=0 and model-
  generated tokens get loss_mask=1.

* **tree-shaped trajectory recording** — Claude Code is *not* a linear agent.
  Sub-agents and compaction both rewrite the message list so that the next
  request's tokens share only a *prefix* with one of the earlier turns. We
  detect this by computing the longest-common-prefix of the new request
  against every recorded turn's ``full_ids`` (= input tokens + output tokens
  the model emitted). The matching turn becomes the parent; the prefix len
  is the branch point. A linear conversation produces a chain (parent_i =
  i-1, prefix grows monotonically); a sub-agent fork produces a sibling
  rooted at the parent turn; a compact produces a NEW root whose parent
  prefix len is small.

* **weight-update abort/resume** — at training time slime calls a global
  ``abort()`` that POSTs ``/abort_request`` to every sglang worker and flips
  ``GenerateState.aborted`` to True. SGLang then returns partial output and
  ``finish_reason.type == "abort"``. We:

  1. record the partial output tokens as a turn (so they're not lost),
  2. wait until the abort flag clears (slime resets it before the next
     rollout, *after* parameters have been pushed to sglang),
  3. re-submit ``/generate`` with the previously-rendered prompt extended
     by the partial output, with ``max_new_tokens`` reduced by the partial
     length.

  Net effect for Claude Code: a single ``/v1/messages`` request can take
  several minutes to complete (it spans a weight update) but it always
  returns a "complete" Anthropic response built from one or more underlying
  sglang turns. The caller never sees a half-message.

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
import hashlib
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web

from slime.utils.aiohttp_threaded import AppHandle, run_app_in_thread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session token store
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _Turn:
    """One round-trip with SGLang.

    ``full_ids`` is the input tokens (the chat-template-rendered prompt for
    this turn) plus the output tokens the model emitted. It's what the NEXT
    turn's prefix-match is computed against; that's why a chain of linear
    turns produces strictly increasing parent_prefix_len.
    """

    id: int
    parent_id: int | None
    parent_prefix_len: int
    input_len: int = 0
    output_len: int = 0
    finish_reason: str = "unknown"
    stop_reason: str = "unknown"
    branch_kind: str = "root"  # "root" | "linear" | "sibling" | "compact"
    request: dict[str, Any] = dataclasses.field(default_factory=dict)
    response: dict[str, Any] = dataclasses.field(default_factory=dict)
    full_ids: list[int] = dataclasses.field(default_factory=list, repr=False)
    # When > 0, this turn was the second half of an abort/resume pair (i.e.
    # the model produced ``output_len`` tokens *after* the resume, and
    # ``aborted_prefix_len`` tokens before it).
    aborted_prefix_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "parent_id": self.parent_id,
            "parent_prefix_len": self.parent_prefix_len,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "branch_kind": self.branch_kind,
            "request": self.request,
            "response": self.response,
        }
        if self.aborted_prefix_len:
            d["aborted_prefix_len"] = self.aborted_prefix_len
        return d


@dataclasses.dataclass
class _SubSession:
    """Independent prefix-tracking state for one subagent dispatch.

    Owns its own prompt_ids / response_ids / loss_mask / msg_hashes /
    seen_msgs / chat_messages / system_hash / initial_prompt_len /
    asst_raw_tokens / pending_raw_tokens. The main session pushes one
    onto ``subagent_stack`` when a Task/Agent dispatch is detected and
    pops it (snapshotting into ``completed_trajectories`` with
    ``kind="subagent"``) when the dispatch returns.

    Per user decision 1 (MASTER_PLAN preamble): the subagent segment's
    prompt prefix is the subagent's own system prompt + initial task
    only, never the main-line prefix from before the dispatch."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)
    initial_prompt_len: int = 0
    asst_raw_tokens: dict[int, tuple[list[int], int]] = dataclasses.field(default_factory=dict)
    pending_raw_tokens: list[tuple[list[int], int]] = dataclasses.field(default_factory=list)
    dispatch_tool_use_id: str = ""  # the tool_use id we are waiting on
    nested_depth: int = 1


@dataclasses.dataclass
class _Session:
    # Canonical chat log. Assistant entries get added by _translate_messages
    # when claude-code echoes our previous generation back in its next request
    # (Anthropic API is stateless; claude-code resends the full conversation
    # each turn). We do NOT append to chat_messages ourselves immediately after
    # /generate -- doing so + claude-code's echo on the next turn would double
    # the assistant message, which previously caused chronic template-rerender
    # mismatch + rebaseline (zeroing loss_mask).
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    system_hash: str = ""
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)
    turns: list[_Turn] = dataclasses.field(default_factory=list)
    sampling_defaults: dict[str, Any] = dataclasses.field(default_factory=dict)
    record_tree: bool = False
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    num_aborts: int = 0

    # === Raw-token preservation (RL on-policy training fix) =================
    # asst_raw_tokens: chat_messages index -> (full_raw_ids, gen_offset).
    # ``full_raw_ids`` = gen_prefix (template-injected, e.g. ``<think>\n``)
    # followed by output_ids (what the model actually emitted). ``gen_offset``
    # = len(gen_prefix); tokens at [gen_offset:] in the raw segment are the
    # ones we want to count as model-generated (loss_mask=1). Splicing the
    # full sequence (prefix + output) makes the rendered ideal_ids byte-
    # identical to what the model originally conditioned on + emitted, so
    # the next-turn cumulative-prefix check always matches.
    asst_raw_tokens: dict[int, tuple[list[int], int]] = dataclasses.field(default_factory=dict)
    # FIFO of (full_raw_ids, gen_offset) tuples awaiting attachment to
    # chat_messages indices. Pushed after each /generate; popped when
    # _translate_messages encounters the corresponding assistant message in
    # claude-code's echo on the next request.
    pending_raw_tokens: list[tuple[list[int], int]] = dataclasses.field(default_factory=list)
    # Cached size of the initial prompt (after first request's render). Used
    # by _classify_branch to recognize autoCompact wipes: when the new request
    # only shares ~initial_prompt_len tokens with its parent, it's a wipe even
    # if the parent itself was already short (post-compact early turn).
    initial_prompt_len: int = 0

    # === list_trajectory: snapshot every chain segment before wiping =========
    # Each entry: (prompt_ids, response_ids, loss_mask, meta). When the chain
    # rebases (autoCompact wipe or sibling fork landing on the same session),
    # we snapshot the previous chain here so generate.py can fan it out into
    # one trainable Sample per segment instead of throwing the pre-wipe tokens
    # away. The current (live) chain is NOT in this list -- it's still in
    # s.prompt_ids/response_ids/loss_mask; pop_session_split() appends it
    # before returning.
    completed_trajectories: list[tuple[list[int], list[int], list[int], dict]] = dataclasses.field(default_factory=list)

    # === subagent dispatch tracking (list_trajectory mode) ===================
    # Stack of currently-active subagent dispatches. The top of stack receives
    # all requests whose body["system"] hashes to its system_hash. When the
    # subagent's tool_result returns on the main line, we pop and snapshot the
    # top into completed_trajectories with kind="subagent". Per user decision 3
    # (MASTER_PLAN preamble): nested subagents (depth >= 2) are merged into
    # the outer sub-session at pop time; only depth-1 pops emit a separate
    # segment.
    subagent_stack: list[_SubSession] = dataclasses.field(default_factory=list)
    # Tool-use id of the most recent Task/Agent dispatch we saw on whichever
    # routing target is "current" (main or top of stack). The next request
    # whose new user messages contain a tool_result with this id triggers the
    # pop. Cleared after pop or when a new dispatch arrives.
    _pending_dispatch_id: str = ""
    # finish_reason of the most recent /generate call on the main line. Used
    # by pop_session_split to annotate the final segment metadata so we can
    # tell task-done from max-turns / length truncation post-hoc.
    last_finish_reason: str = ""


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

    def open_session(self, sid: str, *, defaults: dict[str, Any], record_tree: bool) -> None:
        s = self._s.setdefault(sid, _Session())
        s.sampling_defaults = dict(defaults or {})
        s.record_tree = record_tree


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _strip_cache_control(obj: Any) -> Any:
    # Anthropic prompt-caching attaches "cache_control": {"type": "ephemeral"}
    # to whichever message/system block currently holds the cache breakpoint.
    # The breakpoint migrates across turns (claude-code re-pins it to the most
    # recent few blocks), so the *same* logical message has different hashes
    # in consecutive requests. That spurious mismatch flips is_append=False and
    # forces a full chat_messages rebuild + token-level rebaseline, which then
    # shows up in the trajectory tree as a fake "sibling" branch.
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _hash_obj(obj: Any) -> str:
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True,
                         ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _request_snapshot(body: dict[str, Any]) -> dict[str, Any]:
    keep = ("model", "system", "messages", "tools", "max_tokens", "stop_sequences", "stream")
    return {k: body[k] for k in keep if k in body}


# Threshold (in tokens) for treating a small parent_prefix_len as "essentially
# only the initial system+first-user prompt + autoCompact pack" -> classify as
# compact. claude-code's post-compact m[0] is a list of: 6-ish (read tool
# input/result) blocks + currentDate + a 9-10KB "This session is being
# continued..." summary, which renders to 5-10K tokens beyond the system
# prompt. 16K gives comfortable headroom for the largest packs we have seen.
_COMPACT_PFX_EPS = 16384

# Token tolerance for treating ``parent_prefix_len`` as "essentially equal to"
# the parent's full_ids length. After raw-splice render landed, linear turns
# match exactly (pfx == par.full); pre-splice dumps still drift by a handful
# to a few hundred tokens of chat_template re-render noise. 2048 covers every
# Cat-A drift observed in the 0521 archive while staying well below typical
# sub-agent fork divergence (which is on the order of par.output tokens).
# Also reused by _new_turn's pass-1 neighbor-first parent selection.
LINEAR_DEFICIT_TOK = 2048

# Phrases claude-code uses in its autoCompact "please summarize this
# conversation" request. When both appear in the *last* user message we mark
# the turn as compact_summarization -- it's a one-off summarization call whose
# response goes into a synthetic <system-reminder>, not the trajectory itself.
_SUMMARIZATION_MARKERS = (
    "create a detailed summary",
    "Respond with TEXT ONLY",
)

# Marker claude-code injects as the prefix of the autoCompact "resume" summary
# in messages[0] after it wipes the conversation. Detecting it directly lets
# us classify the new chain as `compact` without relying on token-length
# thresholds that vary by Read-pack size.
_COMPACT_RESUME_MARKER = "This session is being continued from a previous conversation"


def _last_user_text(messages: list[dict]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if last.get("role") != "user":
        return ""
    content = last.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _is_summarization_request(body: dict[str, Any]) -> bool:
    msgs = body.get("messages") or []
    # Main path: claude-code places the summarize prompt as the last user msg.
    text = _last_user_text(msgs)
    if all(m in text for m in _SUMMARIZATION_MARKERS):
        return True
    # Fallback: scan non-last user msgs. Rare cases (observed in flaky compact
    # retries) where claude-code attaches the markers to a non-last user msg.
    # Skip msgs that contain the autoCompact resume marker -- those are the
    # embedded resume pack carrying a prior conversation's summary text, NOT
    # an active summarization request.
    for m in msgs[:-1]:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        body_text = _flatten(m.get("content"))
        if _COMPACT_RESUME_MARKER in body_text:
            continue
        if all(mk in body_text for mk in _SUMMARIZATION_MARKERS):
            return True
    return False


def _is_compact_resume_request(body: dict[str, Any]) -> bool:
    """True when any user message of the request contains claude-code's
    autoCompact "resume from summary" marker. claude-code places the marker in
    a text block right after the conversation gets wiped; depending on whether
    the wipe lands before or after a Read tool call, the marker shows up either
    as a text block of messages[0] *or* of messages[1] (the latter when m[0] is
    a bare `<system-reminder>Called the Read tool ...` and m[1] is the
    corresponding tool result + currentDate + summary pack)."""
    msgs = body.get("messages") or []
    # Only the first couple of user messages can carry the autoCompact pack —
    # claude-code emits the pack inline at the head of the new conversation.
    for m in msgs[:2]:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            if _COMPACT_RESUME_MARKER in content:
                return True
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    if _COMPACT_RESUME_MARKER in b.get("text", ""):
                        return True
    return False


def _classify_branch(
    parent: _Turn | None,
    parent_prefix_len: int,
    parent_full_len: int,
    *,
    initial_prompt_len: int = 0,
    is_summarization: bool = False,
    is_compact_resume: bool = False,
) -> str:
    """Categorize how this turn relates to its parent.

    * ``root`` — no parent (first turn or completely disjoint).
    * ``linear`` — extends the parent's full_ids exactly (the common case for
      a normal assistant turn followed by tool_result followed by next user).
    * ``compact_summarization`` — the request body is itself an autoCompact
      summarization call (last user message contains the standard claude-code
      summarize prompt). The response isn't part of the agent's action stream.
    * ``compact`` — parent_prefix_len is much shorter than the parent's full
      input (either <= initial_prompt_len + ε, meaning autoCompact wiped to
      just the system prompt; or < parent.input_len/2 for older cases).
    * ``sibling`` — shares a proper prefix with the parent but diverges
      before the parent's output ended. Pre-fix this category catches a lot
      of false positives caused by chat_template re-render drift; with the
      raw-token splice in _render_with_raw_splice those false positives go
      away and true siblings only arise from genuine sub-agent forks.
    """
    if parent is None:
        return "root"
    if is_summarization:
        return "compact_summarization"
    # Check the linear-extend path BEFORE the autoCompact marker check: after
    # claude-code wipes the conversation, the resume pack stays embedded in
    # every subsequent request's messages[0..1] for the rest of the chain.
    # Without this ordering, every linear continuation on the post-compact
    # chain would be mis-flagged as `compact` purely because the marker is
    # still floating around in the prefix.
    if parent_prefix_len >= parent_full_len:
        return "linear"
    # [C3] Near-linear tolerance: chat_template re-render drift (Cat A) leaves
    # pfx a few hundred tokens to ~2K short of parent_full_len on otherwise-
    # linear continuations. The auxiliary guard `pfx > parent.input_len`
    # ensures the new turn consumed at least one token of parent's output
    # (real sub-agent forks share only the input prefix and consume zero
    # output, so they fail this guard).
    #
    # NOTE: spec_1 §3.4 originally proposed a stricter `pfx > parent.input +
    # parent.output/2` guard, but that rejects observed-linear cases like
    # sib 14 (pfx=46738, par.in=46401, par.out=1475 → consumed only 337/1475
    # of par.output). Empirically the loose guard is sufficient to separate
    # Cat A drift from sub-agent forks because forks share exactly par.input
    # (or less). Verified by smoke case 13 (true fork, pfx=par.in+1000,
    # par.out=10000 → deficit=9000 > LINEAR_DEFICIT_TOK rejects via deficit).
    if (parent_full_len - parent_prefix_len) <= LINEAR_DEFICIT_TOK \
            and parent_prefix_len > parent.input_len:
        return "linear"
    # [C4] Root-parent special case: when initial_prompt_len is 0 (legacy
    # dumps without C1 serialization) and the parent IS root, treat any pfx
    # that nearly fills the root's full_ids as a compact wipe provided the
    # resume marker is present. This covers sample-11 sib 21 / 32 / 51 / 72
    # where init_pl was lost but parent==root and marker is in m[0..1].
    if parent.id == 0 and parent_prefix_len + LINEAR_DEFICIT_TOK >= parent_full_len and is_compact_resume:
        return "compact"
    # m[0]/m[1] marker is the ground-truth signal that claude-code started a
    # new chain from autoCompact — authoritative for picking compact over
    # sibling when the token geometry is ambiguous (pfx well above
    # initial_prompt_len + _COMPACT_PFX_EPS, e.g. when the resume pack itself
    # is 20K+ tokens of inlined Read results).
    if is_compact_resume:
        return "compact"
    # autoCompact wipes the chat history down to the initial system prompt
    # plus a Read-pack + summary that renders to 5-10K tokens. The shared
    # prefix collapses to ~initial_prompt_len + that pack size, even when the
    # parent itself is short (e.g. a post-compact early turn at ~30K).
    if initial_prompt_len > 0 and parent_prefix_len <= initial_prompt_len + _COMPACT_PFX_EPS:
        return "compact"
    if parent_prefix_len * 2 < parent.input_len:
        return "compact"
    return "sibling"


def _new_turn(s: _Session, ideal_ids: list[int], body: dict[str, Any]) -> _Turn:
    parent_id = None
    parent_prefix_len = 0
    parent_turn: _Turn | None = None
    # [C5] Pass 1: neighbor-first. Walk turns in REVERSE order and pick the
    # most recent prev whose prefix-match is close to its full_ids length
    # (linear continuation) AND that consumed at least part of prev.input
    # (guards against picking a forked sibling whose full_ids happen to
    # overlap the new request in just the input portion). This avoids the
    # longest-prefix-match flattening artifact where multiple linear turns
    # downstream of a compact-init node all reparent to that node.
    for prev in reversed(s.turns):
        prefix_len = _common_prefix_len(prev.full_ids, ideal_ids)
        if prefix_len >= len(prev.full_ids) - LINEAR_DEFICIT_TOK and prefix_len > prev.input_len:
            parent_id = prev.id
            parent_prefix_len = prefix_len
            parent_turn = prev
            break
    # Pass 2: fall back to the original longest-prefix-match if pass 1 found
    # no linear-neighbor candidate (true sub-agent fork or compact wipe).
    if parent_turn is None:
        for prev in s.turns:
            prefix_len = _common_prefix_len(prev.full_ids, ideal_ids)
            if prefix_len > parent_prefix_len:
                parent_id = prev.id
                parent_prefix_len = prefix_len
                parent_turn = prev

    branch_kind = _classify_branch(
        parent_turn,
        parent_prefix_len,
        len(parent_turn.full_ids) if parent_turn is not None else 0,
        initial_prompt_len=s.initial_prompt_len,
        is_summarization=_is_summarization_request(body),
        is_compact_resume=_is_compact_resume_request(body),
    )
    turn = _Turn(
        id=len(s.turns),
        parent_id=parent_id,
        parent_prefix_len=parent_prefix_len,
        input_len=len(ideal_ids),
        request=_request_snapshot(body),
        branch_kind=branch_kind,
    )
    s.turns.append(turn)
    return turn


def _maybe_new_turn(s: _Session, ideal_ids: list[int], body: dict[str, Any]) -> _Turn | None:
    if not s.record_tree:
        return None
    return _new_turn(s, ideal_ids, body)


def _export_tree(s: _Session) -> dict[str, Any]:
    return {
        "version": 3,
        "turns": [t.to_dict() for t in s.turns],
        "num_turns": len(s.turns),
        "prompt_tokens": len(s.prompt_ids),
        "response_tokens": len(s.response_ids),
        "loss_mask_tokens": len(s.loss_mask),
        "num_aborts": s.num_aborts,
        "initial_prompt_len": s.initial_prompt_len,
    }


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

    Thinking blocks are dropped from input; the middleware re-injects them via
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
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": _flatten(content)}]
            for b in blocks:
                if not isinstance(b, dict): continue
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
# Raw-token splice render (RL on-policy training fix)
# ---------------------------------------------------------------------------
# Unique placeholder marker used inside the chat_template render to mark
# where each spliced assistant's raw tokens should go. Surrounding `\x07`
# (BEL) chars keep BPE boundaries clean -- BEL is rare in normal text so
# the tokenizer encodes it consistently regardless of surrounding context.
_RAW_PLACEHOLDER_PREFIX = "\x07RAWSPLICE_"
_RAW_PLACEHOLDER_SUFFIX = "_END\x07"

# Empty-reasoning-stub text that Qwen3 reasoning chat templates auto-inject
# for completed assistant messages WITHOUT a `reasoning_content` field. The
# template emits this stub BEFORE the assistant `content`, so when our
# placeholder render is used and we want to splice the model's raw tokens
# (which already begin with the gen-prefix `<think>\n`) starting at the bare
# `<|im_start|>assistant\n` position, we must extend the splice region LEFT
# to swallow the stub. Otherwise we'd get `<think>\n\n</think>\n\n<think>\n...`
# double-think and the prefix-match across turns fails 95%+ of the time.
_EMPTY_THINK_STUB_TEXT = "<think>\n\n</think>\n\n"

# Token sequence for `<|im_start|>assistant\n` -- used to locate where the
# generation prompt tail begins in ideal_ids so we can capture the chat
# template's "gen prefix" (e.g. `<think>\n` for Qwen3 reasoning models)
# and prepend it to raw output tokens when storing into pending. Without
# this, replaying raw tokens via splice would skip the gen prefix that the
# model actually conditioned on, breaking prefix-match.
_ASSISTANT_MARKER_TEXT = "<|im_start|>assistant\n"


def _detect_gen_prefix(ideal_ids: list[int], marker_ids: list[int]) -> list[int]:
    """Return tokens AFTER the LAST occurrence of ``marker_ids`` in
    ``ideal_ids``. For Qwen3-reasoning this is typically ``<think>\\n`` (2
    tokens). For non-reasoning models it may be empty."""
    if not marker_ids:
        return []
    n = len(marker_ids)
    for start in range(len(ideal_ids) - n, -1, -1):
        if ideal_ids[start:start + n] == marker_ids:
            return list(ideal_ids[start + n:])
    return []


def _attach_pending_raw_tokens(
    s: "_Session | _SubSession",
    base_index: int,
    new_translated_msgs: list[dict],
) -> None:
    """Pop entries from s.pending_raw_tokens to attach raw tokens to each new
    assistant entry we just appended to s.chat_messages.

    Works on both main ``_Session`` and per-dispatch ``_SubSession``; both
    expose the same ``pending_raw_tokens`` / ``asst_raw_tokens`` fields.

    base_index is the chat_messages position of new_translated_msgs[0]; we
    walk in order and for each role==assistant entry, pop one (raw, gen_off)
    tuple from pending and store it under s.asst_raw_tokens[index].
    """
    for offset, m in enumerate(new_translated_msgs):
        if m.get("role") != "assistant":
            continue
        if not s.pending_raw_tokens:
            # claude-code echoed an assistant we didn't generate (resumed
            # session, mid-stream restart, etc.) -- can't attach raw tokens
            # for it, render will fall back to chat_template (may produce
            # Cat A drift for *that one* message but the other assistants
            # still benefit from splice).
            continue
        s.asst_raw_tokens[base_index + offset] = s.pending_raw_tokens.pop(0)


def _render_with_raw_splice(
    tok: Any,
    chat_messages: list[dict],
    tools_schema: list[dict] | None,
    asst_raw_tokens: dict[int, tuple[list[int], int]],
    *,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Render chat_messages but splice in stored raw token sequences for any
    assistant entry whose index is in asst_raw_tokens.

    Returns ``(ideal_ids, raw_ranges)``. Each entry of raw_ranges is
    ``(splice_start, gen_start, splice_end)`` in ideal_ids coords -- gen_start
    marks where the model's generated tokens begin within the splice (after
    the template's gen prefix like ``<think>\\n``). The portion
    [splice_start:gen_start] is template-injected and should get loss_mask=0;
    [gen_start:splice_end] is model-generated and gets loss_mask=1.

    The trick: for each spliced assistant we replace its content with a
    unique placeholder string before rendering. Then we tokenize the
    rendered text with offset_mapping, find each placeholder's token range,
    and substitute the stored raw tokens. The template prefix/suffix
    (`<|im_start|>assistant\\n` and `<|im_end|>`) stays intact -- only the
    assistant body comes from raw tokens.
    """
    valid = {i: tup for i, tup in asst_raw_tokens.items() if 0 <= i < len(chat_messages)}
    if not valid:
        text = tok.apply_chat_template(
            chat_messages, tools=tools_schema, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return tok.encode(text, add_special_tokens=False), []

    # Build a render-only copy with unique placeholder content for each
    # spliced assistant. Drop reasoning_content/tool_calls so the template
    # emits a bare assistant body containing just the placeholder.
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

    # Tokenize once. Prefer offset_mapping for robust token-range location;
    # fall back to segment-by-segment encoding if the tokenizer lacks fast-
    # tokenizer support.
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
        # Pre-compute the empty-think stub token sequence so we can extend
        # placeholder ranges left to swallow it. The stub is template-injected
        # for completed assistants without `reasoning_content`; if not absorbed
        # here the splice would land AFTER the stub, producing a doubled
        # `<think>...</think><think>...` and breaking prefix-match across turns.
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
            # Extend tok_start LEFT to swallow the auto-injected empty-think
            # stub if present immediately before the placeholder.
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
            re = len(ideal_ids)
            raw_ranges.append((rs, rs + gen_off, re))
            cursor = tok_end
        ideal_ids.extend(template_ids[cursor:])
        return ideal_ids, raw_ranges

    # ---- fallback: encode the text in segments around placeholders ----
    # First strip the auto-injected empty-think stub immediately before any
    # placeholder (same reason as the offset path above).
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
        re = len(ideal_ids)
        raw_ranges.append((rs, rs + gen_off, re))
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
    [splice_start:gen_start] (template-injected gen prefix) and tokens
    outside any range get mask=0 (observation)."""
    mask = [0] * response_len
    for splice_start, gen_start, splice_end in raw_ranges:
        a = max(0, gen_start - prompt_len)
        b = max(0, splice_end - prompt_len)
        b = min(b, response_len)
        for k in range(a, b):
            mask[k] = 1
    return mask


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
        if not thinking and "</think>" in body:
            thinking, body = body.split("</think>", 1)

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


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Abort/resume coordinator
# ---------------------------------------------------------------------------
class _AbortCoordinator:
    """Holds the shared "weight update in progress" state for the middleware.

    Two ways to wire it up:

    * **explicit** — caller invokes ``aborted_now()`` / ``resumed_now()`` from
      training code at known checkpoints.
    * **polled** — register a 0-arg ``should_abort_fn`` (e.g.
      ``lambda: GenerateState(args).aborted``) and call ``start_polling()``.
      A background coroutine flips state when the function value changes.

    The handler logic only ever consults ``is_aborted`` / ``await
    wait_for_resume()``; both wiring paths converge here.
    """

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
        """Wire ``should_abort_fn`` to be polled inside the middleware's event loop."""
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


# ---------------------------------------------------------------------------
# /generate call (with abort + resume around weight updates)
# ---------------------------------------------------------------------------
async def _post_generate(
    sglang_url: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    """Single non-streaming POST. Returns the parsed JSON body."""
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


async def _generate_with_abort_resume(
    *,
    sglang_url: str,
    input_ids: list[int],
    sampling_params: dict[str, Any],
    abort: _AbortCoordinator,
    on_partial: Callable[[list[int], str], Awaitable[None]] | None = None,
) -> tuple[list[int], str, dict[str, Any]]:
    """Run ``/generate`` and transparently retry across an abort/resume cycle.

    Returns ``(output_token_ids, finish_reason, meta_info_of_last_call)``.

    Loops at most ``MAX_RESUME_ATTEMPTS`` times. On an aborted upstream call:

    1. Record the partial output via ``on_partial`` if provided.
    2. Wait for the abort coordinator to clear.
    3. Re-issue ``/generate`` with prompt + accumulated partial, with
       ``max_new_tokens`` decremented by the partial length.

    Aborts that the middleware itself observes via the coordinator BEFORE
    issuing the upstream call get the same wait-then-retry treatment, so a
    request that arrives mid-update simply blocks until resume.
    """
    MAX_RESUME_ATTEMPTS = int(os.environ.get("SWE_ABORT_RESUME_MAX_ATTEMPTS", "8"))
    BUDGET_FLOOR = int(os.environ.get("SWE_ABORT_RESUME_MIN_TOKENS", "16"))

    accumulated: list[int] = []
    last_finish = "unknown"
    last_meta: dict[str, Any] = {}

    attempt = 0
    while attempt < MAX_RESUME_ATTEMPTS:
        # If the coordinator already says "aborted", wait it out before issuing.
        if abort.is_aborted:
            await abort.wait_for_resume()

        # Compose the input for this attempt.
        cur_input = input_ids + accumulated
        budget = int(sampling_params.get("max_new_tokens", 4096)) - len(accumulated)
        if budget <= 0:
            last_finish = "length"
            break
        sp = dict(sampling_params)
        sp["max_new_tokens"] = max(budget, BUDGET_FLOOR)

        try:
            data = await _post_generate(sglang_url, cur_input, sp)
        except Exception as e:
            # If we got interrupted *because* of an abort, fall through to the
            # resume path. Otherwise re-raise so the caller can decide.
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
            # sglang aborted this request — most likely because slime POSTed
            # /abort_request on a weight-update. Record what we have, wait
            # for the coordinator to clear (i.e. weights have been pushed
            # back and sglang is hot again), then go around.
            if on_partial is not None:
                try:
                    await on_partial(ids, last_finish)
                except Exception as e:
                    logger.warning("[middleware] on_partial failed: %s", e)
            await abort.wait_for_resume()
            attempt += 1
            continue

        # Normal termination (stop / length / matched stop_token / refusal).
        break
    else:
        logger.warning("[middleware] hit MAX_RESUME_ATTEMPTS=%d; returning whatever was accumulated",
                       MAX_RESUME_ATTEMPTS)

    return accumulated, last_finish, last_meta


# ---------------------------------------------------------------------------
# Subagent dispatch detection (list_trajectory mode)
# ---------------------------------------------------------------------------
# Tool names claude-code uses to dispatch into a subagent. The corresponding
# tool_result on the main line marks the subagent's return.
_SUBAGENT_TOOL_NAMES = ("Task", "Agent")


def _find_dispatch_tool_use_id(blocks: list[dict]) -> str:
    """Return the tool_use id of the most recent Task/Agent block in ``blocks``,
    or "" if none. Used on the main-line response to mark the next request's
    expected subagent dispatch."""
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use" and b.get("name") in _SUBAGENT_TOOL_NAMES:
            return b.get("id", "") or ""
    return ""


def _has_tool_result_for(messages: list[dict], tool_use_id: str) -> bool:
    """True when any of ``messages`` is a user message containing a
    ``tool_result`` block whose ``tool_use_id`` matches ``tool_use_id``.

    Used on the main line to detect that an outstanding Task/Agent dispatch
    has returned (subagent finished, control back to main)."""
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


def _snapshot_subagent(s: _Session, sub: _SubSession) -> None:
    """Append the sub-session as a kind="subagent" segment to s.completed_trajectories.

    Only called when popping the outermost sub-session (depth==1). Sub-sessions
    at deeper nesting are merged into their parent via _merge_into_parent."""
    if not sub.response_ids:
        return
    s.completed_trajectories.append((
        list(sub.prompt_ids),
        list(sub.response_ids),
        list(sub.loss_mask),
        {
            "kind": "subagent",
            "completed_turns": sub.seen_msgs,
            "nested_depth": sub.nested_depth,
        },
    ))


def _merge_into_parent(top: _SubSession, parent: _SubSession) -> None:
    """Merge a nested sub-session (depth >= 2) into its parent at pop time.

    Per user decision 3 (MASTER_PLAN preamble): nested subagents do NOT emit
    their own segment. We concatenate ``top.response_ids`` and
    ``top.loss_mask`` onto the parent's so the inner subagent's tokens stay
    trainable (with loss_mask=1 since they are still model-generated).

    We do NOT touch parent.prompt_ids — the inner subagent's tokens are
    technically observations from the parent's PoV, but accurate chat-
    template splicing would require reconstructing the parent's chain
    around the dispatch boundary, which we have already lost. Marking
    them loss_mask=1 keeps them in the training signal at the cost of a
    minor off-policy bias on tokens generated by the same model in the
    inner sub-context."""
    parent.response_ids.extend(top.response_ids)
    parent.loss_mask.extend([1] * len(top.response_ids))
    parent.nested_depth = max(parent.nested_depth, top.nested_depth)


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
    abort: _AbortCoordinator = A["abort"]

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
        # --- 0) routing: decide whether this request belongs on main line or
        # on top of subagent_stack. Per user decision 4 (MASTER_PLAN preamble):
        # hash body["system"] and compare against the main-line cached system
        # hash; differing hashes are subagent requests. Per user decision 5:
        # any unresolvable ambiguity falls back to main-line routing.
        all_msgs = body.get("messages") or []
        msg_hashes = [_hash_obj(m) for m in all_msgs]
        system_value = body.get("system") if "system" in body else None
        req_system_hash = _hash_obj(system_value) if "system" in body else s.system_hash

        # Check for subagent return on main line BEFORE deciding routing: if
        # the new user messages contain a tool_result for our pending
        # dispatch id, the subagent has returned. Pop top of stack and
        # snapshot it (or merge if nested), then route this request to main.
        if s._pending_dispatch_id and s.subagent_stack \
                and _has_tool_result_for(all_msgs, s._pending_dispatch_id):
            top = s.subagent_stack.pop()
            if s.subagent_stack:
                # nested case: merge into parent sub-session
                _merge_into_parent(top, s.subagent_stack[-1])
            else:
                # outermost subagent return: snapshot as its own segment
                _snapshot_subagent(s, top)
            s._pending_dispatch_id = ""

        # Now pick the routing target. Rules (in order):
        #   1. If subagent_stack is non-empty AND req_system_hash matches the
        #      top of stack → route to top sub-session.
        #   2. If subagent_stack is non-empty AND req_system_hash matches
        #      s.system_hash → subagent already returned (shouldn't normally
        #      happen because the tool_result check above handled it, but
        #      defensive); fall through to main.
        #   3. If s.system_hash is set AND req_system_hash differs AND we
        #      have a pending dispatch marker on the current target → push a
        #      new _SubSession (nested if stack non-empty).
        #   4. Fallback: route to main.
        target_is_sub = False
        target: Any = s

        if s.subagent_stack and req_system_hash == s.subagent_stack[-1].system_hash:
            target = s.subagent_stack[-1]
            target_is_sub = True
        elif s.system_hash and req_system_hash != s.system_hash \
                and req_system_hash != "" \
                and (s._pending_dispatch_id or (s.subagent_stack
                                                and s.subagent_stack[-1].dispatch_tool_use_id)):
            # New subagent dispatch: push a fresh sub-session onto the stack.
            parent_depth = s.subagent_stack[-1].nested_depth if s.subagent_stack else 0
            new_sub = _SubSession(
                system_hash=req_system_hash,
                dispatch_tool_use_id=s._pending_dispatch_id
                    if not s.subagent_stack else s.subagent_stack[-1].dispatch_tool_use_id,
                nested_depth=parent_depth + 1,
            )
            s.subagent_stack.append(new_sub)
            # Clear the pending marker on whoever held it; it's now consumed.
            s._pending_dispatch_id = ""
            target = new_sub
            target_is_sub = True

        # --- 1) ingest new messages (since last call) into the canonical log
        is_append = (
            req_system_hash == target.system_hash
            and len(msg_hashes) >= target.seen_msgs
            and msg_hashes[:target.seen_msgs] == target.msg_hashes[:target.seen_msgs]
        )
        wiped = False
        if target.seen_msgs == 0:
            new = _translate_messages(all_msgs, body.get("system"))
            base_idx = len(target.chat_messages)
            target.chat_messages.extend(new)
            _attach_pending_raw_tokens(target, base_idx, new)
            target.system_hash = req_system_hash
        elif is_append:
            new = _translate_messages(all_msgs[target.seen_msgs:], None)
            base_idx = len(target.chat_messages)
            target.chat_messages.extend(new)
            # Each new role==assistant in `new` is claude-code echoing one of our
            # generated turns back. Pop from pending_raw_tokens and attach so the
            # splice render replaces that assistant body with the model's
            # original raw tokens (preserving Megatron training signal).
            _attach_pending_raw_tokens(target, base_idx, new)
        else:
            # autoCompact wipe or other non-linear update. The previous chain
            # is about to be replaced -- list_trajectory mode snapshots it
            # (prompt/response/mask) into s.completed_trajectories so the
            # downstream Sample fan-out can train on it. tree_trajectory mode
            # simply drops it; either way target.asst_raw_tokens has to go
            # because its chat_messages indices no longer map to the rebuilt
            # chain. Subagent targets share the same snapshot container on s
            # so per-chain wipes inside a subagent also produce a pre_wipe
            # segment (matches user decision 2's whitelist of three kinds).
            if target.response_ids:
                s.completed_trajectories.append((
                    list(target.prompt_ids),
                    list(target.response_ids),
                    list(target.loss_mask),
                    {"kind": "pre_wipe",
                     "completed_turns": len(s.turns),
                     "on_subagent": target_is_sub},
                ))
            logger.info("[middleware] %s non-linear messages update; rebuilding prompt (sub=%s)",
                        session_id, target_is_sub)
            target.chat_messages = _translate_messages(all_msgs, body.get("system"))
            target.system_hash = req_system_hash
            target.asst_raw_tokens.clear()
            target.pending_raw_tokens.clear()
            # Re-attach any assistant in the rebuilt chain (rare: claude-code
            # might include an assistant from before the wipe). Pending is now
            # empty, so this is a no-op unless we added one above.
            _attach_pending_raw_tokens(target, 0, target.chat_messages)
            wiped = True
        target.seen_msgs = len(all_msgs)
        target.msg_hashes = msg_hashes
        if target.tools_schema is None:
            target.tools_schema = _tools_schema(body.get("tools"))

        # --- 2) render with raw-token splice; loss_mask=1 marks model-generated
        # tokens (from stored raw spans), loss_mask=0 marks template-rendered
        # observations. No more chat_template-rerender-mismatch rebaseline:
        # spliced raw tokens are byte-identical to what the model originally
        # emitted, so cumulative prefix always matches.
        ideal_ids, raw_ranges = _render_with_raw_splice(
            tok, target.chat_messages, target.tools_schema, target.asst_raw_tokens,
            add_generation_prompt=True,
        )
        # Tree recording only fires on the main session (subagent forks are
        # represented as their own trajectory in completed_trajectories, not
        # as tree branches).
        turn = _maybe_new_turn(s, ideal_ids, body) if not target_is_sub else None

        if not target.prompt_ids or wiped:
            target.prompt_ids = ideal_ids
            if target.initial_prompt_len == 0:
                # First-ever request on this target: cache initial prompt
                # length for _classify_branch (compact wipe detection later).
                target.initial_prompt_len = len(ideal_ids)
            target.response_ids = []
            target.loss_mask = []
        else:
            # ideal_ids should always start with target.prompt_ids (system +
            # first user prompt is stable across the session unless we hit
            # Cat B). If not, fall back to a clean reset. Per user decision
            # 2 (MASTER_PLAN preamble) we do NOT emit a diverge_reset
            # segment — the divergence is absorbed into the next pre_wipe
            # or final segment instead.
            if len(ideal_ids) < len(target.prompt_ids) or ideal_ids[:len(target.prompt_ids)] != target.prompt_ids:
                # Diagnostic: find first divergence and dump context once per session.
                diag = ""
                try:
                    if len(ideal_ids) < len(target.prompt_ids):
                        diag = f"length-short ideal={len(ideal_ids)} prompt={len(target.prompt_ids)}"
                    else:
                        for di in range(len(target.prompt_ids)):
                            if ideal_ids[di] != target.prompt_ids[di]:
                                a0, a1 = max(0, di - 8), di + 8
                                old_ctx = tok.decode(target.prompt_ids[a0:a1])
                                new_ctx = tok.decode(ideal_ids[a0:a1])
                                diag = (f"diverge@{di}/{len(target.prompt_ids)} "
                                        f"chat_msgs={len(target.chat_messages)} ranges={raw_ranges} "
                                        f"OLD={old_ctx!r} NEW={new_ctx!r}")
                                break
                except Exception as e:
                    diag = f"<diag err: {e}>"
                logger.info("[middleware] %s divergence reset; dropping previous segment [%s]",
                            session_id, diag)
                target.prompt_ids = ideal_ids
                target.response_ids = []
                target.loss_mask = []
            else:
                target.response_ids = ideal_ids[len(target.prompt_ids):]
                target.loss_mask = _build_loss_mask_from_raw_ranges(
                    len(target.response_ids), len(target.prompt_ids), raw_ranges,
                )

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

        max_response_tokens = int(os.environ.get("SWE_MAX_RESPONSE_TOKENS", "0") or 0)
        # Cap counts only model-generated tokens (loss_mask==1), not observation
        # echoes from tool results. Without this, large grep/read outputs eat
        # the budget and the chain short-circuits to length/0 mid-trajectory
        # even though the model itself has generated far fewer tokens.
        remaining = max_response_tokens - sum(target.loss_mask) if max_response_tokens > 0 else None
        if remaining is not None and remaining > 0:
            sp["max_new_tokens"] = min(int(sp["max_new_tokens"]), remaining)

        # --- 4) /generate (with abort/resume; non-streaming upstream) ---------
        # Counter shared with the on_partial callback so the final turn record
        # knows how many of the returned tokens came from the pre-abort
        # attempt. NOTE: the callback MUST NOT extend target.response_ids /
        # target.loss_mask itself — _generate_with_abort_resume already returns
        # the concatenated output across abort/resume cycles, and the extend
        # below adds it once.
        partial_count: list[int] = [0]

        async def _record_partial(ids: list[int], _why: str) -> None:
            partial_count[0] += len(ids)
            s.num_aborts += 1
            if turn is not None:
                turn.aborted_prefix_len += len(ids)

        if remaining is not None and remaining <= 0:
            output_ids: list[int] = []
            finish = "length"
            meta: dict[str, Any] = {}
        else:
            try:
                output_ids, finish, meta = await _generate_with_abort_resume(
                    sglang_url=sglang_url,
                    input_ids=ideal_ids,
                    sampling_params=sp,
                    abort=abort,
                    on_partial=_record_partial,
                )
            except aiohttp.ClientError as e:
                return web.json_response({"type": "error", "error": {
                    "type": "upstream_unreachable", "message": str(e),
                }}, status=502)
            except Exception as e:
                return web.json_response({"type": "error", "error": {
                    "type": "upstream_error", "message": str(e),
                }}, status=502)

        # --- 5) record output tokens; parse; queue raw tokens for next-round attach
        # output_ids is the *full* concatenation across any abort/resume retries
        # (see _generate_with_abort_resume contract).
        target.response_ids.extend(output_ids)
        target.loss_mask.extend([1] * len(output_ids))
        # Always store the finish_reason on the main session so the final
        # segment metadata can surface it (user spec §appendix A: distinguish
        # task done from max_turns / max_response_tokens truncation).
        s.last_finish_reason = finish

        raw_output = tok.decode(output_ids, skip_special_tokens=False)
        thinking, visible, tool_uses = _parse_output(
            raw_output,
            tool_parser_name=tool_parser, reasoning_parser_name=reasoning_parser,
            tools_schema=target.tools_schema,
        )

        # NOTE: we intentionally DO NOT append the structured assistant to
        # target.chat_messages here. Doing so + claude-code echoing the same
        # assistant in its next request would double the message in
        # chat_messages, causing chronic template-rerender mismatch +
        # rebaseline (which previously zeroed the entire loss_mask --
        # leaving 75% of trajectories with sum(loss_mask)=0).
        #
        # Instead we queue the raw token sequence into pending_raw_tokens.
        # On the next request, _attach_pending_raw_tokens will pop this and
        # attach it to the chat_messages index where _translate_messages
        # places claude-code's echo of this assistant. The splice in
        # _render_with_raw_splice then replaces that assistant's body with
        # the model's actual raw output tokens (no normalization, no
        # OpenHands -> qwen25 JSON re-rendering, no token drift).
        #
        # We prepend the gen prefix (e.g. `<think>\n` injected by the chat
        # template for Qwen3 reasoning models) so that what gets spliced
        # equals what the model originally conditioned on + emitted, byte
        # for byte. Without the prefix, prior-asst splice would skip the
        # `<think>\n` opener that the gen prompt added in this turn but
        # that the chat template does NOT auto-inject for completed asst
        # messages -- breaking cumulative prefix match.
        marker_ids = A.get("_assistant_marker_ids")
        if marker_ids is None:
            marker_ids = tok.encode(_ASSISTANT_MARKER_TEXT, add_special_tokens=False)
            A["_assistant_marker_ids"] = marker_ids
        gen_prefix = _detect_gen_prefix(ideal_ids, marker_ids)
        full_raw = gen_prefix + list(output_ids)
        target.pending_raw_tokens.append((full_raw, len(gen_prefix)))

        # --- 6) build Anthropic blocks (shared by JSON + SSE paths)
        blocks: list[dict] = []
        if thinking: blocks.append({"type": "thinking", "thinking": thinking})
        if visible: blocks.append({"type": "text", "text": visible})
        for tu in tool_uses:
            blocks.append({"type": "tool_use", "id": f"toolu_{secrets.token_hex(8)}",
                            "name": tu["name"], "input": tu["input"]})
        if not blocks: blocks.append({"type": "text", "text": ""})

        # --- 6b) subagent dispatch marker: if this response on the main line
        # contains a Task/Agent tool_use, remember the id so the next non-
        # main system_hash request can be matched to this dispatch. Only
        # arm the marker on the main session (depth-0 of subagent_stack);
        # nested dispatches set the marker on the sub-session itself so the
        # nested push can pick it up.
        dispatch_id = _find_dispatch_tool_use_id(blocks)
        if dispatch_id:
            if target_is_sub:
                target.dispatch_tool_use_id = dispatch_id
            else:
                s._pending_dispatch_id = dispatch_id

        stop_reason = "tool_use" if tool_uses else ("max_tokens" if finish == "length" else "end_turn")
        in_tokens = len(ideal_ids)
        out_tokens = len(output_ids)
        model = body.get("model") or "slime-actor"
        if turn is not None:
            turn.output_len = out_tokens
            turn.finish_reason = finish
            turn.stop_reason = stop_reason
            turn.full_ids = ideal_ids + output_ids
            turn.response = {
                "raw_output": raw_output,
                "thinking": thinking,
                "visible": visible,
                "tool_uses": tool_uses,
            }

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
class MiddlewareHandle:
    app_handle: AppHandle
    store: _Store
    abort: _AbortCoordinator
    public_host: str

    @property
    def public_url(self) -> str:
        return f"http://{self.public_host}:{self.app_handle.port}"

    def open_session(
        self,
        session_id: str,
        *,
        sampling_defaults: dict[str, Any] | None = None,
        record_tree: bool = False,
    ) -> None:
        self.store.open_session(session_id, defaults=sampling_defaults or {}, record_tree=record_tree)

    async def _pop_session(self, session_id: str) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
        s = await self.store.pop(session_id)
        if s is None:
            return [], [], [], {}
        async with s.lock:
            return list(s.prompt_ids), list(s.response_ids), list(s.loss_mask), _export_tree(s)

    def pop_session(self, session_id: str) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
        fut = asyncio.run_coroutine_threadsafe(self._pop_session(session_id), self.app_handle.loop)
        return fut.result(timeout=10)

    async def _pop_session_split(self, session_id: str) -> tuple[
        list[tuple[list[int], list[int], list[int], dict[str, Any]]],
        dict[str, Any],
    ]:
        s = await self.store.pop(session_id)
        if s is None:
            return [], {}
        async with s.lock:
            # Any subagent still on the stack never returned (claude-code may
            # have exited mid-dispatch). Snapshot the outermost one as a
            # subagent segment if it has any response tokens; nested ones
            # get merged into their parent first.
            while s.subagent_stack:
                top = s.subagent_stack.pop()
                if s.subagent_stack:
                    _merge_into_parent(top, s.subagent_stack[-1])
                else:
                    _snapshot_subagent(s, top)

            segments: list[tuple[list[int], list[int], list[int], dict[str, Any]]] = [
                (list(p), list(r), list(m), dict(meta))
                for (p, r, m, meta) in s.completed_trajectories
            ]
            if s.response_ids:
                segments.append((
                    list(s.prompt_ids),
                    list(s.response_ids),
                    list(s.loss_mask),
                    {
                        "kind": "final",
                        "completed_turns": len(s.turns),
                        "finish_reason": s.last_finish_reason,
                    },
                ))
            return segments, _export_tree(s)

    def pop_session_split(self, session_id: str) -> tuple[
        list[tuple[list[int], list[int], list[int], dict[str, Any]]],
        dict[str, Any],
    ]:
        """list_trajectory variant of pop_session: returns one (prompt_ids,
        response_ids, loss_mask, meta) tuple per chain segment.

        Segments are emitted in chronological order (only three ``kind``s
        per user decision 2 in MASTER_PLAN preamble):
          * ``pre_wipe`` — one per autoCompact / non-linear wipe on the
            currently routed target (main line OR a sub-session).
          * ``subagent`` — one per outermost Task/Agent dispatch return.
            Nested dispatches (depth >= 2) merge into their parent and
            never emit a standalone segment.
          * ``final`` — the live main-line chain at session pop time.

        ``diverge_reset`` and ``compact_summarization`` are NOT emitted
        as separate segments; divergence simply drops the current chain
        and starts a fresh one whose tokens land in the next ``pre_wipe``
        or ``final`` segment.

        Empty-response segments are dropped (no trainable tokens). The
        trajectory_tree export is returned alongside so the caller can
        attach it to ONE of the fanned-out Samples for downstream viz."""
        fut = asyncio.run_coroutine_threadsafe(self._pop_session_split(session_id), self.app_handle.loop)
        return fut.result(timeout=10)

    # --- Weight-update abort/resume control -----------------------------------
    def aborted_now(self) -> None:
        """Tell the middleware: stop accepting new generate calls; mark
        in-flight ones as ABORTED. Idempotent; safe to call from any thread."""
        asyncio.run_coroutine_threadsafe(self.abort.aborted_now(), self.app_handle.loop).result(timeout=5)

    def resumed_now(self) -> None:
        """Tell the middleware: weights are reloaded; re-issue aborted /generate
        calls with reduced max_new_tokens and let new requests through."""
        asyncio.run_coroutine_threadsafe(self.abort.resumed_now(), self.app_handle.loop).result(timeout=5)

    def install_abort_poll(
        self,
        should_abort_fn: Callable[[], bool],
        *,
        interval_sec: float = 0.5,
        max_wait_sec: float = 1800.0,
    ) -> None:
        """Wire ``should_abort_fn`` to be polled inside the middleware loop.

        Typical usage from the rollout worker process::

            from slime.rollout.sglang_rollout import GenerateState
            state = GenerateState(args)
            middleware.install_abort_poll(lambda: state.aborted)
        """
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
    """Spin up the middleware on a daemon thread; return a handle.

    Args:
        tokenizer:        HF tokenizer that supports apply_chat_template(tools=)
        sglang_url:       slime SGLang router base URL (e.g. ``http://10.0.0.1:30000``)
        tool_parser:      Name in ``FunctionCallParser.ToolCallParserEnum``
                          ('glm47' / 'qwen25' / 'deepseekv3' / ...) or None to disable.
        reasoning_parser: Name in ``ReasoningParser.DetectorMap``
                          ('glm45' / 'qwen3' / 'deepseek-r1' / ...) or None to disable.
    """
    store = _Store()
    abort = _AbortCoordinator()
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
    return MiddlewareHandle(app_handle=handle, store=store, abort=abort, public_host=public_host or host)
