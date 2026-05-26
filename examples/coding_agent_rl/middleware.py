"""Anthropic Messages API shim that translates claude-code requests into
sglang ``/generate`` calls while tracking the token-level training target
for slime RL rollouts.

Each turn:
* fingerprints the incoming Anthropic conversation against per-session state
  to decide whether it continues the main chain or an active sub-agent chain,
  and whether the request appends to / wipes / restarts that chain;
* re-renders the chosen chain through the model's chat template, splicing
  cached raw tokens back in so re-tokenization drift can't corrupt loss_mask;
* posts to sglang ``/generate``, commits output_ids onto the chain's
  response_ids/loss_mask, and verifies per-turn TITO (decode -> encode
  round-trip);
* parses the decoded output back into Anthropic blocks and streams them to
  claude-code as a Messages SSE response.

Public surface:
    start(...)                       build the aiohttp app + store
    open_session(store, sid, ...)    register a session with sampling defaults
    pop_session_split(store, sid)    drain a session's trajectory for training
    Chain, Session                   dataclasses (exposed for type hints)

Read `_handle_request` (§3) top-to-bottom -- that's the whole turn:
    _select_chain      pick main vs active sub; snapshot wipe/sub-done into segments
    _build_prompt      replace/extend chat_messages -> render token ids -> update prompt+mask
    _generate          POST sglang /generate, commit output, per-turn TITO
    _build_reply       output_ids -> Anthropic blocks, queue pending raw tokens

`_build_prompt` is itself a thin orchestrator over three helpers:
    _replace_chat_messages / _extend_chat_messages    translate Anthropic blocks -> chat_messages
    _render_token_ids                                 chat template + raw splice -> (ideal_ids, raw_ranges)
    _update_prompt_and_mask                           update prompt_ids / response_ids / loss_mask

Design notes:
* `kind` (new/wipe/append) is consumed at the dispatch sites:
  `_replace_chat_messages` vs `_extend_chat_messages` is picked at call time,
  and `_update_prompt_and_mask` only branches on append vs full. Nothing
  else sees `kind`.
* `_render_token_ids` only reads target; the caller owns all state mutation.
* `_hash` strips Anthropic `cache_control` keys before hashing so the same
  logical message hashes identically across turns even as cache_control moves.
* Server lifecycle (binding a port, running the loop, `handler_cancellation`)
  lives in the caller, not here -- see `start()` for the required runner
  contract.
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


logger = logging.getLogger(__name__)


# Per-segment hard cap on prompt+response token count. Drops any segment over
# this -- claude-code auto-compact estimates in its own tokenizer space, so a
# 100k autoCompactWindow can produce 130k+ Qwen-tokenized segments after
# sub-agent dispatch reads large files. Such segments OOM fused CE on
# actor_train. 0 disables the cap.
_MAX_SEGMENT_TOKENS = int(os.environ.get("SWE_MAX_SEGMENT_TOKENS", "96000") or 0)

# Tool names claude-code uses to dispatch a sub-agent.
_SUBAGENT_TOOLS = {"Task", "Agent"}

# Qwen3 reasoning chat template auto-injects this before any completed
# assistant `content` that has no `reasoning_content` entry. The raw-splice
# renderer must swallow it so spliced output isn't doubled.
_EMPTY_THINK_STUB_TEXT = "<think>\n\n</think>\n\n"

# Raw-splice placeholder bracket. \x07 (BEL) keeps BPE boundaries clean.
_RAW_PH_PREFIX = "\x07RAWSPLICE_"
_RAW_PH_SUFFIX = "_END\x07"


def _strip_cache_control(obj: Any) -> Any:
    """Drop Anthropic prompt-caching ``cache_control`` keys before hashing -
    cache_control moves across turns so the same logical message would
    otherwise hash differently each request."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(x) for x in obj]
    return obj


def _hash(obj: Any) -> str:
    payload = json.dumps(_strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


# =============================================================================
# 1. Data structures
# =============================================================================

# Raw token slice: (full_tokens_including_template_gen_prefix, gen_prefix_len)
_RawSlice = tuple[list[int], int]


@dataclasses.dataclass
class Chain:
    """One conversation chain (main, or an active sub-agent)."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)

    # Token-level state (the actual training target)
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    response_ids: list[int] = dataclasses.field(default_factory=list)
    loss_mask: list[int] = dataclasses.field(default_factory=list)

    # Raw token bookkeeping for splice rendering
    asst_raw_tokens: dict[int, _RawSlice] = dataclasses.field(default_factory=dict)
    pending_raw_tokens: list[_RawSlice] = dataclasses.field(default_factory=list)

    last_finish_reason: str = ""


@dataclasses.dataclass
class Session:
    main: Chain = dataclasses.field(default_factory=Chain)
    active_sub: Chain | None = None  # at most one sub-agent at a time
    pending_dispatch_id: str = ""  # tool_use_id we're waiting to close
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    segments: list[tuple] = dataclasses.field(default_factory=list)  # frozen output


_Store = dict[str, Session]


def _make_segment(chain: Chain, kind: str) -> tuple:
    """Snapshot the chain's current token state into a segment tuple
    (kind, prompt_ids, response_ids, loss_mask, meta) for the train loop."""
    return (
        kind,
        list(chain.prompt_ids),
        list(chain.response_ids),
        list(chain.loss_mask),
        {"segment_kind": kind, "finish_reason": chain.last_finish_reason},
    )


# =============================================================================
# 2. Per-turn stages
# =============================================================================


def _select_chain(s: Session, body: dict) -> tuple[Chain, bool, str]:
    """Decide which chain this turn operates on.

    1. fingerprint body.messages and body.system into hashes
    2. if main now contains the tool_result for a pending sub dispatch,
       snapshot the sub chain into s.segments and clear s.active_sub
    3. pick main vs s.active_sub based on whether request continues main's prefix
    4. classify as 'new' | 'append' | 'wipe' against the chosen target;
       a wipe also snapshots the target's current state into s.segments

    Returns (target_chain, is_sub, kind).
    """
    all_msgs = body.get("messages") or []
    msg_hashes = [_hash(m) for m in all_msgs]
    req_system_hash = _hash(body.get("system")) if "system" in body else s.main.system_hash

    # Close active sub-agent if its dispatch tool_result has landed on main.
    if s.pending_dispatch_id and s.active_sub is not None:
        tu_id = s.pending_dispatch_id
        for m in all_msgs:
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if not isinstance(content, list):
                continue
            done = any(
                isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tu_id
                for b in content
            )
            if done:
                s.segments.append(_make_segment(s.active_sub, "subagent"))
                s.active_sub = None
                s.pending_dispatch_id = ""
                break

    # Route: main iff request continues main's prefix. Sub system_hash can be
    # "" (armed before sub dialled in), so never route by sub equality alone.
    if s.active_sub is None:
        target, is_sub = s.main, False
    else:
        main_continues = (
            req_system_hash == s.main.system_hash
            and len(msg_hashes) >= s.main.seen_msgs
            and msg_hashes[: s.main.seen_msgs] == s.main.msg_hashes[: s.main.seen_msgs]
        )
        target, is_sub = (s.main, False) if main_continues else (s.active_sub, True)

    # Classify; snapshot a "wipe" segment first if we're discarding work.
    if target.seen_msgs == 0:
        kind = "new"
    else:
        is_append = (
            req_system_hash == target.system_hash
            and len(msg_hashes) >= target.seen_msgs
            and msg_hashes[: target.seen_msgs] == target.msg_hashes[: target.seen_msgs]
        )
        if is_append:
            kind = "append"
        else:
            if target.response_ids:
                s.segments.append(_make_segment(target, "wipe"))
            kind = "wipe"

    return target, is_sub, kind


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
                    tcs.append({"function": {"name": b.get("name", "tool"), "arguments": b.get("input") or {}}})
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": _flatten(content)})
    return translated


def _build_tools_schema(anth_tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tools spec -> chat-template tool schema. Pure function."""
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


def _replace_chat_messages(target: Chain, body: dict) -> None:
    """new/wipe: full reset of chat state + raw token caches."""
    all_msgs = body.get("messages") or []
    target.chat_messages = _translate_anthropic(all_msgs, body.get("system"))
    if "system" in body:
        target.system_hash = _hash(body.get("system"))
    target.asst_raw_tokens.clear()
    target.pending_raw_tokens.clear()
    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _build_tools_schema(body.get("tools"))


def _extend_chat_messages(target: Chain, body: dict) -> None:
    """append: translate only the new tail; promote each pending raw-token
    slice onto target.asst_raw_tokens at the matching assistant index."""
    all_msgs = body.get("messages") or []
    translated = _translate_anthropic(all_msgs[target.seen_msgs :], None)

    base_idx = len(target.chat_messages)
    target.chat_messages.extend(translated)
    for offset, m in enumerate(translated):
        if m.get("role") != "assistant" or not target.pending_raw_tokens:
            continue
        target.asst_raw_tokens[base_idx + offset] = target.pending_raw_tokens.pop(0)

    target.seen_msgs = len(all_msgs)
    target.msg_hashes = [_hash(m) for m in all_msgs]
    if target.tools_schema is None:
        target.tools_schema = _build_tools_schema(body.get("tools"))


def _render_token_ids(target: Chain, tok) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Render target.chat_messages through the chat template. For each
    historical assistant in target.asst_raw_tokens, splice the cached raw
    token slice back in so re-tokenization drift can't corrupt loss_mask.

    Pure read of target. Returns (ideal_ids, raw_ranges) where each raw_range
    is (splice_start, gen_start, splice_end) in ideal_ids space.
    """
    valid = {i: tup for i, tup in target.asst_raw_tokens.items() if 0 <= i < len(target.chat_messages)}
    raw_ranges: list[tuple[int, int, int]] = []

    if not valid:
        # Qwen3.x fast tokenizers return a BatchEncoding here, not a list[int];
        # list(BatchEncoding) yields dict keys (["input_ids", ...]) and poisons
        # sglang /generate. Unwrap input_ids defensively.
        enc = tok.apply_chat_template(
            target.chat_messages,
            tools=target.tools_schema,
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
        return list(ids), raw_ranges

    placeholders: dict[int, str] = {}
    render_msgs: list[dict] = []
    for i, m in enumerate(target.chat_messages):
        if i in valid:
            ph = f"{_RAW_PH_PREFIX}{i}_{secrets.token_hex(6)}{_RAW_PH_SUFFIX}"
            placeholders[i] = ph
            render_msgs.append({"role": "assistant", "content": ph})
        else:
            render_msgs.append(m)

    text = tok.apply_chat_template(
        render_msgs,
        tools=target.tools_schema,
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


def verify_tito_for_turn(tok, decoded_text: str, output_ids: list[int]) -> bool:
    """Per-turn TITO (tokenize-in tokenize-out): tokenizing ``decoded_text``
    must yield ``output_ids`` byte-identical. False means the tokenizer can't
    round-trip these bytes -- caller should zero the loss_mask tail rather
    than train on phantom tokens.

    Module-level so tests can monkeypatch it; pure predicate, no logging or
    state mutation."""
    if not output_ids:
        return True
    retok = tok.encode(decoded_text, add_special_tokens=False)
    if hasattr(retok, "ids"):
        retok = list(retok.ids)
    return list(retok) == list(output_ids)


def verify_tito_cross_turn(target: Chain, ideal_ids: list[int]) -> bool:
    """Cross-turn TITO: ``ideal_ids`` must start with ``target.prompt_ids``
    (the turn-0 anchor) byte-identically. False means the chat template
    drifted between turns -- caller should rebaseline with an all-zero
    loss_mask rather than train on tokens the model never emitted.

    Pure predicate; no logging or state mutation."""
    return ideal_ids[: len(target.prompt_ids)] == target.prompt_ids


def _update_prompt_and_mask(
    target: Chain, ideal_ids: list[int], raw_ranges: list[tuple[int, int, int]], kind: str
) -> None:
    """Update target.prompt_ids / response_ids / loss_mask given freshly
    rendered ideal_ids.

    On append, the re-rendered prefix must be byte-identical to the anchor
    set on turn 0 (cross-turn TITO). On drift we demote the tail mask to 0
    rather than train on tokens the model never emitted.
    """
    if kind != "append":
        target.prompt_ids = ideal_ids
        target.response_ids = []
        target.loss_mask = []
        return

    prompt_len = len(target.prompt_ids)
    if not verify_tito_cross_turn(target, ideal_ids):
        logger.warning("[middleware] template re-render mismatch; rebaselining")
        target.response_ids = ideal_ids[prompt_len:]
        target.loss_mask = [0] * len(target.response_ids)
        return

    response = ideal_ids[prompt_len:]
    mask = [0] * len(response)
    response_len = len(response)
    for _splice_start, gen_start, splice_end in raw_ranges:
        a = max(0, gen_start - prompt_len)
        b = min(response_len, max(0, splice_end - prompt_len))
        for k in range(a, b):
            mask[k] = 1
    target.response_ids = response
    target.loss_mask = mask


def _build_prompt(target: Chain, body: dict, kind: str, tok) -> list[int]:
    """Replace/extend chat_messages -> render token ids -> update prompt+mask.
    Returns ideal_ids to feed sglang.

    Thin orchestrator over `_replace_chat_messages` / `_extend_chat_messages`,
    `_render_token_ids`, and `_update_prompt_and_mask`. Each step is
    independently testable; the only coupling is the (ideal_ids, raw_ranges)
    tuple passed between render and update.
    """
    (_extend_chat_messages if kind == "append" else _replace_chat_messages)(target, body)
    ideal_ids, raw_ranges = _render_token_ids(target, tok)
    _update_prompt_and_mask(target, ideal_ids, raw_ranges, kind)
    return ideal_ids


async def _generate(target: Chain, ideal_ids: list[int], s: Session, body: dict, app) -> tuple[list[int], str]:
    """Call sglang and commit output to target chain.

    1. build sampling_params (session defaults overlaid with body overrides)
    2. POST sglang /generate; on cancel/error fire /abort_request
    3. extend target.response_ids with output_ids; loss_mask += [1]*N
    4. per-turn TITO: decode(output_ids) re-encoded must equal output_ids;
       on mismatch zero out the loss_mask tail for this turn

    Returns (output_ids, finish_reason).
    """
    # ---- (a) Build sampling_params --------------------------------------
    sp: dict[str, Any] = {
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

    # ---- (b) POST sglang /generate (with abort on cancel/error) --------
    # Without abort, a cancelled client + inflight request can race with the
    # next release_memory_occupation and trip sglang's "server is idle" assert.
    sglang_url = app["sglang_url"]
    rid = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": ideal_ids,
                "sampling_params": sp,
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
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError):
        # Best-effort abort with fresh short-timeout session; swallow errors.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    # ---- (c) Commit + per-turn TITO -------------------------------------
    target.response_ids.extend(output_ids)
    target.loss_mask.extend([1] * len(output_ids))
    target.last_finish_reason = finish

    n = len(output_ids)
    if n > 0:
        tok = app["tokenizer"]
        raw = tok.decode(output_ids, skip_special_tokens=False)
        if not verify_tito_for_turn(tok, raw, output_ids):
            target.loss_mask[-n:] = [0] * n
            logger.warning("[middleware] TITO mismatch; loss_mask zeroed (n=%d)", n)

    return output_ids, finish


def _build_reply(target: Chain, output_ids: list[int], finish: str, app) -> tuple[list[dict], str, str]:
    """Turn the model's raw output ids into the reply we send back to claude-code.

    1. parse decoded text -> (thinking, visible, tool_uses) via sglang parsers
    2. pack into Anthropic content blocks; tag dispatch_id when a tool_use
       names Task/Agent (sub-agent trigger)
    3. queue this turn's raw tokens onto target.pending_raw_tokens for the
       next append to splice
    4. derive stop_reason: 'tool_use' | 'max_tokens' | 'end_turn'

    Returns (blocks, stop_reason, dispatch_id).
    """
    tok = app["tokenizer"]
    tool_parser_name = app["tool_parser"]
    reasoning_parser_name = app["reasoning_parser"]
    tools_schema = target.tools_schema

    # (a) Decode raw text. Per-turn TITO already verified inside _generate.
    raw_output = tok.decode(output_ids, skip_special_tokens=False) if output_ids else ""

    # (b) Parse: reasoning -> tool calls -> xml fallback.
    thinking, body_text = "", raw_output
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        r, b = ReasoningParser(model_type=reasoning_parser_name, stream_reasoning=False).parse_non_stream(raw_output)
        thinking, body_text = r or "", b or ""
        if not thinking and "</think>" in body_text:
            thinking, body_text = body_text.split("</think>", 1)

    tool_uses: list[dict] = []
    if tool_parser_name and tools_schema:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        body_text, calls = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name).parse_non_stream(
            body_text
        )
        for c in calls:
            try:
                args = json.loads(c.parameters or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": c.parameters}
            tool_uses.append({"name": c.name or "tool", "input": args})

    # XML fallback when structured parser didn't catch anything (Qwen's
    # occasional Anthropic-style XML tool-call format). re.finditer + manual
    # concat instead of re.sub(repl) so we avoid a nested def for the callback.
    if not tool_uses and tools_schema:
        valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
        cleaned_parts: list[str] = []
        last = 0
        for m in re.finditer(
            r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
            body_text,
            flags=re.DOTALL,
        ):
            name, inner = m.group(1), m.group(2)
            if name in valid_tools:
                args = {
                    p.group(1): p.group(2).strip()
                    for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
                }
                tool_uses.append({"name": name, "input": args})
                cleaned_parts.append(body_text[last : m.start()])
                last = m.end()
        cleaned_parts.append(body_text[last:])
        body_text = "".join(cleaned_parts).replace("<|im_end|>", "")

    visible = (body_text or "").strip()

    # (c) Pack Anthropic content blocks.
    blocks: list[dict] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": thinking})
    if visible:
        blocks.append({"type": "text", "text": visible})
    dispatch_id = ""
    for tu in tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        if tu["name"] in _SUBAGENT_TOOLS:
            dispatch_id = tu_id
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    # (d) Queue raw tokens for the next splice. Reconstruct pre-output ideal_ids
    # = target.prompt_ids + (target.response_ids minus the freshly appended
    # output_ids tail set by _generate). Find the last `<|im_start|>assistant\n`
    # marker; everything after it is the template-injected generation prefix
    # (e.g. `<think>\n` for Qwen3) that we stitch onto the front of raw output_ids.
    n_out = len(output_ids)
    pre_out_response = target.response_ids[:-n_out] if n_out else list(target.response_ids)
    ideal_ids = list(target.prompt_ids) + list(pre_out_response)

    marker_ids = app["assistant_marker_ids"]
    gen_prefix: list[int] = []
    if marker_ids:
        n = len(marker_ids)
        for start in range(len(ideal_ids) - n, -1, -1):
            if ideal_ids[start : start + n] == marker_ids:
                gen_prefix = list(ideal_ids[start + n :])
                break
    target.pending_raw_tokens.append((gen_prefix + list(output_ids), len(gen_prefix)))

    # (e) Stop reason.
    if tool_uses:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    return blocks, stop_reason, dispatch_id


def _start_sub_chain(s: Session, dispatch_id: str) -> None:
    """Start a fresh sub chain on this session and remember the tool_use_id
    we'll watch for on main to know when this sub is done. The matching
    'sub done' step lives inside _select_chain."""
    s.pending_dispatch_id = dispatch_id
    if s.active_sub is None:
        s.active_sub = Chain()


# =============================================================================
# 3. Request handling -- one full turn + SSE wrap
# =============================================================================


async def _handle_request(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    sid = request.headers["Authorization"].removeprefix("Bearer ").strip()
    app = request.app
    s = app["store"].setdefault(sid, Session())

    async with s.lock:  # same sid -> serialized
        target, is_sub, kind = _select_chain(s, body)
        ideal_ids = _build_prompt(target, body, kind, app["tokenizer"])
        output_ids, finish = await _generate(target, ideal_ids, s, body, app)
        blocks, stop, did = _build_reply(target, output_ids, finish, app)
        if did and not is_sub:  # sub doesn't nest
            _start_sub_chain(s, did)
        in_tok, out_tok = len(ideal_ids), len(output_ids)

    return await _stream_response(request, blocks, stop, in_tok, out_tok)


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

    # message_start
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


# =============================================================================
# 4. Public API
# =============================================================================


def open_session(store: _Store, sid: str, *, sampling_defaults: dict | None = None) -> None:
    """Register a new session. Fail-fast on duplicate sid: silently sharing
    state would interleave two independent rollouts into one chain and corrupt
    TITO bookkeeping. `sampling_defaults` seeds the session's default sglang
    sampling_params (overlaid by per-request body in `_generate`)."""
    if sid in store:
        raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
    s = store[sid] = Session()
    s.sampling_defaults = dict(sampling_defaults or {})


def pop_session_split(store: _Store, sid: str) -> list[tuple]:
    """Snapshot whatever chains are still alive (active_sub + main) into
    segments, drop empty and oversized ones. Called by the train loop at
    trajectory end."""
    s = store.pop(sid, None)
    if s is None:
        return []
    if s.active_sub is not None:
        s.segments.append(_make_segment(s.active_sub, "subagent"))
    if s.main.response_ids:
        s.segments.append(_make_segment(s.main, "final"))
    return [
        (p, r, m, meta)
        for kind, p, r, m, meta in s.segments
        if r and (_MAX_SEGMENT_TOKENS <= 0 or len(p) + len(r) <= _MAX_SEGMENT_TOKENS)
    ]


# Trivial endpoints claude-code probes during a session: count_tokens runs
# every turn (return 0 -- client uses it as a hint, not a hard budget),
# healthz/v1/models are startup readiness checks.
async def _count_tokens(request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 0})


async def _ok(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def start(*, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None):
    """Build the aiohttp app + store. Caller is responsible for running the
    server (e.g. `aiohttp.web.run_app` or a daemon-thread wrapper).

    The runner MUST set ``handler_cancellation=True`` so a client disconnect
    actually cancels the handler coroutine, arming the fire-and-forget
    /abort_request inside `_generate`. Without it a cancelled client leaves an
    inflight sglang /generate that races with the next release_memory_occupation
    and trips sglang's "server is idle" assertion -- crashing the scheduler.

    Use `open_session(store, sid, ...)` to register a session before
    claude-code dials in (fail-fast on duplicate sid, seeds sampling defaults).
    Use `pop_session_split(store, sid)` to drain its trajectory at rollout end.
    """
    store: _Store = {}
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["tokenizer"] = tokenizer
    app["sglang_url"] = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
    app["tool_parser"] = tool_parser
    app["reasoning_parser"] = reasoning_parser
    app["store"] = store
    # Search marker for the chat template's assistant role start. _build_reply
    # uses this to locate the last `<|im_start|>assistant\n` in ideal_ids;
    # everything after it is the template-injected generation prefix (e.g.
    # `<think>\n` for Qwen3) we stitch onto raw output_ids when queueing
    # pending_raw_tokens.
    app["assistant_marker_ids"] = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    app.router.add_post("/v1/messages", _handle_request)
    app.router.add_post("/v1/messages/count_tokens", _count_tokens)
    app.router.add_get("/healthz", _ok)
    app.router.add_get("/v1/models", _ok)
    return app, store
