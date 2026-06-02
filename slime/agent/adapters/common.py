"""Shared adapter primitives for token-capturing agent rollouts.

Each HTTP request carries the full conversation history. We render it with the
served model's chat template, call SGLang ``/generate`` with ``input_ids``, and
fold the turn into a per-session :class:`~slime.agent.trajectory_manager.TrajectoryTree`.

The tree does all routing (sub-agent / compaction / history-rewrite branch
automatically by text prefix), so there is no manual new/append/wipe logic. Two
things make this faithful under TITO (text-in-token-out) drift:

* the **current turn's** incremental prompt segment is taken by *segment render
  diff* (``render(through this prompt) - render(through prev response)``), which
  is immune to drift because it compares two re-renders rather than a cache vs a
  re-render (design §5.1); and
* **historical turns'** response tokens come from a per-session *truth cache*
  keyed by the cumulative prompt that produced them, so they equal the tree
  node tokens exactly (no re-tokenization).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.trajectory_manager import PromptSeg, RespSeg, TokenSegment, TrajectoryTree, export_token_segments

ADAPTER_KEY = web.AppKey("adapter", object)
TOKENIZER_KEY = web.AppKey("tokenizer", object)
SGLANG_URL_KEY = web.AppKey("sglang_url", object)
TOOL_PARSER_KEY = web.AppKey("tool_parser", object)
REASONING_PARSER_KEY = web.AppKey("reasoning_parser", object)


@dataclasses.dataclass
class GenResult:
    """One assistant generation from the rollout engine (sglang ``/generate``).

    ``output_text`` is ``decode(output_ids)`` cached once so reply builders and
    the trajectory's response-segment text share a single detokenization.
    """

    output_ids: list[int]
    output_log_probs: list[float]
    finish_reason: str
    output_text: str = ""


@dataclasses.dataclass
class SessionTrajectory:
    """Per-session trajectory state: the turn-node tree plus the truth cache.

    ``resp_truth`` maps the cumulative prompt token tuple that *preceded* a
    response to ``(output_ids, output_log_probs, output_text)``. The same turn,
    seen as history in a later request, renders to the same cumulative prompt and
    so retrieves its exact sampled tokens (a Case 2 append, never a false drift).
    Keying on rendered prompt ids (a deterministic function of the messages),
    rather than on echoed wire text, avoids the parse->serialize round-trip drift
    that would otherwise make a text key miss.

    ``render_memo`` caches ``apply_chat_template`` results within a session so a
    growing trajectory renders each distinct message-prefix once (O(n) renders
    over the whole trajectory instead of O(n^2)).
    """

    tree: TrajectoryTree = dataclasses.field(default_factory=TrajectoryTree)
    resp_truth: dict[tuple[int, ...], tuple[list[int], list[float], str]] = dataclasses.field(default_factory=dict)
    render_memo: dict[tuple[str, bool], list[int]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Session:
    """Per-session adapter state shared by the OpenAI and Anthropic adapters.

    Holds the trajectory (turn-node tree + truth cache), the per-session sampling
    defaults / context budget injected at ``open_session``, and a lock that
    serializes concurrent requests carrying the same session id.
    """

    traj: SessionTrajectory = dataclasses.field(default_factory=SessionTrajectory)
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class BaseAdapter:
    """Base HTTP adapter with per-instance session lifecycle state."""

    session_cls: type = Session

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app[ADAPTER_KEY] = self
        self.app[TOKENIZER_KEY] = tokenizer
        self.app[SGLANG_URL_KEY] = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.app[TOOL_PARSER_KEY] = tool_parser
        self.app[REASONING_PARSER_KEY] = reasoning_parser

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        register_session(
            self.store,
            sid,
            self.session_cls,
            sampling_defaults=sampling_defaults,
            max_context_tokens=max_context_tokens,
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await shutdown_session_tasks(sid, self.closed, self.inflight, wait_timeout=wait_timeout)

    async def finish_session(self, sid: str, *, wait_timeout: float = 5.0) -> list[TokenSegment]:
        """Drain a session: wait out in-flight requests, then export trainable
        ``TokenSegment``s from its trajectory tree. Idempotent -- a second call
        for an already-popped sid returns ``[]``."""
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        s = self.store.pop(sid, None)
        if s is None:
            return []
        return export_token_segments(s.traj.tree)


def strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_cache_control(x) for x in obj]
    return obj


def stable_hash(obj: Any) -> str:
    payload = json.dumps(strip_cache_control(obj), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def json_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _extract_ids(enc: Any) -> list[int]:
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def render_token_ids(
    messages: list[dict], tokenizer, *, tools: list[dict] | None = None, add_generation_prompt: bool = True
) -> list[int]:
    """Render a chat-template message list to token ids."""
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return _extract_ids(enc)


# =============================================================================
# Turn assembly: chat messages -> strictly-alternating (PromptSeg, RespSeg) turns
# =============================================================================


def split_turns(chat_messages: list[dict]) -> list[tuple[list[dict], dict | None]]:
    """Split a chat-template message list into ``(prompt_msgs, assistant_msg)``
    turns at every ``assistant`` boundary.

    ``prompt_msgs`` is the run of non-assistant messages since the previous
    assistant. A trailing ``(prompt_msgs, None)`` turn is always appended: it is
    the current turn whose response is about to be generated.
    """
    specs: list[tuple[list[dict], dict | None]] = []
    buf: list[dict] = []
    for m in chat_messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            specs.append((buf, m))
            buf = []
        else:
            buf.append(m)
    specs.append((buf, None))
    return specs


def _render_memo(
    traj: SessionTrajectory | None,
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None,
    add_generation_prompt: bool,
) -> list[int]:
    """Render with a per-session memo so each distinct message prefix is rendered
    once across a growing trajectory (turns 1..n cost O(n) renders total, not
    O(n^2))."""
    if traj is None:
        return render_token_ids(messages, tokenizer, tools=tools, add_generation_prompt=add_generation_prompt)
    key = (stable_hash([messages, tools]), add_generation_prompt)
    cached = traj.render_memo.get(key)
    if cached is None:
        cached = render_token_ids(messages, tokenizer, tools=tools, add_generation_prompt=add_generation_prompt)
        traj.render_memo[key] = cached
    return list(cached)


def render_prompt(traj: SessionTrajectory, messages: list[dict], tokenizer, tools: list[dict] | None) -> list[int]:
    """Render the full prompt (``add_generation_prompt=True``) to send to sglang,
    memoized on the session so a replayed prefix is not re-rendered next turn."""
    return _render_memo(traj, messages, tokenizer, tools, True)


def assemble_turns(
    traj: SessionTrajectory,
    chat_messages: list[dict],
    tokenizer,
    tools: list[dict] | None,
    gen: GenResult,
    full_prompt_ids: list[int],
) -> tuple[list[tuple[PromptSeg, RespSeg]], tuple[int, ...]]:
    """Build ``record_turn``'s strictly-alternating ``turns`` for this request.

    Every turn's incremental prompt segment comes from a *segment render diff*
    (design §5.1): ``render(through this prompt) - render(through previous
    response)``, comparing two re-renders so it is immune to TITO drift. The
    previous render uses ``add_generation_prompt=False`` (it ends at a response),
    so the diff is a clean length-based suffix.

    History response tokens come from the truth cache keyed by the cumulative
    prompt that produced them (exact sampled ids; a Case 2 append, never a false
    drift). On a cache miss they fall back to re-tokenization, which the tree
    then handles as Case 3/4 (replace + mask). The current turn's response comes
    from ``gen`` and reuses ``full_prompt_ids`` (the prompt already sent to
    sglang) as its prompt segment.

    Does not write the cache -- the caller writes it only after ``record_turn``
    succeeds (no dangling cache on a sglang error). Returns ``(turns,
    pending_key)`` where ``pending_key`` is the cumulative-prompt key under which
    the current turn's response should be cached.
    """
    specs = split_turns(chat_messages)
    turns: list[tuple[PromptSeg, RespSeg]] = []
    prev_cum: list[dict] = []
    prev_ids: list[int] = []
    pending_key: tuple[int, ...] = tuple(full_prompt_ids)

    for prompt_msgs, assistant_msg in specs:
        cum_prompt = prev_cum + prompt_msgs
        if assistant_msg is None:
            # Current turn: reuse the exact prompt already sent to sglang.
            cur_ids = list(full_prompt_ids)
        else:
            cur_ids = _render_memo(traj, cum_prompt, tokenizer, tools, True)
        delta = cur_ids[len(prev_ids) :]
        p_text = tokenizer.decode(delta, skip_special_tokens=False) if delta else ""
        p_seg = PromptSeg(p_text, list(delta))

        if assistant_msg is None:
            r_seg = RespSeg(gen.output_text, list(gen.output_ids), list(gen.output_log_probs))
            pending_key = tuple(cur_ids)
            turns.append((p_seg, r_seg))
            break

        cached = traj.resp_truth.get(tuple(cur_ids))
        if cached is not None:
            r_ids, r_lp, r_text = cached
            r_seg = RespSeg(r_text, list(r_ids), list(r_lp))
        else:
            full = _render_memo(traj, cum_prompt + [assistant_msg], tokenizer, tools, False)
            r_ids = full[len(cur_ids) :]
            r_text = tokenizer.decode(r_ids, skip_special_tokens=False) if r_ids else ""
            r_seg = RespSeg(r_text, list(r_ids), [0.0] * len(r_ids))
        turns.append((p_seg, r_seg))

        prev_cum = cum_prompt + [assistant_msg]
        prev_ids = _render_memo(traj, prev_cum, tokenizer, tools, False)

    return turns, pending_key


def request_session_id(
    request: web.Request,
    *,
    body: dict | None = None,
    include_x_api_key: bool = False,
) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        sid = auth[7:].strip()
        if sid:
            return sid

    if body is not None:
        metadata = body.get("metadata")
        if isinstance(metadata, dict) and metadata.get("session_id"):
            return str(metadata["session_id"])
        if body.get("user"):
            return str(body["user"])

    if include_x_api_key:
        api_key = request.headers.get("X-Api-Key")
        if api_key:
            return api_key.strip()

    return "default"


def register_session(
    store: dict[str, Any],
    sid: str,
    session_factory: Callable[[], Any],
    *,
    sampling_defaults: dict | None = None,
    max_context_tokens: int = 0,
) -> None:
    if sid in store:
        raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
    session = store[sid] = session_factory()
    session.sampling_defaults = dict(sampling_defaults or {})
    session.max_context_tokens = int(max_context_tokens or 0)


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp


async def call_sglang_generate(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    app,
    *,
    max_token_keys: tuple[str, ...],
    stop_keys: tuple[str, ...],
    log_prefix: str,
    logger: logging.Logger,
    session_id: str | None = None,
) -> GenResult:
    sp = _sampling_params(session, body, max_token_keys=max_token_keys, stop_keys=stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] prompt exceeds max_context_tokens (%d >= %d)",
                log_prefix,
                len(prompt_ids),
                session.max_context_tokens,
            )
            return GenResult(output_ids=[], output_log_probs=[], finish_reason="length", output_text="")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    sglang_url = app[SGLANG_URL_KEY]
    rid = uuid.uuid4().hex
    headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": prompt_ids,
                "sampling_params": sp,
                "return_logprob": True,
            },
            headers=headers,
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            data = await r.json(content_type=None)
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    tok = app[TOKENIZER_KEY] if TOKENIZER_KEY in app else None
    output_text = tok.decode(output_ids, skip_special_tokens=False) if (tok is not None and output_ids) else ""
    return GenResult(
        output_ids=output_ids,
        output_log_probs=output_log_probs,
        finish_reason=finish,
        output_text=output_text,
    )


async def shutdown_session_tasks(
    sid: str,
    closed: set[str],
    inflight: dict[str, set[asyncio.Task]],
    *,
    wait_timeout: float = 5.0,
) -> None:
    closed.add(sid)
    tasks = [t for t in inflight.pop(sid, ()) if not t.done()]
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def ok_response(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})
