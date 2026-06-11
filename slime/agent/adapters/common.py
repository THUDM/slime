"""Shared adapter primitives for token-capturing agent rollouts.

A protocol adapter (Anthropic / OpenAI) is a thin subclass of
:class:`BaseAdapter`: it fills in the wire-specific hooks
(:meth:`~BaseAdapter._register_routes`, :meth:`~BaseAdapter._session_id`,
:meth:`~BaseAdapter._translate`, :meth:`~BaseAdapter._build`,
:meth:`~BaseAdapter._respond`, optionally :meth:`~BaseAdapter._preprocess_body`)
plus four class attributes (``logger``, ``log_prefix``, ``max_token_keys``,
``stop_keys``). Everything else -- session lifecycle, the per-sid turn cap, the
inflight-task bookkeeping, the one-turn :meth:`~BaseAdapter._run_turn` pipeline,
the ``on_turn_appended`` hook -- is inherited, so the adapters stay in lockstep.

:mod:`slime.agent.adapters.anthropic` is the canonical template to copy when
adding a harness; :func:`flatten_content`, :func:`tool_call_dict` and
:func:`manager_finish_reason` cover the parts both protocols do identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from collections.abc import Callable
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.parsing import parse_model_output
from slime.agent.trajectory_manager import TrajectoryManager, TurnRecord


__all__ = ["TurnRecord"]


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults, context budget, request lock.

    Trajectory state lives in ``BaseAdapter.manager`` (one shared
    TrajectoryManager keyed by sid across all sessions), not here.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


@dataclasses.dataclass
class Reply:
    """Output of an adapter's ``_build``, consumed by ``_run_turn``.

    ``manager_message`` / ``finish_reason`` feed ``record_turn`` + the
    ``on_turn_appended`` hook (protocol-neutral). ``wire`` is opaque to the
    pipeline -- only the adapter's own ``_respond`` understands it.
    ``skip_append`` drops a turn from the trajectory (e.g. cc title-generation
    meta requests) while still firing the hook.
    """

    manager_message: dict
    finish_reason: str
    wire: Any
    skip_append: bool = False


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


# =============================================================================
# Shared translation helpers (used by every adapter's wire translation)
# =============================================================================


def flatten_content(c: Any) -> str:
    """Flatten a wire content value into a chat-template string.

    Recursive and protocol-neutral (handles Anthropic and OpenAI block shapes).
    NOTE: a non-list value (str / dict / other) returns ``str(c)`` unchanged --
    the per-block handling only applies when iterating an actual content list.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    return "\n".join(p for p in parts if p)


def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call for every adapter's ``manager_message``,
    so ``TrajectoryManager.node_match_key`` hashes a sampled leaf and its
    replayed echo identically.

    ``arguments`` stays a dict (NOT a JSON string): the same list is fed to the
    chat template (Qwen3's ``arguments | items`` needs a mapping -- a string
    raises "Can only get item pairs from a mapping"), and node_match_key's
    ``json.dumps(sort_keys=True)`` hashes equivalent dicts identically. The
    wire-only ``tool_use`` / ``call`` id is dropped so leaf and echo match.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


def manager_finish_reason(tool_uses: list[dict], raw_finish: str) -> str:
    """Finish reason stored on the manager turn (protocol-neutral): a turn with
    tool calls is ``tool_calls`` regardless of the raw sglang finish."""
    return "tool_calls" if tool_uses else (raw_finish or "stop")


class BaseAdapter:
    """Base HTTP adapter: session lifecycle + the shared one-turn pipeline.

    See the module docstring for the four class attributes and five hooks a
    subclass must supply; everything else is inherited.
    """

    session_cls: type = Session
    logger: logging.Logger = logging.getLogger(__name__)
    log_prefix: str = "adapter"
    # sglang sampling-param extraction: which body keys cap max_new_tokens and
    # carry stop sequences, in priority order.
    max_token_keys: tuple[str, ...] = ()
    stop_keys: tuple[str, ...] = ()
    manager: Any

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        on_turn_appended: Callable[..., None] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)

        # ONE manager shared across all sids; per-sid trees live inside.
        # ``fork_threshold_tokens is None`` means "caller did not specify" ->
        # let TrajectoryManager use its own assistant-rewrite merge threshold.
        mgr_kwargs: dict[str, int] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)

        # Optional debug hook invoked after each turn (appended or skipped).
        # Signature: (sid, prompt_messages, tools, response_message,
        #             prompt_ids, response_ids, finish_reason) -> None.
        # Exceptions are swallowed; never block the HTTP response.
        self.on_turn_appended: Callable[..., None] | None = on_turn_appended
        # Per-sid turn cap; None disables. When set, the turn handler returns
        # 429 once a sid has made this many turns, so a runaway agent exits
        # cleanly instead of burning the whole token budget.
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}

        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- wire hooks (subclass overrides) -------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register the protocol's POST route(s); bind ``self._run_turn``."""
        raise NotImplementedError

    def _session_id(self, request: web.Request, body: dict) -> str:
        raise NotImplementedError

    def _preprocess_body(self, body: dict) -> None:
        """Mutate the parsed body in place before sid resolution (default no-op)."""

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Return ``(chat_messages, tools_schema)`` from the wire body."""
        raise NotImplementedError

    def _build(self, parsed, raw_finish: str, translated: list[dict], tools_schema: list[dict] | None) -> Reply:
        """Pack parsed model output into a :class:`Reply`."""
        raise NotImplementedError

    async def _respond(
        self,
        request: web.Request,
        body: dict,
        reply: Reply,
        in_tok: int,
        out_tok: int,
        stream: bool,
    ) -> web.StreamResponse:
        raise NotImplementedError

    # -- session lifecycle ---------------------------------------------------

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        """Register a fresh per-sid :class:`Session`; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        session = self.store[sid] = self.session_cls()
        session.sampling_defaults = dict(sampling_defaults or {})
        session.max_context_tokens = int(max_context_tokens or 0)

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        """Mark a sid closed and drain its in-flight turn tasks."""
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample=None,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list:
        """Drain a session's trajectory into fully-formed Sample objects.

        Waits out in-flight requests for ``sid``, linearises the per-sid tree
        via ``TrajectoryManager.get_trajectory``, then decodes each sample's
        trained tail into ``.response`` (the manager is tokenizer-free, so the
        adapter that owns the tokenizer fills this in). Idempotent -- a second
        call for an already-popped sid returns ``[]``.
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        # Drop the per-sid adapter Session; the trajectory itself lives in the
        # manager's per-sid tree and is popped by get_trajectory(drop=True).
        self.store.pop(sid, None)
        samples = self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
        )
        for s in samples:
            rlen = int(s.response_length or 0)
            s.response = (
                self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
            )
        return samples

    # -- shared request pipeline ---------------------------------------------

    def _check_turn_cap(self, sid: str) -> web.Response | None:
        """Enforce ``max_turns_per_sid``; return a 429 response once exceeded.

        Increments the per-sid counter as a side effect when under the cap.
        """
        cap = self.max_turns_per_sid
        if cap is None:
            return None
        prior = self._sid_turn_count.get(sid, 0)
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
        self._sid_turn_count[sid] = prior + 1
        return None

    def _fire_hook(
        self, sid, translated, tools_schema, manager_message, prompt_ids, output_ids, finish_reason
    ) -> None:
        hook = self.on_turn_appended
        if hook is None:
            return
        try:
            hook(sid, translated, tools_schema, manager_message, prompt_ids, output_ids, finish_reason)
        except Exception:
            self.logger.exception("on_turn_appended hook failed (sid=%s)", sid)

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One full agent turn: translate -> sglang -> parse -> append -> respond.

        The wire-specific steps are delegated to the subclass hooks; everything
        else (sid resolution, closed/cap guards, the per-sid serialisation
        lock, inflight tracking, record_turn, the hook) is identical across
        protocols and lives here.
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:  # session drained; refuse stragglers
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, self.session_cls())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        try:
            async with s.lock:  # same sid -> serialized
                translated, tools_schema = self._translate(body)
                prompt_ids = _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

                turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)

                raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
                parsed = parse_model_output(
                    raw_output,
                    tools_schema=tools_schema,
                    tool_parser_name=self.tool_parser,
                    reasoning_parser_name=self.reasoning_parser,
                )
                reply = self._build(parsed, turn.finish_reason, translated, tools_schema)
                turn = dataclasses.replace(turn, finish_reason=reply.finish_reason)

                if reply.skip_append:
                    # Meta request (e.g. cc title-generation): keep it out of
                    # the tree / RL samples, but still fire the hook below so
                    # per-turn debug dumps keep landing on disk.
                    self.logger.info("skipping record_turn (sid=%s)", sid)
                else:
                    try:
                        self.manager.record_turn(
                            sid,
                            turn=turn,
                            prompt_messages=translated,
                            response_message=reply.manager_message,
                            metadata={"sid": sid},
                        )
                    except Exception:
                        self.logger.exception("record_turn(sid=%s) failed", sid)

                self._fire_hook(
                    sid,
                    translated,
                    tools_schema,
                    reply.manager_message,
                    prompt_ids,
                    turn.output_ids,
                    reply.finish_reason,
                )
                in_tok, out_tok = len(prompt_ids), len(turn.output_ids)

            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")
            return await self._respond(request, body, reply, in_tok, out_tok, stream)
        finally:
            self.inflight.get(sid, set()).discard(task)


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
    *,
    adapter: BaseAdapter,
    session_id: str | None = None,
) -> TurnRecord:
    """POST one turn to sglang ``/generate`` and pack the reply into a TurnRecord.

    Module-level (not a method) so tests can ``monkeypatch.setattr(common,
    "call_sglang_generate", ...)``.
    """
    logger = adapter.logger
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] prompt exceeds max_context_tokens (%d >= %d)",
                adapter.log_prefix,
                len(prompt_ids),
                session.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    sglang_url = adapter.sglang_url
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
        # Free the sglang slot eagerly on client cancel/timeout; otherwise the
        # orphaned generation keeps occupying KV until it hits its own length cap.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    )


async def _health(request: web.Request) -> web.Response:
    """Handler for /healthz and /v1/models readiness probes."""
    return web.json_response({"ok": True})
