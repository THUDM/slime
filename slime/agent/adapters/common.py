"""Shared adapter primitives for token-capturing agent rollouts."""

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

from slime.agent.trajectory import TokenSegment, TurnRecord


ADAPTER_KEY = web.AppKey("adapter", object)
TOKENIZER_KEY = web.AppKey("tokenizer", object)
SGLANG_URL_KEY = web.AppKey("sglang_url", object)
TOOL_PARSER_KEY = web.AppKey("tool_parser", object)
REASONING_PARSER_KEY = web.AppKey("reasoning_parser", object)
# Pooled per-app SGLang client, registered by sglang_client_context and reused across turns.
SGLANG_CLIENT_KEY = web.AppKey("sglang_client", aiohttp.ClientSession)


@dataclasses.dataclass
class AdapterChain:
    """Protocol-neutral chat chain state used by HTTP adapters."""

    system_hash: str = ""
    chat_messages: list[dict] = dataclasses.field(default_factory=list)
    tools_schema: list[dict] | None = None
    seen_msgs: int = 0
    msg_hashes: list[str] = dataclasses.field(default_factory=list)
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)


class BaseAdapter:
    """Base HTTP adapter with per-instance session lifecycle state."""

    session_cls: type

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
        self.app.cleanup_ctx.append(sglang_client_context)

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
        raise NotImplementedError


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


def render_token_ids(chain: AdapterChain, tokenizer) -> list[int]:
    enc = tokenizer.apply_chat_template(
        chain.chat_messages,
        tools=chain.tools_schema,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


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


async def sglang_client_context(app: web.Application):
    # limit=0: connection concurrency is governed by the rollout scheduler, not the pool.
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, keepalive_timeout=60)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    app[SGLANG_CLIENT_KEY] = aiohttp.ClientSession(timeout=timeout, connector=connector)
    try:
        yield
    finally:
        await app[SGLANG_CLIENT_KEY].close()


async def _post_sglang_generate(
    client: aiohttp.ClientSession,
    sglang_url: str,
    *,
    rid: str,
    prompt_ids: list[int],
    sampling_params: dict,
    headers: dict[str, str] | None,
) -> dict:
    async with client.post(
        f"{sglang_url}/generate",
        json={
            "rid": rid,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        },
        headers=headers,
    ) as r:
        if r.status >= 400:
            text = await r.text()
            raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
        return await r.json(content_type=None)


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
) -> TurnRecord:
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
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    sglang_url = app[SGLANG_URL_KEY]
    client = app[SGLANG_CLIENT_KEY]
    rid = uuid.uuid4().hex
    headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
    try:
        try:
            data = await _post_sglang_generate(
                client, sglang_url, rid=rid, prompt_ids=prompt_ids, sampling_params=sp, headers=headers
            )
        except aiohttp.ClientConnectorError:
            # A connector error is raised before any request bytes reach SGLang, so retrying
            # with the SAME rid cannot double-generate. Errors after the request may have
            # reached the server are NOT retried and fall through to the abort path below.
            logger.warning("[%s] retrying SGLang generate after connector failure", log_prefix)
            data = await _post_sglang_generate(
                client, sglang_url, rid=rid, prompt_ids=prompt_ids, sampling_params=sp, headers=headers
            )
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError):
        try:
            async with client.post(
                f"{sglang_url}/abort_request", json={"rid": rid}, timeout=aiohttp.ClientTimeout(total=5)
            ):
                pass
        except Exception:
            pass
        raise

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
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
