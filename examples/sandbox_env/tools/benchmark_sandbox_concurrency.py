#!/usr/bin/env python3
"""Benchmark Inspire sandbox concurrency for SWE-rebench scaffold templates.

The benchmark deliberately separates the expensive rollout path into stages:

* create: Sandbox.create + a trivial command.
* preflight: create + scaffold/wstunnel readiness checks.
* tunnel: preflight + sandbox wstunnel + host reverse tunnel + /v1/models.
* nex: tunnel + one OpenAI chat request through the tunnel to the Nex proxy.
* agent: tunnel + one agent CLI run, using Nex as the model backend.

This makes it easier to tell whether saturation comes from sandbox allocation,
sandbox CPU/runtime startup, wstunnel, or model-service traffic.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import json
import os
import shlex
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


TOOLS_DIR = Path(__file__).resolve().parent
SANDBOX_ENV_DIR = TOOLS_DIR.parent
SLIME_ROOT = SANDBOX_ENV_DIR.parents[1]
WORKSPACE_ROOT = SLIME_ROOT.parent
AVALANCHE_ROOT = WORKSPACE_ROOT.parent
SHARE_WORKSPACE = AVALANCHE_ROOT / "share_workspace"
INSPIRE_SDK = AVALANCHE_ROOT / ".local" / "share" / "inspire_sandbox_site_packages"

for candidate in (SLIME_ROOT, SHARE_WORKSPACE, INSPIRE_SDK):
    value = str(candidate)
    if candidate.exists() and value not in sys.path:
        sys.path.insert(0, value)


DEFAULT_MANIFEST = SANDBOX_ENV_DIR / "data_output" / "swe_rebench_scaffold_template_success.respec_gc2.jsonl"
DEFAULT_OUTPUT_ROOT = SANDBOX_ENV_DIR / "data_output" / "sandbox_concurrency_benchmark"
DEFAULT_PROMPT = (
    "This is a sandbox concurrency probe. Inspect the repository briefly and finish. "
    "Do not make code changes."
)
MODE_ORDER = ("create", "preflight", "tunnel", "nex", "agent")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def monotonic() -> float:
    return time.perf_counter()


def parse_levels(raw: str) -> list[int]:
    levels = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not levels:
        raise ValueError("--concurrency-levels must include at least one integer")
    for level in levels:
        if level <= 0:
            raise ValueError("concurrency levels must be positive")
    return levels


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            alias = str(row.get("inspire_template") or row.get("template_alias") or "").strip()
            if not alias:
                continue
            if not row.get("docker_image_default_user"):
                continue
            if not isinstance(row.get("docker_image_env"), dict):
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no usable template rows in {path}")
    return rows


def repo_workdir(row: dict[str, Any]) -> str:
    explicit = str(row.get("repo_workdir") or "").strip()
    if explicit:
        return explicit
    repo = str(row.get("repo") or "").strip()
    if "/" not in repo:
        raise RuntimeError(f"cannot derive repo workdir from repo={repo!r}")
    return f"/{repo.split('/', 1)[1]}"


def template_alias(row: dict[str, Any]) -> str:
    alias = str(row.get("inspire_template") or row.get("template_alias") or "").strip()
    if not alias:
        raise RuntimeError(f"missing template alias for instance_id={row.get('instance_id')!r}")
    return alias


def sandbox_user(row: dict[str, Any]) -> str | None:
    return str(row.get("docker_image_default_user") or "").strip() or None


def build_sandbox_env(row: dict[str, Any], *, model_port: int, prompt: str, model_name: str, api_key: str) -> dict[str, str]:
    raw_env = row.get("docker_image_env")
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}
    env.update(
        {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{model_port}/v1",
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL": model_name,
            "SWE_INSTANCE_ID": str(row.get("instance_id") or ""),
            "SWE_PROMPT_B64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
            "SWE_PROTOCOL_ROOT": str(row.get("agentic_protocol_root") or "/__avaeval_agentic_protocol_v1__"),
            "SWE_WSTUNNEL_BIN": str(row.get("agentic_wstunnel_bin") or "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"),
            "SWE_NODE_BIN": str(row.get("agentic_node_bin") or "/__avaeval_agentic_protocol_v1__/linux/bin/node"),
        }
    )
    return env


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def summarize_durations(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def summarize_level(records: list[dict[str, Any]], *, level: int, elapsed_s: float, mode: str) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok"]
    errors: dict[str, int] = {}
    for record in records:
        if record.get("status") == "ok":
            continue
        etype = str(record.get("error_type") or "Unknown")
        errors[etype] = errors.get(etype, 0) + 1
    return {
        "mode": mode,
        "concurrency": level,
        "submitted": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "elapsed_s": elapsed_s,
        "throughput_per_s": (len(records) / elapsed_s) if elapsed_s > 0 else None,
        "errors": errors,
        "total_s": summarize_durations(records, "total_s"),
        "create_s": summarize_durations(records, "create_s"),
        "preflight_s": summarize_durations(records, "preflight_s"),
        "prepare_workspace_s": summarize_durations(records, "prepare_workspace_s"),
        "tunnel_s": summarize_durations(records, "tunnel_s"),
        "nex_s": summarize_durations(records, "nex_s"),
        "agent_s": summarize_durations(records, "agent_s"),
        "kill_s": summarize_durations(records, "kill_s"),
    }


class NexProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 4096
    proxy: "NexProxy"


class NexProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def proxy(self) -> "NexProxy":
        return self.server.proxy  # type: ignore[attr-defined,return-value]

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/v1/models":
            self._send_json(
                200,
                {"object": "list", "data": [{"id": self.proxy.model_name, "object": "model", "created": 0}]},
            )
            return
        self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        length = int(self.headers.get("content-length") or "0")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if path == "/v1/chat/completions":
                payload = self.proxy.chat_completions(body)
                self._send_chat(payload, stream=bool(body.get("stream")))
                return
            self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})
        except Exception as exc:
            self._send_json(500, {"error": {"message": str(exc), "type": exc.__class__.__name__}})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_chat(self, payload: dict[str, Any], *, stream: bool) -> None:
        if not stream:
            self._send_json(200, payload)
            return
        choice = payload["choices"][0]
        base = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
        }
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": choice["message"], "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}]},
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


@dataclass
class NexProxy:
    base_url: str
    api_key: str
    model_name: str
    timeout: float
    max_tokens: int
    relay_agent_body: bool
    forward_tools: bool
    server: NexProxyServer | None = field(default=None, init=False)
    thread: threading.Thread | None = field(default=None, init=False)

    @property
    def local_base_url(self) -> str:
        if self.server is None:
            raise RuntimeError("Nex proxy is not running")
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> "NexProxy":
        if self.server is not None:
            return self
        server = NexProxyServer(("127.0.0.1", 0), NexProxyHandler)
        server.proxy = self
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def close(self) -> None:
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self.server = None
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None

    def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        start = monotonic()
        upstream = self._build_upstream_body(body)
        raw = self._post_json("/chat/completions", upstream)
        elapsed = monotonic() - start
        content = self._extract_text(raw)
        if not content.strip():
            content = "Done."
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": raw.get("usage") if isinstance(raw, dict) else None,
            "nex_proxy_elapsed_s": elapsed,
        }

    def _build_upstream_body(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.relay_agent_body:
            upstream = {k: v for k, v in body.items() if k not in {"stream", "model"}}
            if not self.forward_tools:
                upstream.pop("tools", None)
                upstream.pop("tool_choice", None)
            upstream["messages"] = body.get("messages") or [{"role": "user", "content": "Respond with Done."}]
        else:
            upstream = {"messages": [{"role": "user", "content": "Respond with the single word Done."}]}
        upstream["model"] = self.model_name
        upstream["stream"] = False
        upstream["max_tokens"] = self.max_tokens
        upstream["temperature"] = 0
        return upstream

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            url,
            data=raw,
            headers={
                "authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Nex HTTP {exc.code}: {detail[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Nex request failed: {exc}") from exc

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            choice = payload.get("choices", [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        except Exception:
            return ""
        return ""


@dataclass(frozen=True)
class ExternalOpenAIEndpoint:
    local_base_url: str
    model_name: str
    api_key: str

    def close(self) -> None:
        return None


async def run_command(
    sandbox: Any,
    command: str,
    *,
    timeout: float,
    user: str | None,
    cwd: str | None = None,
    envs: dict[str, str] | None = None,
) -> Any:
    def _run() -> Any:
        return sandbox.commands.run(command, timeout=timeout, user=user, cwd=cwd, envs=envs)

    return await asyncio.to_thread(_run)


async def create_sandbox(row: dict[str, Any], env: dict[str, str], *, timeout: int) -> Any:
    from inspire_sandbox import Sandbox

    return await asyncio.to_thread(
        Sandbox.create,
        template=template_alias(row),
        timeout=timeout,
        envs=env,
        network={"allow_public_traffic": True},
    )


async def start_sandbox_wstunnel(sandbox: Any, *, wstunnel_bin: str, port: int, user: str | None) -> Any:
    command = (
        f"test -x {shlex.quote(wstunnel_bin)} && "
        f"{shlex.quote(wstunnel_bin)} server {shlex.quote(f'ws://0.0.0.0:{port}')} "
        f">/tmp/swe-concurrency-wstunnel-server.log 2>&1"
    )

    def _run() -> Any:
        return sandbox.commands.run(command, background=True, timeout=0, request_timeout=120, user=user)

    return await asyncio.to_thread(_run)


async def start_host_tunnel(
    *,
    sandbox: Any,
    sandbox_wstunnel_port: int,
    sandbox_model_port: int,
    host_proxy_base_url: str,
    log_path: Path,
) -> Any:
    from examples.sandbox_env.sandbox_runtime import (
        host_wstunnel_executable,
        proxy_endpoint_from_base_url,
        start_host_reverse_tunnel,
    )

    host, port = proxy_endpoint_from_base_url(host_proxy_base_url)
    url = f"wss://{sandbox.get_host(sandbox_wstunnel_port)}"
    return await asyncio.to_thread(
        start_host_reverse_tunnel,
        sandbox_server_url=url,
        host_model_host=host,
        host_model_port=port,
        sandbox_model_port=sandbox_model_port,
        log_path=log_path,
        live_log=None,
    )


async def run_nex_probe(
    sandbox: Any,
    *,
    node_bin: str,
    timeout: float,
    user: str | None,
    request_count: int,
    max_tokens: int,
    history_growth: bool,
) -> Any:
    script = r"""
const requestCount = Number(process.env.SWE_PROBE_REQUEST_COUNT || "1");
const maxTokens = Number(process.env.SWE_PROBE_MAX_TOKENS || "16");
const historyGrowth = process.env.SWE_PROBE_HISTORY_GROWTH === "1";
const url = `${process.env.OPENAI_BASE_URL}/chat/completions`;
let messages = [{role: "user", content: "Respond with OK only."}];
let totalChars = 0;
for (let i = 0; i < requestCount; i++) {
  const body = {
    model: process.env.OPENAI_MODEL,
    messages,
    max_tokens: maxTokens,
    temperature: 0,
    stream: false
  };
  const started = Date.now();
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": `Bearer ${process.env.OPENAI_API_KEY || "x"}`
    },
    body: JSON.stringify(body)
  });
  const text = await res.text();
  const elapsedMs = Date.now() - started;
  if (!res.ok) {
    console.error(`request ${i + 1}/${requestCount} failed after ${elapsedMs}ms`);
    console.error(text);
    process.exit(1);
  }
  totalChars += text.length;
  let content = "OK";
  try {
    const parsed = JSON.parse(text);
    content = parsed?.choices?.[0]?.message?.content || "OK";
  } catch {
    content = "OK";
  }
  console.log(JSON.stringify({request: i + 1, elapsed_ms: elapsedMs, response_chars: text.length}).slice(0, 200));
  if (historyGrowth) {
    messages.push({role: "assistant", content});
    messages.push({role: "user", content: `Continue. This is synthetic SWE turn ${i + 2}; respond with OK only.`});
  }
}
console.log(JSON.stringify({requests: requestCount, total_response_chars: totalChars, messages: messages.length}));
"""
    command = f"{shlex.quote(node_bin)} --input-type=module -e {shlex.quote(script)}"
    return await run_command(
        sandbox,
        command,
        timeout=timeout,
        user=user,
        envs={
            "SWE_PROBE_REQUEST_COUNT": str(max(1, request_count)),
            "SWE_PROBE_MAX_TOKENS": str(max(1, max_tokens)),
            "SWE_PROBE_HISTORY_GROWTH": "1" if history_growth else "0",
        },
    )


def models_probe_command(node_bin: str) -> str:
    return (
        f"{shlex.quote(node_bin)} "
        "--input-type=module -e "
        + shlex.quote(
            "const r=await fetch(process.env.OPENAI_BASE_URL + '/models');"
            "if(!r.ok){console.error(await r.text());process.exit(1)};"
            "console.log((await r.text()).slice(0,200));"
        )
    )


async def wait_for_model_tunnel(
    sandbox: Any,
    *,
    node_bin: str,
    timeout_s: float,
    interval_s: float,
    command_timeout_s: float,
    user: str | None,
) -> tuple[Any, int]:
    deadline = monotonic() + timeout_s
    attempts = 0
    last_error = ""
    while monotonic() < deadline:
        attempts += 1
        try:
            result = await run_command(
                sandbox,
                models_probe_command(node_bin),
                timeout=command_timeout_s,
                user=user,
            )
            if int(getattr(result, "exit_code", 0) or 0) == 0:
                return result, attempts
            last_error = str(getattr(result, "stderr", "") or getattr(result, "stdout", "") or "")
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(interval_s)
    raise RuntimeError(f"model tunnel did not become ready after {attempts} attempts: {last_error[:500]}")


async def run_agent_probe(
    sandbox: Any,
    row: dict[str, Any],
    *,
    harness: str,
    max_turns: int,
    model_base_url: str,
    model_key: str,
    model_name: str,
    timeout: float,
    user: str | None,
) -> Any:
    from agentic_protocol.command_factory.abc import ModelEndpoint
    from agentic_protocol.command_factory.registry import resolve_agent_command_factory

    factory = resolve_agent_command_factory(harness)
    endpoint = ModelEndpoint(base_url=model_base_url, api_key=model_key, model_name=model_name)
    agent_script = factory.runtime_command(
        max_turns=max_turns,
        cli_args=factory.model_cli_args(endpoint),
        cwd=repo_workdir(row),
    )
    return await run_command(
        sandbox,
        f"bash -lc {shlex.quote(agent_script)}",
        timeout=timeout,
        user=user,
        cwd=repo_workdir(row),
    )


async def preflight_scaffold(sandbox: Any, row: dict[str, Any], *, harness: str, timeout: float, user: str | None) -> Any:
    from agentic_protocol.command_factory.registry import resolve_agent_command_factory

    factory = resolve_agent_command_factory(harness)
    wstunnel_bin = str(row.get("agentic_wstunnel_bin") or "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel")
    command = f"test -x {shlex.quote(wstunnel_bin)} && {factory.readiness_command()}"
    return await run_command(sandbox, command, timeout=timeout, user=user)


async def prepare_workspace(sandbox: Any, row: dict[str, Any], *, timeout: float, user: str | None) -> Any:
    workdir = repo_workdir(row)
    base_commit = str(row.get("base_commit") or "").strip()
    if not base_commit:
        raise RuntimeError(f"missing base_commit for instance_id={row.get('instance_id')!r}")
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(workdir)}",
            f"git cat-file -e {shlex.quote(base_commit)}^{{commit}}",
            f"git reset --hard {shlex.quote(base_commit)}",
            "git clean -fd",
            "git status --short",
        ]
    )
    return await run_command(
        sandbox,
        f"bash -lc {shlex.quote(script)}",
        timeout=timeout,
        user=user,
        cwd=workdir,
    )


async def run_one(
    *,
    sample_idx: int,
    level: int,
    rows: list[dict[str, Any]],
    mode: str,
    args: argparse.Namespace,
    output_dir: Path,
    model_endpoint: NexProxy | ExternalOpenAIEndpoint | None,
) -> dict[str, Any]:
    row = rows[sample_idx % len(rows)]
    prompt = args.prompt
    model_name = str(getattr(model_endpoint, "model_name", "") or os.environ.get("NEX_MODEL_NAME", "nex/nex-n1"))
    model_key = str(getattr(model_endpoint, "api_key", "") or "sandbox-concurrency-probe")
    env = build_sandbox_env(row, model_port=args.model_port, prompt=prompt, model_name=model_name, api_key=model_key)
    record: dict[str, Any] = {
        "run_id": args.run_id,
        "sample_idx": sample_idx,
        "concurrency": level,
        "mode": mode,
        "instance_id": row.get("instance_id"),
        "template_alias": template_alias(row),
        "repo": row.get("repo"),
        "started_at": utc_now(),
    }
    sandbox = None
    wstunnel_handle = None
    host_tunnel_proc = None
    t0 = monotonic()
    user = sandbox_user(row)
    try:
        t = monotonic()
        sandbox = await create_sandbox(row, env, timeout=args.sandbox_timeout)
        record["create_s"] = monotonic() - t
        record["sandbox_id"] = getattr(sandbox, "sandbox_id", "")

        t = monotonic()
        trivial = await run_command(sandbox, args.create_probe_command, timeout=args.command_timeout, user=user)
        record["create_probe_s"] = monotonic() - t
        record["create_probe_exit_code"] = int(getattr(trivial, "exit_code", 0) or 0)
        if int(record["create_probe_exit_code"]) != 0:
            raise RuntimeError(f"create probe failed: {str(getattr(trivial, 'stderr', '') or '')[:500]}")

        if MODE_ORDER.index(mode) >= MODE_ORDER.index("preflight"):
            t = monotonic()
            preflight = await preflight_scaffold(
                sandbox,
                row,
                harness=args.harness,
                timeout=args.command_timeout,
                user=user,
            )
            record["preflight_s"] = monotonic() - t
            record["preflight_exit_code"] = int(getattr(preflight, "exit_code", 0) or 0)
            if int(record["preflight_exit_code"]) != 0:
                raise RuntimeError(f"preflight failed: {str(getattr(preflight, 'stderr', '') or '')[:500]}")

        if args.prepare_workspace or (mode == "agent" and not args.skip_prepare_workspace):
            t = monotonic()
            prepared = await prepare_workspace(
                sandbox,
                row,
                timeout=args.command_timeout,
                user=user,
            )
            record["prepare_workspace_s"] = monotonic() - t
            record["prepare_workspace_exit_code"] = int(getattr(prepared, "exit_code", 0) or 0)
            if int(record["prepare_workspace_exit_code"]) != 0:
                raise RuntimeError(f"prepare workspace failed: {str(getattr(prepared, 'stderr', '') or '')[:500]}")

        if MODE_ORDER.index(mode) >= MODE_ORDER.index("tunnel"):
            if model_endpoint is None:
                raise RuntimeError("a model endpoint is required for tunnel/nex/agent modes")
            t = monotonic()
            wstunnel_handle = await start_sandbox_wstunnel(
                sandbox,
                wstunnel_bin=str(row.get("agentic_wstunnel_bin") or "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"),
                port=args.wstunnel_port,
                user=user,
            )
            await asyncio.sleep(args.tunnel_wait_s)
            tunnel_log = output_dir / "wstunnel" / f"sample_{sample_idx}.log"
            tunnel_log.parent.mkdir(parents=True, exist_ok=True)
            host_tunnel_proc = await start_host_tunnel(
                sandbox=sandbox,
                sandbox_wstunnel_port=args.wstunnel_port,
                sandbox_model_port=args.model_port,
                host_proxy_base_url=model_endpoint.local_base_url,
                log_path=tunnel_log,
            )
            await asyncio.sleep(args.tunnel_wait_s)
            if host_tunnel_proc.poll() is not None:
                raise RuntimeError("host wstunnel client exited early")
            models, attempts = await wait_for_model_tunnel(
                sandbox,
                node_bin=str(row.get("agentic_node_bin") or "/__avaeval_agentic_protocol_v1__/linux/bin/node"),
                timeout_s=args.tunnel_ready_timeout_s,
                interval_s=args.tunnel_ready_interval_s,
                command_timeout_s=args.command_timeout,
                user=user,
            )
            record["tunnel_s"] = monotonic() - t
            record["tunnel_ready_attempts"] = attempts
            record["models_exit_code"] = int(getattr(models, "exit_code", 0) or 0)
            if int(record["models_exit_code"]) != 0:
                raise RuntimeError(f"model tunnel probe failed: {str(getattr(models, 'stderr', '') or '')[:500]}")

        if mode == "nex":
            t = monotonic()
            result = await run_nex_probe(
                sandbox,
                node_bin=str(row.get("agentic_node_bin") or "/__avaeval_agentic_protocol_v1__/linux/bin/node"),
                timeout=args.agent_timeout,
                user=user,
                request_count=args.chat_requests_per_sandbox,
                max_tokens=args.chat_max_tokens,
                history_growth=args.chat_history_growth,
            )
            record["nex_s"] = monotonic() - t
            record["chat_requests_per_sandbox"] = args.chat_requests_per_sandbox
            record["chat_history_growth"] = bool(args.chat_history_growth)
            record["nex_exit_code"] = int(getattr(result, "exit_code", 0) or 0)
            if int(record["nex_exit_code"]) != 0:
                raise RuntimeError(f"nex probe failed: {str(getattr(result, 'stderr', '') or '')[:500]}")

        if mode == "agent":
            t = monotonic()
            result = await run_agent_probe(
                sandbox,
                row,
                harness=args.harness,
                max_turns=args.max_turns,
                model_base_url=f"http://127.0.0.1:{args.model_port}/v1",
                model_key=model_key,
                model_name=model_name,
                timeout=args.agent_timeout,
                user=user,
            )
            record["agent_s"] = monotonic() - t
            record["agent_exit_code"] = int(getattr(result, "exit_code", 0) or 0)
            record["agent_stdout_preview"] = str(getattr(result, "stdout", "") or "")[: args.preview_chars]
            record["agent_stderr_preview"] = str(getattr(result, "stderr", "") or "")[: args.preview_chars]
            if args.require_agent_success and int(record["agent_exit_code"]) != 0:
                raise RuntimeError(f"agent failed with exit={record['agent_exit_code']}")

        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "failed"
        record["error_type"] = exc.__class__.__name__
        record["error"] = str(exc)[:2000]
    finally:
        if host_tunnel_proc is not None and host_tunnel_proc.poll() is None:
            host_tunnel_proc.terminate()
            try:
                await asyncio.to_thread(host_tunnel_proc.wait, timeout=3)
            except Exception:
                host_tunnel_proc.kill()
        if wstunnel_handle is not None:
            try:
                await asyncio.to_thread(wstunnel_handle.kill)
            except Exception:
                pass
        if sandbox is not None and not args.keep_sandboxes:
            t = monotonic()
            try:
                await asyncio.to_thread(sandbox.kill)
                record["kill_s"] = monotonic() - t
            except Exception as exc:
                record["kill_error"] = str(exc)[:1000]
        record["total_s"] = monotonic() - t0
        record["finished_at"] = utc_now()
    return record


async def run_level(
    *,
    level: int,
    rows: list[dict[str, Any]],
    mode: str,
    args: argparse.Namespace,
    output_dir: Path,
    model_endpoint: NexProxy | ExternalOpenAIEndpoint | None,
) -> dict[str, Any]:
    records_path = output_dir / f"records_c{level}_{mode}.jsonl"
    start = monotonic()
    print(f"[{utc_now()}] level start mode={mode} concurrency={level}", flush=True)
    tasks = [
        asyncio.create_task(
            run_one(
                sample_idx=sample_idx,
                level=level,
                rows=rows,
                mode=mode,
                args=args,
                output_dir=output_dir,
                model_endpoint=model_endpoint,
            )
        )
        for sample_idx in range(level)
    ]

    records: list[dict[str, Any]] = []
    next_progress = monotonic() + args.progress_interval_s
    with records_path.open("w", encoding="utf-8") as f:
        for future in asyncio.as_completed(tasks):
            record = await future
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            now = monotonic()
            if now >= next_progress or len(records) == level:
                ok = sum(1 for r in records if r.get("status") == "ok")
                print(
                    f"[{utc_now()}] progress mode={mode} concurrency={level} "
                    f"completed={len(records)}/{level} ok={ok} failed={len(records) - ok}",
                    flush=True,
                )
                next_progress = now + args.progress_interval_s
    elapsed = monotonic() - start
    summary = summarize_level(records, level=level, elapsed_s=elapsed, mode=mode)
    (output_dir / f"summary_c{level}_{mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[{utc_now()}] level done mode={mode} concurrency={level} ok={summary['ok']} "
        f"failed={summary['failed']} elapsed_s={elapsed:.2f}",
        flush=True,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--mode", choices=MODE_ORDER, default="create")
    parser.add_argument("--concurrency-levels", default="1,8,32,64,128,256,512")
    parser.add_argument("--harness", default=os.environ.get("SWE_AGENT_HARNESS", "qwen_code"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sandbox-timeout", type=int, default=int(os.environ.get("SWE_STARTUP_TIMEOUT", "1800")))
    parser.add_argument("--command-timeout", type=float, default=120)
    parser.add_argument("--agent-timeout", type=float, default=300)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--model-port", type=int, default=int(os.environ.get("SWE_MODEL_PROXY_PORT", "30001")))
    parser.add_argument("--wstunnel-port", type=int, default=int(os.environ.get("SWE_WSTUNNEL_SERVER_PORT", "19090")))
    parser.add_argument("--tunnel-wait-s", type=float, default=float(os.environ.get("SWE_WSTUNNEL_WAIT_SECONDS", "2")))
    parser.add_argument("--tunnel-ready-timeout-s", type=float, default=30)
    parser.add_argument("--tunnel-ready-interval-s", type=float, default=2)
    parser.add_argument("--progress-interval-s", type=float, default=30)
    parser.add_argument(
        "--thread-workers",
        type=int,
        default=0,
        help="Max worker threads for blocking SDK calls; default is the largest requested concurrency.",
    )
    parser.add_argument("--preview-chars", type=int, default=2000)
    parser.add_argument("--create-probe-command", default="true")
    parser.add_argument("--prepare-workspace", action="store_true")
    parser.add_argument("--skip-prepare-workspace", action="store_true")
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument("--require-agent-success", action="store_true")
    parser.add_argument("--nex-timeout", type=float, default=120)
    parser.add_argument("--nex-max-tokens", type=int, default=16)
    parser.add_argument(
        "--chat-requests-per-sandbox",
        type=int,
        default=1,
        help="Sequential chat-completion requests sent from each sandbox in mode=nex.",
    )
    parser.add_argument("--chat-max-tokens", type=int, default=16)
    parser.add_argument(
        "--chat-history-growth",
        action="store_true",
        help="Append assistant/user messages after each synthetic chat request so prompt history grows across turns.",
    )
    parser.add_argument(
        "--external-openai-base-url",
        default=os.environ.get("SGLANG_OPENAI_BASE_URL", ""),
        help="Use an existing OpenAI-compatible endpoint directly, for example http://127.0.0.1:30000/v1.",
    )
    parser.add_argument(
        "--external-openai-api-key",
        default=os.environ.get("SGLANG_OPENAI_API_KEY", "sandbox-concurrency-probe"),
    )
    parser.add_argument(
        "--external-openai-model",
        default=os.environ.get("SGLANG_OPENAI_MODEL", ""),
    )
    parser.add_argument(
        "--relay-agent-body",
        action="store_true",
        help="Forward the agent request body to Nex instead of sending a tiny fixed probe prompt.",
    )
    parser.add_argument("--forward-tools", action="store_true", help="Forward tool schemas to Nex when relaying.")
    return parser


async def main_async() -> int:
    args = build_arg_parser().parse_args()
    levels = parse_levels(args.concurrency_levels)
    thread_workers = args.thread_workers if args.thread_workers > 0 else max(levels)
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=max(1, thread_workers))
    )
    rows = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "mode": args.mode,
                "levels": levels,
                "manifest": str(Path(args.manifest).resolve()),
                "manifest_rows": len(rows),
                "harness": args.harness,
                "chat_requests_per_sandbox": args.chat_requests_per_sandbox,
                "chat_history_growth": bool(args.chat_history_growth),
                "nex_base_url_set": bool(os.environ.get("NEX_BASE_URL")),
                "nex_model_name": os.environ.get("NEX_MODEL_NAME", ""),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    model_endpoint: NexProxy | ExternalOpenAIEndpoint | None = None
    if MODE_ORDER.index(args.mode) >= MODE_ORDER.index("tunnel"):
        external_base_url = str(args.external_openai_base_url or "").strip().rstrip("/")
        if external_base_url:
            model_name = str(args.external_openai_model or "").strip() or external_base_url.rsplit("/", 1)[-1]
            model_endpoint = ExternalOpenAIEndpoint(
                local_base_url=external_base_url,
                model_name=model_name,
                api_key=str(args.external_openai_api_key or "sandbox-concurrency-probe"),
            )
            print(f"[{utc_now()}] external OpenAI endpoint: {model_endpoint.local_base_url}", flush=True)
        else:
            base_url = os.environ.get("NEX_BASE_URL", "").strip()
            api_key = os.environ.get("NEX_API_KEY", "").strip()
            model_name = os.environ.get("NEX_MODEL_NAME", "").strip()
            if not base_url or not api_key or not model_name:
                raise RuntimeError(
                    "NEX_BASE_URL, NEX_API_KEY, and NEX_MODEL_NAME must be set, "
                    "or pass --external-openai-base-url for tunnel/nex/agent modes"
                )
            model_endpoint = NexProxy(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                timeout=args.nex_timeout,
                max_tokens=args.nex_max_tokens,
                relay_agent_body=args.relay_agent_body,
                forward_tools=args.forward_tools,
            ).start()
            print(f"[{utc_now()}] nex proxy started at {model_endpoint.local_base_url}", flush=True)

    summaries: list[dict[str, Any]] = []
    try:
        for level in levels:
            summary = await run_level(
                level=level,
                rows=rows,
                mode=args.mode,
                args=args,
                output_dir=output_dir,
                model_endpoint=model_endpoint,
            )
            summaries.append(summary)
            if args.mode != "create" and summary["failed"] == summary["submitted"]:
                print(
                    f"[{utc_now()}] stopping after full failure at concurrency={level}; "
                    "check records before increasing load",
                    flush=True,
                )
                break
    finally:
        if model_endpoint is not None:
            model_endpoint.close()

    (output_dir / "summary.json").write_text(
        json.dumps({"run_id": args.run_id, "summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{utc_now()}] wrote results to {output_dir}", flush=True)
    return 0 if all(s["failed"] == 0 for s in summaries) else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
