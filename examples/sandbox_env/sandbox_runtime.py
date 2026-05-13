"""Sandbox rollout runtime.

This module keeps the sandbox-facing runtime in one place:

- SWE_* config decoding and sandbox env construction.
- Per-sample artifact/log paths.
- Inspire sandbox command/file helpers and wstunnel orchestration.
- Single-sample lifecycle: create sandbox, run scaffold, extract patch, eval.

It intentionally stays Slime-free except for the model proxy callback object
passed in by ``swe_rollout``.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentic_protocol.command_factory.abc import ModelEndpoint
from agentic_protocol.command_factory.layout import AGENTIC_PROTOCOL_ROOT, LINUX_BIN_DIR
from agentic_protocol.command_factory.registry import (
    canonical_agent_harness_name,
    resolve_agent_command_factory,
)
from agentic_protocol.command_factory.wstunnel import SANDBOX_WSTUNNEL

from .sglang_openai_proxy import SGLangOpenAIProxy


AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
SANDBOX_WSTUNNEL_SERVER_LOG = "/tmp/swe-wstunnel-server.log"


def _stringify_content(content: Any) -> str:
    """Flattens nested message content (str or list of blocks) into a single plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(_stringify_content(item.get("content")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def env_or_arg(args: Any, env_name: str, arg_name: str | None, default: Any) -> Any:
    """Reads a config value preferring the args attribute over the environment variable, falling back to default."""
    if arg_name:
        value = getattr(args, arg_name, None)
        if value not in (None, ""):
            return value
    return os.environ.get(env_name) or default


@dataclass(frozen=True)
class SWERolloutConfig:
    """Runtime configuration for one SWE sandbox rollout worker."""

    agent_harness: str
    protocol_root: str
    wstunnel_bin: str
    node_bin: str
    model_proxy_port: int
    wstunnel_server_port: int
    startup_timeout: int
    wait_timeout: int
    agent_finish_timeout: int
    max_turns: int
    preview_limit: int
    keep_containers: bool
    sandbox_start_retry_times: int
    sandbox_start_retry_interval: float
    tunnel_wait_seconds: float
    openai_model: str
    openai_api_key: str
    log_root: str | None


def decode_swe_rollout_config(args: Any) -> SWERolloutConfig:
    """Builds the SWERolloutConfig dataclass from environment variables and args for the current rollout worker.

    Protocol root and wstunnel binary path come from agentic_protocol's shared
    layout (overridable via the AGENTIC_PROTOCOL_ROOT env var, which the
    layout module reads at import time). The legacy SWE_PROTOCOL_ROOT /
    SWE_WSTUNNEL_BIN args/env knobs are no longer honored — set
    AGENTIC_PROTOCOL_ROOT before launching the trainer if you need a non-default
    root, and rebuild templates against that root.
    """
    root = AGENTIC_PROTOCOL_ROOT
    wstunnel_bin = SANDBOX_WSTUNNEL
    harness = canonical_agent_harness_name(
        str(env_or_arg(args, "SWE_AGENT_HARNESS", "swe_agent_harness", "qwen_code"))
    )
    model_name = str(
        env_or_arg(
            args,
            "SWE_OPENAI_MODEL",
            "swe_openai_model",
            getattr(args, "swe_model_name", None) or "default",
        )
    ).strip() or "default"
    return SWERolloutConfig(
        agent_harness=harness,
        protocol_root=root,
        wstunnel_bin=wstunnel_bin,
        node_bin=f"{LINUX_BIN_DIR}/node",
        model_proxy_port=int(env_or_arg(args, "SWE_MODEL_PROXY_PORT", "swe_model_proxy_port", 30001)),
        wstunnel_server_port=int(env_or_arg(args, "SWE_WSTUNNEL_SERVER_PORT", "swe_wstunnel_server_port", 19090)),
        startup_timeout=int(env_or_arg(args, "SWE_STARTUP_TIMEOUT", "swe_startup_timeout", 10800)),
        wait_timeout=int(env_or_arg(args, "SWE_WAIT_TIMEOUT", "swe_wait_timeout", 10800)),
        agent_finish_timeout=int(env_or_arg(args, "SWE_AGENT_FINISH_TIMEOUT", "swe_agent_finish_timeout", 10800)),
        max_turns=int(env_or_arg(args, "SWE_MAX_TURNS", "swe_max_turns", 50)),
        preview_limit=int(env_or_arg(args, "SWE_OUTPUT_PREVIEW_LIMIT", "swe_output_preview_limit", 12000)),
        keep_containers=str(env_or_arg(args, "SWE_KEEP_CONTAINERS", "swe_keep_containers", "0")) == "1",
        sandbox_start_retry_times=int(
            env_or_arg(args, "SWE_SANDBOX_START_RETRY_TIMES", "swe_sandbox_start_retry_times", 10)
        ),
        sandbox_start_retry_interval=float(
            env_or_arg(args, "SWE_SANDBOX_START_RETRY_INTERVAL", "swe_sandbox_start_retry_interval", 5)
        ),
        tunnel_wait_seconds=float(env_or_arg(args, "SWE_WSTUNNEL_WAIT_SECONDS", None, 3)),
        openai_model=model_name,
        openai_api_key=str(env_or_arg(args, "SWE_OPENAI_API_KEY", "swe_openai_api_key", "swe-rollout")),
        log_root=env_or_arg(args, "SWE_LOG_ROOT", "swe_log_root", None),
    )


def prompt_to_text(prompt: Any) -> str:
    """Converts a prompt (plain string or list of role/content dicts) to a flat text string."""
    if isinstance(prompt, list):
        if (
            len(prompt) == 1
            and isinstance(prompt[0], dict)
            and str(prompt[0].get("role") or "user") == "user"
        ):
            return _stringify_content(prompt[0].get("content"))
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role") or "user")
                content = _stringify_content(item.get("content"))
                if content:
                    parts.append(f"{role}: {content}")
        if parts:
            return "\n\n".join(parts)
    return str(prompt)


def build_sandbox_envs(metadata: dict[str, Any], cfg: SWERolloutConfig, prompt_text: str) -> dict[str, str]:
    """Builds the dict of environment variables injected into the Inspire sandbox container at creation."""
    raw_env = metadata.get("docker_image_env")
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}
    sandbox_model_base_url = f"http://127.0.0.1:{cfg.model_proxy_port}/v1"
    env.update(
        {
            "OPENAI_BASE_URL": sandbox_model_base_url,
            "OPENAI_API_KEY": cfg.openai_api_key,
            "OPENAI_MODEL": cfg.openai_model,
            "SWE_INSTANCE_ID": str(metadata.get("instance_id") or ""),
            "SWE_PROMPT_B64": base64.b64encode(prompt_text.encode("utf-8")).decode("ascii"),
            "SWE_PROTOCOL_ROOT": cfg.protocol_root,
            "SWE_WSTUNNEL_BIN": cfg.wstunnel_bin,
            "SWE_NODE_BIN": cfg.node_bin,
        }
    )
    return env


def build_batch_log_dir(log_root: str | None) -> Path | None:
    """Returns the path for the current rollout batch log directory (log_root/current_batch)."""
    if not log_root:
        return None
    return Path(log_root) / "current_batch"


def prepare_batch_log_dir(log_root: str | None) -> Path | None:
    """Creates (and resets) the batch log directory at the start of each rollout batch."""
    batch_log_dir = build_batch_log_dir(log_root)
    if batch_log_dir is None:
        return None
    shutil.rmtree(batch_log_dir, ignore_errors=True)
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    return batch_log_dir


def build_sample_log_dir(log_root: str | None, *, sample_idx: int) -> Path | None:
    """Returns the per-sample subdirectory (current_batch/sample_N) for structured log output."""
    batch_log_dir = build_batch_log_dir(log_root)
    if batch_log_dir is None:
        return None
    return batch_log_dir / f"sample_{sample_idx}"


def build_live_sandbox_log_path(log_root: str | None, *, sample_idx: int) -> Path | None:
    """Returns the path for the live agent stdout/stderr log file (sandbox/agent_output.log)."""
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is None:
        return None
    sandbox_log_dir = sample_log_dir / "sandbox"
    sandbox_log_dir.mkdir(parents=True, exist_ok=True)
    return sandbox_log_dir / "agent_output.log"


def build_eval_log_path(log_root: str | None, *, sample_idx: int) -> Path | None:
    """Returns the path for the eval script stdout/stderr log file (sandbox/eval_output.log)."""
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is None:
        return None
    sandbox_log_dir = sample_log_dir / "sandbox"
    sandbox_log_dir.mkdir(parents=True, exist_ok=True)
    return sandbox_log_dir / "eval_output.log"


def truncate_text(value: str | None, limit: int) -> str:
    """Truncates text to limit characters for storing previews in metadata without blowing up storage."""
    text = value or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]


def write_sample_artifacts_snapshot(
    *,
    log_root: str | None,
    rollout_id: int,
    sample_idx: int,
    sample_prompt: str,
    metadata: dict[str, Any],
    extra_metadata: dict[str, Any],
    turn_responses: list[str],
    trajectory: list[dict[str, Any]],
    final_messages: list[dict[str, Any]] | None,
    last_response_payload: str | None,
) -> str | None:
    """Writes all per-sample artifacts (prompt, trajectory, messages, metadata) to sample_artifacts.json."""
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is None:
        return None

    sample_log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rollout_id": rollout_id,
        "sample_idx": sample_idx,
        "sandbox_id": extra_metadata.get("sandbox_id"),
        "instance_id": metadata.get("instance_id"),
        "repo": metadata.get("repo"),
        "local_image_name": metadata.get("local_image_name"),
        "prompt": sample_prompt,
        "extra_metadata": extra_metadata,
        "turn_responses": turn_responses,
        "trajectory": trajectory,
        "final_messages": final_messages,
        "last_response_payload": last_response_payload,
    }
    # Write atomically: persist to a sibling tmp file then rename, so concurrent
    # snapshots (or a snapshot racing a reader) cannot observe a half-written or
    # interleaved file.
    target = sample_log_dir / "sample_artifacts.json"
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return str(sample_log_dir)


def bootstrap_inspire_sandbox_path() -> None:
    """Push the shared inspire-sandbox site-packages onto sys.path."""
    raw_root = os.environ.get("INSPIRE_SANDBOX_SITE_PACKAGES", "").strip()
    root = Path(raw_root) if raw_root else AVALANCHE_ROOT / ".local" / "share" / "inspire_sandbox_site_packages"
    candidates = [root, *sorted(root.glob("lib/python*/site-packages"))] if root.exists() else [root]
    for candidate in reversed(candidates):
        value = str(candidate)
        if candidate.exists() and value not in sys.path:
            sys.path.insert(0, value)


class LiveLog:
    """Thread-safe append-only log writer with optional mirroring to another log."""

    def __init__(self, path: Path | None, *, mirror: "LiveLog | None" = None) -> None:
        """Stores the output path and optional mirror log, creating the parent directory if needed."""
        self.path = path
        self.mirror = mirror
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        """Returns the UTC timestamp prefix used in sandbox log lines."""
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_kv(value: Any) -> str:
        """Formats structured event field values for a single-line log entry."""
        if isinstance(value, (list, tuple, dict)):
            return repr(value)
        return str(value)

    def event(self, label: str, **fields: Any) -> None:
        """Appends one timestamped structured event line to the log."""
        parts = [f"{key}={self._format_kv(value)}" for key, value in fields.items() if value is not None]
        suffix = (" " + " ".join(parts)) if parts else ""
        self.raw(f"[{self._timestamp()}] [{label}]{suffix}")

    def write(self, label: str, text: str) -> None:
        """Appends labeled lines ([stdout]/[stderr]) to the sample log file under a thread lock."""
        if self.path is None or not text:
            if self.mirror is not None:
                self.mirror.write(label, text)
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                for line in str(text).splitlines():
                    f.write(f"[{self._timestamp()}] [{label}] {line}\n")
        if self.mirror is not None:
            self.mirror.write(label, text)

    def raw(self, line: str) -> None:
        """Appends a preformatted single log line, mirroring it if configured."""
        if self.path is not None:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(str(line).rstrip("\n") + "\n")
        if self.mirror is not None:
            self.mirror.raw(line)


@dataclass
class SandboxCommandResult:
    """Normalized result object returned from sandbox command execution."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        """Returns combined stdout+stderr text; used for error messages and preview metadata."""
        if self.stdout and self.stderr:
            return self.stdout + "\n" + self.stderr
        return self.stdout or self.stderr


async def run_sandbox_command(
    sandbox: Any,
    command: str,
    *,
    timeout: float | None,
    user: str | None,
    cwd: str | None = None,
    envs: dict[str, str] | None = None,
    log: LiveLog | None = None,
) -> SandboxCommandResult:
    """Run a sandbox command and convert command exits into a result object."""
    from inspire_sandbox import CommandExitException

    merged_envs = {str(k): str(v) for k, v in dict(envs or {}).items()}
    if log is not None:
        log.event("command.start", user=user or "<default>", timeout=timeout, cwd=cwd, command=command)
        log.event("command.env", user=user or "<default>", keys=sorted(merged_envs))

    def _on_stdout(chunk: str) -> None:
        """Streams sandbox stdout chunks into the live command log."""
        if log is not None:
            log.write("command.stream.stdout", str(chunk))

    def _on_stderr(chunk: str) -> None:
        """Streams sandbox stderr chunks into the live command log."""
        if log is not None:
            log.write("command.stream.stderr", str(chunk))

    try:
        result = await asyncio.to_thread(
            sandbox.commands.run,
            command,
            cwd=cwd,
            timeout=timeout,
            user=user,
            envs=merged_envs or None,
            on_stdout=_on_stdout,
            on_stderr=_on_stderr,
        )
        command_result = SandboxCommandResult(
            exit_code=int(getattr(result, "exit_code", 0) or 0),
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
        )
        if log is not None:
            log.event(
                "command.end",
                exit_code=command_result.exit_code,
                stdout_bytes=len(command_result.stdout.encode("utf-8")),
                stderr_bytes=len(command_result.stderr.encode("utf-8")),
            )
            if command_result.stdout:
                log.write("command.end.stdout", command_result.stdout)
            if command_result.stderr:
                log.write("command.end.stderr", command_result.stderr)
        return command_result
    except CommandExitException as exc:
        command_result = SandboxCommandResult(
            exit_code=int(getattr(exc, "exit_code", 1) or 1),
            stdout=str(getattr(exc, "stdout", "") or ""),
            stderr=str(getattr(exc, "stderr", "") or ""),
        )
        if log is not None:
            log.event(
                "command.end",
                exit_code=command_result.exit_code,
                stdout_bytes=len(command_result.stdout.encode("utf-8")),
                stderr_bytes=len(command_result.stderr.encode("utf-8")),
                error=str(getattr(exc, "error", "") or ""),
            )
            if command_result.stdout:
                log.write("command.end.stdout", command_result.stdout)
            if command_result.stderr:
                log.write("command.end.stderr", command_result.stderr)
        return command_result


async def write_sandbox_file(
    sandbox: Any,
    path: str,
    data: bytes,
    *,
    user: str | None,
    log: LiveLog | None = None,
) -> None:
    """Writes raw bytes to a file path inside the sandbox (used to stage patch files before eval)."""
    def _write() -> None:
        """Performs the blocking sandbox file write on a worker thread."""
        sandbox.files.write(path, data, user=user)

    if log is not None:
        log.event("file.write.start", user=user or "<default>", path=path, bytes=len(data))
    await asyncio.to_thread(_write)
    if log is not None:
        log.event("file.write.end", user=user or "<default>", path=path, bytes=len(data))


async def create_sandbox_with_retry(
    *,
    template_alias: str,
    timeout: int,
    envs: dict[str, str],
    retry_times: int,
    retry_interval: float,
    log: LiveLog | None = None,
    image_name: str | None = None,
) -> Any:
    """Creates an Inspire sandbox container from the given template alias, retrying on transient failures."""
    bootstrap_inspire_sandbox_path()
    from inspire_sandbox import Sandbox

    last_exc: Exception | None = None
    for attempt in range(1, retry_times + 1):
        try:
            if log is not None:
                log.event("template", action="using existing", template=template_alias)
                log.event(
                    "sandbox.create",
                    template=template_alias,
                    timeout=timeout,
                    image=image_name or "",
                    attempt=f"{attempt}/{retry_times}",
                    env_keys=sorted(envs),
                )
            sandbox = await asyncio.to_thread(
                Sandbox.create,
                template=template_alias,
                timeout=timeout,
                envs=envs,
                network={"allow_public_traffic": True},
            )
            if log is not None:
                log.event(
                    "sandbox.ready",
                    sandbox_id=getattr(sandbox, "sandbox_id", ""),
                    host_name=getattr(sandbox, "host_name", ""),
                    host_ip=getattr(sandbox, "host_ip", ""),
                )
            return sandbox
        except Exception as exc:
            last_exc = exc
            if log is not None:
                log.event(
                    "sandbox.create.failed",
                    template=template_alias,
                    attempt=f"{attempt}/{retry_times}",
                    error=str(exc),
                )
            if attempt >= retry_times:
                break
            print(
                f"[WARN] Sandbox.create failed for template={template_alias} "
                f"attempt {attempt}/{retry_times}: {exc}; retry in {retry_interval}s",
                file=sys.stderr,
            )
            await asyncio.sleep(retry_interval)
    raise RuntimeError(
        f"Sandbox.create failed for template={template_alias} after {retry_times} attempts: {last_exc}"
    ) from last_exc


def host_wstunnel_executable() -> str:
    """Locates the host-side wstunnel binary by checking known paths and PATH; raises if not found."""
    candidates = [
        AVALANCHE_ROOT / ".sii" / "bin" / "wstunnel",
        AVALANCHE_ROOT / "zf_workspace" / "eval" / ".sii" / "bin" / "wstunnel",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    discovered = shutil.which("wstunnel")
    if discovered:
        return str(Path(discovered).resolve())
    raise FileNotFoundError("host wstunnel executable not found")


def proxy_endpoint_from_base_url(base_url: str) -> tuple[str, int]:
    """Parses the proxy server's (host, port) from its base URL for use in the wstunnel reverse-tunnel command."""
    parsed = urlparse(base_url.rstrip("/"))
    if not parsed.hostname:
        raise ValueError(f"invalid proxy base URL: {base_url!r}")
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "http":
        port = 80
    elif parsed.scheme == "https":
        port = 443
    else:
        raise ValueError(f"unsupported proxy scheme: {parsed.scheme!r}")
    return ("127.0.0.1" if parsed.hostname in {"localhost", "0.0.0.0"} else parsed.hostname, port)


def start_host_reverse_tunnel(
    *,
    sandbox_server_url: str,
    host_model_host: str,
    host_model_port: int,
    sandbox_model_port: int,
    log_path: Path | None,
    live_log: LiveLog | None = None,
) -> subprocess.Popen:
    """Launches the host-side wstunnel client that tunnels the model proxy port into the sandbox."""
    executable = host_wstunnel_executable()
    remote_spec = f"tcp://127.0.0.1:{sandbox_model_port}:{host_model_host}:{host_model_port}"
    cmd = [executable, "client", "-R", remote_spec, sandbox_server_url]
    log_file_path = log_path or Path("/tmp/swe-wstunnel-client.log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_file_path.open("ab")
    try:
        if live_log is not None:
            live_log.event("wstunnel.client.start", command=shlex.join(cmd), log_path=str(log_file_path))
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        if live_log is not None:
            live_log.event("wstunnel.client.started", pid=proc.pid, log_path=str(log_file_path))
        return proc
    finally:
        log_file.close()


async def start_sandbox_wstunnel_server(
    sandbox: Any,
    *,
    wstunnel_bin: str,
    server_port: int,
    user: str | None,
    log: LiveLog | None = None,
) -> Any:
    """Starts the wstunnel WebSocket server inside the sandbox as a background process."""
    command = " ".join(
        [
            "test",
            "-x",
            shlex.quote(wstunnel_bin),
            "&&",
            shlex.quote(wstunnel_bin),
            "server",
            shlex.quote(f"ws://0.0.0.0:{server_port}"),
            f">{shlex.quote(SANDBOX_WSTUNNEL_SERVER_LOG)}",
            "2>&1",
        ]
    )

    def _run_background() -> Any:
        """Starts the sandbox-side wstunnel server as a background command."""
        return sandbox.commands.run(
            command,
            background=True,
            timeout=0,
            request_timeout=120,
            user=user,
        )

    if log is not None:
        log.event("wstunnel.server.start", user=user or "<default>", command=command)
    handle = await asyncio.to_thread(_run_background)
    if log is not None:
        log.event("wstunnel.server.started", pid=getattr(handle, "pid", ""), log_path=SANDBOX_WSTUNNEL_SERVER_LOG)
    return handle


async def preflight_scaffold(
    sandbox: Any,
    *,
    wstunnel_bin: str,
    readiness_command: str,
    user: str | None,
    log: LiveLog,
    harness_label: str,
    protocol_root: str,
) -> None:
    """Verify wstunnel and the chosen scaffold are present in the sandbox."""
    checks = [
        f"test -x {shlex.quote(wstunnel_bin)}",
        readiness_command,
    ]
    result = await run_sandbox_command(
        sandbox,
        " && ".join(checks),
        timeout=120,
        user=user,
        log=log,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            "sandbox scaffold preflight failed "
            f"(harness={harness_label}, protocol_root={protocol_root}, "
            f"wstunnel={wstunnel_bin}): {result.output}"
        )


@dataclass
class SandboxSampleResult:
    """Result payload returned after one sample finishes its sandbox lifecycle."""

    final_messages: list[dict[str, Any]] = field(default_factory=list)
    final_tools: list[dict[str, Any]] | None = None
    reward: float = 0.0
    failed: bool = True
    turn_responses: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""


def _setup_sample_log_paths(
    log_root: str | None,
    sample_idx: int,
    extra_metadata: dict[str, Any],
) -> tuple[Path | None, Path | None, Path | None]:
    """Creates log directories for this sample and populates extra_metadata with their paths."""
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is not None:
        sample_log_dir.mkdir(parents=True, exist_ok=True)
        extra_metadata["sample_log_dir"] = str(sample_log_dir)
    agent_log_path = build_live_sandbox_log_path(log_root, sample_idx=sample_idx)
    eval_log_path = build_eval_log_path(log_root, sample_idx=sample_idx)
    tunnel_log_path = None if sample_log_dir is None else sample_log_dir / "sandbox" / "wstunnel_client.log"
    persisted = []
    if agent_log_path is not None:
        extra_metadata["agent_sandbox_api_log_path"] = str(agent_log_path)
        persisted.append("sandbox/agent_output.log")
    if eval_log_path is not None:
        extra_metadata["eval_log_path"] = str(eval_log_path)
        persisted.append("sandbox/eval_output.log")
    if tunnel_log_path is not None:
        tunnel_log_path.parent.mkdir(parents=True, exist_ok=True)
        extra_metadata["wstunnel_client_log_path"] = str(tunnel_log_path)
        persisted.append("sandbox/wstunnel_client.log")
    if persisted:
        extra_metadata["persisted_log_files"] = persisted
    return agent_log_path, eval_log_path, tunnel_log_path


async def run_sample_in_sandbox(
    *,
    args: Any,
    rollout_state: dict[str, Any],
    metadata: dict[str, Any],
    prompt_text: str,
    cfg: SWERolloutConfig,
    evaluation: bool,
    rollout_id: int,
    sample_idx: int,
) -> SandboxSampleResult:
    """Full per-sample lifecycle: create sandbox, set up tunnel, run agent, extract patch, eval, return result."""
    from .rebench import (
        extract_candidate_patch,
        prepare_workspace,
        resolve_rebench_workdir,
        resolve_template_alias,
        run_rebench_eval,
        sandbox_default_user,
    )

    extra_metadata: dict[str, Any] = {
        "raw_reward": 0.0,
        "agent_harness": cfg.agent_harness,
        "swe_protocol_root": cfg.protocol_root,
        "swe_wstunnel_bin": cfg.wstunnel_bin,
        "swe_node_bin": cfg.node_bin,
    }
    agent_log_path, eval_log_path, tunnel_log_path = _setup_sample_log_paths(
        cfg.log_root,
        sample_idx,
        extra_metadata,
    )
    agent_log = LiveLog(agent_log_path)
    sandbox = None
    proxy: SGLangOpenAIProxy | None = None
    wstunnel_handle = None
    host_wstunnel_proc: subprocess.Popen | None = None
    training_messages: list[dict[str, Any]] = []
    training_tools: list[dict[str, Any]] | None = None
    trajectory: list[dict[str, Any]] = []
    turn_responses: list[str] = []
    reward = 0.0
    failed = True
    failure_reason = ""
    user = sandbox_default_user(metadata)

    def persist_sample_snapshot() -> None:
        """Writes the latest per-sample artifact snapshot to disk."""
        write_sample_artifacts_snapshot(
            log_root=cfg.log_root,
            rollout_id=rollout_id,
            sample_idx=sample_idx,
            sample_prompt=prompt_text,
            metadata=metadata,
            extra_metadata=extra_metadata,
            turn_responses=turn_responses,
            trajectory=trajectory,
            final_messages=training_messages or None,
            last_response_payload=None,
        )

    try:
        template_alias = resolve_template_alias(metadata)
        extra_metadata["resolved_inspire_template"] = template_alias
        extra_metadata["effective_sandbox_user"] = user or ""
        sandbox_envs = build_sandbox_envs(metadata, cfg, prompt_text)
        extra_metadata["effective_sandbox_env_keys"] = sorted(sandbox_envs)
        proxy = SGLangOpenAIProxy(
            args=args,
            rollout_state=rollout_state,
            loop=asyncio.get_running_loop(),
            model_name=cfg.openai_model,
            evaluation=evaluation,
            max_assistant_turns=cfg.max_turns
            if cfg.agent_harness in {"codex", "open_code", "openhands"}
            else None,
            metadata={"swe_instance_id": metadata.get("instance_id")},
        ).start()
        proxy_host, proxy_port = proxy_endpoint_from_base_url(proxy.base_url)
        extra_metadata["model_proxy_base_url"] = proxy.base_url
        sandbox = await create_sandbox_with_retry(
            template_alias=template_alias,
            timeout=cfg.startup_timeout,
            envs=sandbox_envs,
            retry_times=cfg.sandbox_start_retry_times,
            retry_interval=cfg.sandbox_start_retry_interval,
            log=agent_log,
            image_name=str(metadata.get("local_image_name") or metadata.get("image_name") or ""),
        )
        extra_metadata["sandbox_id"] = getattr(sandbox, "sandbox_id", "")
        persist_sample_snapshot()

        factory = resolve_agent_command_factory(cfg.agent_harness)
        await preflight_scaffold(
            sandbox,
            wstunnel_bin=cfg.wstunnel_bin,
            readiness_command=factory.readiness_command(),
            user=user,
            log=agent_log,
            harness_label=cfg.agent_harness,
            protocol_root=cfg.protocol_root,
        )
        workspace_output = await prepare_workspace(
            sandbox,
            metadata,
            user=user,
            wait_timeout=cfg.wait_timeout,
            log=agent_log,
        )
        extra_metadata["workspace_prepare_preview"] = truncate_text(workspace_output, cfg.preview_limit)

        wstunnel_handle = await start_sandbox_wstunnel_server(
            sandbox,
            wstunnel_bin=cfg.wstunnel_bin,
            server_port=cfg.wstunnel_server_port,
            user=user,
            log=agent_log,
        )
        await asyncio.sleep(cfg.tunnel_wait_seconds)
        sandbox_wstunnel_url = f"wss://{sandbox.get_host(cfg.wstunnel_server_port)}"
        extra_metadata["sandbox_wstunnel_url"] = sandbox_wstunnel_url
        host_wstunnel_proc = start_host_reverse_tunnel(
            sandbox_server_url=sandbox_wstunnel_url,
            host_model_host=proxy_host,
            host_model_port=proxy_port,
            sandbox_model_port=cfg.model_proxy_port,
            log_path=tunnel_log_path,
            live_log=agent_log,
        )
        await asyncio.sleep(cfg.tunnel_wait_seconds)
        if host_wstunnel_proc.poll() is not None:
            raise RuntimeError("host wstunnel client exited before agent started")

        model_endpoint = ModelEndpoint(
            base_url=f"http://127.0.0.1:{cfg.model_proxy_port}/v1",
            api_key=cfg.openai_api_key,
            model_name=cfg.openai_model,
        )
        agent_script = factory.runtime_command(
            max_turns=cfg.max_turns,
            cli_args=factory.model_cli_args(model_endpoint),
            cwd=resolve_rebench_workdir(metadata),
        )
        extra_metadata["agent_run_cmd"] = truncate_text(agent_script, cfg.preview_limit)
        run_result = await run_sandbox_command(
            sandbox,
            f"bash -lc {shlex.quote(agent_script)}",
            timeout=cfg.agent_finish_timeout,
            user=user,
            cwd=resolve_rebench_workdir(metadata),
            log=agent_log,
        )
        extra_metadata["agent_exit_code"] = run_result.exit_code
        extra_metadata["agent_output_preview"] = truncate_text(run_result.output, cfg.preview_limit)
        if proxy is not None:
            training_messages, training_tools, trajectory, turn_responses = proxy.snapshot()

        candidate_patch = await extract_candidate_patch(
            sandbox,
            metadata,
            user=user,
            wait_timeout=cfg.wait_timeout,
            log=agent_log,
        )
        extra_metadata["candidate_patch_bytes"] = len(candidate_patch.encode("utf-8"))
        extra_metadata["candidate_patch_empty"] = not bool(candidate_patch.strip())

        reward, eval_extras = await run_rebench_eval(
            sandbox,
            metadata,
            user=user,
            eval_log_path=eval_log_path,
            aggregate_log=agent_log,
            wait_timeout=cfg.wait_timeout,
            preview_limit=cfg.preview_limit,
            reached_turn_limit=False,
            last_generation_finish_reason="",
        )
        extra_metadata.update(eval_extras)
        failed = False
        if not training_messages:
            failed = True
            failure_reason = "agent completed without model proxy turns"
    except Exception as exc:
        failure_reason = str(exc)
        extra_metadata["failure_reason"] = failure_reason
        extra_metadata["failure_type"] = exc.__class__.__name__
        reward = 0.0
        failed = True
    finally:
        if proxy is not None:
            latest_messages, latest_tools, latest_trajectory, latest_responses = proxy.snapshot()
            if latest_messages:
                training_messages = latest_messages
            if latest_tools:
                training_tools = latest_tools
            if latest_trajectory:
                trajectory = latest_trajectory
            if latest_responses:
                turn_responses = latest_responses
            proxy.close()
        extra_metadata["proxy_turn_count"] = len(trajectory)
        extra_metadata["turn_responses_empty"] = not bool(turn_responses)
        if host_wstunnel_proc is not None and host_wstunnel_proc.poll() is None:
            agent_log.event("wstunnel.client.terminate", pid=host_wstunnel_proc.pid)
            host_wstunnel_proc.terminate()
            try:
                host_wstunnel_proc.wait(timeout=3)
                agent_log.event("wstunnel.client.terminated", pid=host_wstunnel_proc.pid)
            except subprocess.TimeoutExpired:
                host_wstunnel_proc.kill()
                agent_log.event("wstunnel.client.killed", pid=host_wstunnel_proc.pid)
        if wstunnel_handle is not None:
            try:
                agent_log.event("wstunnel.server.kill", pid=getattr(wstunnel_handle, "pid", ""))
                await asyncio.to_thread(wstunnel_handle.kill)
                agent_log.event("wstunnel.server.killed", pid=getattr(wstunnel_handle, "pid", ""))
            except Exception:
                pass
        if sandbox is not None and not cfg.keep_containers:
            try:
                agent_log.event("sandbox.kill", sandbox_id=getattr(sandbox, "sandbox_id", ""))
                await asyncio.to_thread(sandbox.kill)
                agent_log.event("sandbox.killed", sandbox_id=getattr(sandbox, "sandbox_id", ""))
            except Exception as stop_exc:
                extra_metadata["sandbox_stop_error"] = str(stop_exc)
                agent_log.event("sandbox.kill.failed", sandbox_id=getattr(sandbox, "sandbox_id", ""), error=str(stop_exc))
        persist_sample_snapshot()

    return SandboxSampleResult(
        final_messages=training_messages,
        final_tools=training_tools,
        reward=reward,
        failed=failed,
        turn_responses=turn_responses,
        trajectory=trajectory,
        extra_metadata=extra_metadata,
        failure_reason=failure_reason,
    )
