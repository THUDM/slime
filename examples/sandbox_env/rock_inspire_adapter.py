from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspire_sandbox import CommandExitException, Sandbox as InspireSandbox, SandboxSpecCode, Template

from rock.actions import (
    Action,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command,
    CommandResponse,
    CreateBashSessionRequest,
    CreateBashSessionResponse,
    IsAliveResponse,
    Observation,
    ReadFileRequest,
    ReadFileResponse,
    SandboxStatusResponse,
    UploadMode,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
)
from rock.sdk.sandbox.agent.rock_agent import RockAgent
from rock.sdk.sandbox.client import RunMode
from rock.sdk.sandbox.config import SandboxConfig
from rock.sdk.sandbox.deploy import Deploy
from rock.sdk.sandbox.file_system import LinuxFileSystem
from rock.sdk.sandbox.network import Network
from rock.sdk.sandbox.process import Process
from rock.sdk.sandbox.remote_user import LinuxRemoteUser
from rock.sdk.sandbox.runtime_env.node_runtime_env import NodeRuntimeEnv as _NodeRuntimeEnv
from rock.sdk.sandbox.runtime_env.python_runtime_env import PythonRuntimeEnv as _PythonRuntimeEnv


# ROCK SDK probes the installed runtime with ``test -x node`` / ``test -x python``,
# but POSIX ``test -x`` resolves bare names against cwd (``/root`` in our sandbox),
# not PATH, so the probe fails even though the runtime is installed at
# ``/home/user/.rock/preinstalled/*-runtime/runtime-env/bin``.  The failure used
# to be masked by nohup swallowing exit codes; once nohup started surfacing real
# exit codes, init began failing.  Patch the probe to use ``command -v`` (PATH-aware)
# instead of touching the ROCK source tree.
async def _validate_node_via_path(self) -> None:
    return await self.run(cmd="command -v node >/dev/null")


async def _validate_python_via_path(self) -> None:
    return await self.run(cmd="command -v python >/dev/null")


_NodeRuntimeEnv._validate_node = _validate_node_via_path
_PythonRuntimeEnv._validate_python = _validate_python_via_path


DEFAULT_SANDBOX_USER = os.environ.get("ROCK_INSPIRE_SANDBOX_USER", "root")
DEFAULT_SESSION_USER = os.environ.get("ROCK_INSPIRE_SESSION_USER", "root")
USER_ENV_MERGE_KEYS = {
    "JAVA_HOME",
    "JDK_HOME",
    "GOROOT",
    "GOPATH",
    "GRADLE_HOME",
    "KOTLIN_HOME",
    "M2_HOME",
    "MAVEN_HOME",
    "ANDROID_HOME",
    "ANDROID_SDK_ROOT",
    "SDKMAN_DIR",
    "NVM_DIR",
    "PNPM_HOME",
    "CARGO_HOME",
    "RUSTUP_HOME",
}
VOLATILE_ENV_KEYS = {
    "PWD",
    "OLDPWD",
    "SHLVL",
    "_",
    "HOSTNAME",
    "TERM",
    "LS_COLORS",
}

# Module-level template cache: avoids re-entering the build loop for images already built
# in this process. The inspire backend is idempotent by name, so concurrent builds of the
# same image are harmless but wasteful.
_TEMPLATE_CACHE: set[str] = set()
_TEMPLATE_LOCK = threading.Lock()
DEFAULT_TEMPLATE_PREFIX = os.environ.get("ROCK_INSPIRE_TEMPLATE_PREFIX", "rock-")
DEFAULT_SANDBOX_TIMEOUT = int(os.environ.get("ROCK_INSPIRE_SANDBOX_TIMEOUT", "3600"))
DEFAULT_TEMPLATE_WAIT_SECONDS = int(os.environ.get("ROCK_INSPIRE_TEMPLATE_WAIT_SECONDS", "1800"))

# The default ModelService.anti_call_llm shells out to the ``rock`` CLI, which
# is not present in inspire sandbox images.  We replace it with a Python helper
# that communicates via the model-service log-file protocol.  The response
# payload is passed as a direct shell argument (commands.run supports payloads
# up to ~128 KB; typical response_payload is ≤40 KB).
LOCAL_MODEL_SERVICE_LOG_FILE = "LLMService.log"
ANTI_CALL_HELPER_POLL_INTERVAL = float(os.environ.get("ROCK_INSPIRE_ANTI_CALL_POLL_INTERVAL", "0.1"))
ANTI_CALL_HELPER_TAIL_CHUNK_BYTES = int(os.environ.get("ROCK_INSPIRE_ANTI_CALL_TAIL_CHUNK_BYTES", str(64 * 1024)))


def build_inspire_anti_call_llm_helper_code() -> str:
    return Path(__file__).with_name("anti_call_llm_helper.py").read_text(encoding="utf-8").strip()


@dataclass
class _SessionState:
    env: dict[str, str]
    remote_user: str | None = None
    cwd: str | None = None


@dataclass
class _CommandContext:
    command: str
    user: str
    cwd: str | None = None
    env: dict[str, str] | None = None


def _sanitize_template_name(image: str) -> str:
    digest = hashlib.sha1(image.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^a-z0-9-]+", "-", image.lower()).strip("-")
    stem = stem[-40:] if len(stem) > 40 else stem
    stem = stem or "image"
    return f"{DEFAULT_TEMPLATE_PREFIX}{stem}-{digest}"


def _resolve_spec(config: SandboxConfig, explicit_spec: str | None = None) -> SandboxSpecCode:
    if explicit_spec:
        return SandboxSpecCode[explicit_spec]
    explicit = os.environ.get("ROCK_INSPIRE_SPEC", "").strip()
    if explicit:
        return SandboxSpecCode[explicit]
    cpus = float(config.cpus or 1)
    if cpus <= 1:
        return SandboxSpecCode.G_C1
    if cpus <= 2:
        return SandboxSpecCode.G_C2
    return SandboxSpecCode.G_C4


class InspireRockSandbox:
    """Inspire-sandbox backed replacement for rock.sdk.sandbox.client.Sandbox.

    This adapter intentionally mirrors the concrete surface that ROCK agents,
    RuntimeEnv, Deploy, and ModelService actually use, while removing the hard
    dependency on a separate ROCK admin HTTP control plane.
    """

    def __init__(
        self,
        config: SandboxConfig,
        *,
        inspire_spec: str | None = None,
        template_name: str | None = None,
        image_runtime_env: dict[str, str] | None = None,
    ):
        self.config = config
        self._inspire_spec = inspire_spec
        self._template_name_override = template_name
        self._sandbox: InspireSandbox | None = None
        self._template_name: str | None = None
        self._image_runtime_env: dict[str, str] = {
            str(k): str(v) for k, v in dict(image_runtime_env or {}).items() if str(k).strip()
        }
        self._sandbox_id: str | None = None
        self._host_name: str | None = None
        self._host_ip: str | None = None
        self._cluster: str | None = config.cluster
        self._namespace: str | None = config.namespace
        self._experiment_id: str | None = config.experiment_id
        self._url = "inspire_sandbox://local"
        self._sessions: dict[str, _SessionState] = {}
        self._background_handles: dict[int, Any] = {}
        self._login_env_cache: dict[str, dict[str, str]] = {}
        self.runtime_envs: dict[Any, Any] = {}
        self._model_service = None
        self._live_log_path: Path | None = None
        self._live_log_lock = threading.Lock()
        self._stopping = False
        self.remote_user = LinuxRemoteUser(self)
        self.process = Process(self)
        self.network = Network(self)
        self.fs = LinuxFileSystem(self)
        self.deploy = Deploy(self)
        self.agent = RockAgent(self)

    def set_live_log_path(self, path: str | Path | None) -> None:
        self._live_log_path = Path(path) if path else None
        if self._live_log_path is not None:
            self._live_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._live_log_path.touch(exist_ok=True)
            self._append_live_log(f"[live_log.attached] path={self._live_log_path}")

    def _append_live_log(self, message: str) -> None:
        if self._live_log_path is None:
            return
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message.rstrip()}\n"
        with self._live_log_lock:
            self._live_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._live_log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def _append_live_log_lines(self, label: str, content: str | None) -> None:
        if self._live_log_path is None or not content:
            return
        text = str(content)
        for line in text.splitlines():
            self._append_live_log(f"[{label}] {line}")
        if text.endswith("\n"):
            self._append_live_log(f"[{label}]")

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id or ""

    @property
    def host_name(self) -> str | None:
        return self._host_name

    @property
    def host_ip(self) -> str | None:
        return self._host_ip

    @property
    def cluster(self) -> str | None:
        return self._cluster

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, value: str) -> None:
        self._url = value

    @property
    def model_service(self):
        return self._model_service

    @model_service.setter
    def model_service(self, svc) -> None:
        # RockAgent._init_model_service assigns the freshly built ModelService here.
        # Patch anti_call_llm so it uses a Python helper inside the sandbox
        # instead of shelling out to the ``rock`` CLI (which doesn't exist in
        # inspire images).
        self._model_service = svc
        if svc is not None and not getattr(svc, "_inspire_anti_call_llm_patched", False):
            self._patch_anti_call_llm(svc)

    async def start(self) -> None:
        self._stopping = False
        if self._template_name_override:
            self._template_name = self._template_name_override
            self._append_live_log(f"[template] using existing template={self._template_name}")
        else:
            self._template_name = await asyncio.to_thread(self._ensure_template)
        timeout = int(max(self.config.auto_clear_seconds or 0, DEFAULT_SANDBOX_TIMEOUT))
        self._append_live_log(
            f"[sandbox.create] template={self._template_name} timeout={timeout} image={self.config.image}"
        )
        self._sandbox = await asyncio.to_thread(InspireSandbox.create, template=self._template_name, timeout=timeout)
        self._sandbox_id = getattr(self._sandbox, "sandbox_id", "") or ""
        self._host_name = getattr(self._sandbox, "sandbox_domain", None) or getattr(self._sandbox, "hostname", None)
        self._host_ip = await asyncio.to_thread(self._best_effort_sandbox_ip)
        self._append_live_log(
            f"[sandbox.ready] sandbox_id={self._sandbox_id} host_name={self._host_name} host_ip={self._host_ip}"
        )

    async def stop(self) -> None:
        if self._sandbox is None:
            return
        self._stopping = True
        try:
            self._append_live_log(f"[sandbox.stop] sandbox_id={self._sandbox_id}")
            await self._disconnect_background_handles()
            await asyncio.to_thread(self._sandbox.kill)
        finally:
            self._sandbox = None
            # Clear per-sandbox state so a reused InspireRockSandbox instance (if any caller
            # does that) does not leak stale sessions / runtime_envs / model_service handles.
            self._sessions.clear()
            self._background_handles.clear()
            self.runtime_envs = {}
            self._model_service = None
            self._host_ip = None
            self._host_name = None
            self._sandbox_id = None

    async def close(self) -> CloseResponse:
        await self.stop()
        return CloseResponse()

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._sandbox is None:
            return IsAliveResponse(is_alive=False, message="sandbox not started")
        probe = await self._run_command("true", timeout=timeout or 10, user=DEFAULT_SANDBOX_USER)
        return IsAliveResponse(is_alive=probe.exit_code == 0, message=probe.failure_reason if probe.exit_code else "")

    async def get_status(self) -> SandboxStatusResponse:
        alive = await self.is_alive()
        return SandboxStatusResponse(
            sandbox_id=self.sandbox_id,
            status={"runtime": {"status": "running" if alive.is_alive else "stopped"}},
            port_mapping={},
            host_name=self._host_name,
            host_ip=self._host_ip,
            is_alive=alive.is_alive,
            image=self.config.image,
            user_id=self.config.user_id,
            experiment_id=self._experiment_id,
            namespace=self._namespace,
            cpus=self.config.cpus,
            memory=self.config.memory,
        )

    async def create_session(self, create_session_request: CreateBashSessionRequest) -> CreateBashSessionResponse:
        session_user = create_session_request.remote_user or DEFAULT_SESSION_USER
        env = {}
        if create_session_request.env_enable:
            env = await self._capture_login_env(session_user)
            if session_user == "root":
                env = await self._merge_user_toolchain_env(env)
            env = self._supplement_with_image_runtime_env(env)
            env.update({str(k): str(v) for k, v in dict(create_session_request.env or {}).items()})
        self._sessions[create_session_request.session] = _SessionState(
            env=env,
            remote_user=create_session_request.remote_user,
        )
        self._append_live_log(
            f"[create_session] session={create_session_request.session} "
            f"env_enable={create_session_request.env_enable} remote_user={create_session_request.remote_user} "
            f"captured_user={session_user} env_keys={len(env)}"
        )
        return CreateBashSessionResponse(output="", session_type="bash")

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        self._sessions.pop(request.session, None)
        self._append_live_log(f"[close_session] session={request.session}")
        return CloseSessionResponse(session_type="bash")

    async def run_in_session(self, action: Action) -> Observation:
        ctx = self._resolve_command_context(cmd=action.command, session=action.session)
        self._append_live_log(
            f"[run_in_session.start] session={action.session} timeout={action.timeout} command={action.command}"
        )

        # Persist cwd across turns: capture resulting pwd in a per-session tmp file within the
        # same shell invocation so that `cd` effects are visible to the pwd capture.
        session_key = action.session or ""
        cwd_tmp = f"/tmp/_rock_inspire_cwd_{hash(session_key) & 0xFFFFFFFF:08x}"
        inner = f"{ctx.command}; __rc=$?; pwd > {shlex.quote(cwd_tmp)} 2>/dev/null; exit $__rc"
        obs = await self._run_command(
            inner,
            timeout=action.timeout,
            user=ctx.user,
            cwd=ctx.cwd,
            env=ctx.env,
        )

        # Read back the cwd regardless of exit code (cd may have succeeded even if cmd failed)
        if action.session in self._sessions and self._sandbox is not None:
            try:
                raw = await asyncio.to_thread(
                    self._sandbox.files.read, cwd_tmp, user=DEFAULT_SANDBOX_USER
                )
                new_cwd = (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).strip()
                if new_cwd:
                    self._sessions[action.session].cwd = new_cwd
                    self._append_live_log(
                        f"[run_in_session.cwd] session={action.session} cwd={new_cwd}"
                    )
            except Exception:
                pass

        self._append_live_log(
            f"[run_in_session.end] session={action.session} exit_code={obs.exit_code} "
            f"failure_reason_bytes={len((obs.failure_reason or '').encode('utf-8'))} "
            f"output_bytes={len((obs.output or '').encode('utf-8'))}"
        )
        if obs.failure_reason:
            self._append_live_log_lines(
                f"run_in_session.failure_reason session={action.session}",
                obs.failure_reason,
            )
        if obs.output:
            self._append_live_log_lines(
                f"run_in_session.output session={action.session}",
                obs.output,
            )
        return obs

    async def execute(self, command: Command) -> CommandResponse:
        cmd = shlex.join(command.command) if isinstance(command.command, list) else str(command.command)
        obs = await self._run_command(
            cmd,
            timeout=command.timeout,
            user=DEFAULT_SANDBOX_USER,
            cwd=command.cwd,
            env={str(k): str(v) for k, v in dict(command.env or {}).items()},
        )
        return CommandResponse(stdout=obs.output, stderr=obs.failure_reason, exit_code=obs.exit_code)

    async def arun(
        self,
        cmd: str,
        session: str | None = None,
        wait_timeout: int = 300,
        wait_interval: int = 10,
        mode: str = RunMode.NORMAL,
        response_limited_bytes_in_nohup: int | None = None,
        ignore_output: bool = False,
        output_file: str | None = None,
        user: str | None = None,
    ) -> Observation:
        command_user = user or DEFAULT_SANDBOX_USER
        if mode == RunMode.NORMAL:
            if session is None:
                return await self._run_command(cmd, timeout=wait_timeout, user=command_user)
            return await self.run_in_session(Action(command=cmd, session=session, timeout=wait_timeout))
        if mode != RunMode.NOHUP:
            return Observation(output=f"Unsupported arun mode: {mode}", exit_code=1, failure_reason=f"Unsupported arun mode: {mode}")
        return await self._arun_nohup(
            cmd=cmd,
            session=session,
            wait_timeout=wait_timeout,
            wait_interval=wait_interval,
            response_limited_bytes_in_nohup=response_limited_bytes_in_nohup,
            ignore_output=ignore_output,
            output_file=output_file,
            user=command_user,
        )

    async def write_file(self, request: WriteFileRequest, *, user: str | None = None) -> WriteFileResponse:
        write_user = user or DEFAULT_SANDBOX_USER
        parent = os.path.dirname(request.path)
        if parent:
            await self._run_command(f"mkdir -p {shlex.quote(parent)}", timeout=30, user=write_user)
        # WriteFileRequest.content is str; convert to bytes for a unified transfer path.
        content = request.content
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        await self._write_bytes(request.path, data, user=write_user)
        return WriteFileResponse(success=True, message=f"Successfully write content to file {request.path}")

    async def upload(self, request: UploadRequest) -> UploadResponse:
        return await self.upload_by_path(request.source_path, request.target_path, request.upload_mode)

    async def upload_by_path(
        self,
        file_path: str | Path,
        target_path: str,
        upload_mode: UploadMode = UploadMode.AUTO,
        *,
        user: str | None = None,
    ) -> UploadResponse:
        del upload_mode
        upload_user = user or DEFAULT_SANDBOX_USER
        local_path = Path(file_path)
        if not local_path.exists():
            return UploadResponse(success=False, message=f"File not found: {local_path}")
        parent = os.path.dirname(target_path)
        if parent:
            await self._run_command(f"mkdir -p {shlex.quote(parent)}", timeout=30, user=upload_user)

        await self._write_local_file(local_path, target_path, user=upload_user)
        return UploadResponse(success=True, message=f"Successfully uploaded file {local_path.name} to {target_path}", file_name=local_path.name)

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        # Read as bytes always — avoids server-side text decoding (which can mangle
        # binary content) and keeps the bytes-vs-str fork in one place. inspire_sandbox
        # returns bytearray for format="bytes", so accept both.
        raw = await asyncio.to_thread(
            self._sandbox.files.read, request.path, format="bytes", user=DEFAULT_SANDBOX_USER
        )
        if raw is None:
            content = ""
        elif isinstance(raw, (bytes, bytearray)):
            content = bytes(raw).decode(request.encoding or "utf-8", errors=request.errors or "strict")
        else:
            content = str(raw)
        return ReadFileResponse(content=content)

    async def commit(self, image_tag: str, username: str, password: str) -> CommandResponse:
        del image_tag, username, password
        raise NotImplementedError(
            "InspireRockSandbox.commit: image commit is not supported by the inspire-sandbox "
            "backend. Rebuild/snapshot must be handled at the template level."
        )

    async def _write_bytes(self, target: str, data: bytes, *, user: str) -> None:
        """Write raw bytes into the sandbox via the files.write multipart API."""
        await asyncio.to_thread(self._sandbox.files.write, target, data, user=user)

    async def _write_local_file(self, local_path: Path, target: str, *, user: str) -> None:
        """Stream a local file to the sandbox through files.write multipart upload."""

        def _write_stream() -> None:
            with local_path.open("rb") as f:
                self._sandbox.files.write(target, f, user=user)

        await asyncio.to_thread(_write_stream)

    def _patch_anti_call_llm(self, svc) -> None:
        """Replace ModelService.anti_call_llm with a CLI-free Python wrapper.

        The default implementation shells out to the ``rock`` CLI inside the
        sandbox, which is not available in inspire images.  We run a small
        Python helper that communicates via the model-service log-file protocol
        instead, passing the response payload as a direct shell argument.
        """
        outer = self

        async def patched(index, response_payload=None, call_timeout=600, check_interval=3):
            return await outer._anti_call_llm_via_helper(
                svc, index, response_payload, call_timeout, check_interval
            )

        svc.anti_call_llm = patched
        svc._inspire_anti_call_llm_patched = True

    async def _anti_call_llm_via_helper(
        self, svc, index: int, response_payload: str | None, call_timeout: int, check_interval: int
    ) -> str:
        """CLI-free anti_call_llm via the local model-service log protocol."""
        if not svc.is_started:
            if self._stopping or self._sandbox is None:
                self._append_live_log(
                    f"[anti_call_llm.shutdown_before_start] index={index} sandbox_id={self._sandbox_id}"
                )
                return "SESSION_END"
            raise RuntimeError(
                f"[{self._sandbox_id}] Cannot execute anti-call LLM: ModelService is not started."
            )

        log_file_name = (
            LOCAL_MODEL_SERVICE_LOG_FILE
            if getattr(svc.config, "type", "") == "local"
            else svc.config.logging_file_name
        )
        log_path = f"{svc.config.logging_path.rstrip('/')}/{log_file_name}"

        python_code = build_inspire_anti_call_llm_helper_code()

        cmd_args = [
            shlex.quote(str(index)),
            shlex.quote(log_path),
            shlex.quote(str(ANTI_CALL_HELPER_POLL_INTERVAL)),
            shlex.quote(str(ANTI_CALL_HELPER_TAIL_CHUNK_BYTES)),
        ]
        if response_payload is not None:
            cmd_args.append(shlex.quote(response_payload))

        cmd = "PYTHONWARNINGS=ignore python -c " f"{shlex.quote(python_code)} {' '.join(cmd_args)}"
        bash_cmd = svc.runtime_env.wrapped_cmd(cmd)

        result = await self.arun(
            cmd=bash_cmd,
            mode=RunMode.NOHUP,
            session=None,
            wait_timeout=call_timeout,
            wait_interval=check_interval,
        )
        if result.exit_code != 0:
            detail = (result.failure_reason or "").strip()
            output = (result.output or "").strip()
            if detail and output and detail != output:
                message = f"{detail}; output={output}"
            else:
                message = detail or output or f"exit_code={result.exit_code}"
            lowered_message = message.lower()
            if self._stopping and (
                "sandbox not started" in lowered_message or "incomplete chunked read" in lowered_message
            ):
                self._append_live_log(
                    f"[anti_call_llm.shutdown_after_stop] index={index} message={message}"
                )
                return "SESSION_END"
            raise RuntimeError(f"Anti-call LLM command failed: {message}")
        return result.output

    def _ensure_template(self) -> str:
        image = self.config.image
        name = _sanitize_template_name(image)
        with _TEMPLATE_LOCK:
            if name in _TEMPLATE_CACHE:
                self._append_live_log(f"[template.cache_hit] image={image} template={name}")
                return name
        # Build outside the lock — the inspire backend is idempotent by name, so parallel
        # builds of the same image are safe (though wasteful).
        template = Template().from_image(image)
        build_info = Template.build_in_background(template, name, spec_code=_resolve_spec(self.config, self._inspire_spec))
        self._append_live_log(
            f"[template.build_started] image={image} template={name} "
            f"template_id={build_info.template_id} build_id={build_info.build_id}"
        )
        deadline = time.time() + DEFAULT_TEMPLATE_WAIT_SECONDS
        status = "building"
        offset = 0
        while status == "building" and time.time() < deadline:
            build_status = Template.get_build_status(build_info, logs_offset=offset)
            offset += len(build_status.log_entries)
            for entry in build_status.log_entries:
                self._append_live_log(f"[template.build_log] {entry}")
            status = build_status.status.value
            if status == "building":
                time.sleep(5)
        self._append_live_log(f"[template.build_finished] image={image} template={name} status={status}")
        if status != "ready":
            raise RuntimeError(
                f"Template build for {image!r} ended with status={status!r} "
                f"(name={name!r}, waited up to {DEFAULT_TEMPLATE_WAIT_SECONDS}s). "
                f"Raise ROCK_INSPIRE_TEMPLATE_WAIT_SECONDS or check build logs."
            )
        with _TEMPLATE_LOCK:
            _TEMPLATE_CACHE.add(name)
        return name

    async def _arun_nohup(
        self,
        *,
        cmd: str,
        session: str | None,
        wait_timeout: int,
        wait_interval: int,
        response_limited_bytes_in_nohup: int | None,
        ignore_output: bool,
        output_file: str | None,
        user: str | None,
    ) -> Observation:
        timestamp = str(time.time_ns())
        tmp_file = output_file or f"/tmp/tmp_{timestamp}.out"
        ctx = self._resolve_command_context(cmd=cmd, session=session, user=user)

        self._append_live_log(
            f"[nohup.start] session={session} wait_timeout={wait_timeout} wait_interval={wait_interval} "
            f"ignore_output={ignore_output} output_file={tmp_file} command={ctx.command}"
        )

        if output_file:
            parent = os.path.dirname(output_file)
            if parent:
                mkdir_obs = await self._run_command(
                    f"mkdir -p {shlex.quote(parent)}",
                    timeout=300,
                    user=ctx.user,
                    cwd=ctx.cwd,
                    env=ctx.env,
                )
                if mkdir_obs.exit_code != 0:
                    self._append_live_log(
                        f"[nohup.mkdir_failed] output_file={tmp_file} exit_code={mkdir_obs.exit_code} "
                        f"failure_reason={mkdir_obs.failure_reason!r}"
                    )
                    return mkdir_obs

        pid, submit_error = await self._start_background_process_via_sdk(
            command=ctx.command,
            tmp_file=tmp_file,
            user=ctx.user,
            cwd=ctx.cwd,
            env=ctx.env,
            log_label="nohup",
        )
        if submit_error is not None:
            return submit_error
        if not pid:
            self._append_live_log(
                f"[nohup.pid_missing] output_file={tmp_file}"
            )
            return Observation(output="Failed to obtain PID from background command", exit_code=1, failure_reason="Failed to obtain PID from background command")

        success, message = await self._wait_for_process_completion(
            pid=pid,
            wait_timeout=wait_timeout,
            wait_interval=wait_interval,
        )
        self._append_live_log(
            f"[nohup.wait_end] output_file={tmp_file} pid={pid} success={success} message={message}"
        )
        if ignore_output:
            output = self._build_nohup_detached_message(tmp_file, success, message)
            self._append_live_log(
                f"[nohup.detached] output_file={tmp_file} pid={pid} success={success} message={message}"
            )
            return Observation(output=output, exit_code=0 if success else 1, failure_reason="" if success else message)

        cat_cmd = f"cat {shlex.quote(tmp_file)}"
        if response_limited_bytes_in_nohup:
            cat_cmd = f"head -c {int(response_limited_bytes_in_nohup)} {shlex.quote(tmp_file)}"
        read_back = await self._run_command(cat_cmd, timeout=60, user=ctx.user, cwd=ctx.cwd, env=ctx.env)
        self._append_live_log(
            f"[nohup.read_back] output_file={tmp_file} pid={pid} exit_code={read_back.exit_code} "
            f"failure_reason={read_back.failure_reason!r} output={read_back.output!r}"
        )
        if success:
            return Observation(output=read_back.output, exit_code=0, failure_reason="")
        return Observation(output=read_back.output, exit_code=1, failure_reason=message)

    async def start_nohup_process(self, cmd: str, tmp_file: str, session: str) -> tuple[int | None, Observation | None]:
        """ROCK Sandbox nohup protocol backed by inspire background execution."""
        self._append_live_log(
            f"[start_nohup_process.start] session={session} tmp_file={tmp_file} command={cmd}"
        )
        parent = os.path.dirname(tmp_file)
        if parent:
            mkdir_obs = await self.run_in_session(
                Action(command=f"mkdir -p {shlex.quote(parent)}", session=session, timeout=300)
            )
            if mkdir_obs.exit_code != 0:
                self._append_live_log(
                    f"[start_nohup_process.mkdir_failed] session={session} tmp_file={tmp_file} "
                    f"exit_code={mkdir_obs.exit_code} failure_reason={mkdir_obs.failure_reason!r}"
                    )
                return None, mkdir_obs

        ctx = self._resolve_command_context(cmd=cmd, session=session)
        pid, submit_error = await self._start_background_process_via_sdk(
            command=ctx.command,
            tmp_file=tmp_file,
            user=ctx.user,
            cwd=ctx.cwd,
            env=ctx.env,
            log_label="start_nohup_process",
        )
        if submit_error is not None:
            return None, submit_error
        if not pid:
            self._append_live_log(
                f"[start_nohup_process.pid_missing] session={session} tmp_file={tmp_file}"
            )
            return None, None
        return pid, None

    async def wait_for_process_completion(
        self,
        pid: int,
        session: str,
        wait_timeout: int,
        wait_interval: int,
    ) -> tuple[bool, str]:
        del session
        return await self._wait_for_process_completion(
            pid=pid,
            wait_timeout=wait_timeout,
            wait_interval=wait_interval,
        )

    async def handle_nohup_output(
        self,
        tmp_file: str,
        session: str,
        success: bool,
        message: str,
        ignore_output: bool,
        response_limited_bytes_in_nohup: int | None,
    ) -> Observation:
        if ignore_output:
            return Observation(
                output=self._build_nohup_detached_message(tmp_file, success, message),
                exit_code=0 if success else 1,
                failure_reason="" if success else message,
            )

        cat_cmd = f"cat {shlex.quote(tmp_file)}"
        if response_limited_bytes_in_nohup:
            cat_cmd = f"head -c {int(response_limited_bytes_in_nohup)} {shlex.quote(tmp_file)}"
        read_back = await self.run_in_session(Action(command=cat_cmd, session=session, timeout=60))
        if success:
            return Observation(output=read_back.output, exit_code=0, failure_reason="")
        return Observation(output=read_back.output, exit_code=1, failure_reason=message)

    async def _wait_for_process_completion(
        self,
        *,
        pid: int,
        wait_timeout: int,
        wait_interval: int,
    ) -> tuple[bool, str]:
        del wait_interval
        start_time = time.perf_counter()
        handle = self._background_handles.pop(pid, None)

        if handle is None:
            if self._sandbox is None:
                return False, f"Failed to wait for process {pid}: sandbox not started"
            try:
                handle = await asyncio.to_thread(self._sandbox.commands.connect, pid, 0)
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                return False, f"Failed to reconnect to process {pid} after {elapsed:.1f}s: {exc}"

        try:
            await asyncio.wait_for(asyncio.to_thread(handle.wait), timeout=wait_timeout)
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            try:
                handle.disconnect()
            except Exception:
                pass
            return False, f"Process {pid} did not complete within {elapsed:.1f}s (timeout: {wait_timeout}s)"
        except CommandExitException as exc:
            detail = (exc.stderr or exc.stdout or exc.error or "").strip()
            message = f"Process {pid} exited with code {exc.exit_code}"
            if detail:
                message = f"{message}: {detail}"
            return False, message
        except Exception as exc:
            return False, f"Failed waiting for process {pid}: {exc}"
        finally:
            try:
                handle.disconnect()
            except Exception:
                pass

        elapsed = time.perf_counter() - start_time
        return True, f"Process completed successfully in {elapsed:.1f}s"

    async def _start_background_process_via_sdk(
        self,
        *,
        command: str,
        tmp_file: str,
        user: str,
        cwd: str | None,
        env: dict[str, str] | None,
        log_label: str,
    ) -> tuple[int | None, Observation | None]:
        if self._sandbox is None:
            message = "sandbox not started"
            return None, Observation(output=message, exit_code=1, failure_reason=message)

        redirect_command = f"{command} > {shlex.quote(tmp_file)} 2>&1"
        merged_env = await self._merge_command_env(user=user, env=env)
        self._append_live_log(
            f"[{log_label}.submit] user={user} output_file={tmp_file} command={redirect_command}"
        )
        if cwd:
            self._append_live_log(
                f"[{log_label}.cwd] user={user} cwd={cwd}"
            )
        if merged_env:
            self._append_live_log(
                f"[{log_label}.env] user={user} keys={sorted(merged_env.keys())}"
            )

        def _submit() -> tuple[int, Any]:
            handle = self._sandbox.commands.run(
                redirect_command,
                background=True,
                cwd=cwd,
                timeout=0,
                user=user,
                envs=merged_env or None,
            )
            return int(handle.pid), handle

        try:
            pid, handle = await asyncio.to_thread(_submit)
        except Exception as exc:
            message = str(exc)
            self._append_live_log(
                f"[{log_label}.submit_failed] user={user} output_file={tmp_file} error={message!r}"
            )
            return None, Observation(output=message, exit_code=1, failure_reason=message)

        self._background_handles[pid] = handle
        self._append_live_log(
            f"[{log_label}.submitted] user={user} output_file={tmp_file} pid={pid}"
        )
        return pid, None

    def _resolve_command_context(
        self,
        *,
        cmd: str,
        session: str | None = None,
        user: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _CommandContext:
        merged_env = {str(k): str(v) for k, v in dict(env or {}).items()}
        effective_cwd = cwd

        if session is not None:
            state = self._sessions.get(session, _SessionState(env={}))
            if state.env:
                merged_env = {**state.env, **merged_env}
            if effective_cwd is None:
                effective_cwd = state.cwd
            effective_user = user or state.remote_user or DEFAULT_SESSION_USER
        else:
            effective_user = user or DEFAULT_SANDBOX_USER

        return _CommandContext(
            command=cmd,
            user=effective_user,
            cwd=effective_cwd,
            env=merged_env or None,
        )

    async def _merge_command_env(
        self,
        *,
        user: str | None,
        env: dict[str, str] | None,
    ) -> dict[str, str]:
        merged_env = await self._build_command_env(user=user)
        if env:
            merged_env.update({str(k): str(v) for k, v in dict(env).items()})
        return merged_env

    async def _disconnect_background_handles(self) -> None:
        handles = list(self._background_handles.values())
        self._background_handles.clear()

        def _disconnect_all() -> None:
            for handle in handles:
                try:
                    handle.disconnect()
                except Exception:
                    pass

        if handles:
            await asyncio.to_thread(_disconnect_all)

    async def _run_command(
        self,
        command: str,
        *,
        timeout: float | None,
        user: str | None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Observation:
        if self._sandbox is None:
            return Observation(output="sandbox not started", exit_code=1, failure_reason="sandbox not started")
        log_user = user if user is not None else "<default>"
        self._append_live_log(
            f"[command.start] user={log_user} timeout={timeout} cwd={cwd!r} command={command}"
        )
        merged_env = await self._merge_command_env(user=user, env=env)
        if merged_env:
            self._append_live_log(
                f"[command.env] user={log_user} keys={sorted(merged_env.keys())}"
            )

        def _on_stdout(chunk: str) -> None:
            text = str(chunk or "")
            if text:
                self._append_live_log_lines("command.stream.stdout", text)

        def _on_stderr(chunk: str) -> None:
            text = str(chunk or "")
            if text:
                self._append_live_log_lines("command.stream.stderr", text)

        try:
            result = await asyncio.to_thread(
                self._sandbox.commands.run,
                command,
                cwd=cwd,
                timeout=timeout,
                user=user,
                envs=merged_env or None,
                on_stdout=_on_stdout,
                on_stderr=_on_stderr,
            )
            output = result.stdout if result.exit_code == 0 else (result.stdout or result.stderr)
            self._append_live_log(
                f"[command.end] exit_code={result.exit_code} "
                f"stdout_bytes={len((result.stdout or '').encode('utf-8'))} "
                f"stderr_bytes={len((result.stderr or '').encode('utf-8'))}"
            )
            if result.stdout:
                self._append_live_log_lines("command.end.stdout", result.stdout)
            if result.stderr:
                self._append_live_log_lines("command.end.stderr", result.stderr)
            return Observation(output=output or "", exit_code=result.exit_code, failure_reason=result.stderr or "")
        except CommandExitException as exc:
            output = exc.stdout or exc.stderr or ""
            self._append_live_log(
                f"[command.error] exit_code={exc.exit_code} "
                f"stdout_bytes={len((exc.stdout or '').encode('utf-8'))} "
                f"stderr_bytes={len((exc.stderr or '').encode('utf-8'))}"
            )
            if exc.stdout:
                self._append_live_log_lines("command.error.stdout", exc.stdout)
            if exc.stderr:
                self._append_live_log_lines("command.error.stderr", exc.stderr)
            return Observation(output=output, exit_code=exc.exit_code, failure_reason=exc.stderr or output)
        except Exception as exc:
            self._append_live_log(f"[command.exception] error={exc!r}")
            return Observation(output=str(exc), exit_code=1, failure_reason=str(exc))

    async def _capture_login_env(self, user: str) -> dict[str, str]:
        """Capture the target user's login-shell environment for stable session replay."""
        cached = self._login_env_cache.get(user)
        if cached is not None:
            return dict(cached)
        if self._sandbox is None:
            raise RuntimeError("sandbox not started")
        try:
            probe = await asyncio.to_thread(
                self._sandbox.commands.run,
                "env -0",
                timeout=30,
                user=user,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to capture login env for user={user!r}: {exc!r}"
            ) from exc
        if probe.exit_code != 0:
            raise RuntimeError(
                f"Failed to capture login env for user={user!r}: {probe.stderr or probe.stdout}"
            )
        raw = probe.stdout or ""
        env: dict[str, str] = {}
        for entry in raw.split("\x00"):
            if not entry or "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            if key in {"PWD", "OLDPWD", "SHLVL", "_"}:
                continue
            env[key] = value
        self._login_env_cache[user] = dict(env)
        return env

    async def _merge_user_toolchain_env(self, root_env: dict[str, str]) -> dict[str, str]:
        """Keep root identity semantics, but supplement missing toolchain env from user."""
        try:
            user_env = await self._capture_login_env("user")
        except Exception as exc:
            self._append_live_log(f"[create_session.user_env_skip] error={exc!r}")
            return root_env

        merged = dict(root_env)
        merged["PATH"] = self._merge_path_values(root_env.get("PATH", ""), user_env.get("PATH", ""))
        for key in USER_ENV_MERGE_KEYS:
            if not merged.get(key) and user_env.get(key):
                merged[key] = user_env[key]
        return merged

    @staticmethod
    def _merge_path_values(primary: str, secondary: str) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for raw in (primary, secondary):
            for item in str(raw or "").split(":"):
                entry = item.strip()
                if not entry or entry in seen:
                    continue
                seen.add(entry)
                parts.append(entry)
        return ":".join(parts)

    def _supplement_with_image_runtime_env(self, base_env: dict[str, str]) -> dict[str, str]:
        if not self._image_runtime_env:
            return base_env
        merged = dict(base_env)
        for key, value in self._image_runtime_env.items():
            if key in VOLATILE_ENV_KEYS:
                continue
            if key not in merged or merged.get(key) in {None, ""}:
                merged[key] = value
        return merged

    async def _build_command_env(self, *, user: str | None) -> dict[str, str]:
        """Return only the *supplement* the caller needs to layer on top of login env.

        ``inspire_sandbox.commands.run(envs=...)`` is additive on top of the
        user's default login env, so passing the entire login env back would
        duplicate it.  We capture the login env and return just the
        ``image_runtime_env`` keys that the login shell did not already define
        (and that aren't volatile shell vars).  Callers add their own vars on
        top of this result.
        """
        target_user = user or DEFAULT_SANDBOX_USER
        try:
            login_env = await self._capture_login_env(target_user)
        except Exception as exc:
            self._append_live_log(f"[command.env_capture_skip] user={target_user} error={exc!r}")
            login_env = {}
        return {
            key: value
            for key, value in self._image_runtime_env.items()
            if key not in VOLATILE_ENV_KEYS and login_env.get(key) in (None, "")
        }

    def _best_effort_sandbox_ip(self) -> str | None:
        """Return the sandbox's own primary IP by running hostname -I inside it."""
        if self._sandbox is None:
            return None
        try:
            result = self._sandbox.commands.run(
                "hostname -I | awk '{print $1}'", timeout=5, user=DEFAULT_SANDBOX_USER
            )
            return (result.stdout or "").strip() or None
        except Exception:
            return None

    @staticmethod
    def _build_nohup_detached_message(tmp_file: str, success: bool, detail: str | None) -> str:
        status = "completed" if success else "finished with errors"
        lines = [
            "Command executed in ROCK NOHUP mode using inspire background execution.",
            f"Status: {status}",
            f"Output file: {tmp_file}",
            "Use Sandbox.read_file(...), download APIs, or run 'cat <file>' inside the session to inspect the result.",
        ]
        if detail:
            lines.append(f"Detail: {detail}")
        return "\n".join(lines)
