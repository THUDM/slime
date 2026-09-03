"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import random
import shlex
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path
_RpcResult = TypeVar("_RpcResult")
_DAYTONA_INSTANCE_LABEL = "slime.instance"


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.

    ``idempotent`` is a hint for the backend's transport-retry policy: callers
    mark whether re-sending the command after a severed response is safe to
    replay (see ``E2BSandbox._rpc_retry``). Backends without retries may
    ignore it.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


EXIT_TIME_BUDGET_EXCEEDED = -1


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: int) -> int:
    """Poll a detached command's exit-code marker until it appears, returning the
    exit code (or ``EXIT_TIME_BUDGET_EXCEEDED`` if the budget runs out first).

    The 5s ``test -f && cat`` polls are deliberately short, idempotent RPCs --
    they keep the sandbox alive against idle GC while the detached command runs
    over a stream the gateway can't sever.
    """
    deadline = time.time() + time_budget_sec
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {done_file} && cat {done_file}", user=user, timeout=15, check=False)
        if ec == 0 and (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(
    sb: Sandbox,
    *,
    cmd: str,
    time_budget_sec: int,
    tag: str,
    user: str = "root",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    out_file: str | None = None,
    want_output: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` to completion detached, returning ``(exit_code, output)``.

    A plain ``sb.exec`` keeps an HTTP/2 stream open for the command's whole
    runtime, so a long-running command (build, test suite) outlives what the
    E2B gateway will hold a single response stream open for: the stream gets
    severed mid-run and we lose the exit code with no safe way to retry a
    non-idempotent command. Instead we ``setsid`` the command fully detached,
    redirect its output to a file, and have it drop its exit code into a marker
    file. The caller side then becomes a sequence of short, idempotent RPCs --
    write the launcher, fire-and-forget the spawn, then poll for the marker (see
    ``_await_done_marker``) -- none of which depend on a stream staying alive,
    and the polling doubles as an idle-GC keepalive while the command runs.
    """
    out_file = out_file or f"/tmp/.{tag}.out"
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    prefix = f"cd {workdir}\nexport HOME=/home/{user}\n" if workdir else ""
    launcher_body = f"#!/bin/bash\n{prefix}{cmd}\necho $? > {done_file}\n"
    await sb.write_file(launcher, launcher_body, user=user)

    # Clear the previous invocation's state in its own idempotent RPC, *before*
    # the guarded spawn. The mkdir guard below exists only to dedupe transport
    # retries of this one spawn (a severed response replayed by _rpc_retry); it
    # must not survive into the next logical invocation of the same tag (e.g.
    # install_npm_cli's retry loop), which would skip the spawn entirely and
    # read the previous run's stale exit-code marker. Callers must not overlap
    # two exec_and_wait calls with the same tag.
    await sb.exec(
        f"rm -rf {lock_dir}; rm -f {out_file} {done_file}",
        user=user,
        timeout=30,
        check=True,
        idempotent=True,
    )
    await sb.exec(
        f"chmod +x {launcher}; "
        f"mkdir {lock_dir} 2>/dev/null || exit 0; "
        f"setsid bash {launcher} < /dev/null > {out_file} 2>&1 &",
        user=user,
        env=env,
        timeout=30,
        check=True,
        idempotent=True,
    )
    exit_code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec)
    if exit_code == 0 and not want_output:
        return exit_code, ""
    if want_output:
        return exit_code, await sb.read_file(out_file, user=user)
    _, tail, _ = await sb.exec(f"tail -c 512 {out_file} 2>/dev/null", user=user, timeout=15, check=False)
    return exit_code, tail or ""


def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    size_env = ("SLIME_AGENT_E2B_SANDBOX_SIZE", "SWE_E2B_SANDBOX_SIZE")

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_size = "md"
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
        size: str | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self.size = size if size is not None else self._size_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @classmethod
    def _size_from_env(cls) -> str:
        return _getenv(*cls.size_env, default=cls.default_size)

    # Transient client-side failures safe to retry.
    _TRANSIENT_RPC_ERRORS = frozenset(
        {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }
    )

    @classmethod
    def _is_transient_rpc_error(cls, e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in cls._TRANSIENT_RPC_ERRORS:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        """Run coro_factory() with retries for transient E2B RPC failures.

        :param idempotent: When False, a transient failure is re-raised instead
            of retried: re-running a non-idempotent op (e.g. a process-spawning
            exec) after a severed response could double-execute it. Idempotent
            ops (the default: create / read_file / write_file / short read-only
            execs) retry as before.
        """
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    await self._reset_conn_pool()
                    ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                    backoff = random.uniform(0.0, ceiling)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def _reset_conn_pool(self) -> None:
        """Tear down the sandbox's httpcore pool so the next RPC reconnects."""
        try:
            pool = self._sb._transport.pool  # httpcore.AsyncConnectionPool
            await pool.aclose()
        except Exception as e:
            logger.debug("[agent.sandbox] conn-pool reset skipped: %s", e)

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  # type: ignore

        md = {self.image_metadata_key: self.image}

        if self.size:
            prefix = self.image_metadata_key.rsplit("/", 1)[0] if "/" in self.image_metadata_key else ""
            size_key = f"{prefix}/size" if prefix else "size"
            md[size_key] = self.size

        self._sb = await self._rpc_retry("create", lambda: AsyncSandbox.create(timeout=self.timeout, metadata=md))
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
                idempotent=idempotent,
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


class DaytonaSandbox:
    """Async Daytona implementation of the existing Slime sandbox contract.

    Exactly one of ``image`` or ``snapshot`` is required. Snapshots are treated
    as immutable provider-side environment references; Slime still transfers
    task inputs and outputs through ``write_file`` and ``read_file``.
    """

    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    create_timeout_sec_env = ("SLIME_AGENT_DAYTONA_CREATE_TIMEOUT_SEC",)
    connection_pool_size_env = ("SLIME_AGENT_DAYTONA_CONNECTION_POOL_SIZE",)

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_create_timeout_sec = 600
    default_connection_pool_size = 250
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0
    _TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
    _TRANSIENT_ERROR_NAMES = frozenset(
        {
            "ClientConnectionError",
            "ConnectError",
            "ConnectTimeout",
            "DaytonaRateLimitError",
            "DaytonaTimeoutError",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "ServerDisconnectedError",
            "WriteError",
            "WriteTimeout",
        }
    )

    def __init__(
        self,
        image: str | None = None,
        *,
        snapshot: str | None = None,
        timeout: int | None = None,
        rpc_retries: int | None = None,
        create_timeout: int | None = None,
        connection_pool_size: int | None = None,
        cpu: int | None = None,
        memory_gb: int | None = None,
        disk_gb: int | None = None,
        network_block_all: bool = False,
    ) -> None:
        if image is not None and not image.strip():
            raise ValueError("Daytona image must not be empty")
        if snapshot is not None and not snapshot.strip():
            raise ValueError("Daytona snapshot must not be empty")
        if (image is None) == (snapshot is None):
            raise ValueError("DaytonaSandbox requires exactly one of image or snapshot")
        for name, value in (
            ("timeout", timeout),
            ("rpc_retries", rpc_retries),
            ("create_timeout", create_timeout),
            ("connection_pool_size", connection_pool_size),
            ("cpu", cpu),
            ("memory_gb", memory_gb),
            ("disk_gb", disk_gb),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if snapshot is not None and any(value is not None for value in (cpu, memory_gb, disk_gb)):
            raise ValueError("Daytona snapshot creation does not accept resource overrides")
        self.image = image
        self.snapshot = snapshot
        self.timeout = (
            timeout if timeout is not None else self._int_from_env(self.lifetime_sec_env, self.default_lifetime_sec)
        )
        self.rpc_retries = (
            rpc_retries
            if rpc_retries is not None
            else self._int_from_env(self.rpc_retries_env, self.default_rpc_retries)
        )
        self.create_timeout = (
            create_timeout
            if create_timeout is not None
            else self._int_from_env(self.create_timeout_sec_env, self.default_create_timeout_sec)
        )
        self.connection_pool_size = (
            connection_pool_size
            if connection_pool_size is not None
            else self._int_from_env(self.connection_pool_size_env, self.default_connection_pool_size)
        )
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.disk_gb = disk_gb
        self.network_block_all = network_block_all
        self._instance_label = f"slime-{uuid4().hex}"
        self._client = None
        self._sb = None
        self.sandbox_id = ""

    @staticmethod
    def _int_from_env(names: tuple[str, ...], default: int) -> int:
        value = int(_getenv(*names, default=str(default)))
        if value < 1:
            raise ValueError(f"{names[0]} must be a positive integer")
        return value

    @classmethod
    def _is_transient_rpc_error(cls, error: BaseException) -> bool:
        if type(error).__name__ in cls._TRANSIENT_ERROR_NAMES:
            return True
        status_code = getattr(error, "status_code", None)
        if status_code in cls._TRANSIENT_STATUS_CODES:
            return True
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) in cls._TRANSIENT_STATUS_CODES

    async def _rpc_retry(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[_RpcResult]],
        *,
        idempotent: bool = True,
    ) -> _RpcResult:
        last_error = None
        for attempt in range(self.rpc_retries):
            try:
                return await operation()
            except Exception as error:
                if not idempotent or not self._is_transient_rpc_error(error):
                    raise
                last_error = error
                if attempt + 1 < self.rpc_retries:
                    ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                    backoff = random.uniform(0.0, ceiling)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs",
                        operation_name,
                        type(error).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
        assert last_error is not None
        raise last_error

    async def __aenter__(self) -> DaytonaSandbox:
        sdk = _daytona_sdk()
        self._client = sdk.AsyncDaytona(sdk.DaytonaConfig(connection_pool_maxsize=self.connection_pool_size))
        resources = None
        if any(value is not None for value in (self.cpu, self.memory_gb, self.disk_gb)):
            resources = sdk.Resources(cpu=self.cpu, memory=self.memory_gb, disk=self.disk_gb)
        common_params: dict[str, Any] = {
            "auto_stop_interval": max(1, math.ceil(self.timeout / 60)),
            "auto_delete_interval": 0,
            "network_block_all": self.network_block_all,
            "ephemeral": True,
            "os_user": "root",
            "labels": {_DAYTONA_INSTANCE_LABEL: self._instance_label},
        }
        if self.snapshot is not None:
            params = sdk.CreateSandboxFromSnapshotParams(snapshot=self.snapshot, **common_params)
        else:
            params = sdk.CreateSandboxFromImageParams(image=self.image, resources=resources, **common_params)
        try:
            self._sb = await self._rpc_retry(
                "create", lambda: self._client.create(params=params, timeout=self.create_timeout)
            )
        except asyncio.CancelledError:
            await self._cleanup_failed_create(sdk)
            raise
        except Exception:
            await self._cleanup_failed_create(sdk)
            raise
        self.sandbox_id = self._sb.id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._rpc_retry("delete", lambda: self._sb.delete())
        except Exception as error:
            logger.warning("[agent.sandbox] Daytona delete %s failed: %s", self.sandbox_id[:8], error)
        finally:
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception as error:
                    logger.warning("[agent.sandbox] Daytona client close failed: %s", error)
            self._sb = None
            self._client = None

    async def _cleanup_failed_create(self, sdk: Any) -> None:
        await asyncio.shield(self._reap_orphaned_sandboxes(sdk))
        try:
            await self._client.close()
        except Exception as error:
            logger.warning("[agent.sandbox] Daytona client close after failed create: %s", error)
        self._client = None

    async def _reap_orphaned_sandboxes(self, sdk: Any) -> None:
        """Best-effort cleanup scoped to this creation's unique label."""
        query = sdk.ListSandboxesQuery(labels={_DAYTONA_INSTANCE_LABEL: self._instance_label})
        for attempt in range(6):
            try:
                orphans = [sandbox async for sandbox in self._client.list(query)]
                for orphan in orphans:
                    await orphan.delete()
                if orphans:
                    return
            except Exception as error:
                logger.warning("[agent.sandbox] Daytona orphan cleanup failed: %s", error)
            if attempt < 5:
                await asyncio.sleep(2)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        if self._sb is None:
            raise RuntimeError("Daytona sandbox is not running")
        command = cmd if user == "root" else f"su {shlex.quote(user)} -s /bin/bash -c {shlex.quote(cmd)}"
        result = await self._rpc_retry(
            f"exec({cmd[:60]!r})",
            lambda: self._sb.process.exec(command, env=env, timeout=timeout),
            idempotent=idempotent,
        )
        stdout = result.result or ""
        if check and result.exit_code != 0:
            raise RuntimeError(f"Daytona exec failed (exit={result.exit_code}): {cmd[:120]}\n{stdout[-400:]}")
        return result.exit_code, stdout, ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if self._sb is None:
            raise RuntimeError("Daytona sandbox is not running")
        source: str | bytes = (
            str(content) if isinstance(content, Path) else content.encode() if isinstance(content, str) else content
        )
        await self._rpc_retry("write_file", lambda: self._sb.fs.upload_file(source, sandbox_path))
        if user != "root":
            await self.exec(f"chown {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(sandbox_path)}", check=True)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        del user  # Daytona's byte download API has no user-specific read mode.
        if self._sb is None:
            raise RuntimeError("Daytona sandbox is not running")
        try:
            content = await self._rpc_retry("read_file", lambda: self._sb.fs.download_file(sandbox_path))
        except Exception as error:
            if self._is_not_found_error(error):
                return ""
            raise
        if content is None:
            return ""
        return content.decode(errors="replace") if isinstance(content, bytes) else str(content)

    @staticmethod
    def _is_not_found_error(error: BaseException) -> bool:
        if type(error).__name__ == "DaytonaNotFoundError" or getattr(error, "status_code", None) == 404:
            return True
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) == 404


def _daytona_sdk() -> Any:
    """Load Daytona only when its backend is selected."""
    try:
        import daytona  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("Daytona is not installed. Install daytona>=0.182.0 to use this backend.") from error
    return daytona


def create_sandbox(image: str | None, *, snapshot: str | None = None, backend: str | None = None) -> Sandbox:
    """Create the configured sandbox provider without changing its public protocol."""
    selected_backend = backend or _getenv("SLIME_AGENT_SANDBOX_BACKEND", default="e2b")
    if selected_backend == "e2b":
        if snapshot is not None:
            raise ValueError("E2B sandbox selection does not support Daytona snapshots")
        if image is None:
            raise ValueError("E2B sandbox selection requires an image")
        return E2BSandbox(image)
    if selected_backend == "daytona":
        return DaytonaSandbox(image=image if snapshot is None else None, snapshot=snapshot)
    raise ValueError(f"unknown sandbox backend {selected_backend!r}; expected 'e2b' or 'daytona'")


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent",
        user="root",
        check=True,
        timeout=60,
    )
