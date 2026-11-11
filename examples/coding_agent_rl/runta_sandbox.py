"""Slime example adapter for Runta runtimes (https://runta.com/docs/).

A Runta runtime is a full cloud machine, not a container, and it is not created
from a Docker image. The task image therefore runs as one long-lived container
inside the runtime (Docker is preinstalled there) and every command, read and
write goes through ``docker exec`` — so user switching, the image's own
filesystem and its login environment behave as they do on the other providers.

Install the separately distributed SDK with
``pip install -r examples/coding_agent_rl/requirements-runta.txt`` and
authenticate with ``RUNTA_TOKEN`` (or ``runta login``); ``RUNTA_ENDPOINT``
overrides the API endpoint.

Settings (environment variables):
  RUNTA_VCPUS / RUNTA_MEMORY_MIB   per-sandbox resources (default 2 / 4096)
  RUNTA_CREATE_TIMEOUT_SEC         runtime boot + image pull deadline (default 600)
  SLIME_AGENT_SANDBOX_LIFETIME_SEC server-side lifetime (default 3600): Runta
                                   shuts the runtime down after it even if the
                                   trainer died without cleaning up
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import shlex
from pathlib import Path

from slime.agent.sandbox import ExecResult, FileContent, SandboxTerminatedError

logger = logging.getLogger(__name__)

_CONTAINER = "slime-task"
# Exit code the read helper uses for "no such file", so a missing completion
# marker is told apart from a permission or lifecycle failure.
_MISSING_FILE_EXIT = 44
# The SDK keeps 4 MiB of each output stream unless told otherwise and marks the
# rest as truncated; patches and file contents are data, so allow far more and
# refuse a result that is still cut short.
_READ_LIMIT_BYTES = 64 * 1024 * 1024
# How the SDK reports a command it terminated at its timeout: this exit code
# with this notice ending stderr.
_TIMEOUT_EXIT = 124
_TIMEOUT_NOTICE = re.compile(r"Command timed out after \S+ seconds\.$")
_DELETE_AFTER_SHUTDOWN_SEC = 600


def _lifetime_sec() -> int:
    for name in ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return int(value)
    return 3600


class RuntaSandbox:
    def __init__(self, image: str) -> None:
        self.image = image
        self.sandbox_id = ""
        self._client = None
        self._runtime = None

    async def __aenter__(self) -> RuntaSandbox:
        from runta import AsyncRunta

        create_timeout = float(os.environ.get("RUNTA_CREATE_TIMEOUT_SEC", "600"))
        self._client = AsyncRunta()
        try:
            await self._create_runtime()
            await self._runtime.wait_until_ready(timeout=create_timeout)
            # --network host: the container shares the runtime's network, so the
            # runtime's egress policy applies to the task unchanged.
            started = await self._runtime.exec(
                ["docker", "run", "-d", "--name", _CONTAINER, "--network", "host", self.image, "sleep", "infinity"],
                timeout=create_timeout,
            )
            if started.exit_code != 0:
                raise RuntimeError(f"runta could not start image {self.image}: {started.stderr_text[-400:]}")
        except BaseException:
            await self._close()
            raise
        logger.info("[runta] %s image=%s", self.sandbox_id, self.image)
        return self

    async def _create_runtime(self) -> None:
        """Create the runtime and keep its handle before waiting for it to boot.

        The runtime exists server-side once the request lands, so a boot that
        then fails, times out or is cancelled must still find the handle to
        delete. The request itself is shielded for the same reason: a cancelled
        caller waits for the handle and only then sees the cancellation.
        """

        async def create() -> None:
            self._runtime = await self._client.runtimes.create(
                f"slime-{secrets.token_hex(6)}",
                vcpus=int(os.environ.get("RUNTA_VCPUS", "2")),
                memory_mib=int(os.environ.get("RUNTA_MEMORY_MIB", "4096")),
                auto_shutdown_after_secs=_lifetime_sec(),
                auto_delete_after_secs=_DELETE_AFTER_SHUTDOWN_SEC,
                wait=False,
            )
            self.sandbox_id = self._runtime.id

        request = asyncio.ensure_future(create())
        cancelled = None
        while not request.done():
            try:
                await asyncio.shield(request)
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                break  # the request failed; raised from result() below
        if cancelled is not None:
            raise cancelled
        request.result()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._close()

    async def _close(self) -> None:
        runtime, client = self._runtime, self._client
        self._runtime = self._client = None
        try:
            if runtime is not None:
                await runtime.delete()
        except Exception as error:
            # The server-side lifetime still reclaims the runtime.
            logger.warning("[runta] delete %s failed: %s", self.sandbox_id[:8], error)
        finally:
            if client is not None:
                await client.close()

    async def _run(self, argv: list[str], *, timeout: float, stdin: bytes | None = None, max_output_bytes=None):
        """Run ``argv`` on the runtime; a runtime that has stopped ends the trajectory."""
        from runta import ApiError

        try:
            return await self._runtime.exec(argv, timeout=timeout, stdin=stdin, max_output_bytes=max_output_bytes)
        except ApiError as error:
            if not await self._is_running():
                raise SandboxTerminatedError(f"runta runtime {self.sandbox_id} is not running") from error
            raise

    async def _is_running(self) -> bool:
        from runta import ApiError

        try:
            status = (await self._runtime.refresh_info()).status
        except ApiError as error:
            return error.status_code != 404
        return str(getattr(status, "value", status)).lower() == "running"

    def _docker_exec(self, user: str, env: dict[str, str] | None = None, *, interactive: bool = False) -> list[str]:
        argv = ["docker", "exec", "-u", user]
        if interactive:
            argv.append("-i")
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv.append(_CONTAINER)
        return argv

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
        # The Runta SDK reconnects to the same remote task after a dropped
        # connection instead of re-sending it, so ``idempotent`` needs no handling.
        # A login shell matches the E2B provider's command environment. As there,
        # a command that outlives ``timeout`` raises instead of returning an exit
        # code, so a grader that ran out of time is not scored as a failed test.
        result = await self._run(
            [*self._docker_exec(user, env), "bash", "-lc", cmd], timeout=timeout, max_output_bytes=_READ_LIMIT_BYTES
        )
        if result.exit_code == _TIMEOUT_EXIT and _TIMEOUT_NOTICE.search(result.stderr_text):
            raise RuntimeError(f"runta exec timed out after {timeout}s: {cmd[:120]}")
        if result.stdout_truncated or result.stderr_truncated:
            raise RuntimeError(f"runta exec output exceeds {_READ_LIMIT_BYTES} bytes: {cmd[:120]}")
        if check and result.exit_code != 0:
            raise RuntimeError(f"runta exec failed (exit={result.exit_code}): {cmd[:120]}\n{result.stderr_text[:400]}")
        return result.exit_code, result.stdout_text, result.stderr_text

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            await self._write_host_file(sandbox_path, content, user)
            return
        data = content.encode() if isinstance(content, str) else content
        # The content travels on stdin, so it is never interpolated into a shell.
        writer = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        result = await self._run(
            [*self._docker_exec(user, interactive=True), "sh", "-c", writer, "sh", sandbox_path],
            timeout=600,
            stdin=data,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"runta write_file({sandbox_path}) failed: {result.stderr_text[:400]}")

    async def _write_host_file(self, sandbox_path: str, host_path: Path, user: str) -> None:
        """Stream a host file to the runtime, then copy it into the container as ``user``."""
        staged = f"/tmp/slime-upload-{secrets.token_hex(8)}"
        await self._runtime.files.upload(str(host_path), staged)
        path, target = shlex.quote(sandbox_path), shlex.quote(f"{_CONTAINER}:{sandbox_path}")
        owner = shlex.quote(user)
        # -u root: the image's own USER may not be allowed to create the parent
        # directory or to hand the file to ``user``.
        copy = (
            f'docker exec -u root {_CONTAINER} sh -c \'mkdir -p "$(dirname "$1")"\' sh {path}'
            f" && docker cp {staged} {target}"
            f" && docker exec -u root {_CONTAINER} chown {owner} {path}"
            f"; rc=$?; rm -f {staged}; exit $rc"
        )
        result = await self._run(["sh", "-c", copy], timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(
                f"runta write_file({sandbox_path} <- {host_path.name}) failed: {result.stderr_text[:400]}"
            )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        reader = f'[ -e "$1" ] || exit {_MISSING_FILE_EXIT}; cat "$1"'
        result = await self._run(
            [*self._docker_exec(user), "sh", "-c", reader, "sh", sandbox_path],
            timeout=600,
            max_output_bytes=_READ_LIMIT_BYTES,
        )
        if result.exit_code == _MISSING_FILE_EXIT:
            return ""
        if result.exit_code != 0:
            raise RuntimeError(f"runta read_file({sandbox_path}) failed: {result.stderr_text[:400]}")
        if result.stdout_truncated:
            raise RuntimeError(f"runta read_file({sandbox_path}) exceeds {_READ_LIMIT_BYTES} bytes")
        return result.stdout_text
