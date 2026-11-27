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
  RUNTA_CREATE_TIMEOUT_SEC         runtime boot deadline (default 600)
  RUNTA_IMAGE_PULL_TIMEOUT_SEC     task image pull + container start deadline
                                   (default 1800); every sandbox pulls its image
  RUNTA_LEASE_SEC                  server-side lease (default 600): Runta shuts the
                                   runtime down this long after the last renewal,
                                   and an open sandbox renews every third of it, so
                                   a killed trainer's runtimes stop within one lease
  SLIME_AGENT_SANDBOX_LIFETIME_SEC hard cap on the renewals (default 3600): the
                                   runtime is shut down after it even while in use
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
# The SDK's timeout stops only the ``docker exec`` client on the runtime; the
# process inside the task container keeps running. A command therefore runs under
# ``_TRACK``, which records its process group in a PID file (removed again when
# the command ends in time), and ``_KILL`` stops a timed-out command through
# that file. ``setsid -w`` gives the command a group of its own where the exec'd
# shell does not already lead one. The command itself still runs in a login
# shell at SHLVL 1 with the caller's stderr, as it did without the wrapper; the
# wrapper's own stderr is discarded, or bash would add a notice there whenever
# the command dies from a signal.
_TRACKED = 'exec 3>&2 2>/dev/null; echo $$ > "$1"; SHLVL=0 bash -lc "$2" 2>&3 3>&-; rc=$?; rm -f "$1"; exit $rc'
_TRACK = (
    "if setsid -w true >/dev/null 2>&1; then "
    f'exec setsid -w bash -c {shlex.quote(_TRACKED)} slime-exec "$@"; fi; {_TRACKED}'
)
_KILL = (
    'pid=$(cat "$1" 2>/dev/null) || exit 0; rm -f "$1"; '
    'kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; exit 0'
)


def _lifetime_sec() -> int:
    for name in ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return int(value)
    return 3600


def _lease_sec() -> int:
    lease = int(os.environ.get("RUNTA_LEASE_SEC", "600"))
    if lease < 1:
        # Zero would switch the shutdown policy off instead of shortening it.
        raise ValueError(f"RUNTA_LEASE_SEC must be at least 1, got {lease}")
    return lease


class RuntaSandbox:
    def __init__(self, image: str, *, lease_interval: float | None = None) -> None:
        """``lease_interval``: seconds between lease renewals (default: a third of the lease)."""
        self.image = image
        self.sandbox_id = ""
        self._client = None
        self._runtime = None
        self._lease_interval = lease_interval
        self._lease = 0
        self._lease_deadline = 0.0
        self._lease_task = None

    async def __aenter__(self) -> RuntaSandbox:
        from runta import AsyncRunta

        create_timeout = float(os.environ.get("RUNTA_CREATE_TIMEOUT_SEC", "600"))
        image_timeout = float(os.environ.get("RUNTA_IMAGE_PULL_TIMEOUT_SEC", "1800"))
        self._lease, lifetime = _lease_sec(), _lifetime_sec()
        self._client = AsyncRunta()
        try:
            await self._create_runtime(min(self._lease, lifetime))
            # Renewing starts here, not once the sandbox is ready: booting and
            # pulling the image can take longer than one lease.
            self._lease_deadline = asyncio.get_running_loop().time() + lifetime
            self._lease_task = asyncio.ensure_future(self._keep_lease())
            await self._runtime.wait_until_ready(timeout=create_timeout)
            # --network host: the container shares the runtime's network, so the
            # runtime's egress policy applies to the task unchanged.
            # --entrypoint: an image's own ENTRYPOINT would take ``sleep infinity``
            # as its arguments and exit.
            started = await self._runtime.exec(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    _CONTAINER,
                    "--network",
                    "host",
                    "--entrypoint",
                    "sleep",
                    self.image,
                    "infinity",
                ],
                timeout=image_timeout,
            )
            if started.exit_code != 0:
                raise RuntimeError(f"runta could not start image {self.image}: {started.stderr_text[-400:]}")
        except BaseException:
            await self._close()
            raise
        logger.info("[runta] %s image=%s", self.sandbox_id, self.image)
        return self

    async def _create_runtime(self, shutdown_after_sec: int) -> None:
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
                auto_shutdown_after_secs=shutdown_after_sec,
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

    async def _keep_lease(self) -> None:
        """Re-arm the runtime's shutdown clock until the sandbox closes or its lifetime is used up.

        Nothing in this process survives a hard kill, so the runtime is only ever
        granted one lease: when the renewals stop, Runta shuts it down.
        """
        interval = self._lease / 3 if self._lease_interval is None else self._lease_interval
        while True:
            await asyncio.sleep(interval)
            if not await self._renew_lease():
                return

    async def _renew_lease(self) -> bool:
        """One renewal; False once the lifetime is used up and the last lease is left to expire."""
        remaining = int(self._lease_deadline - asyncio.get_running_loop().time())
        if remaining < 1:
            return False
        try:
            # Saving the policy restarts its clock, even with an unchanged value.
            await self._runtime.update_timeout_policies(auto_shutdown_after_secs=min(self._lease, remaining))
        except Exception as error:
            # The lease outlasts a missed renewal; the next one retries.
            logger.warning("[runta] lease renewal for %s failed, will retry: %s", self.sandbox_id[:8], error)
        return True

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._close()

    async def _close(self) -> None:
        lease, runtime, client = self._lease_task, self._runtime, self._client
        self._lease_task = self._runtime = self._client = None
        try:
            if lease is not None:
                # Waited for, so that no renewal is in flight during the delete.
                lease.cancel()
                await asyncio.gather(lease, return_exceptions=True)
            if runtime is not None:
                await runtime.delete()
        except Exception as error:
            # The lease still reclaims the runtime.
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
        pidfile = f"/tmp/slime-exec-{secrets.token_hex(8)}.pid"
        result = await self._run(
            [*self._docker_exec(user, env), "bash", "-c", _TRACK, "slime-exec", pidfile, cmd],
            timeout=timeout,
            max_output_bytes=_READ_LIMIT_BYTES,
        )
        if result.exit_code == _TIMEOUT_EXIT and _TIMEOUT_NOTICE.search(result.stderr_text):
            await self._kill_timed_out(pidfile)
            raise RuntimeError(f"runta exec timed out after {timeout}s: {cmd[:120]}")
        if result.stdout_truncated or result.stderr_truncated:
            raise RuntimeError(f"runta exec output exceeds {_READ_LIMIT_BYTES} bytes: {cmd[:120]}")
        if check and result.exit_code != 0:
            raise RuntimeError(f"runta exec failed (exit={result.exit_code}): {cmd[:120]}\n{result.stderr_text[:400]}")
        return result.exit_code, result.stdout_text, result.stderr_text

    async def _kill_timed_out(self, pidfile: str) -> None:
        """Kill the process group a timed-out command left running in the task container."""
        # -u root: the group may hold processes that no longer run as the command's user.
        argv = [*self._docker_exec("root"), "bash", "-c", _KILL, "slime-kill", pidfile]
        try:
            killed = await self._runtime.exec(argv, timeout=30)
            failure = killed.stderr_text[-400:] if killed.exit_code != 0 else None
        except Exception as error:
            failure = error
        if failure is not None:
            # Not raised: the caller is about to get the timeout error itself.
            logger.warning("[runta] could not stop the timed-out command in %s: %s", self.sandbox_id[:8], failure)

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
