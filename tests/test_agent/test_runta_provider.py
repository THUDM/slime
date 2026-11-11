"""Runta provider: selection, container routing, file transfer, cleanup and termination mapping."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.runta_sandbox import RuntaSandbox
from examples.coding_agent_rl.sandbox import create_sandbox
from slime.agent.sandbox import Sandbox, SandboxTerminatedError

NUM_GPUS = 0


class ApiError(Exception):
    def __init__(self, message="", *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def result(exit_code=0, stdout="", stderr="", truncated=False, stderr_truncated=False):
    return SimpleNamespace(
        exit_code=exit_code,
        stdout_text=stdout,
        stderr_text=stderr,
        stdout_truncated=truncated,
        stderr_truncated=stderr_truncated,
    )


class FakeRuntime:
    id = "runtime-1234"

    def __init__(self):
        self.calls = []
        self.results = []
        self.uploads = []
        self.deleted = 0
        self.status = "running"
        self.exec_error = None
        self.ready_error = None
        self.ready_timeout = None
        self.files = SimpleNamespace(upload=self._upload)

    async def _upload(self, src, dst):
        self.uploads.append((src, dst))

    async def exec(self, argv, timeout=None, stdin=None, max_output_bytes=None):
        self.calls.append(SimpleNamespace(argv=argv, timeout=timeout, stdin=stdin, max_output_bytes=max_output_bytes))
        if self.exec_error is not None:
            raise self.exec_error
        return self.results.pop(0) if self.results else result()

    async def wait_until_ready(self, *, timeout=240.0, poll_interval=0.5):
        self.ready_timeout = timeout
        if self.ready_error is not None:
            raise self.ready_error

    async def refresh_info(self):
        if isinstance(self.status, Exception):
            raise self.status
        return SimpleNamespace(status=self.status)

    async def delete(self):
        self.deleted += 1


@pytest.fixture
def runta(monkeypatch):
    """Install a fake ``runta`` module; returns the state the fake client records."""
    state = SimpleNamespace(runtime=FakeRuntime(), create_kwargs=None, create_error=None, create_gate=None, closed=0)

    class AsyncRunta:
        def __init__(self):
            self.runtimes = SimpleNamespace(create=self._create)

        async def _create(self, name, *, wait=True, timeout=240.0, poll_interval=0.5, **kwargs):
            state.name, state.create_kwargs = name, {**kwargs, "wait": wait}
            if state.create_error is not None:
                raise state.create_error
            # As in the SDK, the runtime exists from here on: a failed readiness
            # wait raises without handing the caller a handle to delete.
            if state.create_gate is not None:
                await state.create_gate.wait()
            if wait:
                await state.runtime.wait_until_ready(timeout=timeout, poll_interval=poll_interval)
            return state.runtime

        async def close(self):
            state.closed += 1

    monkeypatch.setitem(sys.modules, "runta", SimpleNamespace(AsyncRunta=AsyncRunta, ApiError=ApiError))
    for name in ("RUNTA_VCPUS", "RUNTA_MEMORY_MIB", "RUNTA_CREATE_TIMEOUT_SEC", "SLIME_AGENT_SANDBOX_LIFETIME_SEC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SWE_SANDBOX_LIFETIME_SEC", raising=False)
    return state


@pytest.mark.unit
def test_provider_selection_returns_runta_without_importing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "runta", None)  # importing the SDK would raise
    monkeypatch.setenv("SWE_SANDBOX_PROVIDER", "runta")
    sandbox = create_sandbox("image")
    assert isinstance(sandbox, RuntaSandbox)
    assert isinstance(sandbox, Sandbox)


@pytest.mark.unit
def test_enter_creates_runtime_with_lifetime_and_starts_task_container(monkeypatch, runta):
    monkeypatch.setenv("RUNTA_VCPUS", "4")
    monkeypatch.setenv("RUNTA_MEMORY_MIB", "8192")
    monkeypatch.setenv("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "7200")

    async def run():
        async with RuntaSandbox("repo/task:tag") as sandbox:
            assert sandbox.sandbox_id == "runtime-1234"

    asyncio.run(run())
    assert runta.name.startswith("slime-")
    assert runta.create_kwargs == {
        "vcpus": 4,
        "memory_mib": 8192,
        "auto_shutdown_after_secs": 7200,
        "auto_delete_after_secs": 600,
        "wait": False,
    }
    assert runta.runtime.ready_timeout == 600.0
    start = runta.runtime.calls[0]
    assert start.argv == [
        "docker",
        "run",
        "-d",
        "--name",
        "slime-task",
        "--network",
        "host",
        "repo/task:tag",
        "sleep",
        "infinity",
    ]
    assert runta.runtime.deleted == 1
    assert runta.closed == 1


@pytest.mark.unit
def test_failed_image_start_deletes_runtime(runta):
    runta.runtime.results = [result(125, stderr="manifest unknown")]
    with pytest.raises(RuntimeError, match="manifest unknown"):
        asyncio.run(RuntaSandbox("missing:tag").__aenter__())
    assert runta.runtime.deleted == 1
    assert runta.closed == 1


@pytest.mark.unit
def test_failed_create_closes_client(runta):
    runta.create_error = ApiError("quota", status_code=429)
    with pytest.raises(ApiError):
        asyncio.run(RuntaSandbox("image").__aenter__())
    assert runta.closed == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("runtime runtime-1234 did not become ready within 600.0s"),
        RuntimeError("runtime runtime-1234 entered crashed state"),
        ApiError("bad gateway", status_code=502),
    ],
)
def test_failed_boot_deletes_created_runtime(runta, error):
    runta.runtime.ready_error = error
    with pytest.raises(type(error)):
        asyncio.run(RuntaSandbox("image").__aenter__())
    assert runta.runtime.deleted == 1
    assert runta.closed == 1


@pytest.mark.unit
@pytest.mark.parametrize("during", ["create", "boot"])
def test_cancelled_enter_deletes_created_runtime(runta, during):
    async def run():
        reached = asyncio.Event()

        async def never_ready(**_kwargs):
            reached.set()
            await asyncio.Event().wait()

        if during == "create":
            runta.create_gate = asyncio.Event()
        else:
            runta.runtime.wait_until_ready = never_ready
        task = asyncio.ensure_future(RuntaSandbox("image").__aenter__())
        if during == "create":
            while runta.create_kwargs is None:
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()  # the request is allowed to land first
            runta.create_gate.set()
        else:
            await reached.wait()
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert runta.runtime.deleted == 1
    assert runta.closed == 1


@pytest.mark.unit
def test_exit_deletes_runtime_when_agent_fails_and_survives_delete_error(runta):
    async def failing_delete():
        raise ApiError("gone", status_code=500)

    async def run():
        async with RuntaSandbox("image"):
            runta.runtime.delete = failing_delete
            raise ValueError("agent failed")

    with pytest.raises(ValueError, match="agent failed"):
        asyncio.run(run())
    assert runta.closed == 1


@pytest.mark.unit
def test_exec_runs_login_shell_in_container_as_user(runta):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.results = [result(3, "out", "err")]
            return await sandbox.exec("echo $HOME && false", user="agent", env={"A": "b c"}, timeout=45)

    assert asyncio.run(run()) == (3, "out", "err")
    call = runta.runtime.calls[1]
    assert call.argv == [
        "docker",
        "exec",
        "-u",
        "agent",
        "-e",
        "A=b c",
        "slime-task",
        "bash",
        "-lc",
        "echo $HOME && false",
    ]
    assert call.timeout == 45
    assert call.max_output_bytes == 64 * 1024 * 1024


@pytest.mark.unit
def test_exec_check_raises_on_nonzero_exit(runta):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.results = [result(2, stderr="No such file or directory")]
            await sandbox.exec("ls /missing", check=True)

    with pytest.raises(RuntimeError, match=r"exit=2.*\n.*No such file"):
        asyncio.run(run())


@pytest.mark.unit
def test_exec_timeout_raises_instead_of_returning_exit_code(runta):
    async def run(*results):
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.results = list(results)
            return await sandbox.exec("pytest", timeout=5)

    with pytest.raises(RuntimeError, match="timed out after 5s: pytest") as raised:
        asyncio.run(run(result(124, stderr="collecting\nCommand timed out after 5 seconds.")))
    assert not isinstance(raised.value, (SandboxTerminatedError, TimeoutError))
    # The command's own exit 124 (e.g. a nested ``timeout``) is still an exit code.
    assert asyncio.run(run(result(124, stderr="Terminated\n"))) == (124, "", "Terminated\n")


@pytest.mark.unit
@pytest.mark.parametrize("stream", ["truncated", "stderr_truncated"])
def test_exec_never_returns_truncated_output(runta, stream):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.results = [result(0, "diff --git a/x b/x\n", **{stream: True})]
            await sandbox.exec("git diff --binary")

    with pytest.raises(RuntimeError, match="exceeds 67108864 bytes"):
        asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("content,expected", [("text ✓", "text ✓".encode()), (b"\x00raw", b"\x00raw")])
def test_write_file_sends_content_on_stdin_as_user(runta, content, expected):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            await sandbox.write_file("/home/agent/a b/file", content, user="agent")

    asyncio.run(run())
    call = runta.runtime.calls[1]
    assert call.argv[:6] == ["docker", "exec", "-u", "agent", "-i", "slime-task"]
    assert call.argv[-2:] == ["sh", "/home/agent/a b/file"]  # path is an argument, never shell text
    assert call.stdin == expected


@pytest.mark.unit
def test_write_file_streams_host_path_and_fixes_owner(runta, tmp_path):
    host = tmp_path / "cli.tgz"
    host.write_bytes(b"archive")

    async def run():
        async with RuntaSandbox("image") as sandbox:
            await sandbox.write_file("/opt/cli.tgz", host, user="agent")

    asyncio.run(run())
    source, staged = runta.runtime.uploads[0]
    assert source == str(host) and staged.startswith("/tmp/slime-upload-")
    script = runta.runtime.calls[1].argv[-1]
    # Root regardless of the image's USER, which may be unable to mkdir or chown.
    assert script.startswith("docker exec -u root slime-task sh -c 'mkdir -p")
    assert f"docker cp {staged} slime-task:/opt/cli.tgz" in script
    assert "docker exec -u root slime-task chown agent /opt/cli.tgz" in script
    assert script.count("docker exec") == script.count("docker exec -u root")
    assert script.endswith(f"rm -f {staged}; exit $rc")


@pytest.mark.unit
def test_write_failure_raises(runta):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.results = [result(1, stderr="Permission denied")]
            await sandbox.write_file("/root/x", "data", user="agent")

    with pytest.raises(RuntimeError, match="Permission denied"):
        asyncio.run(run())


@pytest.mark.unit
def test_read_file_returns_empty_only_for_missing_file(runta):
    async def read(*results):
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.results = list(results)
            return await sandbox.read_file("/tmp/.done", user="agent")

    assert asyncio.run(read(result(0, "0\n"))) == "0\n"
    assert asyncio.run(read(result(44))) == ""
    with pytest.raises(RuntimeError, match="Permission denied"):
        asyncio.run(read(result(1, stderr="Permission denied")))
    with pytest.raises(RuntimeError, match="exceeds"):
        asyncio.run(read(result(0, "partial", truncated=True)))


@pytest.mark.unit
@pytest.mark.parametrize(
    "status,terminated",
    [("running", False), ("shutdown", True), (SimpleNamespace(value="suspended"), True)],
)
def test_transport_error_becomes_termination_only_when_runtime_stopped(runta, status, terminated):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.exec_error = ApiError("stream closed", status_code=502)
            runta.runtime.status = status
            await sandbox.exec("true")

    with pytest.raises(SandboxTerminatedError if terminated else ApiError):
        asyncio.run(run())


@pytest.mark.unit
def test_deleted_runtime_is_reported_as_terminated(runta):
    async def run():
        async with RuntaSandbox("image") as sandbox:
            runta.runtime.exec_error = ApiError("not found", status_code=404)
            runta.runtime.status = ApiError("not found", status_code=404)
            await sandbox.read_file("/tmp/.done")

    with pytest.raises(SandboxTerminatedError):
        asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
