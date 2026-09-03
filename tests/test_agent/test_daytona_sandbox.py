import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import slime.agent.sandbox as sandbox_module
from slime.agent.sandbox import DaytonaSandbox, create_sandbox


NUM_GPUS = 0


class FakeCreateParams:
    def __init__(self, **values):
        self.values = values


class FakeFileSystem:
    def __init__(self):
        self.files = {}

    async def upload_file(self, source, destination):
        self.files[destination] = Path(source).read_bytes() if isinstance(source, str) else source

    async def download_file(self, source):
        return self.files.get(source)


class FakeProcess:
    def __init__(self):
        self.commands = []
        self.responses = []

    async def exec(self, command, env=None, timeout=None):
        self.commands.append((command, env, timeout))
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return SimpleNamespace(exit_code=0, result="")


class FakeRemoteSandbox:
    def __init__(self):
        self.id = "daytona-sandbox-id"
        self.fs = FakeFileSystem()
        self.process = FakeProcess()
        self.deleted = False

    async def delete(self):
        self.deleted = True


class FakeDaytonaClient:
    def __init__(self):
        self.configs = []
        self.created = []
        self.remote = FakeRemoteSandbox()
        self.closed = False
        self.orphans = []

    async def create(self, params, timeout):
        self.created.append((params, timeout))
        return self.remote

    async def close(self):
        self.closed = True

    async def list(self, query):
        assert query.values["labels"]
        for orphan in self.orphans:
            yield orphan


@pytest.fixture
def fake_daytona(monkeypatch):
    client = FakeDaytonaClient()
    sdk = SimpleNamespace(
        AsyncDaytona=lambda config: client,
        DaytonaConfig=lambda **values: client.configs.append(values) or values,
        Resources=lambda **values: values,
        ListSandboxesQuery=type("ListSandboxesQuery", (FakeCreateParams,), {}),
        CreateSandboxFromImageParams=type("ImageParams", (FakeCreateParams,), {}),
        CreateSandboxFromSnapshotParams=type("SnapshotParams", (FakeCreateParams,), {}),
    )
    monkeypatch.setattr(sandbox_module, "_daytona_sdk", lambda: sdk)
    return client, sdk


@pytest.mark.parametrize(
    ("sandbox", "parameter_type", "reference_field", "reference"),
    [
        (DaytonaSandbox(image="registry.example/task:1"), "ImageParams", "image", "registry.example/task:1"),
        (
            create_sandbox(None, snapshot="snapshot-123", backend="daytona"),
            "SnapshotParams",
            "snapshot",
            "snapshot-123",
        ),
    ],
)
def test_daytona_context_creates_requested_environment_and_deletes_it(
    fake_daytona, sandbox, parameter_type, reference_field, reference
):
    client, _ = fake_daytona

    async def exercise():
        async with sandbox as running:
            assert running.sandbox_id == "daytona-sandbox-id"

    asyncio.run(exercise())

    params, create_timeout = client.created[0]
    assert type(params).__name__ == parameter_type
    assert params.values[reference_field] == reference
    assert params.values["auto_delete_interval"] == 0
    assert params.values["ephemeral"] is True
    assert params.values["os_user"] == "root"
    assert create_timeout == DaytonaSandbox.default_create_timeout_sec
    assert client.remote.deleted
    assert client.closed


def test_daytona_snapshot_and_image_are_mutually_exclusive():
    with pytest.raises(ValueError, match="exactly one"):
        DaytonaSandbox(image="image", snapshot="snapshot")

    with pytest.raises(ValueError, match="exactly one"):
        DaytonaSandbox()


def test_daytona_snapshot_rejects_sdk_unsupported_resource_overrides():
    with pytest.raises(ValueError, match="does not accept resource overrides"):
        DaytonaSandbox(snapshot="snapshot", cpu=4)


def test_daytona_cancelled_create_reaps_only_its_labelled_orphan(fake_daytona):
    client, _ = fake_daytona
    orphan = FakeRemoteSandbox()
    client.orphans.append(orphan)

    async def cancelled_create(params, timeout):
        del params, timeout
        raise asyncio.CancelledError

    client.create = cancelled_create

    async def exercise():
        with pytest.raises(asyncio.CancelledError):
            async with DaytonaSandbox(snapshot="snapshot-123"):
                pytest.fail("cancelled creation must not enter the sandbox context")

    asyncio.run(exercise())

    assert orphan.deleted
    assert client.closed


def test_daytona_file_round_trip_preserves_text_bytes_and_host_files(fake_daytona, tmp_path):
    client, _ = fake_daytona
    host_file = tmp_path / "payload.bin"
    host_file.write_bytes(b"from-host\x00")

    async def exercise():
        async with DaytonaSandbox(snapshot="snapshot-123") as sandbox:
            await sandbox.write_file("/tmp/text", "hello")
            await sandbox.write_file("/tmp/bytes", b"bytes\x00")
            await sandbox.write_file("/tmp/host", host_file)
            return (
                await sandbox.read_file("/tmp/text"),
                await sandbox.read_file("/tmp/bytes"),
                await sandbox.read_file("/tmp/host"),
            )

    assert asyncio.run(exercise()) == ("hello", "bytes\x00", "from-host\x00")
    assert client.remote.deleted


def test_daytona_read_distinguishes_missing_files_from_provider_failures(fake_daytona):
    client, _ = fake_daytona

    class MissingFileError(Exception):
        status_code = 404

    class AuthenticationError(Exception):
        status_code = 401

    responses = [MissingFileError("missing"), AuthenticationError("invalid token")]

    async def download_file(_source):
        raise responses.pop(0)

    client.remote.fs.download_file = download_file

    async def exercise():
        async with DaytonaSandbox(image="image") as sandbox:
            assert await sandbox.read_file("/tmp/missing") == ""
            with pytest.raises(AuthenticationError):
                await sandbox.read_file("/tmp/private")

    asyncio.run(exercise())


def test_daytona_exec_reports_results_and_honors_check(fake_daytona):
    client, _ = fake_daytona
    client.remote.process.responses = [
        SimpleNamespace(exit_code=0, result="ok\n"),
        SimpleNamespace(exit_code=7, result="failed\n"),
    ]

    async def exercise():
        async with DaytonaSandbox(image="image") as sandbox:
            success = await sandbox.exec("echo ok", env={"NAME": "value"}, timeout=9)
            with pytest.raises(RuntimeError, match="exit=7"):
                await sandbox.exec("false", check=True)
            return success

    assert asyncio.run(exercise()) == (0, "ok\n", "")
    assert client.remote.process.commands[0] == ("echo ok", {"NAME": "value"}, 9)


def test_daytona_retries_only_replay_safe_commands(fake_daytona, monkeypatch):
    client, _ = fake_daytona

    class TransientError(Exception):
        status_code = 503

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(sandbox_module.asyncio, "sleep", no_wait)
    client.remote.process.responses = [TransientError("unavailable"), SimpleNamespace(exit_code=0, result="ok")]

    async def idempotent_command():
        async with DaytonaSandbox(image="image", rpc_retries=2) as sandbox:
            return await sandbox.exec("test -f /tmp/ready", idempotent=True)

    assert asyncio.run(idempotent_command()) == (0, "ok", "")

    client.remote.process.responses = [TransientError("unavailable"), SimpleNamespace(exit_code=0, result="wrong")]

    async def non_idempotent_command():
        async with DaytonaSandbox(image="image", rpc_retries=2) as sandbox:
            await sandbox.exec("start-job", idempotent=False)

    with pytest.raises(TransientError):
        asyncio.run(non_idempotent_command())
    assert len(client.remote.process.responses) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
