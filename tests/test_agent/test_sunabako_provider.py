"""Provider selection, memory enforcement defaults, and cleanup on errors."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.sandbox import create_sandbox
from examples.coding_agent_rl.sunabako_sandbox import SunabakoSandbox
from slime.agent.sandbox import E2BSandbox, SandboxTerminatedError

NUM_GPUS = 0


@pytest.mark.unit
def test_provider_selection_defaults_to_sunabako(monkeypatch):
    monkeypatch.delenv("SWE_SANDBOX_PROVIDER", raising=False)
    assert isinstance(create_sandbox("image"), SunabakoSandbox)
    monkeypatch.setenv("SWE_SANDBOX_PROVIDER", "sunabako")
    assert isinstance(create_sandbox("image"), SunabakoSandbox)
    monkeypatch.setenv("SWE_SANDBOX_PROVIDER", "e2b")
    assert isinstance(create_sandbox("image"), E2BSandbox)
    monkeypatch.setenv("SWE_SANDBOX_PROVIDER", "misspelled")
    with pytest.raises(ValueError, match="Unknown SWE_SANDBOX_PROVIDER"):
        create_sandbox("image")


@pytest.mark.unit
@pytest.mark.parametrize("test_memory,expected_mode", [(None, "cgroup"), ("1", "rss")])
@pytest.mark.parametrize("image_memory", [None, 4096])
@pytest.mark.parametrize("runtime", ["proot", "native"])
def test_memory_mode_requires_explicit_test_opt_in(
    monkeypatch, tmp_path, test_memory, expected_mode, image_memory, runtime
):
    images = tmp_path / "images.json"
    info = {"rootfs": "/images/task/rootfs", "env": {"OMP_NUM_THREADS": "64", "TASK_SETTING": "kept"}}
    if image_memory is not None:
        info["memory_mb"] = image_memory
    images.write_text(json.dumps({"image": info}))
    monkeypatch.setenv("SUNABAKO_MEMORY_MB", "2048")
    monkeypatch.setenv("SUNABAKO_PIDS_MAX", "1024")
    monkeypatch.setenv("SUNABAKO_RUNTIME", runtime)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("GOMAXPROCS", "2")
    monkeypatch.setenv("SUNABAKO_IMAGES", str(images))
    monkeypatch.setenv("SUNABAKO_CLUSTER", "cluster.json")
    monkeypatch.delenv("SUNABAKO_ARTIFACTS", raising=False)
    monkeypatch.delenv("SUNABAKO_ALLOW_TEST_MEMORY", raising=False)
    if test_memory is not None:
        monkeypatch.setenv("SUNABAKO_ALLOW_TEST_MEMORY", test_memory)
    client = SimpleNamespace(sandbox_id="test", node=SimpleNamespace(name="node"), kill=AsyncMock())
    create = AsyncMock(return_value=client)
    monkeypatch.setitem(
        sys.modules,
        "sunabako",
        SimpleNamespace(
            AsyncSandbox=SimpleNamespace(create=create), Cluster=SimpleNamespace(from_file=lambda _: object())
        ),
    )

    async def run():
        sandbox = SunabakoSandbox("image")
        assert not sandbox.emulates_root
        async with sandbox:
            assert sandbox.emulates_root == (runtime == "proot")
            # The capability describes the created runtime, not later changes
            # to the process configuration.
            monkeypatch.setenv("SUNABAKO_RUNTIME", "native" if runtime == "proot" else "proot")
            assert sandbox.emulates_root == (runtime == "proot")

    asyncio.run(run())
    assert create.call_args.kwargs["memory_mode"] == expected_mode
    assert create.call_args.kwargs["runtime"] == runtime
    assert create.call_args.kwargs["memory_mb"] == (image_memory or 2048)
    assert create.call_args.kwargs["pids_max"] == 1024
    assert create.call_args.kwargs["envs"]["OMP_NUM_THREADS"] == "1"
    assert create.call_args.kwargs["envs"]["GOMAXPROCS"] == "2"
    assert create.call_args.kwargs["envs"]["TASK_SETTING"] == "kept"
    client.kill.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.parametrize("defer_cleanup", [False, True])
def test_artifact_write_error_still_destroys_sandbox(monkeypatch, tmp_path, defer_cleanup):
    class SandboxError(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "sunabako", SimpleNamespace(SandboxError=SandboxError))
    blocked_directory = tmp_path / "file"
    blocked_directory.write_text("not a directory")
    monkeypatch.setenv("SUNABAKO_ARTIFACTS", str(blocked_directory))
    monkeypatch.setenv("SUNABAKO_DEFER_ROOTFS_DELETE", "1" if defer_cleanup else "0")
    sandbox = SunabakoSandbox("image")
    sandbox.sandbox_id = "test"
    sandbox._sandbox = SimpleNamespace(kill=AsyncMock())
    asyncio.run(sandbox.__aexit__(RuntimeError, RuntimeError("agent failed"), None))
    sandbox._sandbox.kill.assert_awaited_once_with(defer_cleanup=defer_cleanup)


def test_manifest_failure_does_not_leak_new_sandbox(monkeypatch, tmp_path):
    images = tmp_path / "images.json"
    images.write_text(json.dumps({"image": {"rootfs": "/images/task/rootfs"}}))
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setenv("SUNABAKO_IMAGES", str(images))
    monkeypatch.setenv("SUNABAKO_CLUSTER", "cluster.json")
    monkeypatch.setenv("SUNABAKO_ARTIFACTS", str(blocked))
    client = SimpleNamespace(sandbox_id="test", node=SimpleNamespace(name="node"), kill=AsyncMock())
    monkeypatch.setitem(
        sys.modules,
        "sunabako",
        SimpleNamespace(
            AsyncSandbox=SimpleNamespace(create=AsyncMock(return_value=client)),
            Cluster=SimpleNamespace(from_file=lambda _: object()),
        ),
    )
    with pytest.raises(OSError):
        asyncio.run(SunabakoSandbox("image").__aenter__())
    client.kill.assert_awaited_once()


@pytest.mark.parametrize(
    "message,missing",
    [
        ('{"error":"No such file or directory (os error 2)"}', False),
        (json.dumps({"error": 'sandbox: "No such file or directory (os error 2)"'}), True),
        (json.dumps({"error": 'file operation: "No such file or directory (os error 2)"'}), True),
        (
            json.dumps(
                {"error": "sandbox: " + json.dumps('file operation: "No such file or directory (os error 2)"')}
            ),
            True,
        ),
        (
            json.dumps({"error": "sandbox: " + json.dumps('file operation: "Permission denied (os error 13)"')}),
            False,
        ),
        ('{"error":"sandbox must be running"}', False),
        ('{"error":"Permission denied (os error 13)"}', False),
        ("Transport timed out", False),
        ('["No such file or directory (os error 2)"]', False),
    ],
)
def test_read_file_only_tolerates_missing_files(monkeypatch, message, missing):
    class SandboxError(RuntimeError):
        pass

    error = SandboxError(message)
    monkeypatch.setitem(sys.modules, "sunabako", SimpleNamespace(SandboxError=SandboxError))
    sandbox = SunabakoSandbox("image")
    sandbox._sandbox = SimpleNamespace(read_file=AsyncMock(side_effect=error))
    if missing:
        assert asyncio.run(sandbox.read_file("/tmp/.run.done", user="agent")) == ""
    else:
        with pytest.raises(SandboxError) as caught:
            asyncio.run(sandbox.read_file("/tmp/.run.done", user="agent"))
        assert caught.value is error


@pytest.mark.parametrize(
    "message,terminated",
    [("I/O timeout", False), ('{"error":"Permission denied"}', False), ('{"error":"sandbox must be running"}', True)],
)
def test_context_reports_terminal_execution_failure_after_cleanup(monkeypatch, message, terminated):
    class SandboxError(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "sunabako", SimpleNamespace(SandboxError=SandboxError))
    monkeypatch.delenv("SUNABAKO_ARTIFACTS", raising=False)
    sandbox = SunabakoSandbox("image")
    sandbox._sandbox = SimpleNamespace(kill=AsyncMock())
    error = SandboxError(message)
    if terminated:
        with pytest.raises(SandboxTerminatedError, match="sandbox must be running"):
            asyncio.run(sandbox.__aexit__(type(error), error, None))
    else:
        assert asyncio.run(sandbox.__aexit__(type(error), error, None)) is None
    sandbox._sandbox.kill.assert_awaited_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
