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
from slime.agent.sandbox import E2BSandbox

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
def test_memory_mode_requires_explicit_test_opt_in(monkeypatch, tmp_path, test_memory, expected_mode):
    images = tmp_path / "images.json"
    images.write_text(json.dumps({"image": {"rootfs": "/images/task/rootfs"}}))
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
        async with SunabakoSandbox("image"):
            pass

    asyncio.run(run())
    assert create.call_args.kwargs["memory_mode"] == expected_mode
    client.kill.assert_awaited_once()


@pytest.mark.unit
def test_artifact_write_error_still_destroys_sandbox(monkeypatch, tmp_path):
    blocked_directory = tmp_path / "file"
    blocked_directory.write_text("not a directory")
    monkeypatch.setenv("SUNABAKO_ARTIFACTS", str(blocked_directory))
    sandbox = SunabakoSandbox("image")
    sandbox.sandbox_id = "test"
    sandbox._sandbox = SimpleNamespace(kill=AsyncMock())
    asyncio.run(sandbox.__aexit__(RuntimeError, RuntimeError("agent failed"), None))
    sandbox._sandbox.kill.assert_awaited_once()


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
