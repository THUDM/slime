"""Apply the SGLang patch sets to a supplied, unpatched tokenizer_control_mixin.py and call post_process_weights.

Set SGLANG_TOKENIZER_CONTROL_MIXIN to the unpatched python/sglang/srt/managers/tokenizer_control_mixin.py of the
SGLang the patches target (v0.5.21). No network access and no installed-source mutation: the file is copied to a
temporary tree and patched there. Only the two weight update methods are executed, on a stand-in manager.
"""

import ast
import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
TARGET = "python/sglang/srt/managers/tokenizer_control_mixin.py"
METHODS = ("_weight_update_session_call", "post_process_weights")


@pytest.fixture(scope="module", params=["v0.5.21", "latest"])
def manager_class(request, tmp_path_factory):
    """Patch the supplied source with one patch set and return a class holding its weight update methods."""
    source = os.environ.get("SGLANG_TOKENIZER_CONTROL_MIXIN")
    if not source:
        pytest.skip("Set SGLANG_TOKENIZER_CONTROL_MIXIN to the unpatched tokenizer_control_mixin.py")
    root = tmp_path_factory.mktemp("sglang-patch")
    target = root / TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(Path(source).read_bytes())
    for name in ("sglang.patch", "sglang-pull_weights.patch"):
        patch = str(ROOT / "docker/patch" / request.param / name)
        subprocess.run(["git", "apply", "--check", f"--include={TARGET}", patch], cwd=root, check=True)
        subprocess.run(["git", "apply", f"--include={TARGET}", patch], cwd=root, check=True)
    tree = ast.parse(target.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name in METHODS]
    assert sorted(node.name for node in nodes) == sorted(METHODS)
    for node in nodes:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    namespace = {"FanOutCommunicator": SimpleNamespace(merge_results=lambda results: results)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(target), "exec"), namespace)
    return type("Manager", (), {name: namespace[name] for name in METHODS})


class WriterLock:
    """Writer side of model_update_lock: entering waits until the requests in flight release the reader side."""

    def __init__(self, readers_released):
        self.readers_released = asyncio.Event()
        if readers_released:
            self.readers_released.set()
        self.entered = 0

    async def __aenter__(self):
        await self.readers_released.wait()
        self.entered += 1

    async def __aexit__(self, *exc):
        return False


def call_post_process_weights(manager_class, paused, readers_released):
    """Call post_process_weights on a stand-in manager; return its result, the writer lock and the scheduler calls."""

    async def run():
        """Build the manager inside the event loop and await the call with a deadline."""
        calls = []

        async def communicator(obj):
            """Stand in for the scheduler fan-out."""
            calls.append(obj)
            return ["done"]

        lock = WriterLock(readers_released)
        manager = manager_class()
        manager.auto_create_handle_loop = lambda: None
        manager.is_pause = paused
        manager.is_pause_cond = asyncio.Condition()
        manager.model_update_lock = SimpleNamespace(writer_lock=lock)
        manager.post_process_weights_communicator = communicator
        result = await asyncio.wait_for(manager.post_process_weights("req"), timeout=2)
        return result, lock, calls

    return asyncio.run(run())


def test_paused_post_process_weights_does_not_wait_for_the_writer_lock(manager_class):
    result, lock, calls = call_post_process_weights(manager_class, paused=True, readers_released=False)
    assert result == ["done"] and calls == ["req"]
    assert lock.entered == 0


def test_post_process_weights_takes_the_writer_lock_when_not_paused(manager_class):
    result, lock, calls = call_post_process_weights(manager_class, paused=False, readers_released=True)
    assert result == ["done"] and calls == ["req"]
    assert lock.entered == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
