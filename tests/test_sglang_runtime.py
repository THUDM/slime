import io
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slime.backends.sglang_utils import runtime
from slime.backends.sglang_utils.runtime import (
    launch_server_subprocess,
    prepare_server_environment,
    terminate_server_process,
)

NUM_GPUS = 0


def test_shutdown_does_not_resolve_a_finished_process_pid(monkeypatch):
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = 0

    def lookup(pid):
        raise AssertionError("A reaped PID could now belong to another process")

    monkeypatch.setattr(psutil, "Process", lookup)
    terminate_server_process(process)


@pytest.mark.parametrize("launcher", ["popen", "multiprocessing"])
def test_shutdown_waits_for_owned_tree_without_killing_unrelated_process(tmp_path, launcher):
    child_pid = tmp_path / "child.pid"
    program = """
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    command = [sys.executable, "-c", program, str(child_pid)]
    if launcher == "popen":
        parent = subprocess.Popen(command)
    else:
        parent = multiprocessing.get_context("spawn").Process(target=subprocess.run, args=(command,))
        parent.start()
    try:
        deadline = time.monotonic() + 15
        while not child_pid.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child = psutil.Process(int(child_pid.read_text()))
        terminate_server_process(parent)
        if launcher == "popen":
            assert parent.poll() is not None
        else:
            assert parent.exitcode is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
        terminate_server_process(parent)  # Repeated teardown is harmless.
    finally:
        terminate_server_process(parent)
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_server_subprocess_preserves_arguments_and_isolates_environment(tmp_path, monkeypatch):
    executable = tmp_path / "serving python; literal"
    result_path = tmp_path / "result.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "payload['argv'] = sys.argv[1:]\n"
        "payload['allocator'] = os.environ.get('PYTORCH_ALLOC_CONF')\n"
        "payload['cuda_allocator'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')\n"
        "payload['serving_env'] = os.environ.get('TEST_SERVING_ENV')\n"
        "payload['gpu'] = os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "Path(os.environ['TEST_RESULT']).write_text(json.dumps(payload))\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("TEST_RESULT", str(result_path))
    monkeypatch.setenv("TEST_SERVING_ENV", "/path with spaces/serve")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    allocator = "expandable_segments:True,pinned_use_cuda_host_register:True"
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", allocator)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    server_args = {"model_path": "literal $(echo wrong); 中文\npath", "port": 15000}
    process = launch_server_subprocess(str(executable), server_args)
    assert process.wait(timeout=15) == 0
    result = json.loads(result_path.read_text())
    assert result["server_args"] == server_args
    assert result["parent_pid"] == os.getpid()
    assert result["argv"] == ["-m", "slime.backends.sglang_utils.runtime"]
    assert result["allocator"] == "pinned_use_cuda_host_register:True"
    assert result["cuda_allocator"] is None
    assert result["serving_env"] == "/path with spaces/serve"
    assert result["gpu"] == "2,3"
    assert os.environ["PYTORCH_ALLOC_CONF"] == allocator
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.parametrize(
    "allocator,expected",
    [
        ("expandable_segments:True", None),
        (
            "pinned_use_cuda_host_register:True,expandable_segments:False,max_split_size_mb:128",
            "pinned_use_cuda_host_register:True,max_split_size_mb:128",
        ),
        ("max_split_size_mb:128, expandable_segments:1", "max_split_size_mb:128"),
        ("pinned_use_cuda_host_register:True", "pinned_use_cuda_host_register:True"),
    ],
)
def test_server_environment_preserves_supported_allocator_options(allocator, expected):
    environ = {"PYTORCH_ALLOC_CONF": allocator, "PYTORCH_CUDA_ALLOC_CONF": allocator}
    prepare_server_environment(environ)
    assert environ.get("PYTORCH_ALLOC_CONF") == expected
    assert environ.get("PYTORCH_CUDA_ALLOC_CONF") == expected
    assert environ["MC_MIN_PRC_PORT"] == "27000"
    assert environ["MC_MAX_PRC_PORT"] == "28000"


@pytest.mark.parametrize("name", ["MC_MIN_RPC_PORT", "MC_MAX_RPC_PORT", "MC_MIN_PRC_PORT", "MC_MAX_PRC_PORT"])
def test_server_environment_preserves_explicit_mooncake_range(name):
    environ = {name: "31000"}
    prepare_server_environment(environ)
    assert environ == {name: "31000"}


@pytest.mark.parametrize("encoder_only,parent_alive", [(False, True), (True, True), (True, False)])
def test_runtime_entrypoint_constructs_server_args_in_serving_process(monkeypatch, encoder_only, parent_alive):
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    engine = ModuleType("slime.backends.sglang_utils.sglang_engine")
    engine._launch_server = Mock()
    engine._prepare_server_stdin = Mock()
    process = Mock(exitcode=0)
    engine.launch_server_process = Mock(return_value=process)
    utils = ModuleType("sglang.srt.utils")
    utils.kill_itself_when_parent_died = Mock()
    for module in (server_args, engine, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runtime.os, "getppid", lambda: 123 if parent_alive else 456)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "server_args": {"host": "[::1]", "port": 15000, "encoder_only": encoder_only},
                    "parent_pid": 123,
                }
            )
        ),
    )
    if encoder_only and parent_alive:
        with pytest.raises(SystemExit) as exc:
            runtime.main()
        assert exc.value.code == 0
        args = engine.launch_server_process.call_args.args[0]
        process.join.assert_called_once_with()
        engine._launch_server.assert_not_called()
    elif encoder_only:
        runtime.main()
        engine.launch_server_process.assert_not_called()
        engine._launch_server.assert_not_called()
        return
    else:
        runtime.main()
        args, parent_pid = engine._launch_server.call_args.args
        assert parent_pid == 123
        engine.launch_server_process.assert_not_called()
    assert args.host == "::1"
    assert args.port == 15000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
