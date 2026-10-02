import argparse
import importlib
import sys
import types
from dataclasses import dataclass

import pytest
import requests


@pytest.fixture
def sglang_arguments(monkeypatch):
    sglang = types.ModuleType("sglang")
    sglang.__path__ = []
    sglang_srt = types.ModuleType("sglang.srt")
    sglang_srt.__path__ = []
    server_args = types.ModuleType("sglang.srt.server_args")

    class ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            return parser

    server_args.ServerArgs = ServerArgs
    sglang_router = types.ModuleType("sglang_router")
    sglang_router.__path__ = []
    launch_router = types.ModuleType("sglang_router.launch_router")

    class RouterArgs:
        @staticmethod
        def add_cli_args(parser, **kwargs):
            return parser

    launch_router.RouterArgs = RouterArgs
    accelerator = types.ModuleType("slime.utils.accelerator")
    accelerator.initialize_accelerator = lambda: None
    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils._wrap_ipv6 = lambda host: host

    for name, module in {
        "sglang": sglang,
        "sglang.srt": sglang_srt,
        "sglang.srt.server_args": server_args,
        "sglang_router": sglang_router,
        "sglang_router.launch_router": launch_router,
        "slime.utils.accelerator": accelerator,
        "slime.utils.http_utils": http_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "slime.backends.sglang_utils.arguments"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def sglang_engine(monkeypatch):
    sglang = types.ModuleType("sglang")
    sglang.__path__ = []
    sglang_srt = types.ModuleType("sglang.srt")
    sglang_srt.__path__ = []
    entrypoints = types.ModuleType("sglang.srt.entrypoints")
    entrypoints.__path__ = []
    http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
    http_server.launch_server = lambda server_args: None
    server_args = types.ModuleType("sglang.srt.server_args")
    sglang_utils = types.ModuleType("sglang.srt.utils")

    @dataclass
    class ServerArgs:
        host: str = "127.0.0.1"

    server_args.ServerArgs = ServerArgs
    sglang_utils.kill_process_tree = lambda pid: None

    external = types.ModuleType("slime.backends.sglang_utils.external")
    external.get_server_info = lambda url: {}
    ray_actor = types.ModuleType("slime.ray.ray_actor")
    ray_actor.RayActor = object
    accelerator = types.ModuleType("slime.utils.accelerator")
    accelerator.initialize_accelerator = lambda: None
    accelerator.resolve_visible_device_id = lambda device_id: device_id
    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils.get_host_info = lambda: (None, "127.0.0.1")

    for name, module in {
        "sglang": sglang,
        "sglang.srt": sglang_srt,
        "sglang.srt.entrypoints": entrypoints,
        "sglang.srt.entrypoints.http_server": http_server,
        "sglang.srt.server_args": server_args,
        "sglang.srt.utils": sglang_utils,
        "slime.backends.sglang_utils.external": external,
        "slime.ray.ray_actor": ray_actor,
        "slime.utils.accelerator": accelerator,
        "slime.utils.http_utils": http_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "slime.backends.sglang_utils.sglang_engine"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_sglang_server_startup_timeout_cli(sglang_arguments):
    parser = argparse.ArgumentParser()
    sglang_arguments.add_sglang_arguments(parser)

    assert parser.parse_args([]).sglang_server_startup_timeout == 600
    assert parser.parse_args(["--sglang-server-startup-timeout", "12.5"]).sglang_server_startup_timeout == 12.5


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan"])
def test_sglang_server_startup_timeout_rejects_invalid_values(sglang_arguments, value):
    parser = argparse.ArgumentParser()
    sglang_arguments.add_sglang_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--sglang-server-startup-timeout", value])


def test_wait_server_healthy_times_out_with_bounded_requests(monkeypatch, sglang_engine):
    clock = FakeClock()
    request_timeouts = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers, timeout):
            request_timeouts.append(timeout)
            if len(request_timeouts) == 1:
                return types.SimpleNamespace(status_code=503)
            clock.now += timeout
            raise requests.Timeout("request timed out")

    monkeypatch.setattr(sglang_engine.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sglang_engine.time, "sleep", clock.sleep)
    monkeypatch.setattr(sglang_engine.requests, "Session", FakeSession)

    with pytest.raises(TimeoutError, match=r"http://localhost:30000/health_generate.*6 seconds"):
        sglang_engine._wait_server_healthy(
            base_url="http://localhost:30000",
            api_key="secret",
            is_process_alive=lambda: True,
            startup_timeout=6,
        )

    assert request_timeouts == [5.0, 4.0]


def test_wait_server_healthy_rejects_success_after_deadline(monkeypatch, sglang_engine):
    clock = FakeClock()

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers, timeout):
            clock.now += timeout
            return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(sglang_engine.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(sglang_engine.requests, "Session", FakeSession)

    with pytest.raises(TimeoutError):
        sglang_engine._wait_server_healthy(
            base_url="http://localhost:30000",
            api_key=None,
            is_process_alive=lambda: True,
            startup_timeout=1,
        )


def test_wait_server_healthy_reports_early_process_exit(monkeypatch, sglang_engine):
    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers, timeout):
            raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(sglang_engine.requests, "Session", FakeSession)

    with pytest.raises(RuntimeError, match=r"terminated unexpectedly.*http://localhost:30000/health_generate"):
        sglang_engine._wait_server_healthy(
            base_url="http://localhost:30000",
            api_key=None,
            is_process_alive=lambda: False,
            startup_timeout=30,
        )


def test_launch_server_process_cleans_up_without_masking_wait_failure(monkeypatch, sglang_engine, caplog):
    process = types.SimpleNamespace(pid=4321, start=lambda: None, is_alive=lambda: True)
    wait_failure = ValueError("health check failed")
    killed_pids = []

    def fail_wait(**kwargs):
        assert kwargs["startup_timeout"] == 37
        raise wait_failure

    def fail_cleanup(pid):
        killed_pids.append(pid)
        raise OSError("cleanup failed")

    monkeypatch.setattr(sglang_engine.multiprocessing, "Process", lambda **kwargs: process)
    monkeypatch.setattr(sglang_engine.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_engine, "_wait_server_healthy", fail_wait)
    monkeypatch.setattr(sglang_engine, "kill_process_tree", fail_cleanup)
    server_args = types.SimpleNamespace(
        encoder_only=False,
        host="127.0.0.1",
        node_rank=0,
        api_key=None,
        url=lambda: "http://127.0.0.1:30000",
    )

    with pytest.raises(ValueError) as exc_info:
        sglang_engine.launch_server_process(server_args, startup_timeout=37)

    assert exc_info.value is wait_failure
    assert killed_pids == [4321]
    assert "Failed to clean up SGLang server process tree" in caplog.text


def test_launch_server_process_nonzero_rank_returns_without_waiting(monkeypatch, sglang_engine):
    process = types.SimpleNamespace(pid=4321, start=lambda: None, is_alive=lambda: True)

    monkeypatch.setattr(sglang_engine.multiprocessing, "Process", lambda **kwargs: process)
    monkeypatch.setattr(sglang_engine.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sglang_engine,
        "_wait_server_healthy",
        lambda **kwargs: pytest.fail("nonzero rank must not wait for the server"),
    )
    monkeypatch.setattr(
        sglang_engine,
        "kill_process_tree",
        lambda pid: pytest.fail("nonzero rank must not clean up a healthy process"),
    )
    server_args = types.SimpleNamespace(
        encoder_only=False,
        host="127.0.0.1",
        node_rank=1,
        api_key=None,
        url=lambda: "http://127.0.0.1:30000",
    )

    assert sglang_engine.launch_server_process(server_args, startup_timeout=37) is process


def test_engine_forwards_configured_server_startup_timeout(monkeypatch, sglang_engine):
    engine = sglang_engine.SGLangEngine.__new__(sglang_engine.SGLangEngine)
    engine.args = types.SimpleNamespace(sglang_server_startup_timeout=42.5)
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine._register_to_router = lambda server_args_dict: None
    launched = []

    def fake_launch(server_args, startup_timeout):
        launched.append((server_args, startup_timeout))
        return "process"

    monkeypatch.setattr(sglang_engine, "ServerArgs", lambda **kwargs: kwargs)
    monkeypatch.setattr(sglang_engine, "launch_server_process", fake_launch)

    engine._init_normal({"host": "127.0.0.1", "port": 30000})

    assert engine.process == "process"
    assert launched == [({"host": "127.0.0.1", "port": 30000}, 42.5)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
