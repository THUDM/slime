"""Bearer-auth invariants for SGLang engine control-plane and router calls.

Runs without GPUs or a real SGLang install: the heavy ``sglang`` /
``sglang_router`` / ray imports that ``sglang_engine`` performs at module
scope are stubbed before import, and every HTTP call is monkeypatched.
"""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_GPUS = 0

if "sglang.srt.server_args" not in sys.modules:
    sglang_module = types.ModuleType("sglang")
    sglang_srt_module = types.ModuleType("sglang.srt")
    server_args_module = types.ModuleType("sglang.srt.server_args")
    utils_module = types.ModuleType("sglang.srt.utils")

    class _StubServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def url(self):
            return f"http://{self.host}:{self.port}"

    server_args_module.ServerArgs = _StubServerArgs
    utils_module.kill_process_tree = lambda _pid: None
    sys.modules["sglang"] = sglang_module
    sys.modules["sglang.srt"] = sglang_srt_module
    sys.modules["sglang.srt.server_args"] = server_args_module
    sys.modules["sglang.srt.utils"] = utils_module

if "sglang_router" not in sys.modules:
    sglang_router_module = types.ModuleType("sglang_router")
    sglang_router_module.__version__ = "0.3.0"
    sys.modules["sglang_router"] = sglang_router_module

ray_actor_module = types.ModuleType("slime.ray.ray_actor")
ray_actor_module.RayActor = type("RayActor", (), {})
sys.modules.setdefault("slime.ray.ray_actor", ray_actor_module)

from slime.backends.sglang_utils import server_control  # noqa: E402
from slime.backends.sglang_utils import sglang_engine  # noqa: E402

WORKER_AUTH = {"Authorization": "Bearer worker-secret"}
ROUTER_AUTH = {"Authorization": "Bearer router-secret"}


class _Response:
    text = ""

    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def _capture(calls):
    def request(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(
            {
                "weight_version": "v1",
                "workers": [{"url": "http://10.0.0.2:15000", "id": "w0"}],
            }
        )

    return request


def _engine(*, server_api_key, router_api_key=None):
    args = SimpleNamespace(
        rollout_external=False,
        router_api_key=router_api_key,
        sglang_router_ip="10.0.0.1",
        sglang_router_port=4049,
    )
    engine = sglang_engine.SGLangEngine(args, rank=0)
    engine.node_rank = 0
    engine.router_ip = "10.0.0.1"
    engine.router_port = 4049
    engine.server_host = "10.0.0.2"
    engine.server_port = 15000
    engine.server_api_key = server_api_key
    engine.worker_type = "regular"
    return engine


def _patch_server_start(monkeypatch):
    monkeypatch.setattr(sglang_engine, "ServerArgs", lambda **kwargs: kwargs)
    monkeypatch.setattr(sglang_engine, "launch_server_process", lambda _server_args: object())


@pytest.mark.parametrize("api_key, expected_auth", [(None, None), ("worker-secret", "Bearer worker-secret")])
def test_wait_server_healthy_sends_bearer_only_when_key_set(monkeypatch, api_key, expected_auth):
    captured = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None):
            captured.append((url, headers))
            return _Response()

    monkeypatch.setattr(sglang_engine.requests, "Session", FakeSession)

    sglang_engine._wait_server_healthy("http://10.0.0.2:15000", api_key, lambda: True)

    [(url, headers)] = captured
    assert url == "http://10.0.0.2:15000/health_generate"
    assert headers.get("Authorization") == expected_auth


def test_control_plane_calls_carry_worker_bearer(monkeypatch):
    engine = _engine(server_api_key="worker-secret")
    calls = []
    fake = _capture(calls)
    monkeypatch.setattr(sglang_engine.requests, "get", fake)
    monkeypatch.setattr(sglang_engine.requests, "post", fake)

    engine.health_generate()
    engine.flush_cache()
    engine.get_weight_version()
    engine._make_request("release_memory_occupation")
    engine.pause_generation()
    engine.continue_generation()
    engine.start_profile(output_dir="/tmp/profile")
    engine.stop_profile()

    assert {url.rsplit("/", 1)[-1] for url, _ in calls} == {
        "health_generate",
        "flush_cache",
        "get_weight_version",
        "release_memory_occupation",
        "pause_generation",
        "continue_generation",
        "start_profile",
        "stop_profile",
    }
    for url, kwargs in calls:
        assert kwargs.get("headers") == WORKER_AUTH, url


def test_no_configured_keys_sends_no_authorization(monkeypatch):
    engine = _engine(server_api_key=None, router_api_key=None)
    calls = []
    fake = _capture(calls)
    monkeypatch.setattr(sglang_engine.requests, "get", fake)
    monkeypatch.setattr(sglang_engine.requests, "post", fake)
    _patch_server_start(monkeypatch)
    monkeypatch.setattr(sglang_engine.sglang_router, "__version__", "0.3.0")

    engine._init_normal({})
    engine.health_generate()
    engine.pause_generation()

    assert len(calls) == 3
    for url, kwargs in calls:
        assert not (kwargs.get("headers") or {}).get("Authorization"), url
    register_payload = next(kwargs["json"] for url, kwargs in calls if url.endswith("/workers"))
    assert "api_key" not in register_payload


def test_router_registration_carries_router_bearer_and_worker_key(monkeypatch):
    engine = _engine(server_api_key="worker-secret", router_api_key="router-secret")
    calls = []
    monkeypatch.setattr(sglang_engine.requests, "post", _capture(calls))
    _patch_server_start(monkeypatch)
    monkeypatch.setattr(sglang_engine.sglang_router, "__version__", "0.3.0")

    engine._init_normal({"api_key": "worker-secret"})

    [(url, kwargs)] = calls
    assert url == "http://10.0.0.1:4049/workers"
    assert kwargs["headers"] == ROUTER_AUTH
    assert kwargs["json"]["api_key"] == "worker-secret"


def test_legacy_router_registration_carries_router_bearer(monkeypatch):
    engine = _engine(server_api_key="worker-secret", router_api_key="router-secret")
    calls = []
    monkeypatch.setattr(sglang_engine.requests, "post", _capture(calls))
    _patch_server_start(monkeypatch)
    monkeypatch.setattr(sglang_engine.sglang_router, "__version__", "0.2.1")

    engine._init_normal({"api_key": "worker-secret"})

    [(url, kwargs)] = calls
    assert url == "http://10.0.0.1:4049/add_worker?url=http://10.0.0.2:15000"
    assert kwargs["headers"] == ROUTER_AUTH


def test_shutdown_router_calls_carry_router_bearer(monkeypatch):
    engine = _engine(server_api_key="worker-secret", router_api_key="router-secret")
    engine.process = SimpleNamespace(pid=1234)
    calls = []
    fake = _capture(calls)
    monkeypatch.setattr(sglang_engine.requests, "get", fake)
    monkeypatch.setattr(sglang_engine.requests, "delete", fake)
    monkeypatch.setattr(sglang_engine, "kill_process_tree", lambda _pid: None)
    monkeypatch.setattr(sglang_engine.sglang_router, "__version__", "0.3.0")

    engine.shutdown()

    assert [url for url, _ in calls] == ["http://10.0.0.1:4049/workers", "http://10.0.0.1:4049/workers/w0"]
    for url, kwargs in calls:
        assert kwargs.get("headers") == ROUTER_AUTH, url


def test_abort_until_idle_carries_worker_bearer(monkeypatch):
    posts = []
    gets = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        posts.append((url, headers))
        return {}

    async def fake_get(url, headers=None):
        gets.append((url, headers))
        return {"num_reqs": 0}

    monkeypatch.setattr(server_control, "post", fake_post)
    monkeypatch.setattr(server_control, "get", fake_get)

    asyncio.run(server_control.abort_servers_until_idle(["http://10.0.0.2:15000"], api_key="worker-secret"))

    assert posts == [("http://10.0.0.2:15000/abort_request", WORKER_AUTH)]
    assert gets == [("http://10.0.0.2:15000/v1/loads?include=core", WORKER_AUTH)]


def test_flush_cache_fails_fast_on_auth_rejection(monkeypatch):
    engine = _engine(server_api_key="worker-secret")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(status_code=401)

    monkeypatch.setattr(sglang_engine.requests, "get", get)
    monkeypatch.setattr(sglang_engine.time, "sleep", lambda _seconds: None)

    with pytest.raises(requests.exceptions.HTTPError):
        engine.flush_cache()

    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
