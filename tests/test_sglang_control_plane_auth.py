"""Bearer-auth invariants for SGLang control-plane and router HTTP calls."""

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.sglang_utils import server_control
from slime.utils.http_utils import bearer_auth_headers

NUM_GPUS = 0

WORKER_AUTH = {"Authorization": "Bearer worker-secret"}


def test_bearer_auth_headers_never_sends_bearer_none():
    assert bearer_auth_headers(None) == {}
    assert bearer_auth_headers("") == {}
    assert bearer_auth_headers("worker-secret") == WORKER_AUTH
    wait_headers = {
        "Content-Type": "application/json; charset=utf-8",
        **bearer_auth_headers(None),
    }
    assert "Authorization" not in wait_headers
    assert "Bearer None" not in str(wait_headers)


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


def test_abort_until_idle_omits_authorization_when_key_unset(monkeypatch):
    posts = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        posts.append(headers)
        return {}

    async def fake_get(url, headers=None):
        return {"num_reqs": 0}

    monkeypatch.setattr(server_control, "post", fake_post)
    monkeypatch.setattr(server_control, "get", fake_get)

    asyncio.run(server_control.abort_servers_until_idle(["http://10.0.0.2:15000"]))

    assert posts == [{}]
    assert "Bearer None" not in str(posts)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
