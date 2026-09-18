import asyncio
import json

import pytest

from slime.backends.sglang_utils import server_control
from slime.utils import http_utils

NUM_GPUS = 0


@pytest.mark.unit
def test_num_requests_includes_disaggregated_and_inflight_queues():
    load = {
        "loads": [
            {
                "dp_rank": 0,
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "disaggregation": {"prefill_queue_reqs": 2},
            },
            {"dp_rank": 1, "inflight": [{"name": "transfer", "num_reqs": 3, "reqs": ["a", "b", "c"]}]},
        ]
    }

    assert server_control.num_requests_from_load(load) == 5
    assert server_control.non_idle_request_details(load) == [
        {"dp_rank": 0, "counters": {"disaggregation.prefill_queue_reqs": 2}},
        {"dp_rank": 1, "queue": "transfer", "num_reqs": 3, "reqs": ["a", "b", "c"]},
    ]


@pytest.mark.unit
def test_abort_retries_load_failure_until_idle(monkeypatch):
    calls = 0

    async def abort_once(url, request_timeout):
        return None

    async def get_load(url, request_timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary control-plane failure")
        return {"num_running_reqs": 0, "num_waiting_reqs": 0}

    monkeypatch.setattr(server_control, "_abort_server_once", abort_once)
    monkeypatch.setattr(server_control, "_get_server_load", get_load)

    asyncio.run(server_control.abort_server_until_idle("http://engine", retry_interval=0, timeout=1))
    assert calls == 2


@pytest.mark.unit
def test_abort_servers_surfaces_partial_failure(monkeypatch):
    async def abort(url, **kwargs):
        if url.endswith("bad"):
            raise TimeoutError("not idle")

    monkeypatch.setattr(server_control, "abort_server_until_idle", abort)
    with pytest.raises(RuntimeError, match="bad.*not idle"):
        asyncio.run(server_control.abort_servers_until_idle(["http://good", "http://bad"]))


@pytest.mark.unit
def test_http_post_forwards_timeout_and_closes_response():
    class Response:
        text = ""
        closed = False

        def raise_for_status(self):
            return None

        async def aread(self):
            return json.dumps({"ok": True}).encode()

        async def aclose(self):
            self.closed = True

    response = Response()

    class Client:
        async def post(self, url, **kwargs):
            assert kwargs["timeout"] == 2.5
            return response

    assert asyncio.run(http_utils._post(Client(), "http://engine", {}, max_retries=1, timeout=2.5)) == {"ok": True}
    assert response.closed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
