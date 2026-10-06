import asyncio
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from slime.backends.sglang_utils import server_control
from slime.utils import http_utils

NUM_GPUS = 0


@pytest.mark.parametrize("failure", ["connection", "timeout", "status"])
def test_rollout_boundary_prunes_failed_router_workers(monkeypatch, failure):
    calls = []

    async def respond(request):
        calls.append((request.method, str(request.url)))
        if request.url.path == "/workers":
            return httpx.Response(
                200, json={"workers": [{"id": "good", "url": "http://good"}, {"id": "bad", "url": "http://bad"}]}
            )
        if request.method == "DELETE":
            return httpx.Response(200)
        if request.url.host == "bad":
            if failure == "connection":
                raise httpx.ConnectError("server stopped", request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout("server stuck", request=request)
            return httpx.Response(503)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(server_control.httpx, "AsyncClient", lambda **kwargs: client)
    workers = asyncio.run(server_control.get_live_router_workers("http://router", timeout=0.1))
    assert workers == [{"id": "good", "url": "http://good"}]
    assert ("DELETE", "http://router/workers/bad") in calls
    assert ("DELETE", "http://router/workers/good") not in calls


@pytest.mark.parametrize("actor_failure", ["dead", "wedged"])
def test_boundary_health_check_bypasses_grace_and_cleans_dead_actor(monkeypatch, actor_failure):
    from slime.backends.sglang_utils import engine_group
    from slime.ray.serving import ServingCluster
    from slime.utils import health_monitor

    calls = []
    dead = SimpleNamespace(
        health_generate=SimpleNamespace(remote=lambda **kw: "dead"),
        shutdown=SimpleNamespace(remote=lambda: "shutdown"),
    )
    alive = SimpleNamespace(health_generate=SimpleNamespace(remote=lambda **kw: "alive"))
    group = engine_group.ServerGroup(
        args=SimpleNamespace(num_gpus_per_node=1),
        pg=None,
        num_gpus_per_engine=1,
        num_new_engines=0,
        all_engines=[dead, alive],
        worker_type="regular",
        engine_urls={0: "http://dead", 1: "http://alive"},
        router_ip="router",
        router_port=8000,
    )
    args = SimpleNamespace(
        rollout_health_check_interval=600, rollout_health_check_first_wait=600, rollout_health_check_timeout=0.1
    )
    monitor = health_monitor.RolloutHealthMonitor(group, args)
    serving = object.__new__(ServingCluster.__ray_metadata__.modified_class)
    serving._health_monitors = [monitor]
    serving.servers = {"model": group}

    def wait(refs, *, num_returns, timeout):
        assert timeout == 0.1
        if refs == ["shutdown"]:
            return [], refs
        assert refs == ["dead", "alive"] and num_returns == 2
        return (["alive"], ["dead"]) if actor_failure == "wedged" else (refs, [])

    def get(ref, *, timeout):
        if ref in {"dead", "shutdown"}:
            raise TimeoutError(actor_failure)
        return True

    def unregister(router_url, worker_url, *, timeout):
        calls.append(("unregister", router_url, worker_url))

    monkeypatch.setattr(health_monitor.ray, "wait", wait)
    monkeypatch.setattr(health_monitor.ray, "get", get)
    monkeypatch.setattr(health_monitor.ray, "kill", lambda engine, **kw: calls.append(("kill", engine)))
    monkeypatch.setattr(engine_group, "unregister_worker", unregister)
    monitor.start()
    try:
        snapshot = serving.finish_rollout()
        assert calls == [("unregister", "http://router:8000", "http://dead"), ("kill", dead)]
        assert snapshot["model"].all_engines == [None, alive]
    finally:
        monitor.stop()


@pytest.mark.unit
def test_abort_uses_supported_load_sections_and_waits_for_pd_transfers(monkeypatch):
    abort_calls = 0
    load_calls = 0

    async def post(url, payload, **kwargs):
        nonlocal abort_calls
        assert url == "http://engine/abort_request"
        assert payload == {"abort_all": True}
        abort_calls += 1

    async def get(url, **kwargs):
        nonlocal load_calls
        request = urlsplit(url)
        assert request.path == "/v1/loads"
        sections = set(parse_qs(request.query)["include"][0].split(","))
        # Match the public SGLang LoadSnapshot protocol, which has no
        # standalone "inflight" section. PD transfers live under "disagg".
        assert sections <= {"core", "disagg", "queues", "memory", "spec", "lora", "all"}
        assert {"core", "disagg", "queues"} <= sections
        load_calls += 1
        return {
            "loads": [
                {
                    "dp_rank": 0,
                    "num_running_reqs": 0,
                    "num_waiting_reqs": 0,
                    "disaggregation": {"prefill_inflight_queue_reqs": int(load_calls == 1)},
                    "queues": {"waiting": 0, "grammar": 0},
                }
            ]
        }

    monkeypatch.setattr(server_control, "post", post)
    monkeypatch.setattr(server_control, "get", get)

    asyncio.run(server_control.abort_server_until_idle("http://engine", retry_interval=0, timeout=0.1))
    assert abort_calls == load_calls == 2


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


@pytest.mark.parametrize("concurrency", [2, 257])
def test_http_transport_preserves_streams_timeouts_and_connection_budget(concurrency):
    import httpx

    async def check():
        active = peak = accepted = 0
        tasks = set()

        async def serve(reader, writer):
            nonlocal active, peak, accepted
            task = asyncio.current_task()
            tasks.add(task)
            active += 1
            accepted += 1
            peak = max(peak, active)
            try:
                while True:
                    headers = await reader.readuntil(b"\r\n\r\n")
                    length = next(
                        (
                            int(line.split(b":", 1)[1])
                            for line in headers.split(b"\r\n")
                            if line.lower().startswith(b"content-length:")
                        ),
                        0,
                    )
                    body = await reader.readexactly(length)
                    if b"/slow " in headers:
                        await asyncio.sleep(0.1)
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                active -= 1
                writer.close()
                await writer.wait_closed()
                tasks.discard(task)

        server = await asyncio.start_server(serve, "127.0.0.1", 0, backlog=2 * concurrency)
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            async with httpx.AsyncClient(transport=http_utils._ShardedHTTPTransport(concurrency)) as client:

                async def request(i):
                    body = str(i).encode()
                    async with client.stream("POST", url, content=body) as response:
                        assert await response.aread() == body

                for _ in range(2):
                    await asyncio.gather(*(request(i) for i in range(concurrency + 3)))
                assert peak <= concurrency
                assert accepted <= concurrency  # The second wave reused live sockets.
                with pytest.raises(httpx.ReadTimeout):
                    await client.post(url + "/slow", content=b"x", timeout=0.01)
                assert (await client.post(url, content=b"after-timeout")).content == b"after-timeout"
        finally:
            server.close()
            await server.wait_closed()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert active == 0

    asyncio.run(check())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
