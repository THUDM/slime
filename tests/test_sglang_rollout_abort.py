import asyncio
import types

import pytest

try:
    from plugin_contracts._shared import install_paths, install_stubs
except ImportError:
    from tests.plugin_contracts._shared import install_paths, install_stubs

install_paths()
install_stubs(with_sglang_router=True, with_transformers=True)

from slime.rollout import sglang_rollout as rollout

NUM_GPUS = 0


@pytest.mark.unit
def test_abort_cancels_agent_tasks_before_server_drain(monkeypatch) -> None:
    async def scenario() -> None:
        events = []

        async def agent_task() -> None:
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                events.append("agent-cleanup")

        task = asyncio.create_task(agent_task())
        await asyncio.sleep(0)
        state = types.SimpleNamespace(aborted=False, pendings={task})
        monkeypatch.setattr(rollout, "GenerateState", lambda _args: state)

        async def get_router_workers(_url):
            events.append("list-workers")
            return {"urls": ["http://engine"], "workers": [{"url": "http://engine"}]}

        async def drain_servers(urls):
            assert urls == ["http://engine"]
            assert task.cancelled()
            events.append("server-drain")

        monkeypatch.setattr(rollout, "get", get_router_workers)
        monkeypatch.setattr(rollout, "abort_servers_until_idle", drain_servers)

        args = types.SimpleNamespace(
            partial_rollout=False,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )
        assert await rollout.abort(args, rollout_id=0) == []

        assert events == ["agent-cleanup", "list-workers", "server-drain"]
        assert state.aborted is True
        assert state.pendings == set()

    asyncio.run(scenario())


@pytest.mark.unit
def test_abort_preserves_partial_agent_tasks_until_after_server_drain(monkeypatch) -> None:
    async def scenario() -> None:
        events = []
        server_drained = asyncio.Event()
        sample = types.SimpleNamespace(response="partial response", metadata={})

        async def agent_task():
            await server_drained.wait()
            events.append("agent-finished")
            return [sample]

        task = asyncio.create_task(agent_task())
        await asyncio.sleep(0)
        state = types.SimpleNamespace(aborted=False, pendings={task})
        monkeypatch.setattr(rollout, "GenerateState", lambda _args: state)

        async def get_router_workers(_url):
            return {"urls": ["http://engine"], "workers": [{"url": "http://engine"}]}

        async def drain_servers(urls):
            assert urls == ["http://engine"]
            assert not task.cancelled()
            events.append("server-drain")
            server_drained.set()
            await asyncio.sleep(0)

        monkeypatch.setattr(rollout, "get", get_router_workers)
        monkeypatch.setattr(rollout, "abort_servers_until_idle", drain_servers)

        args = types.SimpleNamespace(
            partial_rollout=True,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )
        assert await rollout.abort(args, rollout_id=7) == [[sample]]

        assert events == ["server-drain", "agent-finished"]
        assert sample.metadata == {"start_rollout_id": 7}
        assert state.pendings == set()

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
