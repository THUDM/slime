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
def test_abort_cancels_pending_groups_before_server_drain(monkeypatch) -> None:
    async def scenario() -> None:
        events = []

        async def pending_group() -> None:
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                events.append("pending-cleanup")

        task = asyncio.create_task(pending_group())
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

        assert events == ["pending-cleanup", "list-workers", "server-drain"]
        assert state.aborted is True
        assert state.pendings == set()

    asyncio.run(scenario())


@pytest.mark.unit
def test_abort_preserves_partial_groups_until_after_server_drain(monkeypatch) -> None:
    async def scenario() -> None:
        events = []
        server_drained = asyncio.Event()
        sample = types.SimpleNamespace(response="partial response", metadata={})

        async def pending_group():
            await server_drained.wait()
            events.append("pending-finished")
            return [sample]

        task = asyncio.create_task(pending_group())
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

        assert events == ["server-drain", "pending-finished"]
        assert sample.metadata == {"start_rollout_id": 7}
        assert state.pendings == set()

    asyncio.run(scenario())


@pytest.mark.unit
def test_abort_cleans_up_partial_groups_when_cancelled_during_drain(monkeypatch) -> None:
    async def scenario() -> None:
        events = []
        drain_started = asyncio.Event()

        async def pending_group():
            try:
                await asyncio.Future()
            finally:
                events.append("pending-cleanup")

        pending_task = asyncio.create_task(pending_group())
        await asyncio.sleep(0)
        state = types.SimpleNamespace(aborted=False, pendings={pending_task})
        monkeypatch.setattr(rollout, "GenerateState", lambda _args: state)

        async def get_router_workers(_url):
            return {"urls": ["http://engine"], "workers": [{"url": "http://engine"}]}

        async def drain_servers(urls):
            assert urls == ["http://engine"]
            events.append("server-drain-started")
            drain_started.set()
            await asyncio.Future()

        monkeypatch.setattr(rollout, "get", get_router_workers)
        monkeypatch.setattr(rollout, "abort_servers_until_idle", drain_servers)

        args = types.SimpleNamespace(
            partial_rollout=True,
            sglang_router_ip="127.0.0.1",
            sglang_router_port=30000,
        )
        abort_task = asyncio.create_task(rollout.abort(args, rollout_id=7))
        await drain_started.wait()
        abort_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await abort_task

        assert events == ["server-drain-started", "pending-cleanup"]
        assert pending_task.cancelled()
        assert state.aborted is True
        assert state.pendings == set()

    asyncio.run(scenario())


@pytest.mark.unit
def test_abort_cleans_up_partial_groups_when_one_fails(monkeypatch) -> None:
    async def scenario() -> None:
        events = []
        server_drained = asyncio.Event()

        async def failed_group():
            await server_drained.wait()
            raise RuntimeError("generation failed")

        async def pending_group():
            try:
                await asyncio.Future()
            finally:
                events.append("pending-cleanup")

        failed_task = asyncio.create_task(failed_group())
        pending_task = asyncio.create_task(pending_group())
        await asyncio.sleep(0)
        state = types.SimpleNamespace(aborted=False, pendings={failed_task, pending_task})
        monkeypatch.setattr(rollout, "GenerateState", lambda _args: state)

        async def get_router_workers(_url):
            return {"urls": ["http://engine"], "workers": [{"url": "http://engine"}]}

        async def drain_servers(urls):
            assert urls == ["http://engine"]
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
        with pytest.raises(RuntimeError, match="generation failed"):
            await rollout.abort(args, rollout_id=7)

        assert events == ["server-drain", "pending-cleanup"]
        assert pending_task.cancelled()
        assert state.pendings == set()

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
