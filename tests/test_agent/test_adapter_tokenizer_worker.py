"""CPU tests for bounded, cancellation-safe adapter prompt rendering."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from functools import partial
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSGLangServer, FakeTokenizer  # noqa: E402

from slime.agent.adapters import anthropic  # noqa: E402
from slime.agent.adapters import common as adapters_common  # noqa: E402
from slime.agent.adapters.common import _render_token_ids  # noqa: E402
from slime.agent.adapters.tokenizer_worker import TokenizerWorker  # noqa: E402

NUM_GPUS = 0


class _BlockingTokenizer(FakeTokenizer):
    def __init__(self, release: threading.Event, *, outputs=None) -> None:
        super().__init__(outputs=outputs)
        self.release = release
        self.started = threading.Event()
        self._lock = threading.Lock()
        self.calls = 0
        self.active = 0
        self.peak_active = 0

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test tokenizer was not released")
        try:
            return super().apply_chat_template(
                messages,
                tools=tools,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        finally:
            with self._lock:
                self.active -= 1


class _Request:
    def __init__(self, sid: str, body: dict) -> None:
        self.headers = {"Authorization": f"Bearer {sid}"}
        self._body = body

    async def json(self) -> dict:
        return self._body


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.005)


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        ([{"role": "user", "content": "hello"}], None),
        (
            [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "look up slime"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "slime"}}}],
                },
                {"role": "tool", "content": "found"},
            ],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        ),
        (
            [
                {"role": "user", "content": "compacted context"},
                {"role": "assistant", "content": "re-rendered answer"},
                {"role": "user", "content": "retry from this branch"},
            ],
            None,
        ),
    ],
)
def test_worker_render_matches_synchronous_full_history(messages, tools):
    async def run_case():
        tokenizer = FakeTokenizer()
        expected = _render_token_ids(messages, tokenizer, tools=tools)
        worker = TokenizerWorker(max_pending=2)
        try:
            actual = await worker.render(partial(_render_token_ids, messages, tokenizer, tools=tools))
        finally:
            await worker.close()
        return expected, actual, tokenizer.rendered

    expected, actual, rendered = asyncio.run(run_case())
    assert actual == expected
    assert rendered == [(messages, tools), (messages, tools)]


def test_worker_bounds_submitted_work_across_cancellation():
    async def run_case():
        worker = TokenizerWorker(max_pending=2)
        release = threading.Event()
        started = threading.Event()
        calls = 0

        def blocking_render() -> list[int]:
            nonlocal calls
            calls += 1
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test render was not released")
            return [1, 2, 3]

        tasks = [asyncio.create_task(worker.render(blocking_render)) for _ in range(6)]
        await _wait_until(lambda: started.is_set() and worker.snapshot()["counters"]["requests_total"] == 6)
        before_cancel = worker.snapshot()
        assert before_cancel["worker"]["active"] == 1
        assert before_cancel["worker"]["outstanding"] == 2
        assert before_cancel["worker"]["admission_waiters"] == 4

        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)

        # The running and executor-queued jobs still hold their permits until
        # they physically finish/dequeue, even though their callers are gone.
        after_cancel = worker.snapshot()
        assert after_cancel["worker"]["outstanding"] == 2
        assert after_cancel["worker"]["outstanding_peak"] == 2
        assert calls == 1

        release.set()
        await _wait_until(lambda: worker.snapshot()["worker"]["outstanding"] == 0)
        final = worker.snapshot()
        await worker.close()
        return calls, final

    calls, metrics = asyncio.run(run_case())
    assert calls == 1
    assert metrics["worker"]["active_peak"] == 1
    assert metrics["counters"]["render_cancelled_total"] == 6
    assert metrics["counters"]["discarded_total"] == 2
    assert metrics["counters"]["discarded_before_render_total"] == 1


def test_cancel_during_admission_handoff_returns_the_permit():
    async def run_case():
        worker = TokenizerWorker(max_pending=1)
        release = threading.Event()
        started = threading.Event()

        def blocking_render() -> list[int]:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test render was not released")
            return [1]

        first = asyncio.create_task(worker.render(blocking_render))
        await _wait_until(started.is_set)
        second = asyncio.create_task(worker.render(lambda: [2]))
        await _wait_until(lambda: len(worker._acquire_tasks) == 1)
        acquire_task = next(iter(worker._acquire_tasks))

        def cancel_twice(_done) -> None:
            second.cancel()
            asyncio.get_running_loop().call_soon(second.cancel)

        acquire_task.add_done_callback(cancel_twice)

        release.set()
        assert await first == [1]
        with pytest.raises(asyncio.CancelledError):
            await second

        # A fresh render proves the handoff race did not permanently consume
        # the worker's only permit.
        assert await asyncio.wait_for(worker.render(lambda: [3]), timeout=0.25) == [3]
        metrics = worker.snapshot()
        await worker.close()
        return metrics

    metrics = asyncio.run(run_case())
    assert metrics["worker"]["outstanding"] == 0
    assert metrics["worker"]["admission_waiters"] == 0


def test_client_cancellation_stays_cancelled_after_worker_is_marked_closed():
    async def run_case():
        worker = TokenizerWorker(max_pending=1)
        release = threading.Event()
        started = threading.Event()

        def blocking_render() -> list[int]:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test render was not released")
            return [1]

        first = asyncio.create_task(worker.render(blocking_render))
        await _wait_until(started.is_set)
        second = asyncio.create_task(worker.render(lambda: [2]))
        await _wait_until(lambda: len(worker._acquire_tasks) == 1)

        # Python 3.10 has no Task.cancelling(). Cancellation ownership is
        # determined by the worker's explicit close marker, not its closed flag.
        worker._closed = True
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

        release.set()
        assert await first == [1]
        await worker.close()
        return worker.snapshot()

    metrics = asyncio.run(run_case())
    assert metrics["worker"]["outstanding"] == 0
    assert metrics["worker"]["admission_waiters"] == 0
    assert metrics["counters"]["render_cancelled_total"] == 1


def test_close_is_cancellation_safe_and_drains_admission_waiters():
    async def run_case():
        worker = TokenizerWorker(max_pending=1)
        release = threading.Event()
        started = threading.Event()

        def blocking_render() -> list[int]:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test render was not released")
            return [1]

        render_tasks = [asyncio.create_task(worker.render(blocking_render)) for _ in range(5)]
        await _wait_until(lambda: started.is_set() and worker.snapshot()["worker"]["admission_waiters"] == 4)
        threads = list(worker._executor._threads)

        first_close = asyncio.create_task(worker.close())
        await _wait_until(lambda: all(task.done() for task in render_tasks[1:]))
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        second_close = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        assert not second_close.done()
        release.set()
        results = await asyncio.gather(*render_tasks, return_exceptions=True)
        await second_close
        await worker.close()
        return results, threads, worker.snapshot()

    results, threads, metrics = asyncio.run(run_case())
    assert results[0] == [1]
    assert all(isinstance(result, RuntimeError) for result in results[1:])
    assert all(not thread.is_alive() for thread in threads)
    assert metrics["worker"]["outstanding"] == 0
    assert metrics["worker"]["active"] == 0
    assert metrics["worker"]["admission_waiters"] == 0


def test_prompt_metrics_report_percentiles_from_bounded_window():
    async def run_case():
        worker = TokenizerWorker()
        try:
            for length in (1, 2, 3, 4):
                await worker.render(lambda length=length: list(range(length)))
            return worker.snapshot()
        finally:
            await worker.close()

    metrics = asyncio.run(run_case())
    prompt = metrics["prompt_tokens"]
    assert prompt == {
        "count": 4,
        "mean": 2.5,
        "max": 4.0,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
    }
    assert metrics["render_latency_ms"]["count"] == 4
    assert metrics["queue_wait_ms"]["count"] == 4
    assert 0 <= metrics["worker"]["utilization"] <= 1


def test_slow_render_keeps_health_and_metrics_responsive():
    async def run_case():
        release = threading.Event()
        tokenizer = _BlockingTokenizer(release, outputs={(901,): "done"})
        async with FakeSGLangServer([[(-0.1, 901)]]) as sglang:
            adapter = anthropic.AnthropicAdapter(
                tokenizer=tokenizer,
                sglang_url=sglang.url,
                event_loop_lag_interval_seconds=0.005,
            )
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                post_task = asyncio.create_task(
                    client.post(
                        "/v1/messages",
                        headers={"Authorization": "Bearer responsive"},
                        json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
                    )
                )
                await _wait_until(tokenizer.started.is_set)

                started_at = time.monotonic()
                health = await asyncio.wait_for(client.get("/healthz"), timeout=0.25)
                assert health.status == 200
                assert time.monotonic() - started_at < 0.25

                await asyncio.sleep(0.015)
                metrics_response = await asyncio.wait_for(client.get("/debug/adapter_metrics"), timeout=0.25)
                metrics = await metrics_response.json()
                assert metrics["tokenizer"]["worker"]["active"] == 1
                assert metrics["tokenizer"]["event_loop_lag_ms"]["count"] > 0

                release.set()
                response = await asyncio.wait_for(post_task, timeout=1)
                assert response.status == 200
                await response.read()
            finally:
                release.set()
                await client.close()

        return adapter, sglang

    adapter, sglang = asyncio.run(run_case())
    assert sglang.requests[0]["input_ids"]
    assert adapter._tokenizer_worker.closed


def test_cancelled_render_never_calls_sglang_or_writes_session(monkeypatch):
    async def run_case():
        release = threading.Event()
        tokenizer = _BlockingTokenizer(release)
        adapter = anthropic.AnthropicAdapter(
            tokenizer=tokenizer,
            sglang_url="http://unused",
            max_turns_per_sid=1,
        )
        sglang_calls: list[list[int]] = []

        async def fake_generate(prompt_ids, session, body, *, adapter, session_id=None):
            sglang_calls.append(list(prompt_ids))
            raise AssertionError("cancelled render reached SGLang")

        monkeypatch.setattr(adapters_common, "call_sglang_generate", fake_generate)
        sid = "cancelled"
        request = _Request(sid, {"model": "m", "messages": [{"role": "user", "content": "slow"}]})
        turn_task = asyncio.create_task(adapter._run_turn(request))
        await _wait_until(tokenizer.started.is_set)
        turn_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn_task

        assert sglang_calls == []
        assert sid not in adapter.store
        assert adapter.manager.turn_count(sid) == 0
        assert not adapter.inflight.get(sid)
        assert sid not in adapter._sid_turn_count

        release.set()
        await _wait_until(lambda: adapter.metrics_snapshot()["tokenizer"]["worker"]["outstanding"] == 0)
        await asyncio.sleep(0)
        assert sglang_calls == []
        assert sid not in adapter.store
        assert adapter.manager.turn_count(sid) == 0
        metrics = adapter.metrics_snapshot()["tokenizer"]
        await adapter._tokenizer_worker.close()
        return sglang_calls, metrics

    sglang_calls, metrics = asyncio.run(run_case())
    assert sglang_calls == []
    assert metrics["counters"]["request_cancelled_total"] == 1
    assert metrics["counters"]["render_cancelled_total"] == 1
    assert metrics["counters"]["discarded_total"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
