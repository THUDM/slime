"""Bounded prompt-render worker and lightweight adapter metrics."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

_METRIC_WINDOW_SIZE = 2048


@dataclasses.dataclass
class _RenderJob:
    render: Callable[[], list[int]]
    queued_at: float
    discarded: threading.Event = dataclasses.field(default_factory=threading.Event)
    dequeued: bool = False


def _distribution(values: deque[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "max": None, "p50": None, "p95": None, "p99": None}

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


class TokenizerWorker:
    """Run full-history prompt renders on one bounded, adapter-owned thread.

    ``ThreadPoolExecutor`` bounds running threads but not its internal queue, so
    an asyncio semaphore limits submitted jobs as well. A request cancellation
    marks its job discarded; the permit is released only when the underlying
    future is physically done. This prevents a cancellation storm from growing
    either worker concurrency or queued tokenizer work without bound.
    """

    def __init__(self, *, max_pending: int = 8) -> None:
        if max_pending < 1:
            raise ValueError("tokenizer max_pending must be at least 1")

        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slime-tokenizer")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._slots: asyncio.BoundedSemaphore | None = None
        self._acquire_tasks: set[asyncio.Task] = set()
        self._close_cancelled_acquires: set[asyncio.Task] = set()
        self._futures: set[Future] = set()
        self._closed = False
        self._close_task: asyncio.Task | None = None

        self._metrics_lock = threading.Lock()
        self._created_at = time.monotonic()
        self._requests_total = 0
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._request_cancelled_total = 0
        self._render_cancelled_total = 0
        self._discarded_total = 0
        self._discarded_before_render_total = 0
        self._waiting = 0
        self._waiting_peak = 0
        self._outstanding = 0
        self._outstanding_peak = 0
        self._active = 0
        self._active_peak = 0
        self._busy_seconds = 0.0
        self._active_started_at: float | None = None
        self._prompt_tokens: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)
        self._render_latency_ms: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)
        self._queue_wait_ms: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)
        self._event_loop_lag_ms: deque[float] = deque(maxlen=_METRIC_WINDOW_SIZE)

    @property
    def closed(self) -> bool:
        return self._closed

    def bind_to_current_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._slots = asyncio.BoundedSemaphore(self.max_pending)
        elif self._loop is not loop:
            raise RuntimeError("TokenizerWorker can only be used from its adapter event loop")

    async def render(self, render: Callable[[], list[int]]) -> list[int]:
        """Run one render without blocking the adapter event loop."""
        self.bind_to_current_loop()
        assert self._slots is not None
        assert self._loop is not None
        if self._closed:
            raise RuntimeError("tokenizer worker is closed")

        job = _RenderJob(render=render, queued_at=time.monotonic())
        self._request_queued()
        acquire_task = asyncio.create_task(self._slots.acquire())
        self._acquire_tasks.add(acquire_task)
        try:
            await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            close_cancelled = acquire_task in self._close_cancelled_acquires
            if not acquire_task.done():
                acquire_task.cancel()
            while not acquire_task.done():
                try:
                    await asyncio.shield(acquire_task)
                except asyncio.CancelledError:
                    pass
            if acquire_task.done() and not acquire_task.cancelled() and acquire_task.exception() is None:
                self._slots.release()
            if close_cancelled:
                self._request_rejected_after_close()
                raise RuntimeError("tokenizer worker is closed") from None
            self._request_cancelled_before_submit()
            raise
        finally:
            self._acquire_tasks.discard(acquire_task)
            self._close_cancelled_acquires.discard(acquire_task)

        if self._closed:
            self._slots.release()
            self._request_rejected_after_close()
            raise RuntimeError("tokenizer worker is closed")

        self._job_submitted()
        try:
            future = self._executor.submit(self._execute, job)
        except BaseException:
            self._slots.release()
            self._submission_failed()
            raise

        self._futures.add(future)
        future.add_done_callback(lambda done: self._schedule_future_done(done, job))
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(_consume_future_exception)
        try:
            result = await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            job.discarded.set()
            self._request_cancelled_after_submit()
            raise

        if result is None:
            raise RuntimeError("tokenizer render result was discarded")
        return result

    def _execute(self, job: _RenderJob) -> list[int] | None:
        started_at = time.monotonic()
        job.dequeued = True
        if job.discarded.is_set():
            self._job_discarded_before_render(started_at - job.queued_at)
            return None

        self._job_started(started_at - job.queued_at, started_at)
        try:
            prompt_ids = job.render()
        except BaseException:
            self._job_finished(started_at, failed=True)
            raise
        self._job_finished(started_at, prompt_tokens=len(prompt_ids))
        return prompt_ids

    def _future_done(self, future: Future, job: _RenderJob) -> None:
        self._futures.discard(future)
        assert self._slots is not None
        self._slots.release()
        with self._metrics_lock:
            self._outstanding -= 1
            if not job.dequeued:
                self._waiting -= 1

    def _schedule_future_done(self, future: Future, job: _RenderJob) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._future_done, future, job)
        except RuntimeError:
            # The app loop can be force-stopped after its shutdown timeout while
            # a third-party tokenizer call is still returning.
            pass

    async def monitor_event_loop_lag(self, interval_seconds: float) -> None:
        """Sample scheduling delay while the adapter application is running."""
        if interval_seconds <= 0:
            raise ValueError("event-loop lag interval must be positive")
        loop = asyncio.get_running_loop()
        while True:
            expected = loop.time() + interval_seconds
            await asyncio.sleep(interval_seconds)
            lag_ms = max(0.0, loop.time() - expected) * 1000.0
            with self._metrics_lock:
                self._event_loop_lag_ms.append(lag_ms)

    async def close(self) -> None:
        """Stop accepting renders and drain the bounded executor."""
        self.bind_to_current_loop()
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._drain(), name="tokenizer-worker-close")
            self._close_task.add_done_callback(_consume_future_exception)
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        while self._acquire_tasks:
            acquire_tasks = list(self._acquire_tasks)
            for task in acquire_tasks:
                self._close_cancelled_acquires.add(task)
                task.cancel()
            await asyncio.gather(*acquire_tasks, return_exceptions=True)
            await asyncio.sleep(0)

        self._executor.shutdown(wait=False, cancel_futures=False)
        futures = list(self._futures)
        if futures:
            await asyncio.gather(*(asyncio.wrap_future(future) for future in futures), return_exceptions=True)
        self._executor.shutdown(wait=True)
        await asyncio.sleep(0)

    def snapshot(self) -> dict[str, object]:
        """Return a thread-safe JSON-serializable metrics snapshot."""
        now = time.monotonic()
        with self._metrics_lock:
            busy_seconds = self._busy_seconds
            if self._active and self._active_started_at is not None:
                busy_seconds += now - self._active_started_at
            elapsed_seconds = max(now - self._created_at, 1e-9)
            queue_depth = max(self._outstanding - self._active, 0)
            admission_waiters = max(self._waiting - queue_depth, 0)
            return {
                "worker": {
                    "threads": 1,
                    "max_outstanding": self.max_pending,
                    "outstanding": self._outstanding,
                    "outstanding_peak": self._outstanding_peak,
                    "active": self._active,
                    "active_peak": self._active_peak,
                    "queue_depth": queue_depth,
                    "admission_waiters": admission_waiters,
                    "waiting_peak": self._waiting_peak,
                    "utilization": min(busy_seconds / elapsed_seconds, 1.0),
                },
                "counters": {
                    "requests_total": self._requests_total,
                    "submitted_total": self._submitted_total,
                    "completed_total": self._completed_total,
                    "failed_total": self._failed_total,
                    "request_cancelled_total": self._request_cancelled_total,
                    "render_cancelled_total": self._render_cancelled_total,
                    "discarded_total": self._discarded_total,
                    "discarded_before_render_total": self._discarded_before_render_total,
                },
                "prompt_tokens": _distribution(self._prompt_tokens),
                "render_latency_ms": _distribution(self._render_latency_ms),
                "queue_wait_ms": _distribution(self._queue_wait_ms),
                "event_loop_lag_ms": _distribution(self._event_loop_lag_ms),
                "sample_window_size": _METRIC_WINDOW_SIZE,
            }

    def _request_queued(self) -> None:
        with self._metrics_lock:
            self._requests_total += 1
            self._waiting += 1
            self._waiting_peak = max(self._waiting_peak, self._waiting)

    def _request_cancelled_before_submit(self) -> None:
        with self._metrics_lock:
            self._render_cancelled_total += 1
            self._waiting -= 1

    def _request_rejected_after_close(self) -> None:
        with self._metrics_lock:
            self._waiting -= 1

    def _job_submitted(self) -> None:
        with self._metrics_lock:
            self._submitted_total += 1
            self._outstanding += 1
            self._outstanding_peak = max(self._outstanding_peak, self._outstanding)

    def _submission_failed(self) -> None:
        with self._metrics_lock:
            self._failed_total += 1
            self._waiting -= 1
            self._outstanding -= 1

    def _request_cancelled_after_submit(self) -> None:
        with self._metrics_lock:
            self._render_cancelled_total += 1
            self._discarded_total += 1

    def _job_discarded_before_render(self, queue_wait_seconds: float) -> None:
        with self._metrics_lock:
            self._waiting -= 1
            self._discarded_before_render_total += 1
            self._queue_wait_ms.append(queue_wait_seconds * 1000.0)

    def _job_started(self, queue_wait_seconds: float, started_at: float) -> None:
        with self._metrics_lock:
            self._waiting -= 1
            self._active += 1
            self._active_peak = max(self._active_peak, self._active)
            self._active_started_at = started_at
            self._queue_wait_ms.append(queue_wait_seconds * 1000.0)

    def _job_finished(self, started_at: float, *, prompt_tokens: int | None = None, failed: bool = False) -> None:
        finished_at = time.monotonic()
        with self._metrics_lock:
            self._active -= 1
            self._active_started_at = None
            self._busy_seconds += finished_at - started_at
            self._completed_total += 1
            self._failed_total += int(failed)
            self._render_latency_ms.append((finished_at - started_at) * 1000.0)
            if prompt_tokens is not None:
                self._prompt_tokens.append(float(prompt_tokens))

    def record_request_cancellation(self) -> None:
        with self._metrics_lock:
            self._request_cancelled_total += 1


def _consume_future_exception(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()
