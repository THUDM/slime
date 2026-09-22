"""CPU unit tests for the fully-async rollout worker's queue contract.

The module docstring of ``slime.rollout.fully_async_rollout`` promises that the
worker's output queue "stays warm" across ``generate_rollout`` calls: each call
takes ``rollout_batch_size`` completed groups and leaves the rest queued.

Three behaviours are pinned here:

  1. ``_generate_rollout_async`` consumes exactly ``rollout_batch_size`` groups
     and leaves the surplus in the queue. (It used to drain the whole queue and
     slice — throwing away fully generated, reward-scored groups whose prompts
     had already been consumed from the data buffer.)
  2. The task done-callback never blocks. It runs on the event-loop thread, so
     a bounded queue that filled up would freeze every in-flight generation.
  3. Backpressure exists anyway: ``_loop`` stops pulling new prompts while a
     full pool of completed groups is already waiting to be consumed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from collections import deque
from types import SimpleNamespace

# ``fully_async_rollout`` imports ``sglang_rollout``, which needs sglang_router
# and (transitively) transformers — both deliberately absent from the CPU CI
# env. The tests below never dial a server or touch a tokenizer, so stub the
# imports, same as tests/test_agent/test_agent_rollout_cpu.py.
if "sglang_router" not in sys.modules:
    _router_stub = types.ModuleType("sglang_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = _router_stub
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import slime.rollout.fully_async_rollout as fa
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.staleness import compute_staleness_metrics, sample_staleness
from slime.utils.types import Sample


NUM_GPUS = 0


class _FakeGenerateState:
    def __init__(self, args):
        self.sampling_params = {}


class _FakeDataBuffer:
    """Finite fuel: one group per ``get_samples`` call until exhausted."""

    def __init__(self, groups):
        self._groups = deque(groups)
        self.requeued = []

    def get_samples(self, n):
        assert n == 1
        if not self._groups:
            return []
        return [self._groups.popleft()]

    def add_samples(self, groups):
        self.requeued.extend(groups)


def _make_group(index: int) -> list[Sample]:
    sample = Sample(index=index, prompt=f"p{index}")
    sample.status = Sample.Status.COMPLETED
    return [sample]


def _make_worker(monkeypatch, data_buffer=None, concurrency=4) -> fa.AsyncRolloutWorker:
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4, n_samples_per_prompt=1)
    return fa.AsyncRolloutWorker(args, data_buffer or _FakeDataBuffer([]), concurrency=concurrency)


@pytest.mark.unit
def test_rollout_takes_target_groups_and_leaves_surplus_queued(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(10):
        worker.output_queue.put((gid, _make_group(gid)))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4)
    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert len(out) == 4
    # FIFO: the oldest four groups ship first.
    assert [group[0].index for group in out] == [0, 1, 2, 3]
    # The other six are still queued for the next rollout, not thrown away.
    assert worker.queue_size() == 6
    assert [gid for gid, _ in worker.get_completed_groups()] == [4, 5, 6, 7, 8, 9]


@pytest.mark.unit
def test_get_completed_groups_limit(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(5):
        worker.output_queue.put((gid, _make_group(gid)))

    assert [gid for gid, _ in worker.get_completed_groups(limit=2)] == [0, 1]
    assert [gid for gid, _ in worker.get_completed_groups()] == [2, 3, 4]
    assert worker.get_completed_groups(limit=3) == []


@pytest.mark.unit
def test_concurrency_is_scaled_by_samples_per_prompt(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(n_samples_per_prompt=4)
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=10).concurrency == 2
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=2).concurrency == 1


@pytest.mark.unit
def test_dynamic_filter_drops_groups_and_refills(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(8):
        group = _make_group(gid)
        group[0].reward = float(gid % 2)
        worker.output_queue.put((gid, group))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)

    def keep_odd(args, group):
        return DynamicFilterOutput(keep=bool(group[0].reward), reason="even")

    monkeypatch.setattr(fa, "load_function", lambda path: keep_odd)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=3,
        dynamic_sampling_filter_path="test.keep_odd",
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=7, data_buffer=None))

    assert [group[0].index for group in result.samples] == [1, 3, 5]
    dropped = result.metrics["_dropped_samples"]
    assert [group[0].index for group in dropped] == [0, 2, 4]
    assert all(sample.remove_sample for group in dropped for sample in group)
    assert result.metrics["rollout/dynamic_filter/drop_even"] == 3
    assert [gid for gid, _ in worker.get_completed_groups()] == [6, 7]


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions,expected",
    [(["10"], 1), (["8"], 3), (["3", "10"], 8), ([], None), (["bad"], None), (["11"], None), (["²"], None)],
)
def test_sample_staleness_uses_oldest_valid_version(versions, expected):
    assert sample_staleness(SimpleNamespace(weight_versions=versions), 10) == expected


@pytest.mark.unit
def test_staleness_metrics_use_serving_snapshot():
    samples = [
        SimpleNamespace(weight_versions=["2"]),
        SimpleNamespace(weight_versions=["3"]),
        SimpleNamespace(weight_versions=[]),
    ]
    assert compute_staleness_metrics(samples, 10) == {
        "staleness/unknown_count": 1,
        "staleness/mean": 8.5,
        "staleness/max": 9,
    }
    assert compute_staleness_metrics(samples, None) == {}


@pytest.mark.unit
def test_done_callback_never_blocks_event_loop_thread(monkeypatch):
    """The callback runs on the loop thread; blocking there freezes every
    in-flight generation. Push more results than the old bounded-queue cap
    (1000) through it and require completion."""
    worker = _make_worker(monkeypatch)

    class _DoneTask:
        def __init__(self, gid):
            self._result = _make_group(gid)

        def result(self):
            return self._result

    def _push_all():
        for gid in range(1001):
            worker._make_done_cb(gid)(_DoneTask(gid))

    pusher = threading.Thread(target=_push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=30)

    assert not pusher.is_alive(), "done-callback blocked on a full output queue"
    assert worker.queue_size() == 1001


@pytest.mark.unit
def test_loop_backpressure_stops_topping_up_when_queue_is_full(monkeypatch):
    """With instantly-completing generations and plenty of fuel, the queue must
    plateau around ``concurrency`` instead of absorbing the whole dataset."""
    concurrency = 3
    fuel = 60
    data_buffer = _FakeDataBuffer([_make_group(i) for i in range(fuel)])

    async def _instant_generate(args, group, sampling_params, evaluation):
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _instant_generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=concurrency)
    worker.poll_interval = 0.01

    worker.start()
    try:
        # Give the loop ample iterations to overshoot if it is going to.
        deadline = time.time() + 3.0
        max_seen = 0
        while time.time() < deadline:
            max_seen = max(max_seen, worker.queue_size())
            if max_seen > 2 * concurrency:
                break
            time.sleep(0.02)
    finally:
        worker.stop()

    # In-flight tasks may still land after the gate check, so allow one pool
    # beyond the gate — but nothing near the unthrottled fuel size.
    assert 0 < max_seen <= 2 * concurrency, f"queue grew to {max_seen} with concurrency={concurrency}"


@pytest.mark.unit
@pytest.mark.parametrize("failure_site", ["generate", "data_source", "result_type", "requeue"])
def test_worker_errors_reach_waiting_rollout(monkeypatch, failure_site):
    """A failed producer must fail the consumer instead of waiting forever."""
    error = ValueError("broken rollout dependency")
    data_buffer = _FakeDataBuffer([_make_group(0)])

    def fail(*args):
        raise error

    async def generate(args, group, sampling_params, evaluation):
        if failure_site == "generate":
            raise error
        if failure_site == "result_type":
            return None
        group[0].status = Sample.Status.ABORTED
        return group

    if failure_site == "data_source":
        data_buffer.get_samples = fail
    if failure_site == "requeue":
        data_buffer.add_samples = fail
    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=1)
    worker.poll_interval = 0.01
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)

    async def collect():
        return await asyncio.wait_for(fa._generate_rollout_async(args, 0, data_buffer), timeout=2)

    worker.start()
    try:
        with pytest.raises(RuntimeError, match="fully-async rollout worker failed") as exc:
            asyncio.run(collect())
        if failure_site == "result_type":
            assert isinstance(exc.value.__cause__, TypeError)
        else:
            assert exc.value.__cause__ is error
        assert not worker.running
    finally:
        worker.stop()


@pytest.mark.unit
def test_failed_global_worker_is_not_silently_restarted(monkeypatch):
    worker = _make_worker(monkeypatch)
    error = ValueError("reward failed before collection")

    def fail():
        raise error

    task = SimpleNamespace(result=fail)
    worker._make_done_cb(0)(task)
    worker.worker_thread = threading.Thread(target=lambda: None)
    worker.worker_thread.start()
    worker.worker_thread.join()
    monkeypatch.setattr(fa, "_global_worker", worker)

    with pytest.raises(RuntimeError, match="fully-async rollout worker failed") as exc:
        fa._get_global_worker(SimpleNamespace(), worker.data_buffer)
    assert exc.value.__cause__ is error


@pytest.mark.unit
def test_aborted_group_is_still_requeued(monkeypatch):
    worker = _make_worker(monkeypatch)
    group = _make_group(0)
    group[0].status = Sample.Status.ABORTED
    worker._make_done_cb(0)(SimpleNamespace(result=lambda: group))

    assert worker.data_buffer.requeued == [group]
    assert worker.get_completed_groups() == []
    assert worker.running


@pytest.mark.unit
@pytest.mark.parametrize("stopping", [False, True])
def test_cancelled_task_is_only_an_error_while_running(monkeypatch, stopping):
    worker = _make_worker(monkeypatch)

    def cancelled():
        raise asyncio.CancelledError

    if stopping:
        worker.stop()
    worker._make_done_cb(0)(SimpleNamespace(result=cancelled))
    if stopping:
        assert worker.get_completed_groups() == []
    else:
        with pytest.raises(RuntimeError, match="fully-async rollout worker failed") as exc:
            worker.get_completed_groups()
        assert "cancelled" in str(exc.value.__cause__)


@pytest.mark.unit
def test_worker_failure_cancels_other_inflight_generations(monkeypatch):
    blocked_started = threading.Event()
    blocked_cancelled = threading.Event()
    data_buffer = _FakeDataBuffer([_make_group(0), _make_group(1)])

    async def generate(args, group, sampling_params, evaluation):
        if group[0].index == 0:
            while not blocked_started.is_set():
                await asyncio.sleep(0)
            raise ValueError("generation failed")
        blocked_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            blocked_cancelled.set()

    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, concurrency=2)
    worker.poll_interval = 0.01
    worker.start()
    try:
        assert blocked_cancelled.wait(timeout=2), "in-flight task survived a fatal worker error"
        with pytest.raises(RuntimeError) as exc:
            worker.get_completed_groups()
        assert isinstance(exc.value.__cause__, ValueError)
    finally:
        worker.stop()
    assert not worker.worker_thread.is_alive()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
