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
from slime.rollout.base_types import RolloutFnTrainOutput
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


def _make_worker(
    monkeypatch,
    data_buffer=None,
    concurrency=4,
    policy_version=0,
    max_policy_version_lag=None,
) -> fa.AsyncRolloutWorker:
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=4,
        max_policy_version_lag=max_policy_version_lag,
    )
    return fa.AsyncRolloutWorker(
        args,
        data_buffer or _FakeDataBuffer([]),
        concurrency=concurrency,
        policy_version=policy_version,
    )


@pytest.mark.unit
def test_rollout_takes_target_groups_and_leaves_surplus_queued(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(10):
        worker.output_queue.put(fa.CompletedSampleRecord(gid, _make_group(gid), policy_version=0))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4)
    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert len(out) == 4
    # FIFO: the oldest four groups ship first.
    assert [group[0].index for group in out] == [0, 1, 2, 3]
    # The other six are still queued for the next rollout, not thrown away.
    assert worker.queue_size() == 6
    assert [record.gid for record in worker.get_completed_groups()] == [4, 5, 6, 7, 8, 9]


@pytest.mark.unit
def test_get_completed_groups_limit(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(5):
        worker.output_queue.put(fa.CompletedSampleRecord(gid, _make_group(gid), policy_version=0))

    assert [record.gid for record in worker.get_completed_groups(limit=2)] == [0, 1]
    assert [record.gid for record in worker.get_completed_groups()] == [2, 3, 4]
    assert worker.get_completed_groups(limit=3) == []


@pytest.mark.unit
def test_stale_groups_are_requeued_and_exact_lag_boundary_is_accepted(monkeypatch):
    data_buffer = _FakeDataBuffer([])
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, policy_version=5)

    stale_retry = [Sample(index=30, prompt="retry-30")]
    worker.output_queue.put(
        fa.CompletedSampleRecord(
            gid=3,
            group=_make_group(3),
            policy_version=3,
            retry_group=stale_retry,
        )
    )
    worker.output_queue.put(
        fa.CompletedSampleRecord(
            gid=4,
            group=_make_group(4),
            policy_version=4,
            retry_group=[Sample(index=40, prompt="retry-40")],
        )
    )

    records = worker.get_completed_groups(limit=1, max_policy_version_lag=1)

    assert [(record.gid, record.policy_version) for record in records] == [(4, 4)]
    assert data_buffer.requeued == [stale_retry]
    assert data_buffer.requeued[0][0].status is Sample.Status.PENDING
    metrics = worker.snapshot_metrics(reset=False)
    assert metrics["fully_async/count/stale_rejected"] == 1
    assert metrics["fully_async/count/stale_requeued"] == 1


@pytest.mark.unit
def test_unset_staleness_budget_preserves_legacy_queue_behavior(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=5)
    worker.output_queue.put(fa.CompletedSampleRecord(gid=3, group=_make_group(3), policy_version=1))

    records = worker.get_completed_groups(limit=1)

    assert [(record.gid, record.policy_version) for record in records] == [(3, 1)]


@pytest.mark.unit
def test_rollout_never_returns_stale_group_when_budget_is_enabled(monkeypatch):
    data_buffer = _FakeDataBuffer([])
    worker = _make_worker(monkeypatch, data_buffer=data_buffer, policy_version=2)
    stale_retry = [Sample(index=10, prompt="retry-10")]
    worker.output_queue.put(
        fa.CompletedSampleRecord(
            gid=1,
            group=_make_group(1),
            policy_version=1,
            retry_group=stale_retry,
        )
    )
    worker.output_queue.put(
        fa.CompletedSampleRecord(
            gid=2,
            group=_make_group(2),
            policy_version=2,
            retry_group=[Sample(index=20, prompt="retry-20")],
        )
    )
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        max_policy_version_lag=0,
    )

    out = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=data_buffer))

    assert [group[0].index for group in out] == [2]
    assert data_buffer.requeued == [stale_retry]


@pytest.mark.unit
def test_stale_retry_uses_pristine_admission_snapshot(monkeypatch):
    admitted_group = [Sample(index=6, prompt="p6")]
    data_buffer = _FakeDataBuffer([admitted_group])

    async def _complete_generation(args, group, sampling_params, evaluation):
        group[0].response = "generated"
        group[0].response_length = 1
        group[0].reward = 1.0
        group[0].status = Sample.Status.COMPLETED
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", _complete_generation)
    worker = _make_worker(
        monkeypatch,
        data_buffer=data_buffer,
        concurrency=1,
        policy_version=2,
        max_policy_version_lag=0,
    )
    worker.poll_interval = 0.01
    worker.start()
    try:
        deadline = time.time() + 3.0
        while worker.queue_size() == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert worker.queue_size() == 1

        worker.publish_policy_version(3)
        assert worker.get_completed_groups(limit=1, max_policy_version_lag=0) == []
    finally:
        worker.stop()

    assert len(data_buffer.requeued) == 1
    retry = data_buffer.requeued[0][0]
    assert retry.prompt == "p6"
    assert retry.response == ""
    assert retry.response_length == 0
    assert retry.reward is None
    assert retry.status is Sample.Status.PENDING
    assert retry.policy_version is None


@pytest.mark.unit
def test_negative_staleness_budget_is_rejected(monkeypatch):
    worker = _make_worker(monkeypatch)

    with pytest.raises(ValueError, match="must be non-negative"):
        worker.get_completed_groups(max_policy_version_lag=-1)


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
            worker._make_done_cb(gid, worker.policy_version)(_DoneTask(gid))

    pusher = threading.Thread(target=_push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=30)

    assert not pusher.is_alive(), "done-callback blocked on a full output queue"
    assert worker.queue_size() == 1001


@pytest.mark.unit
def test_metrics_use_fixed_lag_buckets_and_reset_counters(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=5)
    groups = [
        [Sample(index=0, policy_version=5)],
        [Sample(index=1, policy_version=4)],
        [Sample(index=2, policy_version=2)],
        [Sample(index=3, policy_version=0)],
        [Sample(index=4, policy_version=None)],
    ]

    worker.record_processed_groups(groups)
    metrics = worker.snapshot_metrics(reset=True)

    assert metrics == {
        "fully_async/version_lag/count_0": 1,
        "fully_async/version_lag/count_1": 1,
        "fully_async/version_lag/count_2_to_3": 1,
        "fully_async/version_lag/count_4_plus": 1,
        "fully_async/version_lag/count_unknown": 1,
        "fully_async/count/stale_rejected": 0,
        "fully_async/count/stale_requeued": 0,
        "fully_async/count/aborted_requeued": 0,
        "fully_async/queue/completed_groups": 0,
        "fully_async/policy/current_version": 5,
    }
    assert worker.snapshot_metrics(reset=False)["fully_async/version_lag/count_0"] == 0


@pytest.mark.unit
def test_aborted_group_increments_requeue_counter(monkeypatch):
    worker = _make_worker(monkeypatch)
    aborted = _make_group(7)
    aborted[0].status = Sample.Status.ABORTED

    class _DoneTask:
        def result(self):
            return aborted

    worker._make_done_cb(gid=7, policy_version=0)(_DoneTask())

    assert worker.queue_size() == 0
    assert worker.data_buffer.requeued == [aborted]
    assert worker.snapshot_metrics(reset=False)["fully_async/count/aborted_requeued"] == 1


@pytest.mark.unit
def test_generate_entrypoint_returns_samples_with_metrics(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=3)
    samples = [_make_group(1)]
    samples[0][0].policy_version = 3
    worker.record_processed_groups(samples)
    monkeypatch.setattr(fa, "run", lambda awaitable: (awaitable.close(), samples)[1])
    monkeypatch.setattr(fa, "_global_worker", worker)

    output = fa.generate_rollout_fully_async(SimpleNamespace(), 0, None)

    assert isinstance(output, RolloutFnTrainOutput)
    assert output.samples == samples
    assert output.metrics["fully_async/version_lag/count_0"] == 1


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
def test_out_of_order_completion_keeps_each_admission_policy_version(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=3)

    class _DoneTask:
        def __init__(self, index):
            self.index = index

        def result(self):
            return _make_group(self.index)

    old_admission_version = worker.policy_version
    worker.publish_policy_version(4)
    new_admission_version = worker.policy_version

    # The newer-policy request finishes first. Neither record may consult the
    # worker's mutable current version when its callback eventually runs.
    worker._make_done_cb(gid=12, policy_version=new_admission_version)(_DoneTask(8))
    worker._make_done_cb(gid=11, policy_version=old_admission_version)(_DoneTask(7))

    records = worker.get_completed_groups(limit=2)
    assert [(record.gid, record.policy_version) for record in records] == [(12, 4), (11, 3)]
    assert [record.group[0].policy_version for record in records] == [4, 3]
    assert worker.policy_version == 4


@pytest.mark.unit
def test_failed_weight_update_does_not_publish_policy_version(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=5)
    monkeypatch.setattr(fa, "_global_worker", worker)
    monkeypatch.setattr(fa, "_published_policy_version", 5)

    fa.after_weight_update(policy_version=5, succeeded=False)
    assert fa._published_policy_version == 5
    assert worker.policy_version == 5

    fa.after_weight_update(policy_version=6, succeeded=True)
    assert fa._published_policy_version == 6
    assert worker.policy_version == 6


@pytest.mark.unit
def test_policy_version_rejects_regression(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=2)

    with pytest.raises(ValueError, match="policy version must be monotonic"):
        worker.publish_policy_version(1)
