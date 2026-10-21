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
try:
    import sglang_router  # noqa: F401
except ImportError:
    _router_stub = types.ModuleType("sglang_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = _router_stub
try:
    import transformers  # noqa: F401
except ImportError:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import slime.rollout.fully_async_rollout as fa
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.staleness import compute_staleness_metrics, sample_staleness
from slime.utils.types import Sample

NUM_GPUS = 0


def test_custom_data_source_keeps_original_entrypoint(monkeypatch):
    args = SimpleNamespace()
    # A custom data source may have its own unrelated scheduler attribute.
    source = SimpleNamespace(scheduler=object())
    expected = object()

    async def generate(received_args, rollout_id, received_source):
        assert received_args is args and received_source is source
        assert rollout_id == 3
        return expected

    monkeypatch.setattr(fa, "_get_worker", lambda args, source: _make_worker(monkeypatch))
    monkeypatch.setattr(fa, "_generate_rollout_async", generate)
    monkeypatch.setattr(fa, "run", asyncio.run)
    result = fa.generate_rollout_fully_async(args, 3, source)
    assert isinstance(result, RolloutFnTrainOutput)
    assert result.samples is expected


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
        n_samples_per_prompt=1,
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
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(
        rollout_batch_size=4, rollout_data_transport="object-store", rollout_sample_filter_path=None
    )
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
def test_concurrency_is_scaled_by_samples_per_prompt(monkeypatch):
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    args = SimpleNamespace(n_samples_per_prompt=4)
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=10).concurrency == 2
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=2).concurrency == 1
    args.fully_async_group_prefetch_factor = 2
    assert fa.AsyncRolloutWorker(args, _FakeDataBuffer([]), concurrency=10).concurrency == 4


@pytest.mark.unit
def test_dynamic_filter_drops_groups_and_refills(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(8):
        group = _make_group(gid)
        group[0].reward = float(gid % 2)
        worker.output_queue.put(fa.CompletedSampleRecord(gid, group, policy_version=0))
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)

    def keep_odd(args, group):
        return DynamicFilterOutput(keep=bool(group[0].reward), reason="even")

    monkeypatch.setattr(fa, "load_function", lambda path: keep_odd)
    args = SimpleNamespace(
        rollout_batch_size=3,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path="test.keep_odd",
        reward_key=None,
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=7, data_buffer=None))

    assert [group[0].index for group in result.samples] == [1, 3, 5]
    assert "_dropped_samples" not in result.metrics
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 3
    assert result.metrics["rollout/dynamic_filter/dropped_ratio"] == 0.5
    assert result.metrics["rollout/dynamic_filter/drop_even"] == 3
    assert result.metrics["rollout/raw_reward_before_filters"] == 0.5
    assert [record.gid for record in worker.get_completed_groups()] == [6, 7]


@pytest.mark.unit
@pytest.mark.parametrize("segmented", [False, True])
def test_uniform_reward_filter_replenishes_eight_sample_groups(monkeypatch, segmented):
    worker = _make_worker(monkeypatch)
    rewards = [[0.0] * 8, [1.0] * 8, [0.0, 1.0] * 4, [1.0, 0.0] * 4, [0.0] * 8]
    groups = []
    for gid, values in enumerate(rewards):
        group = []
        for index, reward in enumerate(values):
            sample = Sample(index=gid * 8 + index, reward=reward, status=Sample.Status.COMPLETED)
            # Multiple context windows still represent one trajectory reward.
            group.append([sample, sample] if segmented and index % 2 else [sample] if segmented else sample)
        groups.append(group)
        worker.output_queue.put(fa.CompletedSampleRecord(gid, group, policy_version=0))
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)
    args = SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=8,
        reward_key=None,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=0, data_buffer=None))
    assert result.samples == groups[2:4]
    assert all(len(group) == 8 for group in result.samples)
    assert result.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1
    assert result.metrics["rollout/dynamic_filter/drop_zero_std_1.0"] == 1
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 2
    assert result.metrics["rollout/raw_reward_before_filters"] == 0.5
    assert [record.gid for record in worker.get_completed_groups()] == [4]


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions,expected",
    [(["10"], 0), (["8"], 2), (["3", "10"], 7), ([], None), (["bad"], None), (["11"], None), (["²"], None)],
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
        "staleness/mean": 7.5,
        "staleness/max": 8,
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
            worker._make_done_cb(gid, _make_group(gid), worker.policy_version)(_DoneTask(gid))

    pusher = threading.Thread(target=_push_all, daemon=True)
    pusher.start()
    pusher.join(timeout=30)

    assert not pusher.is_alive(), "done-callback blocked on a full output queue"
    assert worker.queue_size() == 1001


@pytest.mark.unit
def test_interrupted_segmented_group_retains_successful_siblings(monkeypatch):
    worker = _make_worker(monkeypatch)
    original = [Sample(index=i, prompt=f"original-{i}") for i in range(8)]
    callback = worker._make_done_cb(0, original, worker.policy_version)
    original[0].remove_sample = True
    original[0].prompt = "mutated by aborted generator"
    result = [[Sample(index=i, status=Sample.Status.COMPLETED)] for i in range(8)]
    result[3].append(Sample(index=3, status=Sample.Status.ABORTED))
    callback(SimpleNamespace(result=lambda: result))
    assert worker.queue_size() == 0
    [requeued] = worker.data_buffer.requeued
    assert len(requeued) == 8
    for index, sample in enumerate(requeued):
        if index == 3:
            assert isinstance(sample, Sample)
            assert sample.prompt == "original-3"
            assert sample.status == Sample.Status.PENDING and not sample.remove_sample
        else:
            assert sample is result[index]


def test_oldest_buffer_order_uses_retained_segment_versions():
    from slime.data.data_source import pop_oldest

    older = [[Sample(weight_versions=["2", "7"])], Sample()]
    newer = [[Sample(weight_versions=["4"])], Sample()]
    buffer = [newer, older]
    assert pop_oldest(None, None, buffer, 1) == [older]
    assert buffer == [newer]


@pytest.mark.parametrize("status,reward", [(Sample.Status.ABORTED, 0.0), (Sample.Status.COMPLETED, None)])
def test_retry_rejects_unfinished_segments_before_starting_generation(monkeypatch, status, reward):
    from slime.rollout import sglang_rollout

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: types.SimpleNamespace(aborted=False))
    group = [Sample(), [Sample(status=status, reward=reward)]]
    with pytest.raises(ValueError, match="Only completed segmented"):
        asyncio.run(sglang_rollout.generate_and_rm_group(types.SimpleNamespace(group_rm=False), group, {}))


@pytest.mark.unit
@pytest.mark.parametrize("status", [Sample.Status.PENDING, Sample.Status.ABORTED])
def test_unfinished_flat_group_is_requeued_without_discarding_partial_response(monkeypatch, status):
    worker = _make_worker(monkeypatch)
    group = [Sample(index=0, status=status, response="partial", response_length=7)]
    worker._make_done_cb(0, group, worker.policy_version)(SimpleNamespace(result=lambda: group))
    assert worker.queue_size() == 0
    assert worker.data_buffer.requeued == [group]
    assert worker.data_buffer.requeued[0][0] is group[0]


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
        worker.close()

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
    worker._make_done_cb(gid=12, input_group=_make_group(8), policy_version=new_admission_version)(_DoneTask(8))
    worker._make_done_cb(gid=11, input_group=_make_group(7), policy_version=old_admission_version)(_DoneTask(7))

    records = worker.get_completed_groups(limit=2)
    assert [(record.gid, record.policy_version) for record in records] == [(12, 4), (11, 3)]
    assert [record.group[0].policy_version for record in records] == [4, 3]
    assert worker.policy_version == 4


@pytest.mark.unit
def test_failed_weight_update_does_not_publish_policy_version(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=5)
    source = SimpleNamespace(policy_version=5, consumers={"fully_async": worker})

    fa.after_weight_update(data_source=source, policy_version=5, succeeded=False)
    assert source.policy_version == 5
    assert worker.policy_version == 5

    fa.after_weight_update(data_source=source, policy_version=6, succeeded=True)
    assert source.policy_version == 6
    assert worker.policy_version == 6


@pytest.mark.unit
def test_policy_version_rejects_regression(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=2)

    with pytest.raises(ValueError, match="policy version must be monotonic"):
        worker.publish_policy_version(1)


def test_policy_publication_is_scoped_to_source_before_and_after_start(monkeypatch):
    monkeypatch.setattr(fa.AsyncRolloutWorker, "start", lambda worker: None)
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(fa, "get_rollout_num_engines", lambda args: 1)
    args = SimpleNamespace(sglang_server_concurrency=1)
    first, second = _FakeDataBuffer([]), _FakeDataBuffer([])
    fa.after_weight_update(data_source=first, policy_version=3, succeeded=True)
    a, b = fa._get_worker(args, first), fa._get_worker(args, second)
    assert (a.policy_version, b.policy_version) == (3, 0)
    fa.after_weight_update(data_source=second, policy_version=1, succeeded=True)
    fa.after_weight_update(data_source=first, policy_version=4, succeeded=False)
    assert (a.policy_version, b.policy_version) == (3, 1)


def test_retry_keeps_completed_sibling_policy_and_stamps_new_segments(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=4)
    retained = Sample(index=0, status=Sample.Status.COMPLETED, reward=1, policy_version=2)
    pending = Sample(index=1, status=Sample.Status.ABORTED, policy_version=3)
    callback = worker._make_done_cb(0, [[retained], pending], worker.policy_version)
    assert pending.policy_version == 4
    worker.publish_policy_version(5)
    replacement = Sample(index=1, status=Sample.Status.COMPLETED, reward=1)
    callback(SimpleNamespace(result=lambda: [[retained], [replacement]]))
    [record] = worker.get_completed_groups()
    assert record.policy_version == 4
    assert (retained.policy_version, replacement.policy_version) == (2, 4)


def test_restore_legacy_ready_queue_keeps_unknown_policy(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=4)
    group = _make_group(7)
    worker.load_state_dict({"ready": [(11, group)]})
    [record] = worker.get_completed_groups()
    assert (record.gid, record.group) == (11, group)
    assert record.policy_version is None and group[0].policy_version is None


def test_admission_pause_keeps_unfinished_agent_group_alive(monkeypatch):
    started, finish = threading.Event(), threading.Event()
    group = _make_group(7)
    invocations = []

    async def generate(args, received, sampling_params, evaluation):
        invocations.append(received)
        started.set()
        while not finish.is_set():
            await asyncio.sleep(0.01)
        return [[received[0]]]

    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    worker = _make_worker(monkeypatch, data_buffer=_FakeDataBuffer([group]), concurrency=1)
    worker.start()
    try:
        assert started.wait(5)
        assert worker.pause(drain=False) is False
        assert worker.active == 1 and worker.queue_size() == 0
        # Weight publication pauses new admission; the existing sandbox task is
        # still the same coroutine when the next training round resumes it.
        worker.resume()
        finish.set()
        deadline = time.monotonic() + 5
        while worker.queue_size() == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert invocations == [group]
        assert worker.get_completed_groups()[0].group == [[group[0]]]
        assert not worker.data_buffer.requeued
    finally:
        finish.set()
        worker.close()


@pytest.mark.parametrize("transport", ["object-store", "straw"])
def test_local_worker_lifecycle_is_scoped_to_source(monkeypatch, transport):
    from slime.data.data_source import DataSource

    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(fa, "get_rollout_num_engines", lambda args: 1)

    async def generate(args, group, sampling_params, evaluation):
        await asyncio.sleep(0.01)
        return group

    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    args = SimpleNamespace(n_samples_per_prompt=1, sglang_server_concurrency=2, rollout_data_transport=transport)
    sources = [_FakeDataBuffer([_make_group(i) for i in range(20)]) for _ in range(2)]
    workers = [fa._get_worker(args, source) for source in sources]
    try:
        assert workers[0] is not workers[1]
        deadline = time.monotonic() + 5
        while not all(worker.queue_size() for worker in workers):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        for worker in workers:
            assert worker.pause() is False
            saved = worker.state_dict()
            queued = worker.queue_size()
            time.sleep(0.03)
            assert worker.queue_size() == queued and not worker.active
            replacement = fa.AsyncRolloutWorker(args, worker.data_buffer, concurrency=2)
            replacement.load_state_dict(saved)
            assert replacement.get_completed_groups() == worker.get_completed_groups()
    finally:
        for source in sources:
            DataSource.close(source)
    assert all(not worker.worker_thread.is_alive() for worker in workers)


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
    monkeypatch.setattr(fa, "_get_worker", lambda args, data_buffer: worker)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        max_policy_version_lag=0,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
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
        worker.close()

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

    worker._make_done_cb(gid=7, input_group=aborted, policy_version=0)(_DoneTask())

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
    monkeypatch.setattr(fa, "_get_worker", lambda args, source: worker)

    output = fa.generate_rollout_fully_async(SimpleNamespace(), 0, None)

    assert isinstance(output, RolloutFnTrainOutput)
    assert output.samples == samples
    assert output.metrics["fully_async/version_lag/count_0"] == 1


@pytest.mark.parametrize("segmented", [False, True])
def test_stale_retry_after_partial_checkpoint_uses_original_inputs(monkeypatch, segmented):
    import pickle

    worker = _make_worker(monkeypatch, policy_version=2, max_policy_version_lag=0)
    original = [Sample(index=i, prompt=f"original-{i}") for i in range(2)]
    callback = worker._make_done_cb(0, original, worker.policy_version)
    original[0].prompt = "mutated during generation"
    finished = Sample(index=0, status=Sample.Status.COMPLETED, reward=1, response="old result")
    partial = Sample(index=1, status=Sample.Status.ABORTED, response="partial tokens")
    result = [[finished], [partial]] if segmented else [finished, partial]
    callback(SimpleNamespace(result=lambda: result))
    [retry] = worker.data_buffer.requeued
    # Ordinary retries retain completed siblings and flat partial responses.
    assert retry[0] is result[0]
    if not segmented:
        assert retry[1] is partial
    else:
        assert retry[1].prompt == "original-1" and retry[1].status is Sample.Status.PENDING

    # Source checkpoints pickle both the buffer and the consumer state. Resume
    # the partial group under newer weights, then checkpoint its ready record.
    retry = pickle.loads(pickle.dumps(retry))
    restored = _make_worker(monkeypatch, policy_version=3, max_policy_version_lag=0)
    callback = restored._make_done_cb(1, retry, restored.policy_version)
    new = Sample(index=1, status=Sample.Status.COMPLETED, reward=1, response="new result")
    completed = [retry[0], [new] if segmented else new]
    callback(SimpleNamespace(result=lambda: completed))
    restored.admission_paused = True
    state = pickle.loads(pickle.dumps(restored.state_dict()))
    final = _make_worker(monkeypatch, policy_version=3, max_policy_version_lag=0)
    final.load_state_dict(state)
    assert final.get_completed_groups(max_policy_version_lag=0) == []
    [fresh] = final.data_buffer.requeued
    assert [s.prompt for s in fresh] == ["original-0", "original-1"]
    assert all(s.status is Sample.Status.PENDING and not s.response and s.reward is None for s in fresh)
    assert all(s.policy_version is None for s in fresh)

    # Regeneration can now complete at the current version without looping on
    # retained old trajectories or relabeling an old result as newly generated.
    callback = final._make_done_cb(2, fresh, final.policy_version)
    for sample in fresh:
        sample.status, sample.reward = Sample.Status.COMPLETED, 1
    callback(SimpleNamespace(result=lambda: fresh))
    [accepted] = final.get_completed_groups(max_policy_version_lag=0)
    assert accepted.policy_version == 3
    assert final.snapshot_metrics(reset=False)["fully_async/count/stale_requeued"] == 1


def test_lag_budget_rejects_restored_unknown_policy(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=4, max_policy_version_lag=0)
    worker.load_state_dict({"ready": [(11, _make_group(7))]})
    with pytest.raises(RuntimeError, match="unknown policy version"):
        worker.get_completed_groups(max_policy_version_lag=0)


def test_metrics_count_nested_sample_leaves(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=4)
    worker.record_processed_groups([[[Sample(policy_version=4), Sample(policy_version=3)], Sample()]])
    metrics = worker.snapshot_metrics(reset=False)
    assert metrics["fully_async/version_lag/count_0"] == 1
    assert metrics["fully_async/version_lag/count_1"] == 1
    assert metrics["fully_async/version_lag/count_unknown"] == 1


def test_entrypoint_metrics_preserve_transport_acceptance_fields(monkeypatch):
    worker = _make_worker(monkeypatch)
    original = RolloutFnTrainOutput(samples=[], metrics={"existing": 2}, on_accepted=lambda: None, sample_refs=[])
    monkeypatch.setattr(fa, "_get_worker", lambda args, source: worker)
    monkeypatch.setattr(fa, "run", lambda awaitable: (awaitable.close(), original)[1])
    result = fa.generate_rollout_fully_async(SimpleNamespace(), 0, None)
    assert result.samples is original.samples
    assert result.on_accepted is original.on_accepted and result.sample_refs is original.sample_refs
    assert result.metrics["existing"] == 2 and "fully_async/policy/current_version" in result.metrics
    assert original.metrics == {"existing": 2}


def test_policy_publication_survives_source_checkpoint(monkeypatch):
    import copy

    from slime.data.data_source import RolloutDataSourceWithBuffer

    # Exercise the built-in source's metadata save/load without loading a model
    # or tokenizer. Restart before constructing the lazy rollout consumer.
    source = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    source.metadata, source.consumers, source._restored_consumers, source.buffer = {}, {}, {}, []
    for name in ("sample_offset", "epoch_id", "sample_group_index", "sample_index"):
        setattr(source, name, 0)
    fa.after_weight_update(data_source=source, policy_version=7, succeeded=True)
    state = copy.deepcopy(source.state_dict())
    restored = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    restored.args = SimpleNamespace(rollout_shuffle=False)
    restored.consumers = {}
    restored.load_state_dict(state)
    assert fa.before_weight_update(data_source=restored, policy_version=0) == 7
    fa.after_weight_update(data_source=restored, policy_version=8, succeeded=True)
    assert restored.metadata["fully_async_policy_version"] == 8
    monkeypatch.setattr(fa, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(fa, "get_rollout_num_engines", lambda args: 1)
    monkeypatch.setattr(fa.AsyncRolloutWorker, "start", lambda self: None)
    worker = fa._get_worker(SimpleNamespace(sglang_server_concurrency=1), restored)
    assert worker.policy_version == 8


def test_ready_checkpoint_restores_policy_without_constructor_override(monkeypatch):
    worker = _make_worker(monkeypatch, policy_version=5, max_policy_version_lag=0)
    worker.admission_paused = True
    worker.output_queue.put(fa.CompletedSampleRecord(1, _make_group(1), 5))
    restored = _make_worker(monkeypatch, max_policy_version_lag=0)
    restored.load_state_dict(worker.state_dict())
    assert restored.policy_version == 5
    assert restored.get_completed_groups(max_policy_version_lag=0)[0].policy_version == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
