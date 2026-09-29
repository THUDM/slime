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
import contextlib
import copy
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
import slime.rollout.sglang_rollout as sr
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

    monkeypatch.setattr(fa, "_generate_rollout_async", generate)
    monkeypatch.setattr(fa, "run", asyncio.run)
    assert fa.generate_rollout_fully_async(args, 3, source) is expected


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
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=1)
    return fa.AsyncRolloutWorker(args, data_buffer or _FakeDataBuffer([]), concurrency=concurrency)


@pytest.mark.unit
def test_rollout_takes_target_groups_and_leaves_surplus_queued(monkeypatch):
    worker = _make_worker(monkeypatch)
    for gid in range(10):
        worker.output_queue.put((gid, _make_group(gid)))
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(
        rollout_batch_size=4, rollout_data_transport="object-store", rollout_sample_filter_path=None
    )
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
        rollout_batch_size=3,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        dynamic_sampling_filter_path="test.keep_odd",
    )
    result = asyncio.run(fa._generate_rollout_async(args, rollout_id=7, data_buffer=None))

    assert [group[0].index for group in result.samples] == [1, 3, 5]
    assert "_dropped_samples" not in result.metrics
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 3
    assert result.metrics["rollout/dynamic_filter/dropped_ratio"] == 0.5
    assert result.metrics["rollout/dynamic_filter/drop_even"] == 3
    assert [gid for gid, _ in worker.get_completed_groups()] == [6, 7]


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
        "staleness/multi_version_frac": 0.0,
    }
    assert compute_staleness_metrics(samples, None) == {}


@pytest.mark.unit
def test_staleness_reports_samples_spanning_several_versions():
    samples = [
        SimpleNamespace(weight_versions=["2", "3"]),
        SimpleNamespace(weight_versions=["3", "3"]),
        SimpleNamespace(weight_versions=[]),
    ]
    assert compute_staleness_metrics(samples, 3)["staleness/multi_version_frac"] == pytest.approx(1 / 3)


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


# --- Trajectories continued across weight updates -------------------------
#
# A weight update pauses the engines with SGLang's default "abort" mode: each
# in-flight request returns HTTP 200 with its partial output and
# finish_reason "abort". The tests below drive the real generate_and_rm_group
# -> generate path against a scripted server to pin how such a sample is
# requeued and continued.


async def _no_wait():
    return None


class _FakeServerState:
    """The parts of GenerateState that generate_and_rm / generate touch, without a tokenizer."""

    def __init__(self, args):
        self.tokenizer = self.processor = None
        self.semaphore = asyncio.Semaphore(8)
        self.generation_pacer = SimpleNamespace(wait=_no_wait)
        self.aborted = False
        self.cancellable_tasks = set()
        self.active_server_generations = 0

    @contextlib.contextmanager
    def dp_rank_context(self):
        yield 0


def _chunk(tokens, version, finish):
    return {
        "text": "t" * len(tokens),
        "meta_info": {
            "output_token_logprobs": [[-0.1 * token, token] for token in tokens],
            "weight_version": version,
            "finish_reason": {"type": finish},
        },
    }


def _server_args(**overrides):
    values = dict(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=0,
        use_rollout_routing_replay=False,
        rollout_top_p=1.0,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        custom_generate_function_path=None,
    )
    return SimpleNamespace(**(values | overrides))


def _pending(index):
    # Prompt ids are already tokenized, so generate() never needs a tokenizer.
    return Sample(index=index, prompt=f"p{index}", tokens=[11, 12, 13])


@pytest.fixture
def fake_server(monkeypatch):
    """Answers each /generate request with the next scripted chunk and counts reward calls."""
    server = SimpleNamespace(replies=deque(), payloads=[], rewards=0, group_rewards=0)

    async def post(url, payload, **kwargs):
        server.payloads.append(copy.deepcopy(payload))
        return server.replies.popleft()

    async def async_rm(args, sample):
        server.rewards += 1
        return 1.0

    async def batched_async_rm(args, samples):
        server.group_rewards += 1
        return [1.0] * len(samples)

    monkeypatch.setattr(sr, "GenerateState", _FakeServerState)
    monkeypatch.setattr(sr, "post", post)
    monkeypatch.setattr(sr, "async_rm", async_rm)
    monkeypatch.setattr(sr, "batched_async_rm", batched_async_rm)
    return server


def _generate(args, group, max_new_tokens=5):
    return asyncio.run(sr.generate_and_rm_group(args, group, {"max_new_tokens": max_new_tokens}))


@pytest.mark.unit
def test_aborted_sample_is_requeued_and_continues_from_its_partial_response(monkeypatch, fake_server):
    data_buffer = _FakeDataBuffer([])
    worker = _make_worker(monkeypatch, data_buffer=data_buffer)
    sample = _pending(0)
    fake_server.replies.extend([_chunk([1, 2], "1", "abort"), _chunk([3], "2", "stop")])

    group = _generate(_server_args(), [sample])
    worker._make_done_cb(0)(SimpleNamespace(result=lambda: group))
    assert data_buffer.requeued == [[sample]] and worker.queue_size() == 0
    assert sample.status == Sample.Status.ABORTED and sample.reward is None

    group = _generate(_server_args(), data_buffer.requeued.pop())
    worker._make_done_cb(1)(SimpleNamespace(result=lambda: group))
    assert worker.get_completed_groups() == [(1, [sample])]
    # The second request resends the partial response with the remaining budget.
    assert fake_server.payloads[1]["input_ids"] == [11, 12, 13, 1, 2]
    assert fake_server.payloads[1]["sampling_params"]["max_new_tokens"] == 3
    assert sample.tokens == [11, 12, 13, 1, 2, 3]
    assert sample.rollout_log_probs == pytest.approx([-0.1, -0.2, -0.3])
    assert sample.loss_mask == [1, 1, 1]
    assert sample.weight_versions == ["1", "2"]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 1.0 and fake_server.rewards == 1


@pytest.mark.unit
def test_resume_regenerates_only_the_aborted_member(fake_server):
    finished, cut = _pending(0), _pending(1)
    fake_server.replies.extend([_chunk([7], "1", "stop"), _chunk([1, 2], "1", "abort")])
    group = _generate(_server_args(), [finished, cut])
    assert finished.status == Sample.Status.COMPLETED and cut.status == Sample.Status.ABORTED

    fake_server.replies.append(_chunk([3], "2", "stop"))
    _generate(_server_args(), group)
    assert len(fake_server.payloads) == 3
    assert fake_server.payloads[2]["input_ids"] == [11, 12, 13, 1, 2]
    assert finished.tokens == [11, 12, 13, 7]
    assert fake_server.rewards == 2


@pytest.mark.unit
def test_resume_with_an_exhausted_budget_truncates_without_a_request(fake_server):
    sample = _pending(0)
    fake_server.replies.append(_chunk([1, 2, 3], "1", "abort"))
    _generate(_server_args(), [sample], max_new_tokens=3)
    assert sample.status == Sample.Status.ABORTED

    _generate(_server_args(), [sample], max_new_tokens=3)
    assert len(fake_server.payloads) == 1
    assert sample.status == Sample.Status.TRUNCATED and fake_server.rewards == 1


@pytest.mark.unit
def test_group_reward_waits_until_aborted_members_finish(fake_server):
    args = _server_args(group_rm=True)
    fake_server.replies.extend([_chunk([7], "1", "stop"), _chunk([1], "1", "abort")])
    group = _generate(args, [_pending(0), _pending(1)])
    assert fake_server.group_rewards == 0
    assert [sample.reward for sample in group] == [None, None]

    fake_server.replies.append(_chunk([2], "2", "stop"))
    group = _generate(args, group)
    assert fake_server.group_rewards == 1
    assert [sample.reward for sample in group] == [1.0, 1.0]


@pytest.mark.unit
def test_resumed_multimodal_sample_resends_its_partial_response(monkeypatch, fake_server):
    monkeypatch.setattr(sr, "encode_image_for_rollout_engine", lambda image: "img")
    sample = _pending(0)
    sample.multimodal_inputs = {"images": [object()]}
    sample.multimodal_train_inputs = {"pixel_values": None}  # set by the processor on the first request
    fake_server.replies.extend([_chunk([1, 2], "1", "abort"), _chunk([3], "2", "stop")])

    _generate(_server_args(), [sample])
    _generate(_server_args(), [sample])

    first, second = fake_server.payloads
    # A fresh request sends text so SGLang expands the image itself; the resumed one must
    # send the processor ids plus the partial response instead of dropping it.
    assert first["text"] == "p0" and "input_ids" not in first
    assert second["input_ids"] == [11, 12, 13, 1, 2] and "text" not in second
    assert first["image_data"] == second["image_data"] == ["img"]
    assert sample.tokens == [11, 12, 13, 1, 2, 3]


@pytest.mark.unit
def test_evaluation_runs_on_the_worker_event_loop(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker.poll_interval = 0.01
    seen = {}

    async def eval_rollout(args, rollout_id):
        seen["loop"] = asyncio.get_running_loop()
        return f"eval-{rollout_id}", []

    monkeypatch.setattr(fa, "eval_rollout", eval_rollout)
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)
    worker.start()
    try:
        output = fa.generate_rollout_fully_async(SimpleNamespace(), 5, None, evaluation=True)
    finally:
        worker.stop()
    assert output == "eval-5"
    assert seen["loop"] is worker.event_loop


# --- Batches formed like molt's asynchronous trainer (--fully-async-pool-size) ---


class _PromptSource:
    """Like RolloutDataSource: one fresh group per prompt, wrapping into the next epoch."""

    def __init__(self, size=1000):
        self.dataset = list(range(size))
        self.sample_offset = 0
        self.epoch_id = 0
        self.dispatched = 0

    def __len__(self):
        return len(self.dataset)

    def get_samples(self, n):
        groups = []
        for _ in range(n):
            if self.sample_offset == len(self.dataset):
                self.epoch_id += 1
                self.sample_offset = 0
            sample = _pending(self.dispatched)
            sample.metadata = {"epoch": self.epoch_id}
            groups.append([sample])
            self.sample_offset += 1
            self.dispatched += 1
        return groups


class _PoolGenerateState:
    def __init__(self, args):
        self.sampling_params = {"max_new_tokens": 5}


async def _finish_at_once(args, group, sampling_params, evaluation):
    for sample in group:
        sample.status = Sample.Status.COMPLETED
    return group


def _pooled_worker(monkeypatch, data_buffer, **overrides) -> fa.PooledRolloutWorker:
    monkeypatch.setattr(fa, "GenerateState", _PoolGenerateState)
    values = dict(
        fully_async_pool_size=4,
        fully_async_max_queued_batches=1,
        fully_async_drain_each_epoch=False,
        rollout_batch_size=2,
        rollout_seed=0,
        dynamic_sampling_filter_path=None,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
    )
    return fa.PooledRolloutWorker(_server_args(**(values | overrides)), data_buffer)


@contextlib.contextmanager
def _running(worker):
    worker.start()
    try:
        yield worker
    finally:
        worker.stop()


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _Keys:
    """Collection-order keys handed out in dispatch order."""

    def __init__(self, keys):
        self.keys = list(keys)

    def random(self):
        return self.keys.pop(0)


@pytest.mark.unit
def test_pooled_worker_refills_only_while_forming_a_batch(monkeypatch):
    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)
    source = _PromptSource()
    worker = _pooled_worker(monkeypatch, source)
    with _running(worker):
        assert _wait_until(lambda: worker.formed.qsize() == 1)
        time.sleep(0.2)
        # Four prompts fill the pool and one refills it after the first rollout is taken. The
        # second completes the batch; its slot is the only one, so nothing more is dispatched.
        assert source.dispatched == 5 and worker.formed.qsize() == 1
        groups, metrics = worker.next_batch()
        assert len(groups) == 2
        assert metrics["rollout/fully_async/queued_batches"] == 0
        assert metrics["rollout/fully_async/pool_finished"] == 2
        assert _wait_until(lambda: worker.formed.qsize() == 1)
        time.sleep(0.2)
        assert source.dispatched == 7


@pytest.mark.unit
def test_pooled_worker_takes_finished_rollouts_in_their_dispatch_key_order(monkeypatch):
    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)
    worker = _pooled_worker(monkeypatch, _PromptSource())
    worker._rng = _Keys([0.9, 0.1, 0.5, 0.3, 0.2, 0.7, 0.8, 0.6, 0.4])
    with _running(worker):
        first, _ = worker.next_batch()
        second, _ = worker.next_batch()
    # Prompts 0-3 fill the pool and finish: the lowest keys go first. Prompt 4 (key 0.2) refills
    # the pool but has not finished when the first batch takes its second rollout.
    assert sorted(group[0].index for group in first) == [1, 3]
    assert sorted(group[0].index for group in second) == [2, 4]


@pytest.mark.unit
def test_pooled_worker_drains_each_epoch_into_a_smaller_last_batch(monkeypatch):
    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)
    source = _PromptSource(size=7)
    worker = _pooled_worker(monkeypatch, source, rollout_batch_size=3, fully_async_drain_each_epoch=True)
    with _running(worker):
        batches = [worker.next_batch() for _ in range(6)]

    assert [len(groups) for groups, _ in batches] == [3, 3, 1, 3, 3, 1]
    assert [metrics["rollout/fully_async/epoch_tail"] for _, metrics in batches] == [0, 0, 1, 0, 0, 1]
    for epoch, chunk in enumerate((batches[:3], batches[3:])):
        samples = [group[0] for groups, _ in chunk for group in groups]
        # Every prompt of the epoch is trained once, and no batch mixes epochs.
        assert len(samples) == 7 and {sample.metadata["epoch"] for sample in samples} == {epoch}


@pytest.mark.unit
def test_pooled_worker_drops_an_epoch_tail_smaller_than_the_data_parallel_size(monkeypatch):
    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)
    worker = _pooled_worker(
        monkeypatch,
        _PromptSource(size=7),
        rollout_batch_size=3,
        fully_async_drain_each_epoch=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
    )
    with _running(worker):
        batches = [worker.next_batch() for _ in range(4)]
    # The 1-rollout tail cannot give both data-parallel ranks a sample, so it is dropped.
    assert [len(groups) for groups, _ in batches] == [3, 3, 3, 3]
    assert [group[0].metadata["epoch"] for group in batches[2][0]] == [1, 1, 1]


@pytest.mark.unit
def test_pooled_worker_resumes_an_aborted_rollout_in_place(monkeypatch, fake_server):
    monkeypatch.setattr(fa, "_ABORT_RETRY_DELAY", 0)
    source = _PromptSource()
    worker = _pooled_worker(monkeypatch, source, fully_async_pool_size=1, rollout_batch_size=1)
    # The third reply serves the prompt dispatched once the batch is taken.
    fake_server.replies.extend([_chunk([1, 2], "1", "abort"), _chunk([3], "2", "stop"), _chunk([4], "2", "stop")])
    with _running(worker):
        assert _wait_until(lambda: worker.formed.qsize() == 1)
        # The aborted request was resent with its partial response, without taking a new prompt.
        assert source.dispatched == 1 and len(fake_server.payloads) == 2 and fake_server.rewards == 1
        (group,), _ = worker.next_batch()

    (sample,) = group
    assert fake_server.payloads[1]["input_ids"] == [11, 12, 13, 1, 2]
    assert fake_server.payloads[1]["sampling_params"]["max_new_tokens"] == 3
    assert sample.weight_versions == ["1", "2"]
    assert sample.status == Sample.Status.COMPLETED and sample.reward == 1.0


@pytest.mark.unit
def test_pooled_worker_drops_a_failed_rollout_and_refills(monkeypatch):
    async def generate(args, group, sampling_params, evaluation):
        if group[0].index == 0:
            raise RuntimeError("reward model down")
        return await _finish_at_once(args, group, sampling_params, evaluation)

    monkeypatch.setattr(fa, "generate_and_rm_group", generate)
    worker = _pooled_worker(monkeypatch, _PromptSource())
    with _running(worker):
        groups, metrics = worker.next_batch()
    assert 0 not in [group[0].index for group in groups] and len(groups) == 2
    assert metrics["rollout/fully_async/failed_rollouts"] == 1


@pytest.mark.unit
def test_pooled_worker_applies_the_dynamic_filter(monkeypatch):
    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)

    def keep_odd(args, group):
        return DynamicFilterOutput(keep=group[0].index % 2 == 1, reason="even")

    monkeypatch.setattr(fa, "load_function", lambda path: keep_odd)
    worker = _pooled_worker(monkeypatch, _PromptSource(), dynamic_sampling_filter_path="test.keep_odd")
    with _running(worker):
        groups, metrics = worker.next_batch()
    assert len(groups) == 2 and all(group[0].index % 2 == 1 for group in groups)
    dropped = metrics["rollout/dynamic_filter/dropped_groups"]
    assert dropped == metrics["rollout/dynamic_filter/drop_even"] >= 1
    assert metrics["rollout/dynamic_filter/dropped_ratio"] == dropped / (dropped + 2)


@pytest.mark.unit
def test_pooled_worker_pauses_batch_formation_during_evaluation(monkeypatch):
    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)
    worker = _pooled_worker(monkeypatch, _PromptSource(), rollout_batch_size=1, fully_async_max_queued_batches=10**6)
    seen = {}

    async def evaluation():
        seen["loop"] = asyncio.get_running_loop()
        seen["before"] = worker.formed.qsize()
        await asyncio.sleep(0.3)
        seen["after"] = worker.formed.qsize()
        return "eval"

    with _running(worker):
        assert worker.evaluate(evaluation()) == "eval"
        assert _wait_until(lambda: worker.formed.qsize() > seen["after"] + 5)
    # A batch already being formed may finish, but no other starts while evaluating.
    assert seen["after"] - seen["before"] <= 1
    assert seen["loop"] is worker.event_loop


@pytest.mark.unit
def test_rollout_function_takes_a_formed_batch(monkeypatch):
    from slime.rollout.base_types import RolloutFnTrainOutput

    monkeypatch.setattr(fa, "generate_and_rm_group", _finish_at_once)
    worker = _pooled_worker(monkeypatch, _PromptSource(), fully_async_max_queued_batches=2)
    monkeypatch.setattr(fa, "_get_global_worker", lambda args, data_buffer: worker)
    with _running(worker):
        assert _wait_until(lambda: worker.formed.qsize() == 2)
        output = fa.generate_rollout_fully_async(worker.args, 0, None)
    assert isinstance(output, RolloutFnTrainOutput)
    indices = [group[0].index for group in output.samples]
    assert len(indices) == 2 and indices == sorted(indices)
    assert output.metrics["rollout/fully_async/queued_batches"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
