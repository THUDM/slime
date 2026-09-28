"""Persistent queue input ownership and global batch processing."""

import asyncio
import copy
import json
import os
import random
import sys
import threading
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.base_types import iter_samples
from slime.rollout.data_source import RolloutDataSource
from slime.rollout.queue_data_source import QueueDataSource, QueueReader, RolloutQueueController
from slime.utils.data import Dataset, process_rollout_data
from slime.utils.rollout_transport import unpack_rollout_payload
from slime.utils.tensor_store import TensorRef
from slime.utils.types import Sample

NUM_GPUS = 0

# No HTTP server is used by these CPU tests, including their Ray workers.
try:
    import sglang_router  # noqa: F401
except ImportError:
    sys.modules["sglang_router"] = SimpleNamespace(__version__="0.3.0")


class _LocalHandle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.target, name))


@pytest.fixture
def source_factory(monkeypatch, tmp_path, request):
    import ray

    def init(source, args):
        source.args = args
        source.metadata = {}
        source.sample_offset = source.epoch_id = source.sample_group_index = source.sample_index = 0
        source.dataset = Dataset.__new__(Dataset)
        source.dataset.origin_samples = [Sample(prompt=f"prompt-{i}") for i in range(7)]
        source.dataset.samples = source.dataset.origin_samples
        source.dataset.seed = 31
        source.dataset.epoch_id = -1
        source.dataset.shuffle(0)

    monkeypatch.setattr(RolloutDataSource, "__init__", init)
    monkeypatch.setattr(ray, "get", lambda value: value)
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        rollout_seed=31,
        rollout_shuffle=True,
        buffer_filter_path=None,
        buffer_sort_by_staleness=False,
        rollout_data_transport="straw",
        rollout_queue_online_gc=getattr(request, "param", False),
        rollout_data_dir=str(tmp_path),
    )
    controller = RolloutQueueController(args)
    sources = []

    def create(worker, take=None):
        handle = _LocalHandle(controller)
        if take is not None:
            handle.take = SimpleNamespace(remote=take)
        source = QueueReader(copy.copy(args), handle, str(worker), 7)
        sources.append(source)
        return source

    create.controller = controller
    create.args = args
    yield create
    for source in sources:
        source.close()
    controller.close()


def test_task_claims_and_producer_cursor_survive_recovery(source_factory):

    controller = source_factory.controller
    first = controller.take("first", 2)
    second = controller.take("second", 2)
    ids = [a.task.task_id for a in first.assignments + second.assignments]
    assert len(ids) == len(set(ids)) == 4
    state = controller.queue.producer_state("dataset")
    controller.close()
    args = copy.copy(source_factory.args)
    args.rollout_queue_resume = True
    restored = RolloutQueueController(args)
    try:
        assert restored.queue.producer_state("dataset") == state
        assert restored.queue.outstanding_reads() == ()
        reclaimed = restored.take("recovered", 4)
        assert [a.task.task_id for a in reclaimed.assignments] == ids
        assert restored.heartbeat([first.assignments[0].lease]) == ["StaleAttempt"]
        assert restored.queue.producer_state("dataset") == state
    finally:
        restored.close()


@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_whole_job_recovery_replays_accepted_results_and_durable_continuations(
    source_factory,
):
    from dataclasses import asdict

    from slime.utils.rollout_transport import pack_rollout_group

    reader = source_factory("fully_async_0")
    accepted_group, partial_group = reader.get_samples(2)
    accepted = pack_rollout_group(accepted_group, reader.args, 0)
    partial_group[0].tokens = [1, 7, 9]
    partial_group[0].response_length = 2
    partial_group[0].status = Sample.Status.ABORTED
    partial_group[0].rollout_routed_experts = torch.arange(8).reshape(2, 2, 2)
    reader.add_samples([partial_group])
    task_id = partial_group[0]._queue_lease["task_id"]
    # Drop local state, leaving only durable task progress and accepted receipts.
    reader.close()
    controller = source_factory.controller
    controller.close()
    args = copy.copy(reader.args)
    args.rollout_queue_resume = True
    restored = RolloutQueueController(args)
    try:
        replay = restored.codec.load(restored.recover_pending_rollout())
        assert replay == [asdict(accepted.receipt)]
        assignment = restored.queue.acquire("new-reader", 1, task_ids=[task_id]).assignments[0]
        assert assignment.lease.coordinator_epoch != partial_group[0]._queue_lease["coordinator_epoch"]
        actual = restored.codec.load(assignment.task.input_ref)
        assert actual[0].tokens == [1, 7, 9]
        assert actual[0].response_length == 2
        assert actual[0].rollout_routed_experts.load().flatten().tolist() == list(range(8))
        assert len(restored.queue.commits) == 1
        # Once a batch exists, queue state alone cannot select model/optimizer state.
        plan = restored.codec.publish({"test": True}, submission_id="recovery-plan")
        restored.plan_batch("batch:test", [accepted.receipt.position], plan)
        with pytest.raises(RuntimeError, match="matching model/optimizer"):
            restored.recover_pending_rollout()
    finally:
        restored.close()


@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_dynamic_rejection_releases_spilled_tensor_staging(source_factory):
    from straw.tensor import publish_tensors

    from slime.utils.rollout_transport import discard_rollout_group, rollout_store

    reader = source_factory("filter")
    group = reader.get_samples(1)[0]
    source_factory.controller.store.seal()
    store, _, _ = rollout_store(reader.args)
    refs = publish_tensors(
        store,
        {
            "rollout_routed_experts": torch.arange(8),
            "rollout_topk_token_ids": torch.arange(4),
        },
        submission_id="discarded",
    )
    group[0].rollout_routed_experts, group[0].rollout_topk_token_ids = refs
    source_factory.controller.store.seal()
    discard_rollout_group(group, reader.args)
    assert source_factory.controller.queue.collect_garbage()["reclaimed_files"] >= 1


@pytest.mark.parametrize("missing", ["rollout_routed_experts", "rollout_topk_token_ids"])
def test_incomplete_server_abort_retries_last_valid_durable_prefix(source_factory, missing):
    reader = source_factory("partial")
    reader.args.use_rollout_routing_replay = True
    reader.args.use_score_centering = True
    group = reader.get_samples(1)[0]
    sample = group[0]
    sample.tokens, sample.response_length = [7, 8, 9], 2
    sample.status = Sample.Status.ABORTED
    sample.rollout_routed_experts = torch.arange(8).reshape(2, 2, 2)
    sample.rollout_topk_token_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    sample.rollout_topk_log_probs = torch.zeros(2, 2)
    reader.add_samples([group])
    group = reader.get_samples(1)[0]
    lease = group[0]._queue_lease
    last_input = source_factory.controller.status(lease["task_id"])["spec"]["input_ref"]
    setattr(group[0], missing, None)  # An abort response lacks part of its capture.
    reader.add_samples([group])
    assert reader.get_buffer_length() == 1
    status = source_factory.controller.status(lease["task_id"])
    assert status["state"] == "pending" and status["failures"] == 0
    assert status["spec"]["input_ref"] == last_input
    resumed = reader.get_samples(1)[0][0]
    assert resumed.tokens == [7, 8, 9] and resumed.response_length == 2
    assert resumed._queue_lease["attempt_id"] != lease["attempt_id"]
    assert resumed.rollout_routed_experts.load().flatten().tolist() == list(range(8))
    assert resumed.rollout_topk_token_ids.load().tolist() == [[1, 2], [3, 4]]


def test_generation_workers_cannot_claim_manager_collection_tasks(source_factory):
    from straw.protocol import TaskSpec

    controller = source_factory.controller
    task_id = "collection:owned-by-manager"
    controller.queue.submit_tasks("manager-submit", [TaskSpec(task_id, allow_empty=True)])
    groups = source_factory(0).get_samples(2)
    assert len(groups) == 2
    assert controller.status(task_id)["state"] == "pending"
    collection = controller.queue.acquire("manager", 1, task_ids=[task_id])
    assert collection.assignments[0].task.task_id == task_id


def test_group_reply_loss_returns_original_receipt_and_detects_changed_content(
    source_factory,
):
    from straw.errors import IdempotencyConflict

    from slime.utils.rollout_transport import pack_rollout_group

    reader = source_factory(0)
    group = reader.get_samples(1)[0]
    group[0].rollout_routed_experts = torch.arange(8).reshape(2, 2, 2)
    first = pack_rollout_group(group, reader.args, 0)
    retried = pack_rollout_group(group, reader.args, 0)
    assert retried == first
    assert retried.manifest == retried.receipt.result_ref
    group[0].rollout_routed_experts[0, 0, 0] += 1
    with pytest.raises(IdempotencyConflict):
        pack_rollout_group(group, reader.args, 0)
    assert source_factory.controller.queue.read_commits().cursor == 1


def test_collection_request_reuses_lease_after_response_loss(source_factory):
    controller = source_factory.controller
    lease = controller.begin_collection("rollout-1")
    assert controller.begin_collection("rollout-1") == lease
    ref = controller.codec.publish(
        [],
        submission_id="collection-test",
        metadata={"task_id": lease.task_id, "attempt_id": lease.attempt_id},
    )
    receipt = controller.complete(lease, ref)
    assert controller.begin_collection("rollout-1") == lease
    assert controller.complete(lease, ref) == receipt
    assert controller.queue.read_commits().cursor == 1


def test_failed_initial_refill_does_not_advance_dataset_cursor(source_factory, monkeypatch):
    controller = source_factory.controller

    def fail_publication(*args, **kwargs):
        raise OSError("injected input publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(controller.codec, "publish_many", fail_publication)
        with pytest.raises(OSError, match="input publication failure"):
            controller.take("reader", 1)
    assert controller.queue.producer_state("dataset") is None
    group = source_factory(0).get_samples(1)[0]
    assert [sample.index for sample in group] == [0, 1]
    order = list(range(7))
    random.Random(31).shuffle(order)
    assert group[0].prompt == f"prompt-{order[0]}"


def test_workers_read_locally_across_epochs_without_duplicate_ids(source_factory):
    first, second = source_factory(0), source_factory(1)
    groups = []
    for source, count in ((first, 4), (second, 9), (first, 12), (second, 3)):
        groups.extend(source.get_samples(count))
    indices = [sample.index for group in groups for sample in group]
    assert len(indices) == len(set(indices))
    for group in groups:
        position = group[0].group_index
        epoch, offset = divmod(position, 7)
        order = list(range(7))
        random.Random(31 + epoch).shuffle(order)
        assert [sample.prompt for sample in group] == [f"prompt-{order[offset]}"] * 2
        assert [sample.index for sample in group] == [2 * position, 2 * position + 1]


def test_returned_partial_is_available_to_another_reader(source_factory):
    source, other = source_factory(0), source_factory(1)
    group = source.get_samples(1)[0]
    group[0].tokens = [1, 2, 3]
    group[0].response_length = 2
    group[0].status = Sample.Status.ABORTED
    lease = group[0]._queue_lease
    source.add_samples([group])
    assert not hasattr(source, "buffer")
    assert not source._leases
    assert source.get_buffer_length() == 1
    restored = other.get_samples(1)[0]
    assert restored[0].index == group[0].index
    assert restored[0].tokens == [1, 2, 3]
    assert restored[0].status == Sample.Status.ABORTED
    assert restored[0]._queue_lease["worker_id"] == "1"
    assert restored[0]._queue_lease["attempt_id"] != lease["attempt_id"]
    assert source.get_buffer_length() == 0


def test_queue_orders_stage_then_staleness_then_fifo(source_factory):
    reader, other = source_factory("producer"), source_factory("consumer")
    groups = reader.get_samples(6)
    for group, version in zip(groups, [8, 2, 2, None, 9, 1], strict=True):
        for sample in group:
            sample.tokens, sample.response_length = [1, 2], 1
            sample.status = Sample.Status.ABORTED
            sample.weight_versions = [str(version)] if version is not None else []
    # Completed groups are deliverable immediately, even if their version is newer.
    for sample in groups[4]:
        sample.status, sample.reward = Sample.Status.COMPLETED, 1
    # A mixed-version prefix is as stale as its oldest generated segment.
    groups[5][0].weight_versions = ["1", "12"]
    reader.add_samples(groups)
    expected = [groups[i][0].index for i in (4, 5, 1, 2, 0, 3)]
    assert [group[0].index for group in other.get_samples(6)] == expected
    assert other.get_samples(1)[0][0].response_length == 0


@pytest.mark.parametrize(
    "path",
    [
        "custom.filter",
        "slime.rollout.data_source.pop_first",
        "slime.rollout.data_source.pop_oldest",
    ],
)
def test_straw_rejects_all_buffer_filter_paths(source_factory, path):
    source_factory.args.buffer_filter_path = path
    with pytest.raises(ValueError, match="--buffer-filter-path is not supported"):
        source_factory("unsupported")


def test_lost_return_reply_preserves_one_pending_task(source_factory):
    reader = source_factory("producer")
    group = reader.get_samples(1)[0]
    group[0].tokens, group[0].response_length = [1, 2], 1
    original = reader.controller.return_groups.remote

    def lose_reply(updates, deliveries):
        original(updates, deliveries)
        raise ConnectionError("reply lost")

    reader.controller.return_groups = SimpleNamespace(remote=lose_reply)
    with pytest.raises(ConnectionError, match="reply lost"):
        reader.add_samples([group])
    reader.close()
    other = source_factory("replacement")
    resumed = other.get_samples(1)[0]
    assert resumed[0].tokens == [1, 2]
    assert resumed[0].index == group[0].index
    assert other.get_buffer_length() == 0


def test_completed_return_uses_a_delivery_without_rewriting_accepted_history(
    source_factory,
):
    from slime.utils.rollout_transport import pack_rollout_group

    reader = source_factory(0)
    group = reader.get_samples(1)[0]
    for sample in group:
        sample.status, sample.reward = Sample.Status.COMPLETED, 1
    result = pack_rollout_group(group, reader.args, 0)
    reader.add_samples([group])
    returned = reader.get_samples(1)[0]
    assert [sample.index for sample in returned] == [sample.index for sample in group]
    assert all(sample._queue_source_positions == [result.receipt.position] for sample in returned)
    assert all(not hasattr(sample, "_queue_receipt") for sample in returned)
    assert source_factory.controller.status(result.receipt.task_id)["state"] == "completed"
    delivery = pack_rollout_group(returned, reader.args, 0)
    assert delivery.receipt.task_id != result.receipt.task_id
    assert source_factory.controller.queue.read_commits().cursor == 2
    source_factory.controller.args.rollout_queue_resume = True
    replay = source_factory.controller.codec.load(source_factory.controller.recover_pending_rollout())
    assert [receipt["position"] for receipt in replay] == [delivery.receipt.position]

    # Discarding the delivery also releases the predecessor's accepted capacity.
    from slime.utils.rollout_transport import discard_rollout_group, load_rollout_samples

    discard_rollout_group(load_rollout_samples([delivery])[0], reader.args, "test selection")
    assert source_factory.controller._training_state()["processed_cursor"] == 2


def test_older_pending_snapshot_restores_prefix_as_a_delivery(source_factory):
    from slime.utils.rollout_transport import pack_rollout_group

    reader = source_factory("owner")
    group = reader.get_samples(1)[0]
    for sample in group:
        sample.tokens, sample.response_length = [1, 2], 1
        sample.status = Sample.Status.ABORTED
    reader.add_samples([group])
    state = reader.state_dict()
    group = reader.get_samples(1)[0]
    for sample in group:
        sample.tokens.append(3)
        sample.status = Sample.Status.COMPLETED
    accepted = pack_rollout_group(group, reader.args, 0)
    reader.load_state_dict(state)
    restored = reader.get_samples(1)[0]
    assert all(sample.tokens == [1, 2] for sample in restored)
    assert all(sample._queue_source_positions == [accepted.receipt.position] for sample in restored)
    assert all(sample._queue_generation_start == 2 for sample in restored)
    assert all(sample._queue_lease["task_id"] != accepted.receipt.task_id for sample in restored)
    assert source_factory.controller.queue.read_commits().cursor == 1


def test_legacy_rollout_return_completes_borrowed_inputs(source_factory):
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.rollout_transport import accept_raw_rollout, load_rollout_samples

    reader = source_factory(0)
    groups = reader.get_samples(2)
    leases = [group[0]._queue_lease for group in groups]
    ref = accept_raw_rollout(RolloutFnTrainOutput(samples=groups), reader.args, 0)
    reader.close()
    assert all(source_factory.controller.status(lease["task_id"])["state"] == "completed" for lease in leases)
    restored = load_rollout_samples(ref)
    assert [[sample.index for sample in group] for group in restored] == [
        [0, 1],
        [2, 3],
    ]


def test_checkpoint_buffer_and_latest_continuation_survive_reader_release(
    source_factory,
):
    source = source_factory(0)
    used = source.get_samples(2)
    used[0][0].tokens = [7, 8]
    used[0][0].response_length = 1
    source.add_samples([used[0]])
    state = source.state_dict()
    source.close()
    restored = source_factory(0)
    restored.load_state_dict(state)
    group = restored.get_samples(1)[0]
    assert [s.index for s in group] == [s.index for s in used[0]]
    assert group[0].tokens == [7, 8]
    assert group[0]._queue_lease != used[0][0]._queue_lease
    # Closing returned the other unfinished task, without reserving/discarding a range.
    assert [s.index for s in restored.get_samples(1)[0]] == [s.index for s in used[1]]


@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_buffer_checkpoint_reuses_durable_groups_and_protects_them_from_gc(
    source_factory,
):
    reader = source_factory("compact-buffer")
    groups = reader.get_samples(32)
    for group in groups:
        for sample in group:
            sample.tokens = list(range(8192))
            sample.response_length = 8191
            sample.status = Sample.Status.ABORTED
    root = Path(reader.args.rollout_data_dir)

    def pack_bytes():
        return sum(path.stat().st_size for path in root.rglob("*.pack"))

    before = pack_bytes()
    reader.add_samples(groups)
    continuation_bytes = pack_bytes() - before
    before = pack_bytes()
    saved = reader.state_dict()
    assert pack_bytes() - before < continuation_bytes / 10
    checkpoint = saved.load()
    assert checkpoint["version"] == 3
    tasks = [task for task in checkpoint["pending"].load()["tasks"] if task["metadata"].get("returned")]
    assert len(tasks) == len(groups)
    pending = {spec.task_id: spec.input_ref for spec in source_factory.controller.queue.pending_tasks()}
    for group, spec in zip(groups, tasks, strict=True):
        task_id = group[0]._queue_lease["task_id"]
        assert spec["task_id"] == task_id
        assert spec["input_ref"].manifest == pending[task_id]
    source_factory.controller.reader_state("snapshot", saved.manifest)
    reader.close()
    source_factory.controller.store.seal()
    source_factory.controller.queue.collect_garbage()
    restored = source_factory("compact-buffer")
    restored.load_state_dict(saved)
    assert all(sample.tokens == list(range(8192)) for group in restored.get_samples(32) for sample in group)


def test_pending_checkpoint_reuses_inputs_without_reserving_leases(source_factory):
    reader = source_factory("lease-snapshot")
    group = reader.get_samples(1)[0]
    group[0].tokens, group[0].response_length = [1, 7, 9], 2
    reader.add_samples([group])
    saved = reader.state_dict()
    reader.close()
    restored = source_factory("lease-snapshot")
    restored.load_state_dict(saved)
    assert not restored._leases
    saved_again = restored.state_dict()
    assert saved_again.load()["pending"].load()["tasks"] == saved.load()["pending"].load()["tasks"]
    restored.load_state_dict(saved_again)
    returned = restored.get_samples(1)[0]
    assert returned[0].tokens == [1, 7, 9]
    assert returned[0]._queue_lease != group[0]._queue_lease


def test_legacy_reader_local_checkpoint_requires_explicit_migration(source_factory):
    from slime.utils.rollout_transport import pack_rollout_payload

    reader = source_factory("legacy-buffer")
    saved = pack_rollout_payload({"version": 1, "buffer": [], "metadata": {}}, reader.args, -1)
    with pytest.raises(ValueError, match="require migration"):
        reader.load_state_dict(saved)


def test_slow_task_acquire_does_not_block_async_generation(source_factory):
    entered, release = threading.Event(), threading.Event()

    def take(reader, count):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release task acquire")
        return source_factory.controller.take(reader, count)

    source = source_factory(0, take)

    async def exercise():
        task = asyncio.create_task(source.get_samples_async(1))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert not task.done()
        finally:
            release.set()
        assert len(await task) == 1

    asyncio.run(exercise())


def test_shuffle_does_not_reseed_user_random_state(source_factory):
    source = source_factory(0)
    state = random.getstate()
    source.get_samples(20)
    assert random.getstate() == state


@pytest.fixture
def scheduler_factory(monkeypatch):
    import ray

    from slime.rollout.filter_hub.base_types import DynamicFilterOutput
    from slime.rollout.fully_async_distributed import RolloutScheduler, _GenerationActor
    from slime.utils.async_utils import get_async_loop

    schedulers, actors, releases = [], [], []
    loop = get_async_loop().loop

    def remote(method):
        return SimpleNamespace(remote=lambda *args: asyncio.run_coroutine_threadsafe(method(*args), loop))

    def ray_wait(refs, num_returns, timeout):
        done, pending = wait(refs, timeout=timeout, return_when=FIRST_COMPLETED)
        return list(done)[:num_returns], list(pending)

    def kill(worker, no_restart=True):
        if worker.actor.producer is not None:
            loop.call_soon_threadsafe(worker.actor.producer.cancel)

    monkeypatch.setattr(ray, "wait", ray_wait)
    monkeypatch.setattr(ray, "get", lambda ref: ref.result())
    monkeypatch.setattr(ray, "cancel", lambda ref: ref.cancel())
    monkeypatch.setattr(ray, "kill", kill)

    def create(
        batch,
        capacities,
        *,
        slow_worker=None,
        reject=lambda index: False,
        crash=False,
        failed_workers=(),
        keep_when_insufficient=False,
    ):
        release = threading.Event()
        releases.append(release)
        submitted, workers = [], []
        args = SimpleNamespace(
            rollout_batch_size=batch,
            rollout_sample_filter_path=None,
            rollout_all_samples_process_path=None,
            rollout_data_transport="object-store",
        )

        def make_generator(worker):
            async def execute(rollout_id, group=None):
                index = len(submitted)
                submitted.append((worker, index))
                if worker == slow_worker:
                    while not release.is_set():
                        await asyncio.sleep(0.01)
                if crash:
                    raise RuntimeError("generation failed")
                if worker in failed_workers:
                    raise ray.exceptions.RayActorError("worker process exited")
                group = [Sample(index=index, group_index=index, metadata={"worker": worker})]
                return group, DynamicFilterOutput(
                    keep=not reject(index),
                    reason="test",
                    keep_when_insufficient=keep_when_insufficient,
                )

            return execute

        async def initial_groups(count):
            return [[None] for _ in range(count)]

        for i, capacity in enumerate(capacities):
            actor = _GenerationActor.__new__(_GenerationActor)
            actor.capacity = capacity
            actor.producer = None
            actor.running = False
            actor._generate_group = make_generator(i)
            actor.data_source = SimpleNamespace(get_samples_async=initial_groups)
            actors.append(actor)
            workers.append(
                SimpleNamespace(
                    start=remote(actor.start),
                    next=remote(actor.next),
                    pause=remote(actor.pause),
                    actor=actor,
                )
            )
        scheduler = RolloutScheduler(args, workers, capacities)
        schedulers.append(scheduler)
        return scheduler, submitted, release

    yield create
    for release in releases:
        release.set()
    for scheduler in schedulers:
        scheduler.close()

    async def cleanup():
        producers = [actor.producer for actor in actors if actor.producer is not None]
        for producer in producers:
            producer.cancel()
        await asyncio.gather(*producers, return_exceptions=True)

    asyncio.run_coroutine_threadsafe(cleanup(), loop).result(timeout=10)


def test_scheduler_replaces_rejected_groups_and_bounds_prefetch(scheduler_factory):
    scheduler, submitted, _ = scheduler_factory(17, [3, 3], reject=lambda i: i % 3 == 0)
    result = scheduler.generate(0, prefetch=6)
    assert len(result.samples) == 17
    assert all(group[0].index % 3 for group in result.samples)
    time.sleep(0.05)
    # Collector window + local slots + one queued output per worker.
    assert len(submitted) <= 17 + result.metrics["rollout/dynamic_filter/dropped_groups"] + 6 + 6 + 2
    scheduler.pause()
    assert not scheduler.pending and not scheduler.running_workers


def test_rejected_groups_are_released_while_filling_batch(scheduler_factory, monkeypatch):
    import ray

    samples = []

    def reject(index):
        if index == 512:
            assert sum(ref() is not None for ref in samples) < 8
        return index < 512

    scheduler, _, _ = scheduler_factory(1, [1], reject=reject)
    get = ray.get

    def record(ref):
        output = get(ref)
        if output is not None:
            samples.append(weakref.ref(output[0][0]))
        return output

    monkeypatch.setattr(ray, "get", record)
    result = scheduler.generate(0, prefetch=1)
    assert result.samples[0][0].index == 512
    assert result.metrics == {
        "rollout/dynamic_filter/drop_test": 512,
        "rollout/dynamic_filter/dropped_groups": 512,
        "rollout/dynamic_filter/dropped_ratio": 512 / 513,
    }


def test_distributed_rollout_rejects_all_samples_hook_before_starting_workers():
    from slime.rollout.fully_async_distributed import DistributedRollout

    args = SimpleNamespace(rollout_all_samples_process_path="test.all_samples")
    with pytest.raises(ValueError, match="rollout-all-samples-process-path.*not supported"):
        DistributedRollout(args, None)


def test_async_scheduler_fast_worker_fills_batch_without_waiting_for_slow_worker(
    scheduler_factory,
):
    scheduler, submitted, release = scheduler_factory(5, [1, 1], slow_worker=0, reject=lambda i: i > 5)
    with ThreadPoolExecutor(1) as consumer:
        future = consumer.submit(scheduler.generate, 0, prefetch=2)
        try:
            result = future.result(timeout=5)
            assert len(result.samples) == 5
            assert all(group[0].metadata["worker"] == 1 for group in result.samples)
            time.sleep(0.1)
            # Even rejected prefetches occupy the bounded completion queue.
            assert len(submitted) <= 11
            assert len(scheduler.pending) + len(scheduler.ready) <= 2
        finally:
            release.set()


@pytest.mark.parametrize("control", ["start", "pause"])
def test_slow_worker_control_does_not_block_collection_or_close(scheduler_factory, control):
    scheduler, _, _ = scheduler_factory(3, [1, 1])
    stalled = Future()
    entered = threading.Event()
    original = getattr(scheduler.workers[0], control).remote

    def stall(*args):
        entered.set()
        return stalled

    if control == "pause":
        scheduler.generate(0, prefetch=2)
    getattr(scheduler.workers[0], control).remote = stall
    with ThreadPoolExecutor(2) as callers:
        request = (
            callers.submit(scheduler.generate, 0, prefetch=2)
            if control == "start"
            else callers.submit(scheduler.pause)
        )
        closing = None
        try:
            assert entered.wait(5)
            if control == "start":
                result = request.result(timeout=2)
                assert all(group[0].metadata["worker"] == 1 for group in result.samples)
            closing = callers.submit(scheduler.close)
            closing.result(timeout=2)
            if control == "pause":
                with pytest.raises(RuntimeError, match="scheduler is closed"):
                    request.result(timeout=2)
        finally:
            # Also let the old, blocking implementation exit after test failure.
            if not stalled.done():
                stalled.set_result(None)
            getattr(scheduler.workers[0], control).remote = original
            scheduler.close()


def test_fully_async_filter_never_keeps_rejected_groups_to_fill_batch(
    scheduler_factory,
):
    scheduler, _, _ = scheduler_factory(12, [1], reject=lambda index: index < 16, keep_when_insufficient=True)
    result = scheduler.generate(0, prefetch=2)
    assert len(result.samples) == 12
    assert all(group[0].index >= 16 for group in result.samples)
    assert result.metrics["rollout/dynamic_filter/dropped_groups"] == 16


def test_scheduler_retires_failed_worker_and_survivor_fills_batches(scheduler_factory):
    scheduler, _, _ = scheduler_factory(12, [4, 4], failed_workers=(0,))
    first = scheduler.generate(0, prefetch=8)
    second = scheduler.generate(1, prefetch=scheduler.capacity)
    scheduler.pause()
    assert scheduler.capacity == 4
    assert scheduler.capacities == [0, 4]
    assert scheduler.error is None
    groups = first.samples + second.samples
    assert len(groups) == 24
    assert len({group[0].index for group in groups}) == 24
    assert all(group[0].metadata["worker"] == 1 for group in groups)


def test_scheduler_reports_all_workers_lost(scheduler_factory):
    scheduler, _, _ = scheduler_factory(12, [4, 4], failed_workers=(0, 1))
    with pytest.raises(RuntimeError, match="All rollout workers are unavailable"):
        scheduler.generate(0, prefetch=8)


def test_scheduler_surfaces_worker_failure(scheduler_factory):
    scheduler, _, _ = scheduler_factory(2, [1, 1], crash=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        scheduler.generate(0, prefetch=0)


def test_scheduler_close_unblocks_waiting_consumer(scheduler_factory):
    scheduler, _, release = scheduler_factory(1, [1], slow_worker=0)
    with ThreadPoolExecutor(1) as consumer:
        future = consumer.submit(scheduler.generate, 0, prefetch=1)
        try:
            deadline = time.monotonic() + 5
            with scheduler.condition:
                while not scheduler.pending:
                    assert time.monotonic() < deadline, "worker was not dispatched"
                    scheduler.condition.wait(timeout=0.01)
            scheduler.close()
            with pytest.raises(RuntimeError, match="scheduler is closed"):
                future.result(timeout=5)
            with pytest.raises(RuntimeError, match="scheduler is closed"):
                scheduler.generate(1, prefetch=1)
        finally:
            release.set()


def test_reader_adopts_checkpoint_tensors_from_another_pool(tmp_path, source_factory):
    from straw import SharedFilesystemStore
    from straw.tensor import publish_tensors

    with SharedFilesystemStore(tmp_path / "source", "source", codecs=("tensor.v1",)) as store:
        (reference,) = publish_tensors(store, {"routes": torch.tensor([1, 2])}, submission_id="buffer")
    path = Path(reference.path)
    reader = source_factory(0)
    group = reader.get_samples(1)[0]
    group[0].rollout_routed_experts = reference
    group[0].response_length = 1
    reader.add_samples([group])
    saved = reader.state_dict()
    path.unlink()
    checkpoint = saved.load()
    saved_group = unpack_rollout_payload(checkpoint["pending"].load()["tasks"][0]["input_ref"])
    assert saved_group[0].rollout_routed_experts.load().tolist() == [1, 2]
    transferred = reader.materialize_samples(reader.get_samples(1))
    assert transferred[0][0].rollout_routed_experts.tolist() == [1, 2]


@pytest.mark.parametrize("fanout", [False, True])
def test_distributed_completed_groups_keep_rewards_and_masks(monkeypatch, fanout):
    from slime.rollout import fully_async_distributed, sglang_rollout
    from slime.rollout.fully_async_distributed import _GenerationActor

    sample = Sample(
        index=1,
        rollout_id=1,
        tokens=[1, 2, 3],
        response="answer",
        response_length=2,
        reward=0.0,
        loss_mask=[1, 1],
        status=Sample.Status.COMPLETED,
    )
    group = [[sample]] if fanout else [sample]
    sample._queue_receipt = {"position": 7}

    async def get_samples(count):
        assert count == 1
        return [group]

    async def unexpected_generation(*args):
        raise AssertionError("completed groups must bypass generation and group reward")

    async def republish_buffered(*args, **kwargs):
        return SimpleNamespace(receipt=None)

    monkeypatch.setattr(sglang_rollout, "generate_and_rm_group", unexpected_generation)
    monkeypatch.setattr(fully_async_distributed, "publish_rollout_async", republish_buffered)
    worker = _GenerationActor.__new__(_GenerationActor)
    worker.running = True
    worker.args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        rollout_data_transport="object-store",
        _rollout_queue_controller=object(),
    )
    worker.dynamic_filter = None
    worker.data_source = SimpleNamespace(get_samples_async=get_samples, materialize_samples=lambda value: value)
    result, _ = asyncio.run(worker._generate_group(3))
    assert result is group
    assert sample.reward == 0.0
    assert sample.loss_mask == [1, 1]
    assert sample._queue_receipt == {"position": 7}


def test_worker_refills_locally_and_slow_group_does_not_block_results():
    from slime.rollout.fully_async_distributed import _GenerationActor

    worker = _GenerationActor.__new__(_GenerationActor)
    worker.capacity = 3
    worker.producer = None
    worker.running = False
    release = asyncio.Event()
    generated = []
    in_flight = 0
    maximum = 0

    async def generate_group(rollout_id, group=None):
        nonlocal in_flight, maximum
        index = len(generated)
        generated.append(index)
        in_flight += 1
        maximum = max(maximum, in_flight)
        try:
            if index == 0:
                await release.wait()
            else:
                await asyncio.sleep(0)
            return index
        finally:
            in_flight -= 1

    worker._generate_group = generate_group
    claims = []

    async def initial_groups(count):
        claims.append(count)
        return [[None] for _ in range(count)]

    worker.data_source = SimpleNamespace(get_samples_async=initial_groups)

    async def exercise():
        await worker._start(0)
        try:
            first = await asyncio.wait_for(worker._next(), timeout=5)
            second = await asyncio.wait_for(worker._next(), timeout=5)
            assert first > 0 and second > 0 and first != second
            assert claims == [worker.capacity]
            assert not release.is_set()
            # No manager-issued credits: consuming results lets slots pull more.
            received = [first, second]
            for _ in range(10):
                output = await asyncio.wait_for(worker._next(), timeout=5)
                assert output > 0
                received.append(output)
            await asyncio.sleep(0.02)
            count = len(generated)
            await asyncio.sleep(0.02)
            assert len(generated) == count  # Backpressure stops local refills.
            assert count <= 12 + worker.capacity + 1
            assert maximum <= worker.capacity
            await worker._pause()
            release.set()
            drained = []
            while (output := await asyncio.wait_for(worker._next(), timeout=5)) is not None:
                drained.append(output)
            assert 0 in drained
            assert worker.producer.done()
            assert len(generated) == count
            assert sorted(received + drained) == generated
        finally:
            worker.producer.cancel()
            await asyncio.gather(worker.producer, return_exceptions=True)

    asyncio.run(exercise())


def _skip_rollout_log(*args):
    return True


def _eval_locally(args, rollout_id, data_source, evaluation=False):
    assert evaluation
    sample = Sample(index=0, reward=float(args.rollout_batch_size), tokens=[1, 2], response_length=1)
    return {"test": {"rewards": [sample.reward], "samples": [sample]}}


async def _generate_locally(args, sample, sampling_params):
    while Path(args.test_generation_gate).exists():
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.01 * (sample.group_index % 3))
    sample.tokens = [sample.index + 1, 2, 3]
    sample.response = "answer"
    sample.response_length = 2
    sample.loss_mask = [1, 1]
    sample.rollout_log_probs = [-0.5, -0.7]
    for key, tensor in (
        ("rollout_topk_token_ids", torch.tensor([[2, 4], [3, 5]], dtype=torch.int32)),
        (
            "rollout_topk_log_probs",
            torch.tensor([[-0.5, -2.0], [-0.7, -2.5]], dtype=torch.float32),
        ),
    ):
        setattr(sample, key, tensor)
    sample.status = Sample.Status.COMPLETED
    if sample.index % 2 == 0:
        sample.multimodal_train_inputs = {"pixel_values": torch.ones(1, 3, 2, 2)}
    sample.rollout_routed_experts = torch.full((2, 1, 1), sample.index % 8, dtype=torch.uint8)
    if args.test_fanout:
        # A real custom hook can construct new Samples, without private queue fields.
        children = [
            Sample(**{field.name: copy.deepcopy(getattr(sample, field.name)) for field in fields(Sample)})
            for _ in range(2)
        ]
        for offset, child in enumerate(children):
            child.rollout_id = sample.index
            child.index = 2 * sample.index + offset
            child.tokens[0] = child.index + 1
        return children
    return sample


async def _score_fanout(args, samples):
    for sample in samples:
        assert "reward_calls" not in sample.metadata
        sample.metadata["reward_calls"] = 1
    return [float(sample.index % 2) for sample in samples]


async def _score_group(args, samples):
    assert len(samples) == args.n_samples_per_prompt
    assert len({sample.group_index for sample in samples}) == 1
    for sample in samples:
        assert "reward_calls" not in sample.metadata, "completed groups must not be scored twice"
        sample.metadata["reward_calls"] = 1
    return [float(sample.index % 2) for sample in samples]


def _filter_complete_group(args, group):
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    assert len(group) == args.n_samples_per_prompt
    samples = list(iter_samples(group))
    assert len({sample.group_index for sample in samples}) == 1
    assert all(sample.reward is not None and sample.status == Sample.Status.COMPLETED for sample in samples)
    return DynamicFilterOutput(keep=True)


def _read_training_locally(refs, rank):
    import ray

    data = process_rollout_data(refs, rank, 2)
    for key in (
        "rollout_routed_experts",
        "rollout_topk_token_ids",
        "rollout_topk_log_probs",
    ):
        for ref in data[key]:
            value = ref.load() if isinstance(ref, TensorRef) else ref
            assert value.shape[0] == 2
            assert torch.isfinite(value).all()
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "samples": len(data["tokens"]),
    }


def _load_queue_checkpoint(args, rollout_id):
    from straw.protocol import RecordSetRef

    from slime.utils.rollout_transport import DiskPayloadRef

    path = Path(args.save) / "rollout" / f"queue_state_{rollout_id}.json"
    index = json.loads(path.read_text())
    return DiskPayloadRef(RecordSetRef.from_dict(index["manifest"]), index["root"]).load()


def _rollout_args(tmp_path, *, fanout=False, transport="straw"):
    """Bounded local dataset, tokenizer and R3/SC configuration shared by Ray tests."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer_dir = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    ).save_pretrained(tokenizer_dir)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps({"text": f"hello {i}"}) for i in range(7)))
    return SimpleNamespace(
        rollout_batch_size=4,
        data_source_path="slime.rollout.queue_data_source.QueueDataSource",
        debug_train_only=False,
        test_fanout=fanout,
        test_generation_gate=str(tmp_path / "generation-gate"),
        hf_checkpoint=str(tokenizer_dir),
        prompt_data=str(dataset),
        input_key="text",
        rollout_max_prompt_len=None,
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=31,
        buffer_filter_path=None,
        buffer_sort_by_staleness=False,
        rollout_shuffle=True,
        n_samples_per_prompt=2,
        rollout_num_engines=1,
        sglang_server_concurrency=24,
        over_sampling_batch_size=4,
        eval_function_path="test_distributed_rollout._eval_locally",
        custom_rollout_log_function_path="test_distributed_rollout._skip_rollout_log",
        custom_eval_rollout_log_function_path="test_distributed_rollout._skip_rollout_log",
        log_passrate=False,
        wandb_always_use_train_step=False,
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        use_score_centering=True,
        score_centering_top_k=2,
        rollout_top_p=1.0,
        use_rollout_routing_replay=True,
        rollout_data_transport=transport,
        rollout_data_dir=str(tmp_path / "queue"),
        use_distributed_post=False,
        debug_rollout_only=False,
        save_debug_rollout_data=str(tmp_path / "debug" / "rollout_{rollout_id}.pt"),
        ci_test=False,
        load_debug_rollout_data=None,
        save=str(tmp_path / "checkpoint"),
        load=None,
        dump_details=None,
        global_batch_size=8,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        balance_data=False,
        balance_by_flops=False,
        num_experts=8,
        num_layers=1,
        moe_router_topk=1,
        use_wandb=False,
        use_tensorboard=False,
        rollout_function_path="slime.rollout.fully_async_rollout.generate_rollout_fully_async",
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        custom_generate_function_path="test_distributed_rollout._generate_locally",
        custom_rm_path=(
            "test_distributed_rollout._score_fanout" if fanout else "test_distributed_rollout._score_group"
        ),
        dynamic_sampling_filter_path="test_distributed_rollout._filter_complete_group",
        sglang_dp_size=1,
        group_rm=not fanout,
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        rollout_temperature=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=2,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rollout_sample_hook_path=None,
    )


@pytest.mark.parametrize(
    "fanout,transport",
    [(False, "straw"), (True, "straw")],
)
def test_two_ray_nodes_generate_transfer_and_restore(tmp_path, fanout, transport):
    multiplier = 2 if fanout else 1
    import ray
    from ray.cluster_utils import Cluster

    from slime.ray.rollout import RolloutManager

    args = _rollout_args(tmp_path, fanout=fanout, transport=transport)
    config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    node_count = 2
    cluster = Cluster()
    source = None
    try:
        for _ in range(node_count):
            cluster.add_node(
                num_cpus=2,
                num_gpus=0,
                object_store_memory=128 * 1024**2,
                include_dashboard=False,
            )
        ray.init(
            address=cluster.address,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join(
                        [
                            str(Path(__file__).parent),
                            str(Path(__file__).parent.parent),
                            os.environ.get("PYTHONPATH", ""),
                        ]
                    ),
                }
            },
        )
        from slime.utils.misc import load_function
        from slime.utils.rollout_transport import check_rollout_storage

        if transport == "straw":
            check_rollout_storage(args)
        source = load_function(args.data_source_path)(args)
        assert not source.consumers
        returned = source.get_samples(1)[0]
        if transport == "straw":
            for sample in returned:
                sample.tokens = [1, 2, 3]
                sample.response_length = 2
                sample.loss_mask = [1, 1]
                sample.status = Sample.Status.ABORTED
                sample.rollout_routed_experts = torch.tensor([[[1]], [[2]]], dtype=torch.int32)
                sample.rollout_topk_token_ids = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32).numpy()
                sample.rollout_topk_log_probs = torch.tensor([[-0.5, -2], [-0.7, -2.5]]).numpy()
        source.add_samples([returned])

        cls = RolloutManager.__ray_metadata__.modified_class
        manager = cls.__new__(cls)
        manager.args = args
        from slime.utils.misc import load_function

        rollout_function = load_function(args.rollout_function_path)
        from slime.rollout.batch_builder import BatchBuilder

        manager.batch_builder = BatchBuilder(args)
        manager.set_train_parallel_config(config)
        manager.health_monitoring_resume = lambda: None
        manager._get_updatable_server = lambda: None
        calls = []

        def custom_rollout(global_args, rollout_id, data_source, evaluation=False):
            assert global_args is args
            assert global_args.rollout_batch_size == 4
            assert data_source is source
            assert not evaluation
            calls.append(("rollout", 4))
            output = rollout_function(global_args, rollout_id, data_source, evaluation=evaluation)
            if args.rollout_data_transport == "straw":
                from slime.utils.rollout_transport import DiskPayloadRef

                assert isinstance(output.samples, DiskPayloadRef)
            return output

        manager.generate_rollout = custom_rollout
        manager.eval_generate_rollout = load_function(args.eval_function_path)

        def postprocess(global_args, samples):
            assert global_args.rollout_batch_size == 4
            assert len(samples) == 8 * multiplier
            assert len({sample.group_index for sample in samples}) == 4
            assert sum(sample.multimodal_train_inputs is not None for sample in samples) == 4 * multiplier
            calls.append(("rewards", len(samples)))
            raw = [sample.reward for sample in samples]
            return raw, [value - sum(raw) / len(raw) for value in raw]

        def convert(global_args, samples):
            assert global_args is args
            assert len(samples) == 8 * multiplier
            calls.append(("convert", len(samples)))
            manager.batch_builder.custom_convert_samples_to_train_data_func = None
            try:
                return manager.batch_builder.convert(samples)
            finally:
                manager.batch_builder.custom_convert_samples_to_train_data_func = convert

        manager.batch_builder.custom_reward_post_process_func = postprocess
        manager.batch_builder.custom_convert_samples_to_train_data_func = convert

        def fetch(rollout_id):
            manager.data_source = source
            previous = len(calls)
            refs = manager.generate(rollout_id)
            assert calls[previous:] == [
                ("rollout", 4),
                ("convert", 8 * multiplier),
                ("rewards", 8 * multiplier),
            ]
            dump = torch.load(tmp_path / "debug" / f"rollout_{rollout_id}.pt", weights_only=False)
            assert len(dump["samples"]) == 8 * multiplier
            assert all(
                isinstance(sample["rollout_routed_experts"], (torch.Tensor, TensorRef)) for sample in dump["samples"]
            )
            assert not list((tmp_path / "debug").glob("worker_*"))
            if rollout_id == 0:
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

                remote_read = ray.remote(_read_training_locally)
                nodes = [node for node in ray.nodes() if node["Alive"]]
                assert len(nodes) == node_count
                reports = ray.get(
                    [
                        remote_read.options(
                            num_cpus=0,
                            scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
                        ).remote(refs, i % 2)
                        for i, node in enumerate(nodes)
                    ]
                )
                assert {report["node_id"] for report in reports} == {node["NodeID"] for node in nodes}
                assert all(report["samples"] == 4 * multiplier for report in reports)
            batches = [process_rollout_data(refs, rank, 2) for rank in range(2)]
            rows = {}
            multimodal_samples = 0
            for batch in batches:
                assert isinstance(batch["rollout_mask_sums"], torch.Tensor)
                assert batch["rollout_mask_sums"].tolist() == [2.0 * multiplier] * (4 * multiplier)
                for i, index in enumerate(batch["sample_indices"]):
                    if batch["multimodal_train_inputs"][i] is not None:
                        multimodal_samples += 1
                        assert batch["multimodal_train_inputs"][i]["pixel_values"].shape == (1, 3, 2, 2)
                    assert index not in rows
                    assert isinstance(
                        batch["rollout_routed_experts"][i],
                        (torch.Tensor, TensorRef),
                    )
                    rows[index] = batch["tokens"][i].tolist()
                    assert rows[index][0] == index + 1
            assert len(rows) == 8 * multiplier
            assert multimodal_samples == 4 * multiplier
            # Mock trainer completion after every rank has read the same batch.
            # This tests runtime capacity only, not model/optimizer checkpointing.
            manager.training_completed(rollout_id)
            if transport == "straw":
                assert ray.get(source.controller.metrics.remote())["usage_with_reservations"]["ready_bytes"] == 0
            return rows

        first = fetch(0)
        assert source.get_buffer_length() == 0  # Workers claimed the owner's returned group.
        runtime = source.consumers["fully_async"]
        assert len(runtime.workers) == node_count
        manager.save(0)
        state = _load_queue_checkpoint(args, 0)["consumers"]["fully_async"]
        assert len(state["scheduler"]["ready"]) >= 4
        assert all(
            sample.metadata["reward_calls"] == 1
            for group, _ in state["scheduler"]["ready"]
            for sample in iter_samples(unpack_rollout_payload(group))
        )
        # Claim priority does not impose completion order across workers. After
        # checkpointing drains them, the returned group must be in the first
        # batch or the saved ready results, but need not be in the second batch.
        returned_indices = {multiplier * sample.index + offset for sample in returned for offset in range(multiplier)}
        saved_indices = {
            sample.index
            for group, _ in state["scheduler"]["ready"]
            for sample in iter_samples(unpack_rollout_payload(group))
        }
        assert returned_indices <= (first.keys() | saved_indices)
        expected = fetch(1)
        assert not first.keys() & expected.keys()
        source.close()
        source = None
        args.load = args.save
        args.rollout_queue_resume = True
        source = QueueDataSource(args)
        source.data_config["n_samples_per_prompt"] += 1
        with pytest.raises(ValueError, match="n_samples_per_prompt"):
            source.load(0)
        source.data_config["n_samples_per_prompt"] -= 1
        source.load(0)
        if transport == "straw":
            manager.batch_builder.load(0)
        assert not source.consumers  # Execution state is restored when fully async starts.
        source.save(0)  # Saving before the first generate must preserve the restored queue.
        actual = fetch(1)
        assert actual == expected
        restored_group = source.get_samples(1)[0]
        assert not {sample.index for sample in restored_group} & (first.keys() | actual.keys())
        manager.eval(1)
        evaluation = torch.load(tmp_path / "debug" / "rollout_eval_1.pt", weights_only=False)
        assert [sample["reward"] for sample in evaluation["samples"]] == [4.0]  # Eval keeps the original global quota.

        # Lose an actor (or a whole non-manager Ray node) while both have work.
        runtime = source.consumers["fully_async"]
        runtime.pause()
        runtime.ready.clear()
        gate = Path(args.test_generation_gate)
        gate.touch()
        runtime.resume()
        with ThreadPoolExecutor(1) as consumer:
            future = consumer.submit(fetch, 2)
            try:
                deadline = time.monotonic() + 10
                with runtime.condition:
                    while len(runtime.pending) != node_count:
                        assert time.monotonic() < deadline, "all workers should receive work"
                        runtime.condition.wait(timeout=0.01)
                failed = 0
                if fanout:
                    node = next(node for node in cluster.list_all_nodes() if node != cluster.head_node)
                    nodes = sorted(
                        ray.nodes(),
                        key=lambda node: (node["NodeManagerAddress"], node["NodeID"]),
                    )
                    failed = next(i for i, item in enumerate(nodes) if item["NodeID"] == node.node_id)
                    cluster.remove_node(node, allow_graceful=False)
                else:
                    ray.kill(runtime.workers[failed], no_restart=True)
                deadline = time.monotonic() + 30
                with runtime.condition:
                    while runtime.capacities[failed] and runtime.error is None:
                        assert time.monotonic() < deadline, "failed worker was not retired"
                        runtime.condition.wait(timeout=0.01)
                    assert runtime.error is None
            finally:
                gate.unlink()
            after_failure = future.result(timeout=30)
        runtime.pause()
        assert runtime.capacities[failed] == 0
        assert runtime.capacity == 12 - 12 // node_count
        assert runtime.error is None
        assert not after_failure.keys() & actual.keys()
        source.save(2)
        saved = _load_queue_checkpoint(args, 2)
        assert saved["consumers"]["fully_async"]["readers"][failed] is None
        assert len(fetch(3)) == 8 * multiplier

        if not fanout:
            # A restart with the same topology preserves retired worker slots.
            source.close()
            source = QueueDataSource(args)
            source.load(2)
            assert len(fetch(3)) == 8
            runtime = source.consumers["fully_async"]
            assert runtime.capacities[failed] == 0
            assert runtime.capacity == 12 - 12 // node_count
            runtime.pause()
            runtime.ready.clear()
            for i, worker in enumerate(runtime.workers):
                if runtime.capacities[i]:
                    ray.kill(worker, no_restart=True)
            runtime.resume()
            # Killed workers can still have committed results in the journal.
            # Wait for death notifications before checking exhausted capacity;
            # an earlier generate may correctly deliver a recovered warm batch.
            with runtime.condition:
                deadline = time.monotonic() + 30
                while runtime.error is None and time.monotonic() < deadline:
                    runtime.condition.wait(timeout=0.1)
                assert runtime.capacity == 0
            with pytest.raises(RuntimeError, match="All rollout workers are unavailable"):
                runtime.generate(4, prefetch=runtime.capacity)
    finally:
        if source is not None:
            source.close()
        ray.shutdown()
        cluster.shutdown()


def test_filtered_accepts_and_lost_delivery_are_accounted(source_factory, monkeypatch):
    from slime.rollout.base_types import finalize_rollout_groups
    from slime.utils import misc
    from slime.utils.rollout_transport import DiskPayloadRef, pack_rollout_group, pack_rollout_payload

    source = source_factory("worker")
    controller = source_factory.controller
    groups = source.get_samples(4)
    refs = [pack_rollout_group(group, source.args, 0) for group in groups[:3]]
    source.args.rollout_sample_filter_path = "test.filter"
    monkeypatch.setattr(misc, "load_function", lambda _: lambda args, groups: groups.pop(0))
    output = finalize_rollout_groups(source.args, 0, refs[:2])
    assert len(output.samples.load()) == 1
    assert controller.training_state()["processed_cursor"] == 1
    # One reply arrived; another was accepted but the process died before replying.
    delivered = pack_rollout_payload([refs[1].receipt.position], source.args, 0)
    recovered = controller.recover_reader_results("worker", delivered.manifest)
    receipts = DiskPayloadRef(recovered, source.args.rollout_data_dir).load()
    assert [r["position"] for r in receipts] == [refs[2].receipt.position]
    assert controller.status(groups[3][0]._queue_lease["task_id"])["state"] == "pending"
    from straw.errors import StaleAttempt

    with pytest.raises(StaleAttempt):
        pack_rollout_group(groups[3], source.args, 0)
    assert len(controller.queue.read_commits().commits) == 3
    from slime.utils.rollout_transport import discard_rollout_group, load_rollout_samples

    # A previously accepted warm group can fail a later dynamic filter too.
    discard_rollout_group(load_rollout_samples([refs[2]])[0], source.args, "warm group rejected")
    assert controller.queue._usage()["accepted_unprocessed"]["records"] == 1
    assert len(controller.queue.read_commits().commits) == 3


def test_fresh_samples_inherit_authorization_and_reject_conflicting_hooks(
    source_factory,
):
    from slime.utils.rollout_transport import inherit_queue_context, pack_rollout_group

    source = source_factory("worker")
    group = source.get_samples(1)[0]
    children = [[Sample(index=2 * parent.index + i, tokens=[1, 2]) for i in range(2)] for parent in group]
    for parent, output in zip(group, children, strict=True):
        parent.queue_generation_requests = [{"sampling_params": {"temperature": 0.7}}]
        inherit_queue_context(parent, output)
    ref = pack_rollout_group(children, source.args, 0)
    assert ref.receipt is not None
    assert all(
        s.queue_generation_requests[0]["sampling_params"]["temperature"] == 0.7 for s in iter_samples(ref.load())
    )
    children[0][0]._queue_lease = {"task_id": "another-task"}
    with pytest.raises(ValueError, match="different queue task authorization"):
        inherit_queue_context(group[0], children)


def _restore_source_fixture(controller, payload):
    store, codec = controller.store, controller.codec
    source = codec.publish({"old_buffer": payload}, submission_id="old-source")
    state = codec.publish(
        dict(
            version=1,
            fetch_cursor=0,
            processed_cursor=0,
            processed_positions=[],
            batches=[],
            finished_batches=[],
        ),
        submission_id="old-training-state",
    )
    store.retain("checkpoint:old", [source, state])
    store.release_publications([source, state])
    store.seal()
    controller.restore_training_state(state, 0, 0, retained_ref=source)
    return source, state


@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_restore_source_handoff_releases_obsolete_snapshot_graphs(source_factory):
    from dataclasses import asdict

    from slime.utils.rollout_transport import DiskPayloadRef

    controller = source_factory.controller
    store, codec = controller.store, controller.codec
    old = codec.publish({"tensor": torch.arange(4096)}, submission_id="obsolete")
    old_pack = store.backend.root / old.manifest.segment.path
    store.seal()
    source, saved_state = _restore_source_fixture(controller, DiskPayloadRef(old, str(store.backend.root)))
    restored = controller._training_state()
    decision = restored["disposition"].load()
    assert decision["checkpoint_state"] == asdict(saved_state)
    assert decision["retained_source"] == asdict(source)
    assert restored["restored_source"].manifest == source
    # Replacing the latest filter audit must not drop recovery ownership.
    filtered = codec.publish({"reason": "filter", "positions": []}, submission_id="filter")
    controller.record_dispositions(filtered)
    assert controller._training_state()["restored_source"].manifest == source
    for index in range(2):
        batch = f"completed-{index}"
        plan = codec.publish({"batch_id": batch}, submission_id=f"plan-{index}")
        controller.plan_batch(batch, [], plan)
        ready = codec.publish({"batch_id": batch}, submission_id=f"ready-{index}")
        controller.ready_batch(batch, ready)
        controller.finish_batch(batch)
    replacement = codec.publish({"buffer": []}, submission_id="empty-source")
    store.retain("checkpoint:new", [replacement])
    store.release_publications([replacement])
    controller.handoff_restored_source(replacement)
    store.seal()
    controller._collect_storage()
    assert old_pack.exists()  # The original checkpoint has not been retired.
    store.release("checkpoint:old")
    result = controller._collect_storage()
    assert result["reclaimed_files"] > 0
    assert not old_pack.exists()
    state = controller._training_state()
    assert state["restored_source"].manifest == replacement
    assert state["batches"] == []
    assert state["disposition"].load() == {"reason": "filter", "positions": []}


@pytest.mark.parametrize("failure", [None, "retain", "report", "persist"])
@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_restore_source_handoff_preserves_lazy_consumers_and_failed_save(
    source_factory, monkeypatch, tmp_path, failure
):
    import json
    from dataclasses import asdict

    import straw.reporting

    from slime.rollout.queue_data_source import QueueDataSource
    from slime.utils.rollout_transport import DiskPayloadRef

    controller = source_factory.controller
    store, codec = controller.store, controller.codec
    source = source_factory("owner")
    source.__class__ = QueueDataSource
    source._owns_controller = False
    source.data_config = {"test": True}
    source.args.save = str(tmp_path / "checkpoints")
    source.consumers = {}
    prefix = codec.publish({"r3": torch.arange(4096)}, submission_id="lazy-prefix")
    prefix_pack = store.backend.root / prefix.manifest.segment.path
    store.seal()
    source._restored_consumers = {"not_started": {"prefix": DiskPayloadRef(prefix, str(store.backend.root))}}
    original, _ = _restore_source_fixture(controller, source._restored_consumers)
    source.args._queue_restored_source_ref = original
    before = controller.training_state()
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("injected save failure")

        if failure == "retain":
            patch.setattr(store, "retain", fail)
        elif failure == "report":
            patch.setattr(straw.reporting, "write_report", fail)
        elif failure == "persist":
            patch.setattr(controller._queue, "save_consumer_state", fail)
        if failure:
            with pytest.raises(RuntimeError, match="injected save failure"):
                source.save(0)
            assert controller.training_state() == before
            store.seal()
            controller._collect_storage()
            assert prefix_pack.exists()
    source.save(1)
    saved = json.loads((Path(source.args.save) / "rollout/queue_state_1.json").read_text())
    assert asdict(controller._training_state()["restored_source"].manifest) == saved["manifest"]
    store.release("checkpoint:old")
    store.seal()
    controller._collect_storage()
    current = controller._training_state()["restored_source"].load()
    assert current["consumers"].keys() == {"not_started"}
    actual = current["consumers"]["not_started"]["prefix"].load()["r3"]
    torch.testing.assert_close(actual, torch.arange(4096), rtol=0, atol=0)
    # Once the lazy consumer has consumed its buffer, a subsequent cut drops it.
    source._restored_consumers = {}
    source.save(2)
    store.release(saved["storage_owner"])
    if failure in {"persist", None}:
        # A failed prepared snapshot remains protected until explicitly retired.
        store.seal()
        controller._collect_storage()
        assert prefix_pack.exists() == (failure == "persist")


@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_restore_source_handoff_preserves_prefix_before_later_accepted_result(
    source_factory,
):
    from slime.rollout.queue_data_source import QueueDataSource
    from slime.utils.rollout_transport import pack_rollout_group

    controller = source_factory.controller
    reader = source_factory("old-reader")
    group = reader.get_samples(1)[0]
    for sample in group:
        sample.tokens = [7, 8, 9]
        sample.response_length = 2
        sample.status = Sample.Status.ABORTED
        sample.rollout_routed_experts = torch.arange(8).reshape(2, 2, 2)
    reader.add_samples([group])
    prefix = reader.state_dict()
    original, _ = _restore_source_fixture(controller, prefix)
    later = reader.get_samples(1)[0]
    for sample in later:
        sample.tokens = [7, 8, 9, 10]
        sample.response_length = 3
        sample.rollout_routed_experts = torch.arange(12).reshape(3, 2, 2)
    accepted = pack_rollout_group(later, reader.args, 0)
    restored = source_factory("owner")
    restored.__class__ = QueueDataSource
    restored._owns_controller = False
    restored.data_config = {"test": True}
    restored.consumers, restored._restored_consumers = {}, {}
    restored.args.save = str(Path(reader.args.rollout_data_dir) / "saved")
    restored.args._queue_restored_source_ref = original
    restored.load_state_dict(prefix)
    pending = controller.queue.pending_tasks()[0]
    assert pending.metadata["source_positions"] == [accepted.receipt.position]
    assert not restored._leases
    restored.save(0)
    controller.store.release("checkpoint:old")
    controller.store.seal()
    controller._collect_storage()
    current = controller._training_state()["restored_source"].load()
    checkpoint = current["reader"].load()["pending"].load()["tasks"][0]["input_ref"].load()
    assert [sample.tokens for sample in checkpoint] == [[7, 8, 9]] * 2
    for sample in checkpoint:
        torch.testing.assert_close(
            sample.rollout_routed_experts.load(),
            torch.arange(8).reshape(2, 2, 2),
            rtol=0,
            atol=0,
        )


def test_rebuffer_traffic_is_linear_and_continuations_survive_worker_loss(
    source_factory,
):
    from slime.utils.rollout_transport import pack_rollout_payload

    reader = source_factory("buffered")
    groups = reader.get_samples(64)
    for group in groups:
        for sample in group:
            sample.tokens = [1, 2, 3]
            sample.response_length = 2
            sample.status = Sample.Status.ABORTED
    root = Path(reader.args.rollout_data_dir)

    def size():
        return sum(p.stat().st_size for p in root.rglob("*.pack"))

    before = size()
    for group in groups[:32]:
        reader.add_samples([group])
    halfway = size() - before
    for group in groups[32:]:
        reader.add_samples([group])
    assert size() - before < halfway * 2.5
    controller = source_factory.controller
    delivered = pack_rollout_payload([], reader.args, -1)
    controller.recover_reader_results("buffered", delivered.manifest)
    replacement = source_factory("replacement")
    resumed = replacement.get_samples(64)
    assert all(sample.tokens == [1, 2, 3] for group in resumed for sample in group)


def test_cancelled_rebuffer_waiter_does_not_drop_its_continuation():
    from slime.rollout.fully_async_distributed import _GenerationActor

    actor = _GenerationActor.__new__(_GenerationActor)
    persisted = []
    actor.data_source = SimpleNamespace(add_samples=lambda groups: persisted.append(groups))

    async def exercise():
        waiters = [asyncio.create_task(actor._rebuffer(i)) for i in range(130)]
        await asyncio.sleep(0.001)
        waiters[0].cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        assert sorted(group for batch in persisted for group in batch) == list(range(130))
        assert all(len(batch) <= 64 for batch in persisted)
        assert len(persisted) == 3

    asyncio.run(exercise())


def test_reader_retries_when_another_reader_takes_the_just_refilled_batch(
    source_factory,
):
    from straw.protocol import AcquireResult

    calls = []
    controller = source_factory.controller

    def contested_take(reader_id, count):
        calls.append(count)
        if len(calls) == 1:
            return AcquireResult("empty")
        return controller.take(reader_id, count)

    reader = source_factory("contended", take=contested_take)
    groups = reader.get_samples(64)
    assert len(groups) == 64
    assert len({group[0].group_index for group in groups}) == 64
    assert calls[:2] == [64, 64]


@pytest.mark.parametrize("timeout", [False, True])
def test_shutdown_allows_large_buffers_and_propagates_deadline_failure(monkeypatch, timeout):
    import ray

    from slime.rollout.fully_async_distributed import DistributedRollout, RolloutScheduler

    scheduler = DistributedRollout.__new__(DistributedRollout)
    scheduler.workers = [SimpleNamespace(close=SimpleNamespace(remote=lambda i=i: i)) for i in range(2)]
    scheduler.capacities = [4096, 4096]
    monkeypatch.setattr(RolloutScheduler, "close", lambda self: None)
    clock = iter([100, 100, 140])
    monkeypatch.setattr(
        "slime.rollout.fully_async_distributed.time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )
    waits, killed = [], []

    def get(ref, *, timeout):
        waits.append(timeout)
        if fail:
            raise ray.exceptions.GetTimeoutError("unfinished durable writes")
        assert timeout > 40  # A valid large flush exceeds the old 30-second limit.

    fail = timeout
    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))
    if timeout:
        with pytest.raises(ray.exceptions.GetTimeoutError):
            scheduler.close()
    else:
        scheduler.close()
        assert waits == [300, 260]
    assert killed == scheduler.workers


@pytest.mark.parametrize("source_factory", [True], indirect=True)
def test_online_gc_waits_for_training_completion_and_checkpoint_release(source_factory):
    from straw import SharedFilesystemStore
    from straw.protocol import RecordSetRef
    from straw.tensor import publish_tensors

    from slime.rollout.queue_codec import CODECS, SampleCodec
    from slime.utils.rollout_transport import DiskPayloadRef

    controller = source_factory.controller
    assignment = controller.take("worker", 1).assignments[0]
    lease = assignment.lease
    writer = SharedFilesystemStore(controller.store.backend.root, controller.store.run_id, codecs=CODECS)
    tensor = publish_tensors(writer, {"routes": torch.arange(1024)}, submission_id="routes")[0]
    raw = SampleCodec(writer).publish(
        {"routes": tensor},
        submission_id="result",
        metadata={"task_id": lease.task_id, "attempt_id": lease.attempt_id},
    )
    receipt = controller.complete(lease, raw)
    writer.seal()
    plan = controller.codec.publish({"raw": DiskPayloadRef(raw, str(writer.backend.root))}, submission_id="plan")
    controller.plan_batch("batch", [receipt.position], plan)
    ready = controller.codec.publish(
        {"batch_id": "batch", "raw": DiskPayloadRef(raw, str(writer.backend.root))},
        submission_id="ready",
    )
    controller.ready_batch("batch", ready)
    controller._collect_storage()
    assert torch.equal(tensor.load(), torch.arange(1024))
    state = RecordSetRef.from_dict(controller.training_state()["state_ref"])
    controller.store.retain("checkpoint:test", [state])
    controller.finish_batch("batch")
    assert controller._training_state()["batches"] == []
    assert torch.equal(tensor.load(), torch.arange(1024))
    controller.store.release("checkpoint:test")
    report = controller._collect_storage()
    assert report["reclaimed_files"] >= 1
    assert not Path(tensor.path).exists()


@pytest.mark.parametrize("failure", [None, "cursor", "persist"])
def test_joint_restore_starts_gc_only_after_consumer_state_is_durable(tmp_path, monkeypatch, failure):
    from straw.errors import StorageUnavailable

    args = SimpleNamespace(
        rollout_data_transport="straw",
        rollout_data_dir=str(tmp_path),
        rollout_queue_online_gc=True,
        _joint_resume={"rollout_id": 0},
    )
    controller = RolloutQueueController(args)
    try:
        assert controller._gc_thread is None
        state = dict(
            version=1,
            fetch_cursor=0,
            processed_cursor=0,
            processed_positions=[],
            batches=[],
            finished_batches=[],
        )
        ref = controller.codec.publish(state, submission_id="saved-training")
        save = controller._queue.save_consumer_state
        persisted = []

        def save_before_gc(*a, **kw):
            assert controller._gc_thread is None
            if failure == "persist":
                raise StorageUnavailable("injected storage failure")
            result = save(*a, **kw)
            persisted.append(result)
            return result

        monkeypatch.setattr(controller._queue, "save_consumer_state", save_before_gc)
        if failure:
            with pytest.raises((ValueError, StorageUnavailable)):
                controller.restore_training_state(ref, 1 if failure == "cursor" else 0, 0)
            assert controller._gc_thread is None
            assert not persisted
        else:
            restored = controller.restore_training_state(ref, 0, 0)
            assert persisted and controller.codec.load(restored) == state
            assert controller._gc_thread.is_alive()
            thread = controller._gc_thread
            controller._start_gc()
            assert controller._gc_thread is thread
    finally:
        controller.close()


@pytest.mark.parametrize("error_name", ["UnsafeRecovery", "StorageUnavailable"])
def test_background_gc_failure_stops_queue_work_and_fails_close(tmp_path, monkeypatch, error_name):
    from straw import Record, TaskSpec, errors

    args = SimpleNamespace(rollout_data_transport="straw", rollout_data_dir=str(tmp_path))
    controller = RolloutQueueController(args)
    queue = controller.queue
    ref = controller.store.publish([Record("input", b"still protected")], submission_id="input")
    queue.submit_tasks("input", [TaskSpec("prompt:0", input_ref=ref)])
    lease = queue.acquire("reader").assignments[0].lease
    controller.store.seal()
    before = queue.journal.path.read_bytes()
    error = getattr(errors, error_name)("injected GC failure")
    calls = []

    def fail():
        calls.append("collect")
        raise error

    monkeypatch.setattr(queue, "collect_garbage", fail)
    monkeypatch.setattr(controller._gc_stop, "wait", lambda timeout: False)
    controller._gc_thread = threading.Thread(target=controller._gc_loop)
    controller._gc_thread.start()
    controller._gc_thread.join(5)
    assert not controller._gc_thread.is_alive()
    assert controller._gc_stop.is_set() and calls == ["collect"]

    for operation in (
        controller.configuration,
        controller.identity,
        lambda: controller.take("another-reader", 1),
        lambda: controller.complete(lease, ref),
        lambda: controller.return_groups([{"lease": lease, "input_ref": ref}], []),
        lambda: controller.finish_batch("batch"),
        controller.training_state,
        lambda: controller._save_training_state({}),
    ):
        with pytest.raises(RuntimeError, match="GC failed") as raised:
            operation()
        assert raised.value.__cause__ is error
    assert queue.journal.path.read_bytes() == before
    closed = []
    original_close = queue.close

    def close_queue():
        original_close()
        closed.append(True)

    monkeypatch.setattr(queue, "close", close_queue)
    with pytest.raises(RuntimeError, match="GC failed") as raised:
        controller.close()
    assert raised.value.__cause__ is error and closed == [True]
    assert next(controller.store.read(ref)).payload == b"still protected"


def test_admission_pause_acknowledges_workers_without_waiting_for_responses(
    scheduler_factory,
):
    scheduler, submitted, release = scheduler_factory(1, [1, 1], slow_worker=0)
    scheduler.generate(0, prefetch=2)
    with ThreadPoolExecutor(1) as pool:
        paused = pool.submit(scheduler.pause, drain=False)
        try:
            assert paused.result(timeout=3) is False
            assert not release.is_set()
            assert all(not worker.actor.running for worker in scheduler.workers)
            count = len(submitted)
        finally:
            release.set()
    assert scheduler.pause() is True
    assert len(submitted) == count
    assert not scheduler.pending and not scheduler.running_workers
    scheduler.resume()
    assert len(scheduler.generate(1, prefetch=2).samples) == 1


def test_queue_fetch_finishing_after_pause_preserves_group_without_new_generation(
    monkeypatch,
):
    from slime.rollout import sglang_rollout
    from slime.rollout.fully_async_distributed import _GenerationActor

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        sample = Sample(tokens=[1, 2], status=Sample.Status.ABORTED)
        sample._queue_lease = {"task_id": "borrowed"}
        group = [sample]
        buffered = []

        async def fetch(count):
            entered.set()
            await release.wait()
            return [group]

        async def rebuffer(value):
            buffered.append(value)

        async def unexpected(*args):
            raise AssertionError("Paused worker admitted a generation request")

        monkeypatch.setattr(sglang_rollout, "generate_and_rm_group", unexpected)
        worker = _GenerationActor.__new__(_GenerationActor)
        worker.running = True
        worker.data_source = SimpleNamespace(get_samples_async=fetch)
        worker._rebuffer = rebuffer
        task = asyncio.create_task(worker._generate_group(0))
        await entered.wait()
        await worker._pause()
        release.set()
        assert await task is None
        assert buffered == [group]
        assert sample._queue_lease == {"task_id": "borrowed"}
        assert sample.tokens == [1, 2]

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
