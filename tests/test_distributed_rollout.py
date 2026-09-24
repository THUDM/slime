"""Distributed index ownership and global batch processing."""

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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.base_types import iter_samples
from slime.rollout.data_source import RolloutDataSource
from slime.rollout.distributed_data_source import (
    DistributedDataSourceWithBuffer,
    RolloutIndexAllocator,
    _DistributedReader,
)
from slime.utils.data import Dataset, process_rollout_data
from slime.utils.tensor_store import DiskTensorRef
from slime.utils.types import Sample

NUM_GPUS = 0

# No HTTP server is used by these CPU tests, including their Ray workers.
try:
    import sglang_router  # noqa: F401
except ImportError:
    sys.modules["sglang_router"] = SimpleNamespace(__version__="0.3.0")


@pytest.fixture
def source_factory(monkeypatch):
    def init(source, args):
        source.args = args
        source.metadata = {}
        source.dataset = Dataset.__new__(Dataset)
        source.dataset.origin_samples = [Sample(prompt=f"prompt-{i}") for i in range(7)]
        source.dataset.samples = source.dataset.origin_samples
        source.dataset.seed = 31
        source.dataset.epoch_id = -1

    monkeypatch.setattr(RolloutDataSource, "__init__", init)
    sources = []

    def create(allocator, worker, claim=None):
        args = SimpleNamespace(
            n_samples_per_prompt=2, rollout_shuffle=True, buffer_filter_path=None, buffer_sort_by_staleness=False
        )
        source = _DistributedReader(
            args, claim or (lambda request, count: allocator.claim(worker, request, count)), prefetch_size=3
        )
        sources.append(source)
        return source

    yield create
    for source in sources:
        source.close()


def test_allocator_claims_are_disjoint_and_retries_are_idempotent():
    allocator = RolloutIndexAllocator()
    assert allocator.claim(0, 0, 10) == (0, 10)
    assert allocator.claim(1, 0, 5) == (10, 15)
    assert allocator.claim(0, 0, 10) == (0, 10)
    assert allocator.claim(0, 1, 10) == (15, 25)
    with pytest.raises(ValueError, match="in order"):
        allocator.claim(0, 3, 10)
    restored = RolloutIndexAllocator()
    restored.load_state_dict(allocator.state_dict())
    assert restored.claim(1, 1, 5) == (25, 30)


def test_workers_read_locally_across_epochs_without_duplicate_ids(source_factory):
    allocator = RolloutIndexAllocator()
    first, second = source_factory(allocator, 0), source_factory(allocator, 1)
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


def test_buffered_partial_group_stays_with_its_local_reader(source_factory):
    allocator = RolloutIndexAllocator()
    source = source_factory(allocator, 0)
    other = source_factory(allocator, 1)
    group = source.get_samples(1)[0]
    group[0].tokens = [1, 2, 3]
    group[0].status = Sample.Status.ABORTED
    source.add_samples([group])
    assert other.get_samples(1)[0][0].group_index != group[0].group_index
    assert source.get_buffer_length() == 1
    assert source.get_samples(1)[0] is group
    assert group[0].tokens == [1, 2, 3]


def test_checkpoint_preserves_unused_claims_and_buffer(source_factory):
    allocator = RolloutIndexAllocator()
    source = source_factory(allocator, 0)
    used = source.get_samples(2)
    source.add_samples([used[0]])
    state = source.state_dict()
    global_state = allocator.state_dict()
    expected = source.get_samples(12)
    restored_allocator = RolloutIndexAllocator()
    restored_allocator.load_state_dict(global_state)
    restored = source_factory(restored_allocator, 0)
    restored.load_state_dict(state)
    actual = restored.get_samples(12)
    assert [[s.index for s in g] for g in actual] == [[s.index for s in g] for g in expected]
    assert [[s.prompt for s in g] for g in actual] == [[s.prompt for s in g] for g in expected]


def test_slow_index_claim_does_not_block_async_generation(source_factory):
    entered, release = threading.Event(), threading.Event()
    allocator = RolloutIndexAllocator()

    def claim(request, count):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release index allocation")
        return allocator.claim(0, request, count)

    source = source_factory(allocator, 0, claim)

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
    source = source_factory(RolloutIndexAllocator(), 0)
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
            rollout_batch_size=batch, rollout_sample_filter_path=None, rollout_all_samples_process_path=None
        )

        def make_generator(worker):
            async def execute(rollout_id):
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
                    keep=not reject(index), reason="test", keep_when_insufficient=keep_when_insufficient
                )

            return execute

        for i, capacity in enumerate(capacities):
            actor = _GenerationActor.__new__(_GenerationActor)
            actor.capacity = capacity
            actor.producer = None
            actor.running = False
            actor._generate_group = make_generator(i)
            actors.append(actor)
            workers.append(
                SimpleNamespace(
                    start=remote(actor.start), next=remote(actor.next), pause=remote(actor.pause), actor=actor
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


def test_async_scheduler_fast_worker_fills_batch_without_waiting_for_slow_worker(scheduler_factory):
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


def test_fully_async_filter_never_keeps_rejected_groups_to_fill_batch(scheduler_factory):
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


def test_reader_materializes_checkpoint_and_transferred_samples(tmp_path):
    path = tmp_path / "spill" / "buffer.safetensors"
    reference = DiskTensorRef.write(torch.tensor([1, 2]), path)
    sample = Sample(index=0, rollout_routed_experts=reference)
    reader = _DistributedReader.__new__(_DistributedReader)
    reader.args = SimpleNamespace(
        rollout_routed_experts_store_dir=str(path.parent),
        keep_rollout_routed_experts_files=False,
    )
    reader._lock = threading.RLock()
    reader._buffer_files = {str(path)}
    reader.buffer = [[sample]]
    reader._prefetch = None
    reader._ranges = []
    reader._request_id = 0
    reader.metadata = {}
    saved = reader.state_dict()
    assert saved["buffer"][0][0].rollout_routed_experts.tolist() == [1, 2]
    assert path.exists()  # Snapshotting preserves the live buffer's files.
    groups = reader.materialize_samples([[sample]])
    assert groups[0][0].rollout_routed_experts.tolist() == [1, 2]
    assert sample.rollout_routed_experts is reference
    assert not path.exists() and not reader._buffer_files


@pytest.mark.parametrize("fanout", [False, True])
def test_distributed_completed_groups_keep_rewards_and_masks(monkeypatch, fanout):
    from slime.rollout import sglang_rollout
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

    async def get_samples(count):
        assert count == 1
        return [group]

    async def unexpected_generation(*args):
        raise AssertionError("completed groups must bypass generation and group reward")

    monkeypatch.setattr(sglang_rollout, "generate_and_rm_group", unexpected_generation)
    worker = _GenerationActor.__new__(_GenerationActor)
    worker.args = SimpleNamespace(partial_rollout=True, mask_offpolicy_in_partial_rollout=True)
    worker.dynamic_filter = None
    worker.data_source = SimpleNamespace(get_samples_async=get_samples, materialize_samples=lambda value: value)
    result, _ = asyncio.run(worker._generate_group(3))
    assert result is group
    assert sample.reward == 0.0
    assert sample.loss_mask == [1, 1]


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

    async def generate_group(rollout_id):
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

    async def exercise():
        await worker._start(0)
        try:
            first = await asyncio.wait_for(worker._next(), timeout=5)
            second = await asyncio.wait_for(worker._next(), timeout=5)
            assert first > 0 and second > 0 and first != second
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
    sample.status = Sample.Status.COMPLETED
    if sample.index % 2 == 0:
        sample.multimodal_train_inputs = {"pixel_values": torch.ones(1, 3, 2, 2)}
    sample.rollout_routed_experts = DiskTensorRef.write(
        torch.full((2, 1, 1), sample.index % 8, dtype=torch.uint8),
        Path(args.rollout_routed_experts_store_dir) / f"{sample.index}.safetensors",
    )
    if args.test_fanout:
        children = [copy.deepcopy(sample) for _ in range(2)]
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


@pytest.mark.parametrize("fanout", [False, True])
def test_two_ray_nodes_generate_transfer_and_restore(tmp_path, fanout):
    multiplier = 2 if fanout else 1
    import ray
    from ray.cluster_utils import Cluster
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from slime.ray.rollout import RolloutManager

    tokenizer_dir = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    ).save_pretrained(tokenizer_dir)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps({"text": f"hello {i}"}) for i in range(7)))
    args = SimpleNamespace(
        rollout_batch_size=4,
        data_source_path="slime.rollout.distributed_data_source.DistributedDataSourceWithBuffer",
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
        use_score_centering=False,
        rollout_top_p=1.0,
        use_rollout_routing_replay=False,
        rollout_data_transport="object-store",
        use_distributed_post=False,
        debug_rollout_only=False,
        save_debug_rollout_data=str(tmp_path / "debug" / "rollout_{rollout_id}.pt"),
        ci_test=False,
        load_debug_rollout_data=None,
        keep_rollout_routed_experts_files=False,
        save=str(tmp_path / "checkpoint"),
        load=None,
        dump_details=None,
        rollout_routed_experts_store_dir=str(tmp_path / "spill"),
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
        custom_rm_path="test_distributed_rollout._score_fanout" if fanout else "test_distributed_rollout._score_group",
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
    config = {"dp_size": 2, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}
    cluster = Cluster()
    source = None
    try:
        for _ in range(2):
            cluster.add_node(num_cpus=2, num_gpus=0, object_store_memory=128 * 1024**2, include_dashboard=False)
        ray.init(
            address=cluster.address,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parent.parent)]),
                }
            },
        )
        from slime.utils.misc import load_function

        source = load_function(args.data_source_path)(args)
        assert not source.consumers
        returned = source.get_samples(1)[0]
        source.add_samples([returned])

        cls = RolloutManager.__ray_metadata__.modified_class
        manager = cls.__new__(cls)
        manager.args = args
        from slime.utils.misc import load_function

        rollout_function = load_function(args.rollout_function_path)
        manager.train_parallel_config = config
        manager._active_routed_experts_rollouts = set()
        manager.health_monitoring_resume = lambda: None
        manager._get_updatable_server = lambda: None
        calls = []

        def custom_rollout(global_args, rollout_id, data_source, evaluation=False):
            assert global_args is args
            assert global_args.rollout_batch_size == 4
            assert data_source is source
            assert not evaluation
            calls.append(("rollout", 4))
            return rollout_function(global_args, rollout_id, data_source, evaluation=evaluation)

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
            manager.custom_convert_samples_to_train_data_func = None
            try:
                return manager._convert_samples_to_train_data(samples)
            finally:
                manager.custom_convert_samples_to_train_data_func = convert

        manager.custom_reward_post_process_func = postprocess
        manager.custom_convert_samples_to_train_data_func = convert

        def fetch(rollout_id):
            manager.data_source = source
            previous = len(calls)
            refs = manager.generate(rollout_id)
            assert calls[previous:] == [("rollout", 4), ("convert", 8 * multiplier), ("rewards", 8 * multiplier)]
            dump = torch.load(tmp_path / "debug" / f"rollout_{rollout_id}.pt", weights_only=False)
            assert len(dump["samples"]) == 8 * multiplier
            assert all(isinstance(sample["rollout_routed_experts"], torch.Tensor) for sample in dump["samples"])
            assert not list((tmp_path / "debug").glob("worker_*"))
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
                    assert isinstance(batch["rollout_routed_experts"][i], torch.Tensor)
                    rows[index] = batch["tokens"][i].tolist()
                    assert rows[index][0] == index + 1
            assert len(rows) == 8 * multiplier
            assert multimodal_samples == 4 * multiplier
            return rows

        first = fetch(0)
        assert source.get_buffer_length() == 1  # The owner's buffer stays local.
        runtime = source.consumers["fully_async"]
        assert len(runtime.workers) == 2
        manager.save(0)
        path = Path(args.save) / "rollout" / "distributed_data_source_0.pt"
        state = torch.load(path, weights_only=False)["consumers"]["fully_async"]
        assert len(state["scheduler"]["ready"]) >= 4
        assert all(
            sample.metadata["reward_calls"] == 1
            for group, _ in state["scheduler"]["ready"]
            for sample in iter_samples(group)
        )
        manager.cleanup_rollout_data(0)
        expected = fetch(1)
        assert not first.keys() & expected.keys()
        source.close()
        source = None
        args.load = args.save
        source = DistributedDataSourceWithBuffer(args)
        source.data_config["n_samples_per_prompt"] += 1
        with pytest.raises(ValueError, match="samples per prompt"):
            source.load(0)
        source.data_config["n_samples_per_prompt"] -= 1
        source.load(0)
        assert not source.consumers  # Execution state is restored when fully async starts.
        source.save(0)  # Saving before the first generate must preserve the restored queue.
        actual = fetch(1)
        assert actual == expected
        restored_group = source.get_samples(1)[0]
        assert [sample.index for sample in restored_group] == [sample.index for sample in returned]
        assert not {sample.index for sample in returned} & (first.keys() | actual.keys())
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
                    while len(runtime.pending) != 2:
                        assert time.monotonic() < deadline, "both workers should receive work"
                        runtime.condition.wait(timeout=0.01)
                failed = 0
                if fanout:
                    node = next(node for node in cluster.list_all_nodes() if node != cluster.head_node)
                    nodes = sorted(ray.nodes(), key=lambda node: (node["NodeManagerAddress"], node["NodeID"]))
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
        assert runtime.capacity == 6
        assert runtime.error is None
        assert not after_failure.keys() & actual.keys()
        source.save(2)
        saved = torch.load(Path(args.save) / "rollout" / "distributed_data_source_2.pt", weights_only=False)
        assert saved["consumers"]["fully_async"]["readers"][failed] is None
        assert len(fetch(3)) == 8 * multiplier

        if not fanout:
            # A restart with the same topology preserves retired worker slots.
            source.close()
            source = DistributedDataSourceWithBuffer(args)
            source.load(2)
            assert len(fetch(3)) == 8
            runtime = source.consumers["fully_async"]
            assert runtime.capacities[failed] == 0
            assert runtime.capacity == 6
            runtime.pause()
            runtime.ready.clear()
            for i, worker in enumerate(runtime.workers):
                if runtime.capacities[i]:
                    ray.kill(worker, no_restart=True)
            with pytest.raises(RuntimeError, match="All rollout workers are unavailable"):
                runtime.generate(4, prefetch=runtime.capacity)
    finally:
        if source is not None:
            source.close()
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
