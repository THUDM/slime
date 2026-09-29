import os
import pickle
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from straw.errors import StorageUnavailable
from slime.rollout.queue_data_source import RolloutQueueController
from slime.utils.rollout_transport import (
    DiskPayloadRef,
    RolloutGroupRef,
    load_rollout_samples,
    pack_rollout_group,
    pack_rollout_payload,
    resolve_rollout_data_dir,
    seal_rollout_store,
    unpack_rollout_payload,
)
from slime.utils.types import Sample

NUM_GPUS = 0


def test_cpu_rollout_imports_do_not_require_sglang():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoSGLang(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sglang', 'sglang_router'}:
            raise ModuleNotFoundError(f'CPU rollout imported {fullname}', name=fullname)

sys.meta_path.insert(0, NoSGLang())
from slime.ray.rollout import RolloutManager
from slime.rollout.sglang_rollout import generate_and_rm
from slime.rollout.fully_async_distributed import RolloutScheduler
assert not any(name.split('.')[0] in {'sglang', 'sglang_router'} for name in sys.modules)
""",
        ],
        check=True,
        timeout=60,
    )


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        rollout_data_transport="straw", rollout_data_dir=str(tmp_path), rollout_sample_filter_path=None, save=None
    )


def test_directory_requires_explicit_shared_root_or_checkpoint(tmp_path):
    args = SimpleNamespace(rollout_data_dir=None, save=None)
    with pytest.raises(ValueError, match="--rollout-data-dir or --save"):
        resolve_rollout_data_dir(args)
    args.save = str(tmp_path)
    resolve_rollout_data_dir(args)
    assert args.rollout_data_dir == str(tmp_path / "rollout_data")
    args.rollout_data_dir = str(tmp_path / "explicit")
    resolve_rollout_data_dir(args)
    assert args.rollout_data_dir == str(tmp_path / "explicit")


@pytest.mark.parametrize(
    "options",
    [
        "",
        "--save-debug-rollout-data x.pt",
        "--rollout-data-transport straw",
        "--save=/shared/ckpt",
        "--rollout-data-dir /shared/data",
    ],
)
def test_local_test_launcher_does_not_invent_storage_paths(options, monkeypatch):
    from slime.utils.external_utils import command_utils

    commands = []
    monkeypatch.setattr(command_utils, "exec_command", commands.append)
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)
    monkeypatch.setenv("SLIME_SCRIPT_EXTERNAL_RAY", "0")
    monkeypatch.setenv("SLIME_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    command_utils.execute_train(options, num_gpus_per_node=1, megatron_model_type=None)
    assert commands[-1].count("--rollout-data-dir") == options.count("--rollout-data-dir")
    assert options in commands[-1]


def test_payload_reference_is_small_and_readable_by_independent_process(args, tmp_path):
    tensor = torch.arange(1_000_000, dtype=torch.int32)
    sample = Sample(tokens=[1, 2], response_length=1, rollout_routed_experts=tensor)
    packed = pack_rollout_payload(dict(sample=sample, tensor=tensor, alias=tensor, text="x" * 1_000_000), args, 0)
    assert isinstance(packed, DiskPayloadRef)
    assert len(pickle.dumps(packed)) < 1024
    path = tmp_path / "reference.pkl"
    path.write_bytes(pickle.dumps(packed))
    # Reading while the writer is open must work, with no format environment variable.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pickle, sys
from slime.utils.tensor_store import TensorRef
value = pickle.load(open(sys.argv[1], 'rb')).load()
assert value['tensor'] is value['alias']
assert value['tensor'][-1].item() == 999999
assert len(value['text']) == 1_000_000
assert isinstance(value['sample'].rollout_routed_experts, TensorRef)
assert value['sample'].rollout_routed_experts[999990:].tolist() == list(range(999990, 1000000))
""",
            str(path),
        ],
        check=True,
        timeout=60,
        env=dict(os.environ),
    )


def test_batch_manifest_reuses_sorted_group_refs_without_reading_samples(args, monkeypatch):
    from slime.rollout.base_types import finalize_rollout_groups

    groups = [pack_rollout_group([[Sample(index=i, rollout_id=i, response="x" * 1_000_000)]], args, 0) for i in (2, 1)]

    def unexpected_read(ref):
        raise AssertionError("batch assembly must not load sample payloads")

    with monkeypatch.context() as m:
        m.setattr(DiskPayloadRef, "load", unexpected_read)
        result = finalize_rollout_groups(args, 0, groups)
        assert pack_rollout_payload(result.samples, args, 0) is result.samples
    assert isinstance(result.samples, DiskPayloadRef)
    assert result.samples.manifest.manifest.segment.size < 8192
    stored = result.samples.load()
    assert all(isinstance(ref, RolloutGroupRef) for ref in stored)
    assert [ref.index for ref in stored] == [1, 2]
    assert stored == groups
    restored = load_rollout_samples(result.samples)
    assert [group[0][0].index for group in restored] == [1, 2]
    assert len(restored[0][0][0].response) == 1_000_000


def test_batch_hook_changes_are_saved_once_after_loading_groups(args, monkeypatch):
    from slime.rollout.base_types import finalize_rollout_groups
    from slime.utils import misc

    groups = [pack_rollout_group([Sample(index=i, reward=i)], args, 0) for i in (2, 1)]
    calls = []

    def filter_batch(received_args, samples):
        assert received_args is args
        calls.append([group[0].index for group in samples])
        samples.pop()
        samples[0][0].reward = 42

    args.rollout_sample_filter_path = "test.batch_filter"
    monkeypatch.setattr(misc, "load_function", lambda path: filter_batch)
    result = finalize_rollout_groups(args, 0, groups)
    assert calls == [[1, 2]]
    restored = load_rollout_samples(result.samples)
    assert len(restored) == 1
    assert restored[0][0].reward == 42


def test_fully_async_stores_groups_while_collecting_the_batch(args, monkeypatch):
    import asyncio

    from slime.rollout import fully_async_rollout as fa

    args.rollout_batch_size = 2
    args.dynamic_sampling_filter_path = None
    saved = []

    def pack(group, *a):
        ref = pack_rollout_group(group, *a)
        saved.append(ref)
        return ref

    def take(limit):
        assert limit == 2 - len(saved)
        # Producing the next group requires the previous one to be on disk.
        if saved:
            assert saved[0].load()[0].index == 2
        return [(len(saved), [Sample(index=2 - len(saved), reward=1)])]

    worker = SimpleNamespace(queue_size=lambda: 0, get_completed_groups=take)
    monkeypatch.setattr(fa, "_get_global_worker", lambda *a: worker)
    from slime.utils import rollout_transport

    monkeypatch.setattr(rollout_transport, "pack_rollout_group", pack)
    output = asyncio.run(fa._generate_rollout_async(args, 0, None))
    assert len(saved) == 2
    assert isinstance(output.samples, DiskPayloadRef)
    assert [group[0].index for group in load_rollout_samples(output.samples)] == [1, 2]


@pytest.mark.parametrize("all_samples_hook", [False, True])
def test_synchronous_rollout_stores_during_generation_and_preserves_legacy_hook(args, monkeypatch, all_samples_hook):
    import asyncio

    from slime.rollout import sglang_rollout as sr

    args.rollout_batch_size = 2
    args.n_samples_per_prompt = 1
    args.over_sampling_batch_size = 2
    args.dynamic_sampling_filter_path = None
    args.rollout_all_samples_process_path = "test.all_samples" if all_samples_hook else None
    saved = []
    hook_calls = []
    state = SimpleNamespace(remaining_batch_size=0, pendings=set(), reset=lambda: None)

    async def exercise():
        second = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def generate(index):
            if index == 1 and not all_samples_hook:
                await second.wait()
                assert saved[0].load()[0].index == 2
            return [Sample(index=index, prompt="p", response="r", reward=index)]

        def submit(groups):
            state.remaining_batch_size += len(groups)
            state.pendings.update(asyncio.create_task(generate(index)) for index in (2, 1))

        def pack(group, *a):
            ref = pack_rollout_group(group, *a)
            saved.append(ref)
            loop.call_soon_threadsafe(second.set)
            return ref

        async def abort(*a):
            return []

        def hook(received_args, samples, source):
            hook_calls.append([group[0].index for group in samples])
            samples[0][0].reward = 42

        state.submit_generate_tasks = submit
        monkeypatch.setattr(sr, "GenerateState", lambda args: state)
        from slime.utils import rollout_transport

        monkeypatch.setattr(rollout_transport, "pack_rollout_group", pack)
        monkeypatch.setattr(sr, "abort", abort)
        monkeypatch.setattr(sr, "load_function", lambda path: hook)
        return await asyncio.wait_for(sr.generate_rollout_async(args, 0, lambda count: [None] * count), timeout=20)

    output, aborted = asyncio.run(exercise())
    assert isinstance(output.samples, DiskPayloadRef)
    assert aborted == []
    restored = load_rollout_samples(output.samples)
    assert [group[0].index for group in restored] == [1, 2]
    assert hook_calls == ([[1, 2]] if all_samples_hook else [])
    assert restored[0][0].reward == (42 if all_samples_hook else 1)
    assert len(saved) == 2


@pytest.mark.parametrize("form", ["raw", "wrapped", "stored", "accepted"])
def test_manager_adapts_legacy_samples_and_validates_accepted_refs(args, monkeypatch, form):
    from slime.ray import rollout
    from slime.rollout.base_types import RolloutFnTrainOutput, finalize_rollout_groups
    from slime.rollout.queue_data_source import RolloutQueueController
    from slime.utils.rollout_transport import RawRolloutRef, accept_raw_rollout

    groups = [[Sample(index=1, tokens=[1, 2], response_length=1, reward=1)]]
    result = (
        finalize_rollout_groups(args, 0, [pack_rollout_group(groups[0], args, 0)])
        if form in {"stored", "accepted"}
        else RolloutFnTrainOutput(samples=groups, metrics={"custom": 1})
    )
    controller = RolloutQueueController(args)
    args._rollout_queue_controller = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(controller, name))
            for name in ("begin_collection", "complete", "accepted")
        }
    )
    monkeypatch.setattr(rollout.ray, "get", lambda value: value)
    manager = object.__new__(rollout.RolloutManager.__ray_metadata__.modified_class)
    manager.args = args
    args.load_debug_rollout_data = None
    manager.data_source = object()
    manager.batch_builder = SimpleNamespace()
    if form == "accepted":
        result = accept_raw_rollout(result, args, 0)
    manager.generate_rollout = lambda *a, **kw: groups[0] if form == "raw" else result
    previous_files = {path: path.read_bytes() for path in Path(args.rollout_data_dir).rglob("*.pack")}
    try:
        samples, metrics = manager._get_rollout_data(0)
        assert [sample.tokens for sample in samples] == [[1, 2]]
        ref = manager.batch_builder.raw_ref
        assert isinstance(ref, RawRolloutRef)
        assert controller.accepted(ref.receipt) == ref.receipt
        assert all(path.read_bytes()[: len(contents)] == contents for path, contents in previous_files.items())
        assert metrics == ({"custom": 1} if form == "wrapped" else None)
        if form == "accepted":
            assert ref is result
            assert len(previous_files) == len(list(Path(args.rollout_data_dir).rglob("*.pack")))
    finally:
        controller.close()


@pytest.mark.parametrize("transport", ["object-store", "nixl", "straw"])
def test_train_partitions_preserve_top_p_and_multimodal(args, monkeypatch, transport):
    from slime.rollout import batch_builder as rollout
    from slime.utils.data import process_rollout_data

    args.rollout_data_transport = transport
    if transport != "straw":
        args.rollout_data_dir = None
    manager = object.__new__(rollout.BatchBuilder)
    manager.args = SimpleNamespace(**vars(args), global_batch_size=2)
    manager.rollout_id = 0
    manager.train_parallel_config = {"dp_size": 2}
    monkeypatch.setattr(rollout, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    sent = []

    def put(value, **kwargs):
        if transport == "straw":
            assert isinstance(value, DiskPayloadRef)
            assert len(pickle.dumps(value)) < 1024
        else:
            assert isinstance(value, dict)
            assert kwargs == ({"_tensor_transport": "nixl"} if transport == "nixl" else {})
        sent.append(value)
        return value

    monkeypatch.setattr(rollout.ray, "put", put)
    monkeypatch.setattr(rollout.ray, "get", lambda value: value)
    refs = manager.split_by_dp(
        dict(
            tokens=[[10, 11], [20, 21, 22]],
            rollout_ids=[0, 1],
            response_lengths=[1, 2],
            loss_masks=[[1], [0, 1]],
            rollout_top_p_token_ids=[[11], [21, 22]],
            rollout_top_p_token_offsets=[[0, 1], [0, 1, 2]],
            rollout_top_p_log_probs=[[-0.2], [-0.3, -0.4]],
            rollout_log_probs=[[-0.2], [-0.3, -0.4]],
            raw_reward=[1.0, 2.0],
            rewards=[-1.0, 1.0],
            multimodal_train_inputs=[None, {"pixel_values": torch.ones(3, 20, 20)}],
        )
    )
    assert len(sent) == 2
    batch = process_rollout_data(refs, 0, 2)
    assert batch["tokens"][0].tolist() == [20, 21, 22]
    assert batch["loss_masks"][0].tolist() == [0, 1]
    assert batch["total_lengths"] == [3]
    assert batch["local_raw_reward"] == [2.0]
    assert batch["rollout_top_p_token_offsets"][0].tolist() == [0, 1, 2]
    torch.testing.assert_close(batch["rollout_top_p_log_probs"][0], torch.tensor([-0.3, -0.4]))
    assert batch["multimodal_train_inputs"][0]["pixel_values"].shape == (3, 20, 20)


def test_durable_batch_replays_one_plan_and_rejects_mixed_ranks(args, monkeypatch):
    from dataclasses import replace

    from slime.rollout import batch_builder as module
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.queue_data_source import RolloutQueueController
    from slime.utils.data import process_rollout_data
    from slime.utils.misc import Box
    from slime.utils.rollout_transport import TrainBatchRef, accept_raw_rollout

    args.custom_reward_post_process_path = None
    args.custom_convert_samples_to_train_data_path = None
    args.global_batch_size = 2
    samples = [Sample(index=i, rollout_id=i, tokens=[1, 2], response_length=1) for i in range(2)]
    controller = RolloutQueueController(args)
    args._rollout_queue_controller = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(controller, name))
            for name in (
                "begin_collection",
                "complete",
                "accepted",
                "batch",
                "plan_batch",
                "ready_batch",
                "training_state",
                "restore_training_state",
                "finish_batch",
            )
        }
    )
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "put", lambda value: value)
    monkeypatch.setattr(module, "build_dp_schedule", lambda *a, **kw: ([[0], [1]], [[[0]], [[0]]], [1], [2]))
    try:
        builder = module.BatchBuilder(args)
        builder.train_parallel_config = {"dp_size": 2}
        builder.raw_ref = accept_raw_rollout(RolloutFnTrainOutput(samples=samples), args, 0)
        assert builder.begin(samples) is None
        plan = controller.batch(builder.batch_id)
        assert not plan["ready"]
        assert "rng" not in builder._plan.load()
        # A crash before conversion commits reuses the selection without RNG state.
        assert builder.begin(samples) is None
        assert controller.batch(builder.batch_id) == plan
        refs = builder.split_by_dp({"tokens": [[1, 2], [1, 2]], "rollout_ids": [0, 1], "rewards": [0.0, 1.0]})
        assert controller.batch(builder.batch_id)["ready"]
        assert all(isinstance(box.inner, TrainBatchRef) for box in refs)
        assert len({box.inner.batch_id for box in refs}) == 1
        before = set(Path(args.rollout_data_dir).rglob("*.pack"))
        replay = builder.begin(samples)
        assert [box.inner for box in replay] == [box.inner for box in refs]
        assert before == set(Path(args.rollout_data_dir).rglob("*.pack"))
        assert process_rollout_data(refs, 0, 2)["tokens"][0].tolist() == [1, 2]
        mixed = [refs[0], Box(replace(refs[1].inner, batch_id="another-batch"))]
        with pytest.raises(ValueError, match="inconsistent batch identities"):
            process_rollout_data(mixed, 0, 2)
        args.save = str(Path(args.rollout_data_dir) / "checkpoint")
        builder.save(0)
        assert controller.queue._usage()["ready_bytes"] > 0
        builder.training_completed(builder.rollout_id)
        assert controller.queue._usage()["ready_bytes"] == 0
        assert not controller.queue.checkpoints  # runtime completion is not a checkpoint
        # Later production facts survive restoring the earlier training view.
        later = accept_raw_rollout(RolloutFnTrainOutput(samples=samples), args, 1)
        assert later.receipt.position == 1
        args.load = args.save
        builder.load(0)
        assert controller.queue._usage()["ready_bytes"] > 0  # earlier consumer view is restored
        assert controller.training_state()["processed_cursor"] == 1
        assert controller.queue.read_commits().cursor == 2
        args.global_batch_size = 4
        with pytest.raises(ValueError, match="different selection or conversion plan"):
            builder.begin(samples)
    finally:
        controller.close()


def test_packed_tensors_handle_scalar_empty_and_bfloat16(args):
    original = {"scalar": torch.tensor(2.5), "empty": torch.empty(0, 3), "bf16": torch.ones(3, dtype=torch.bfloat16)}
    restored = unpack_rollout_payload(pack_rollout_payload(original, args, 0))
    for key in original:
        torch.testing.assert_close(restored[key], original[key])


def test_debug_dump_survives_removal_of_queue_storage(args, tmp_path):
    from straw.tensor import TensorRef

    from slime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data

    args.num_layers, args.num_experts, args.moe_router_topk = 1, 8, 2
    args.use_score_centering, args.score_centering_top_k, args.rollout_top_p = True, 2, 1.0
    args.use_rollout_routing_replay = True
    sample = Sample(
        index=0,
        tokens=[1, 2, 3],
        response_length=2,
        status=Sample.Status.COMPLETED,
        rollout_routed_experts=torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.uint8),
        rollout_topk_token_ids=[[1, 2], [3, 4]],
        rollout_topk_log_probs=[[-1.0, -2.0], [-1.0, -2.0]],
    )
    sample = pack_rollout_payload(sample, args, 0).load()
    refs = [sample.rollout_routed_experts, sample.rollout_topk_token_ids, sample.rollout_topk_log_probs]
    assert all(isinstance(ref, TensorRef) for ref in refs)
    assert len({ref.path for ref in refs}) == 1
    debug_path = str(tmp_path / "debug_{rollout_id}.pt")
    save_debug_rollout_data(debug_path, [sample], rollout_id=0, evaluation=False)
    seal_rollout_store(args)
    assert sample.rollout_topk_token_ids.load().tolist() == [[1, 2], [3, 4]]
    paths = {Path(ref.path) for ref in refs}
    assert len(paths) == 1
    for path in paths:
        path.unlink()
    [restored] = load_debug_rollout_data(debug_path, rollout_id=0)
    assert restored.tokens == sample.tokens
    assert isinstance(restored.rollout_routed_experts, torch.Tensor)
    assert restored.rollout_routed_experts.tolist() == [[[1, 2]], [[3, 4]]]
    assert restored.rollout_topk_token_ids.tolist() == [[1, 2], [3, 4]]
    assert not list(tmp_path.rglob("*.pack"))
    assert not list(tmp_path.glob("*.bin"))


def test_shared_storage_probe_reports_missing_mount(args, monkeypatch):
    from slime.utils import rollout_transport

    monkeypatch.setattr(rollout_transport.ray, "nodes", lambda: [])

    def fail(*a, **kw):
        raise FileNotFoundError("node cannot see shared file")

    monkeypatch.setattr(rollout_transport.ray, "get", fail)
    with pytest.raises(RuntimeError, match="All rollout/training nodes must share the run"):
        rollout_transport.check_rollout_storage(args)


def test_failed_flush_does_not_publish_a_reference(args, tmp_path, monkeypatch):
    from slime.utils import rollout_transport

    store, _, _ = rollout_transport.rollout_store(args)

    def fail(phase):
        if phase == "before_directory_sync":
            raise OSError("shared storage flush failed")

    with monkeypatch.context() as m:
        m.setattr(store.backend, "fault", fail)
        with pytest.raises(StorageUnavailable, match="shared storage flush failed"):
            pack_rollout_payload({"tensor": torch.arange(16)}, args, 0)
    # No commit marker was published. A later successful write remains readable.
    assert not list(tmp_path.rglob("*.pack"))
    restored = pack_rollout_payload({"tensor": torch.arange(16)}, args, 0).load()
    assert restored["tensor"].tolist() == list(range(16))


def test_straw_debug_archive_keys_lazy_tensors_gc_and_legacy_export(args, tmp_path):
    from slime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data
    from slime.utils.rollout_archive import RolloutArchive
    from slime.utils.rollout_transport import rollout_store
    from slime.utils.tensor_store import TensorRef

    args.rollout_queue_online_gc = True
    tensor = torch.arange(8, dtype=torch.int32).reshape(2, 2, 2)
    sample = Sample(index=42, tokens=[1, 2, 3], response_length=2, reward=1.0, rollout_routed_experts=tensor)
    sample._queue_lease = {"task_id": "prompt:21", "queue_id": "parent"}
    sample._queue_source_positions = [999]
    original = pack_rollout_payload([sample], args, 0)
    [sample] = original.load()
    template = str(tmp_path / "archive_{rollout_id}.straw.json")
    save_debug_rollout_data(template, [sample], rollout_id=7, evaluation=False, args=args)
    store, _, _ = rollout_store(args)
    store.release_publications([original.manifest])
    store.seal()
    store.collect_garbage()
    with RolloutArchive(template.format(rollout_id=7)) as archive:
        assert archive.keys() == [("sample:42", "prompt:21")]
        [restored] = archive.load_samples(sample_key="sample:42", task_key="prompt:21")
        assert not hasattr(restored, "_queue_lease") and not hasattr(restored, "_queue_source_positions")
        assert isinstance(restored.rollout_routed_experts, TensorRef)
        assert restored.rollout_routed_experts.record_ref == sample.rollout_routed_experts.record_ref
        assert torch.equal(restored.rollout_routed_experts.load(), tensor)
        with pytest.raises(KeyError):
            archive.load_samples(sample_key="sample:404")
        legacy = tmp_path / "export.pt"
        archive.export_pt(legacy)
        [loaded] = load_debug_rollout_data(template, rollout_id=7)
        assert loaded.tokens == restored.tokens
        archive.release()
    store.collect_garbage()
    [exported] = load_debug_rollout_data(str(legacy), rollout_id=7)
    assert torch.equal(exported.rollout_routed_experts, tensor)
    with pytest.raises(FileExistsError):
        save_debug_rollout_data(template, [sample], rollout_id=7, evaluation=False, args=args)


def test_archive_index_reads_only_selected_chunk_and_preserves_duplicate_keys(args, tmp_path, monkeypatch):
    from slime.utils.rollout_archive import RolloutArchive

    samples = [Sample(index=i, tokens=[i]) for i in range(130)]
    samples[129].index = 128  # Compact trajectories can reuse a logical sample ID.
    path = tmp_path / "indexed.straw.json"
    RolloutArchive.save(path, samples, rollout_id=0, args=args)
    with RolloutArchive(path) as archive:
        read = []
        load = archive.codec.load

        def tracked(ref, **kwargs):
            if not kwargs:
                read.append(ref)
            return load(ref, **kwargs)

        monkeypatch.setattr(archive.codec, "load", tracked)
        assert [s.tokens for s in archive.load_samples(sample_key="sample:128")] == [[128], [129]]
        assert len(read) == 1


def test_straw_debug_archive_uses_existing_train_only_conversion(args, tmp_path):
    from slime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data
    from slime.rollout.batch_builder import BatchBuilder

    args.custom_reward_post_process_path = args.custom_convert_samples_to_train_data_path = None
    args.reward_key = None
    args.advantage_estimator = "grpo"
    args.rewards_normalization = False
    args.use_score_centering = args.use_rollout_routing_replay = False
    sample = Sample(index=4, tokens=[1, 2, 3], response_length=2, reward=0.5, loss_mask=[1, 0])
    paths = [str(tmp_path / "rollout_{rollout_id}.pt"), str(tmp_path / "rollout_{rollout_id}.straw.json")]
    batches = []
    for path in paths:
        save_debug_rollout_data(path, [sample], rollout_id=7, evaluation=False, args=args)
        builder = BatchBuilder(args)
        loaded = load_debug_rollout_data(path, rollout_id=7)
        assert builder.begin(loaded) is None
        batches.append(builder.convert(loaded))
    assert batches[0] == batches[1]
    # A real CPU optimizer step on either loader's output must match.
    weights = []
    for batch in batches:
        model = torch.nn.Embedding(4, 1)
        torch.nn.init.ones_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.tensor(batch["tokens"][0])).sum() * batch["rewards"][0]
        loss.backward()
        optimizer.step()
        weights.append(model.weight.detach().clone())
    torch.testing.assert_close(*weights)


@pytest.mark.parametrize("step", [0, 7])
def test_fork_requires_committed_model_and_source_and_restores_weight_version(tmp_path, step):
    import json
    from slime.utils.rollout_checkpoint import commit_checkpoint, resolve_checkpoint

    parent = tmp_path / "parent"
    model = parent / f"iter_{step:07d}"
    model.mkdir(parents=True)
    (model / "weights.pt").write_bytes(b"model-and-optimizer")
    (parent / "latest_checkpointed_iteration.txt").write_text(str(step))
    rollout = parent / "rollout"
    rollout.mkdir()
    for name in ("queue_state", "builder_state"):
        (rollout / f"{name}_{step}.json").write_text(json.dumps({"test": name}))
    args = SimpleNamespace(save=str(parent), rollout_data_dir=str(tmp_path / "pool"), rollout_queue_run_id="run")
    commit_checkpoint(args, step, model_args=[args])
    restore = SimpleNamespace(
        _rollout_queue_fork=True,
        rollout_data_transport="straw",
        load=str(parent),
        save=str(tmp_path / "child"),
        rollout_data_dir=None,
        rollout_queue_run_id="run",
        ckpt_step=step,
        start_rollout_id=None,
    )
    resolve_checkpoint(restore)
    assert restore.ckpt_step == step and restore.start_rollout_id == step + 1
    assert restore.update_weight_start_version == step + 1
    assert restore.rollout_data_dir == args.rollout_data_dir
    assert not (tmp_path / "child").exists()  # Selection does not mutate the parent or create a job.
    (rollout / f"queue_state_{step}.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        resolve_checkpoint(restore)


@pytest.mark.parametrize("fail_at", [None, "actor", "critic", "rollout"])
@pytest.mark.parametrize("start", [0, 8])
def test_training_commits_only_after_save_calls_return(tmp_path, monkeypatch, fail_at, start):
    import importlib.util
    import json
    from unittest.mock import Mock

    # Exercise the real training loop with CPU stand-ins for the GPU services.
    # Argument parsing is not used here and must not pull in SGLang on CPU CI.
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", SimpleNamespace(parse_args=None))
    spec = importlib.util.spec_from_file_location("checkpoint_train", Path(__file__).parents[1] / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        save=str(tmp_path),
        rollout_data_dir=str(tmp_path / "pool"),
        rollout_data_transport="straw",
        release_train=False,
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        num_rollout=start + 3,
        start_rollout_id=start,
        eval_interval=None,
        save_interval=1,
        use_critic=True,
        num_critic_only_steps=0,
        debug_train_only=False,
        debug_rollout_only=False,
    )
    saved = []

    def save_model(role, directory, step, *, force_sync):
        assert force_sync
        saved.append(role)
        if fail_at == role:
            raise OSError(f"{role} save failed")
        model = directory / f"iter_{step:07d}"
        model.mkdir(parents=True)
        (model / "weights.pt").write_bytes(role.encode())

    actor, critic = Mock(), Mock()
    actor.update_weights.return_value = None
    actor.args = SimpleNamespace(save=str(tmp_path))
    critic.args = SimpleNamespace(save=str(tmp_path / "critic"))
    actor.save_model.side_effect = lambda step, **kw: save_model("actor", tmp_path, step, **kw)
    critic.save_model.side_effect = lambda step, **kw: save_model("critic", tmp_path / "critic", step, **kw)

    def save_rollout(step):
        saved.append("rollout")
        if fail_at == "rollout":
            raise OSError("rollout save failed")
        rollout = tmp_path / "rollout"
        rollout.mkdir(exist_ok=True)
        for name in ("queue_state", "builder_state"):
            (rollout / f"{name}_{step}.json").write_text("{}")

    manager = Mock()
    manager.save.remote.side_effect = save_rollout
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module, "configure_logger", lambda: None)
    monkeypatch.setattr(module, "init_tracking", lambda args: None)
    monkeypatch.setattr(module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(module, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(module, "create_rollout_manager", lambda *a: (manager, None))
    monkeypatch.setattr(module, "create_training_models", lambda *a: (actor, critic))
    marker = tmp_path / "rollout" / f"committed_{start}.json"
    if fail_at:
        with pytest.raises(OSError, match=f"{fail_at} save failed"):
            module.train(args)
        assert saved == ["actor", "critic", "rollout"][: ["actor", "critic", "rollout"].index(fail_at) + 1]
        assert not marker.exists()
    else:
        module.train(args)
        assert saved == ["actor", "critic", "rollout"] * 3
        assert actor.update_weights.call_count == 4  # Initial sync, then once per rollout.
        for step in range(start, start + 3):
            checkpoint = json.loads((tmp_path / "rollout" / f"committed_{step}.json").read_text())
            assert checkpoint["weight_version"] == step + 1
            assert set(checkpoint["files"]) == {
                f"iter_{step:07d}/weights.pt",
                f"critic/iter_{step:07d}/weights.pt",
                f"rollout/queue_state_{step}.json",
                f"rollout/builder_state_{step}.json",
            }


@pytest.mark.parametrize("mode", ["nccl", "disk", "release"])
def test_disk_weight_updates_do_not_require_actor_return_values(tmp_path, monkeypatch, mode):
    from unittest.mock import Mock
    from slime.ray.actor_group import RayTrainGroup

    args = SimpleNamespace(
        update_weight_start_version=12,
        update_weight_mode="full",
        update_weight_transport="nccl" if mode == "nccl" else "disk",
        update_weight_disk_dir=str(tmp_path / "weights"),
        release_train=mode == "release",
        save=str(tmp_path / "model"),
        no_save_optim=False,
    )
    group = RayTrainGroup(args, 1, 1, pg=None)
    worker = Mock()
    worker.update_weights.remote.return_value = None
    worker.save_model.remote.return_value = None
    worker.init.remote.return_value = 8
    group._actor_handlers = [worker]
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda value: value)
    reloaded = []
    monkeypatch.setattr(
        group, "_reload_rollout_weights_from_disk", lambda path, version: reloaded.append((path, version))
    )
    monkeypatch.setattr(group, "release", lambda: group._actor_handlers.clear())
    monkeypatch.setattr(group, "_allocate_gpus_for_actor", lambda *a: group._actor_handlers.append(worker))

    for version in (13, 14, 15, 16):
        assert group.update_weights() is None
        if mode == "release" and not group._actor_handlers:
            assert group.create() == [8]
            assert worker.init.remote.call_args.args[0].update_weight_start_version == version
        assert group.save_model(7, force_sync=True) is None
    expected = (
        []
        if mode == "nccl"
        else [(tmp_path / "weights" / f"weight_v{version:06d}", str(version)) for version in (13, 14, 15, 16)]
    )
    assert reloaded == expected


@pytest.mark.parametrize("mode", ["straw", "object-store", "critic-only", "debug-train", "debug-rollout"])
def test_checkpoint_helper_preserves_save_modes(tmp_path, monkeypatch, mode):
    from unittest.mock import Mock
    from slime.utils.rollout_checkpoint import save_checkpoint

    args = SimpleNamespace(
        rollout_data_transport="object-store" if mode == "object-store" else "straw",
        release_train=False,
        num_rollout=10,
        use_critic=mode == "critic-only",
        debug_train_only=mode == "debug-train",
        debug_rollout_only=mode == "debug-rollout",
        save=str(tmp_path),
    )
    actor, critic, manager = Mock(), Mock(), Mock()
    monkeypatch.setattr("ray.get", lambda value: value)
    # A pre-existing joint marker must not affect non-joint save modes.
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout/committed_0.json").write_text("old")
    if mode == "straw":
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            save_checkpoint(args, 0, actor, critic, manager, actor_trains=True)
        actor.save_model.assert_not_called()
        critic.save_model.assert_not_called()
        manager.save.remote.assert_not_called()
        return
    save_checkpoint(args, 0, actor, critic, manager, actor_trains=mode != "critic-only")
    if mode == "critic-only":
        actor.save_model.assert_not_called()
        critic.save_model.assert_called_once_with(0, force_sync=False)
    else:
        actor.save_model.assert_called_once_with(0, force_sync=False)
        critic.save_model.assert_not_called()
    manager.save.remote.assert_called_once_with(0)
    assert (tmp_path / "rollout/committed_0.json").read_text() == "old"


@pytest.mark.parametrize("omitted", ["no_save_optim", "no_save_rng"])
def test_checkpoint_retains_critic_save_policy(tmp_path, omitted):
    import json
    from slime.utils.rollout_checkpoint import commit_checkpoint, resolve_checkpoint

    root, pool = tmp_path / "run", tmp_path / "pool"
    _checkpoint_fixture(root, 0, pool, queue=False)
    _checkpoint_fixture(root / "critic", 0, pool, queue=False)
    for name in ("queue_state", "builder_state"):
        (root / "rollout" / f"{name}_0.json").write_text("{}")
    args = SimpleNamespace(save=str(root), rollout_data_dir=str(pool))
    critic = SimpleNamespace(save=str(root / "critic"), **{omitted: True})
    commit_checkpoint(args, 0, model_args=[args, critic])
    assert not json.loads((root / "rollout/committed_0.json").read_text())["resumable"]
    with pytest.raises(ValueError, match="omitted optimizer/RNG"):
        resolve_checkpoint(_checkpoint_args(root, tmp_path / "child", step=0))


def _checkpoint_fixture(root, step, pool, *, queue=True, checkpoint_args=None):
    """Write real joint-commit metadata around tiny model payload fixtures."""
    import json
    from slime.utils.rollout_checkpoint import commit_checkpoint

    model = root / f"iter_{step:07d}"
    model.mkdir(parents=True)
    (model / "weights.pt").write_bytes(f"model-{root.name}-{step}".encode())
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    rollout = root / "rollout"
    rollout.mkdir(exist_ok=True)
    if queue:
        for name in ("queue_state", "builder_state"):
            (rollout / f"{name}_{step}.json").write_text(json.dumps({"test": name}))
        args = SimpleNamespace(save=str(root), rollout_data_dir=str(pool), rollout_queue_run_id="rollout")
        if checkpoint_args is not None:
            args._checkpoint_plan = checkpoint_args._checkpoint_plan
        commit_checkpoint(args, step, model_args=[args])


def _checkpoint_args(load, save, *, step=None):
    return SimpleNamespace(
        rollout_data_transport="straw",
        load=str(load) if load is not None else None,
        save=str(save),
        ckpt_step=step,
        start_rollout_id=None,
        rollout_data_dir=None,
        _rollout_queue_fork=False,
        _rollout_queue_resume=False,
    )


def test_automatic_checkpoint_branches_repeat_save_and_resume_current(tmp_path):
    import json
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root, pool = tmp_path / "run", tmp_path / "pool"
    _checkpoint_fixture(root, 1, pool)
    _checkpoint_fixture(root, 2, pool)
    old_weight = (root / "iter_0000002/weights.pt").read_bytes()
    branches = []
    for _ in range(2):
        args = _checkpoint_args(root, root, step=1)
        resolve_checkpoint(args)
        assert args.ckpt_step == 1 and args.start_rollout_id == 2
        assert args._rollout_queue_fork and not args._rollout_queue_resume
        destination = Path(args.save)
        assert destination.parent == root / "branches" and destination not in branches
        with closing(RolloutQueueController(args)):
            assert json.loads((root / "rollout/current.json").read_text())["directory"] == str(destination)
            _checkpoint_fixture(destination, 2, pool)
        branches.append(destination)
    assert (root / "iter_0000002/weights.pt").read_bytes() == old_weight
    # A newer but unfinished model write must not beat the last joint commit.
    (branches[-1] / "iter_0000003").mkdir()
    (branches[-1] / "latest_checkpointed_iteration.txt").write_text("3")
    resumed = _checkpoint_args(root, root)
    resolve_checkpoint(resumed)
    assert resumed.load == str(branches[-1]) and resumed.ckpt_step == 2
    assert resumed.save not in map(str, branches)
    assert resumed.rollout_data_dir == str(pool)


def test_manual_branch_and_commit_selection_exclude_abandoned_future(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root, pool = tmp_path / "run", tmp_path / "pool"
    _checkpoint_fixture(root, 1, pool)
    _checkpoint_fixture(root, 3, pool)
    fork = _checkpoint_args(root, root, step=1)
    resolve_checkpoint(fork)
    with closing(RolloutQueueController(fork)):
        _checkpoint_fixture(Path(fork.save), 2, pool)
    with pytest.raises(ValueError, match="No committed"):
        resolve_checkpoint(_checkpoint_args(root, tmp_path / "other", step=3))
    exact = _checkpoint_args(root / "rollout/committed_3.json", tmp_path / "manual")
    resolve_checkpoint(exact)
    assert exact.load == str(root) and exact.ckpt_step == 3
    branch = _checkpoint_args(fork.save, tmp_path / "branch", step=2)
    resolve_checkpoint(branch)
    assert branch.load == fork.save and branch.ckpt_step == 2
    with pytest.raises(ValueError, match="differs"):
        resolve_checkpoint(_checkpoint_args(root / "rollout/committed_3.json", tmp_path / "bad", step=1))


def test_edited_tracker_rolls_back_and_explicit_step_takes_precedence(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root, pool = tmp_path / "run", tmp_path / "pool"
    for step in (0, 1, 2):
        _checkpoint_fixture(root, step, pool)
    (root / "latest_checkpointed_iteration.txt").write_text("0")
    args = _checkpoint_args(root, root)
    resolve_checkpoint(args)
    assert args.ckpt_step == 0 and args.start_rollout_id == 1
    explicit = _checkpoint_args(root, root, step=1)
    resolve_checkpoint(explicit)
    assert explicit.ckpt_step == 1
    with closing(RolloutQueueController(explicit)):
        assert (root / "latest_checkpointed_iteration.txt").read_text() == "1"
        _checkpoint_fixture(Path(explicit.save), 2, pool, checkpoint_args=explicit)
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "2"
    # A normal restart selects the child, without treating its parent's model
    # tracker (at the same logical root) as an instruction to leave the branch.
    resumed = _checkpoint_args(root, root)
    resolve_checkpoint(resumed)
    assert resumed.ckpt_step == 2 and resumed.load == explicit.save
    (root / "latest_checkpointed_iteration.txt").write_text("1")
    edited = _checkpoint_args(root, root)
    resolve_checkpoint(edited)
    assert edited.ckpt_step == 1 and edited.load == str(root)
    # Users can also edit a physical branch's tracker.
    _checkpoint_fixture(Path(explicit.save), 3, pool, checkpoint_args=explicit)
    (Path(explicit.save) / "latest_checkpointed_iteration.txt").write_text("2")
    physical = _checkpoint_args(explicit.save, tmp_path / "physical")
    resolve_checkpoint(physical)
    assert physical.ckpt_step == 2
    (root / "latest_checkpointed_iteration.txt").write_text("3\n")
    through_logical_root = _checkpoint_args(root, root)
    resolve_checkpoint(through_logical_root)
    assert through_logical_root.ckpt_step == 2 and through_logical_root.load == explicit.save


def test_edited_tracker_with_missing_queue_snapshot_starts_empty(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root, pool = tmp_path / "run", tmp_path / "pool"
    _checkpoint_fixture(root, 1, pool, queue=False)
    _checkpoint_fixture(root, 2, pool)
    (root / "latest_checkpointed_iteration.txt").write_text("1")
    args = _checkpoint_args(root, root)
    resolve_checkpoint(args)
    assert args.ckpt_step == 1 and args._rollout_queue_empty_restore
    assert args.update_weight_start_version == 2
    (root / "latest_checkpointed_iteration.txt").write_text("0")
    with pytest.raises(ValueError, match="No committed"):
        resolve_checkpoint(_checkpoint_args(root, root))


def test_branch_without_first_commit_recovers_its_parent(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    parent, child = tmp_path / "parent", tmp_path / "child"
    _checkpoint_fixture(parent, 7, tmp_path / "pool")
    args = _checkpoint_args(parent, child, step=7)
    resolve_checkpoint(args)
    with closing(RolloutQueueController(args)):
        pass  # Crash before the child has produced a complete checkpoint.
    again = _checkpoint_args(child, child)
    resolve_checkpoint(again)
    assert again.load == str(parent) and again.ckpt_step == 7
    assert Path(again.save).parent == child / "branches"


def test_initial_run_recovers_wal_without_queue_flags(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "run"
    first = _checkpoint_args(None, root)
    first.hf_checkpoint = "initial-model"
    resolve_checkpoint(first)
    first.rollout_data_dir = str(root / "rollout_data")
    with closing(RolloutQueueController(first)):
        pass
    for load in (root, None):
        again = _checkpoint_args(load, root)
        again.hf_checkpoint = "initial-model"
        resolve_checkpoint(again)
        assert again._rollout_queue_resume and not again._rollout_queue_fork
        assert again._rollout_queue_id == first._rollout_queue_id
        assert again.load is None and again.save == first.save
        assert again.rollout_data_dir == first.rollout_data_dir
    changed = _checkpoint_args(root, root)
    changed.hf_checkpoint = "different-model"
    with pytest.raises(ValueError, match="Initial model/input"):
        resolve_checkpoint(changed)


def test_checkpoint_directory_lock_and_stale_selection(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "run"
    _checkpoint_fixture(root, 1, tmp_path / "pool")
    first, stale = (_checkpoint_args(root, root) for _ in range(2))
    resolve_checkpoint(first)
    resolve_checkpoint(stale)
    with closing(RolloutQueueController(first)):
        with pytest.raises(RuntimeError, match="Another training job"):
            with closing(RolloutQueueController(stale)):
                pytest.fail("Concurrent job acquired the save directory")
    with pytest.raises(RuntimeError, match="changed during startup"):
        with closing(RolloutQueueController(stale)):
            pytest.fail("Stale selection replaced the current branch")
    # Both failed constructors must release their descriptors, even while their
    # exceptions may still be retained by the caller.
    again = _checkpoint_args(root, root)
    resolve_checkpoint(again)
    with closing(RolloutQueueController(again)):
        pass


@pytest.mark.parametrize("stage", ["publication", "coordinator", "after_coordinator"])
def test_checkpoint_lock_released_when_controller_initialization_fails(tmp_path, monkeypatch, stage):
    from slime.rollout import queue_data_source
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "run"
    args = _checkpoint_args(None, root)
    resolve_checkpoint(args)

    def fail(*args, **kwargs):
        raise OSError("interrupted controller initialization")

    with monkeypatch.context() as patch:
        if stage == "publication":
            patch.setattr(queue_data_source, "write_report", fail)
        elif stage == "coordinator":
            patch.setattr(queue_data_source.Coordinator, "__init__", fail)
        else:
            patch.setattr(RolloutQueueController, "_start_gc", fail)
        with pytest.raises(OSError, match="interrupted controller initialization") as error:
            RolloutQueueController(args)
    # Keep the failed constructor's traceback alive: release cannot rely on GC.
    assert error.value.__traceback__ is not None
    again = _checkpoint_args(None, root)
    resolve_checkpoint(again)
    with closing(RolloutQueueController(again)):
        pass


def test_checkpoint_lock_covers_cleanup_and_releases_on_seal_failure(tmp_path, monkeypatch):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "run"
    args = _checkpoint_args(None, root)
    resolve_checkpoint(args)
    controller = RolloutQueueController(args)
    again = _checkpoint_args(None, root)
    resolve_checkpoint(again)

    def fail_seal():
        with pytest.raises(RuntimeError, match="Another training job"):
            RolloutQueueController(again)
        raise OSError("seal failed")

    with monkeypatch.context() as patch:
        patch.setattr(controller.store, "seal", fail_seal)
        with pytest.raises(OSError, match="seal failed"):
            controller.close()
    with closing(RolloutQueueController(again)):
        pass


def test_checkpoint_lock_released_when_controller_process_is_killed(tmp_path):
    import select
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "run"
    args = _checkpoint_args(None, root)
    resolve_checkpoint(args)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import pickle, sys
from slime.rollout.queue_data_source import RolloutQueueController

args = pickle.load(sys.stdin.buffer)
controller = RolloutQueueController(args)
print("ready", flush=True)
sys.stdin.buffer.read(1)
""",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        pickle.dump(args, process.stdin)
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 60)[0], "Controller did not start"
        assert process.stdout.readline() == b"ready\n"
        again = _checkpoint_args(None, root)
        resolve_checkpoint(again)
        with pytest.raises(RuntimeError, match="Another training job"):
            RolloutQueueController(again)
    finally:
        process.kill()
        process.wait(timeout=30)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
    # No close() ran in the killed process; the kernel must release its lock.
    with closing(RolloutQueueController(again)):
        pass


@pytest.mark.parametrize("has_cursor", [False, True])
def test_missing_queue_snapshot_starts_empty_without_touching_old_model(tmp_path, has_cursor, caplog):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "legacy"
    _checkpoint_fixture(root, 7, tmp_path / "pool", queue=False)
    cursor = root / "rollout/global_dataset_state_dict_7.pt"
    if has_cursor:
        torch.save({"sample_offset": 30}, cursor)
    before = (root / "iter_0000007/weights.pt").read_bytes()
    args = _checkpoint_args(root, root, step=7)
    resolve_checkpoint(args)
    assert args._rollout_queue_empty_restore and not args._rollout_queue_fork
    assert args._rollout_dataset_cursor == (str(cursor) if has_cursor else None)
    assert args.start_rollout_id == 8 and args.load == str(root)
    assert Path(args.save).parent == root / "branches"
    assert "new empty queue" in caplog.text
    if not has_cursor:
        assert "offset 0" in caplog.text
    with closing(RolloutQueueController(args)):
        pass
    resumed = _checkpoint_args(root, root)
    resolve_checkpoint(resumed)
    assert resumed._rollout_queue_empty_restore and resumed.ckpt_step == 7
    assert (root / "iter_0000007/weights.pt").read_bytes() == before


def test_missing_model_or_broken_queue_snapshot_never_falls_back(tmp_path):
    from slime.utils.rollout_checkpoint import resolve_checkpoint

    root = tmp_path / "legacy"
    with pytest.raises(FileNotFoundError, match="model checkpoint tracker"):
        resolve_checkpoint(_checkpoint_args(root, tmp_path / "absent", step=7))
    _checkpoint_fixture(root, 7, tmp_path / "pool", queue=False)
    with pytest.raises(FileNotFoundError, match="model checkpoint"):
        resolve_checkpoint(_checkpoint_args(root, tmp_path / "missing", step=8))
    (root / "rollout/queue_state_7.json").write_text("{}")
    with pytest.raises(ValueError, match="without a complete joint commit"):
        resolve_checkpoint(_checkpoint_args(root, tmp_path / "broken", step=7))
    _checkpoint_fixture(root, 9, tmp_path / "pool")
    (root / "rollout/queue_state_9.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(_checkpoint_args(root, tmp_path / "lost", step=9))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
