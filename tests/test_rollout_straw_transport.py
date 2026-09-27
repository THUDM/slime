import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from straw.errors import StorageUnavailable
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
    from pathlib import Path
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
