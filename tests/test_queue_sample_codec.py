import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from straw import SharedFilesystemStore
from straw.errors import CorruptData
from straw.protocol import decode
from straw.tensor import CHUNK_BYTES, publish_tensors

from slime.data.codec import CODECS, QueueTensorRef, SampleCodec
from slime.data.tensor import TensorRef
from slime.utils.score_centering import validate_sampler_topk
from slime.utils.types import Sample

NUM_GPUS = 0
# Independent v1 frame fixture for deliberate on-disk corruption: u32 metadata
# length, u64 payload length, both little-endian. Storage framing lives in Rust.
FRAME = struct.Struct("<IQ")


@pytest.fixture
def codec(tmp_path):
    return SampleCodec(SharedFilesystemStore(tmp_path / "run", "codec-test", codecs=CODECS))


def test_nested_groups_fanout_status_and_custom_fields_roundtrip(codec):
    sample = Sample(
        index=12,
        group_index=3,
        rollout_id=9,
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 0],
        reward={"score": 0.5},
        status=Sample.Status.ABORTED,
    )
    sample.custom_field = {"binary": b"\0\xff", "tuple": (2, 3), 5: float("-inf")}
    sample.weight_versions = ["policy-a", "policy-b"]
    sample.spec_info.completion_token_num = 2
    nested = [[sample, [Sample(index=13, group_index=3, rollout_id=9)]]]
    ref = codec.publish(nested, submission_id="group")
    restored = codec.load(ref)
    assert isinstance(restored[0][1], list)
    a = restored[0][0]
    assert vars(a) == vars(sample)
    assert a is not sample
    assert restored[0][1][0].rollout_id == 9


def test_partial_r3_sc_remain_lazy_across_repeated_buffer_snapshots(codec):
    routes = torch.arange(1024 * 48 * 8, dtype=torch.int32).reshape(1024, 48, 8)
    ids = torch.arange(128, dtype=torch.int32).repeat(1024, 1)
    logps = torch.full((1024, 128), -6.0)
    sample = Sample(
        tokens=[1] * 1025,
        response_length=1024,
        status=Sample.Status.ABORTED,
        rollout_routed_experts=[routes[:512], routes[512:]],
        rollout_topk_token_ids=ids.numpy(),
        rollout_topk_log_probs=logps.numpy(),
    )
    sample.custom_alias = logps
    first = codec.load(codec.publish(sample, submission_id="partial"))
    for key, expected in (
        ("rollout_routed_experts", routes),
        ("rollout_topk_token_ids", ids),
        ("rollout_topk_log_probs", logps),
    ):
        value = getattr(first, key)
        assert isinstance(value, QueueTensorRef) and not value.validated
        assert value.kind == key and torch.equal(value.load(), expected)
    assert torch.equal(first.custom_alias, logps)
    # Unrelated custom fields retain their eager type; remove this alias so
    # subsequent snapshots only contain references plus small sample metadata.
    del first.custom_alias
    paths = list(codec.store.backend.root.rglob("*.pack"))
    before = sum(p.stat().st_size for p in paths)
    restored = first
    for i in range(5):
        restored = codec.load(codec.publish(restored, submission_id=f"snapshot-{i}"))
    assert sum(p.stat().st_size for p in paths) - before < 128 * 1024
    assert restored.rollout_routed_experts.record_ref == first.rollout_routed_experts.record_ref
    restored._append_rollout_routed_experts_chunk(routes[:1])
    assert restored.get_rollout_routed_experts_length() == 1025
    assert torch.equal(restored.materialize_rollout_routed_experts()[-1:], routes[:1])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float32, torch.bfloat16, torch.bool])
def test_typed_tensors_empty_scalar_noncontiguous_and_aliases(codec, dtype):
    value = torch.arange(24).reshape(4, 6).T.to(dtype)
    scalar = torch.tensor(1, dtype=dtype)
    empty = torch.empty((0, 2), dtype=dtype)
    ref = codec.publish(
        {"tensor": value, "alias": value, "scalar": scalar, "empty": empty},
        submission_id="tensors",
    )
    restored = codec.load(ref)
    assert torch.equal(restored["tensor"], value)
    assert restored["tensor"] is restored["alias"]
    assert torch.equal(restored["scalar"], scalar)
    assert restored["empty"].shape == empty.shape
    assert all(item.dtype == dtype for item in restored.values())


def test_identical_tensor_payload_has_stable_identity_across_writers(codec):
    value = {"tensor": torch.arange(32).reshape(8, 4), "array": np.array([1.0, 2.0])}
    first = codec.publish(value, submission_id="same-request")
    other = SampleCodec(SharedFilesystemStore(codec.store.backend.root, codec.store.run_id, codecs=CODECS))
    second = other.publish(value, submission_id="same-request")
    assert first.manifest != second.manifest
    assert first.digest == second.digest
    value["tensor"][0, 0] += 1
    changed = SampleCodec(SharedFilesystemStore(codec.store.backend.root, codec.store.run_id, codecs=CODECS)).publish(
        value, submission_id="same-request"
    )
    assert changed.digest != first.digest


def test_numpy_and_multimodal_image(codec):
    from PIL import Image

    sample = Sample(
        multimodal_inputs={"image": Image.new("RGB", (3, 4), (10, 20, 30))},
        multimodal_train_inputs={"pixels": np.arange(12, dtype=np.float32).reshape(3, 4)},
    )
    ref = codec.publish([sample], submission_id="multimodal")
    (restored,) = codec.load(ref)
    assert restored.multimodal_inputs["image"].tobytes() == sample.multimodal_inputs["image"].tobytes()
    np.testing.assert_array_equal(
        restored.multimodal_train_inputs["pixels"],
        sample.multimodal_train_inputs["pixels"],
    )


def test_r3_sc_tensors_survive_source_pool_removal_and_remain_lazy(codec, tmp_path):
    routes = torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2)
    topk_ids = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)
    topk_logps = torch.tensor([[-0.5, -2], [-0.7, -2.5]], dtype=torch.float32)
    values = {
        "rollout_routed_experts": routes,
        "rollout_topk_token_ids": topk_ids,
        "rollout_topk_log_probs": topk_logps,
    }
    with SharedFilesystemStore(tmp_path / "source", "source", codecs=CODECS) as source:
        refs = dict(zip(values, publish_tensors(source, values, submission_id="source"), strict=True))
    sample = Sample(
        tokens=[0, 1, 2],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.5, -0.7],
        **refs,
    )
    ref = codec.publish([sample], submission_id="r3-sc")
    for path in {Path(old.path) for old in refs.values()}:
        path.unlink()
    (restored,) = codec.load(ref)
    for field, expected in values.items():
        actual = getattr(restored, field)
        assert isinstance(actual, QueueTensorRef)
        assert actual.validated
        assert torch.equal(actual.load(), expected)
        assert torch.equal(actual[1:], expected[1:])
    validate_sampler_topk(restored, 2)
    # A second batch manifest reuses queue-owned blobs without duplicating payloads.
    old_tensor_segments = {getattr(restored, field).record_ref.segment for field in values}
    second = codec.publish([restored], submission_id="training-batch")
    (again,) = codec.load(second)
    assert {getattr(again, field).record_ref.segment for field in values} == old_tensor_segments


def test_top_p_ragged_support_and_behavior_probabilities_stay_distinct(codec):
    sample = Sample(
        tokens=[8, 1, 3],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.2, -0.7],
        teacher_log_probs=[-0.3, -0.9],
        rollout_top_p_token_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2, 5], dtype=torch.int32),
        rollout_top_p_log_probs=torch.tensor([-0.2, -1.9, -0.7, -1.4, -2.0]),
    )
    (restored,) = codec.load(codec.publish([sample], submission_id="ragged"))
    assert restored.rollout_log_probs == sample.rollout_log_probs
    assert restored.teacher_log_probs == sample.teacher_log_probs
    assert torch.equal(restored.rollout_top_p_token_ids.load(), sample.rollout_top_p_token_ids)
    assert torch.equal(restored.rollout_top_p_token_offsets.load(), sample.rollout_top_p_token_offsets)
    assert torch.equal(restored.rollout_top_p_log_probs.load(), sample.rollout_top_p_log_probs)


def test_lazy_rows_verify_only_touched_chunks(codec, tmp_path):
    tensor = torch.arange(CHUNK_BYTES * 3, dtype=torch.int32).reshape(-1, 4)
    with SharedFilesystemStore(tmp_path / "source", "source", codecs=CODECS) as source:
        (old,) = publish_tensors(source, {"tensor": tensor}, submission_id="large")
    ref = codec.publish(old, submission_id="large")
    lazy = codec.load(ref)
    assert torch.equal(lazy[-2:], tensor[-2:])
    assert lazy[:0].shape == (0, 4)
    entry = codec.store.inspect_segment(lazy.record_ref.segment)["records"][lazy.record_ref.ordinal]
    path = codec.store.backend.path(lazy.record_ref.segment.path)
    with path.open("r+b") as stream:
        stream.seek(entry["offset"])
        envelope_len, _ = FRAME.unpack(stream.read(FRAME.size))
        stream.seek(envelope_len + CHUNK_BYTES, os.SEEK_CUR)
        stream.write(b"corrupt")
    assert torch.equal(lazy[:1], tensor[:1])
    with pytest.raises(CorruptData, match="chunk checksum"):
        lazy[CHUNK_BYTES // 16 : CHUNK_BYTES // 16 + 1]
    with pytest.raises(CorruptData):
        lazy.load()


def test_paths_in_codec_payload_are_relative(codec, tmp_path):
    with SharedFilesystemStore(tmp_path / "source", "source", codecs=CODECS) as source:
        (old,) = publish_tensors(source, {"tensor": torch.arange(8)}, submission_id="relative")
    ref = codec.publish(old, submission_id="relative")
    record = next(codec.store.read(ref))
    assert str(tmp_path).encode() not in record.payload
    assert decode(record.payload)["tree"][0] == "tensor"


def test_unknown_custom_fields_fail_without_silent_loss(codec):
    sample = Sample()
    sample.extra = object()
    with pytest.raises(TypeError, match="Unsupported durable Sample field"):
        codec.publish(sample, submission_id="unknown")
    sample.extra = sample
    with pytest.raises(TypeError, match="Cyclic"):
        codec.publish(sample, submission_id="cycle")


def test_batch_builder_preserves_group_rewards_masks_and_r3_sc_after_replay(codec, tmp_path):
    import copy

    from slime.data.batch_builder import BatchBuilder

    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        use_score_centering=True,
        score_centering_top_k=2,
        rollout_top_p=1.0,
        use_rollout_routing_replay=True,
        num_experts=16,
        num_layers=4,
        moe_router_topk=2,
    )
    # Two rollouts in one prompt group; rollout 7 has nonadjacent segments
    # sharing one outcome, which must count only once in reward statistics.
    segments = [(7, 1.0, [1, 0]), (8, 3.0, [1, 1]), (7, 1.0, [1, 1])]
    samples = []
    for index, (rollout_id, reward, loss_mask) in enumerate(segments):
        values = {
            "rollout_routed_experts": torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2),
            "rollout_topk_token_ids": torch.tensor([[1, 3], [2, 4]], dtype=torch.int32),
            "rollout_topk_log_probs": torch.tensor([[-0.5, -2], [-0.7, -2.5]], dtype=torch.float32),
        }
        refs = dict(zip(values, publish_tensors(codec.store, values, submission_id=f"source:{index}"), strict=True))
        samples.append(
            Sample(
                index=index,
                group_index=0,
                rollout_id=rollout_id,
                tokens=[0, 1, 2],
                response_length=2,
                loss_mask=loss_mask,
                reward=reward,
                rollout_log_probs=[-0.5, -0.7],
                teacher_log_probs=[-2.0, -3.0],
                status=Sample.Status.COMPLETED,
                **refs,
            )
        )
    ref = codec.publish([samples], submission_id="filtered-group")
    builder = BatchBuilder(args)
    expected = builder.convert(copy.deepcopy(samples))
    replayed = builder.convert(codec.load(ref)[0])

    def materialize(value):
        if isinstance(value, TensorRef):
            return value.load().tolist()
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, dict):
            return {key: materialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [materialize(item) for item in value]
        return value

    assert materialize(expected) == materialize(replayed)
    assert replayed["raw_reward"] == [1.0, 3.0, 1.0]
    assert replayed["rewards"] == [-1.0, 1.0, -1.0]
    assert replayed["loss_masks"] == [[1, 0], [1, 1], [1, 1]]
    assert replayed["rollout_mask_sums"] == [3, 2, 3]
    assert replayed["rollout_ids"] == [7, 8, 7]
    assert all(isinstance(ref, QueueTensorRef) for ref in replayed["rollout_routed_experts"])


@pytest.mark.parametrize("invalid_field", ["rollout_routed_experts", "rollout_topk_token_ids"])
def test_completed_capture_is_validated_before_publication(tmp_path, invalid_field):
    args = SimpleNamespace(
        use_rollout_routing_replay=True,
        use_score_centering=True,
        num_layers=2,
        moe_router_topk=2,
        num_experts=4,
        score_centering_top_k=2,
    )
    store = SharedFilesystemStore(tmp_path, "validate", codecs=CODECS)
    codec = SampleCodec(store, args=args)
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        rollout_routed_experts=torch.tensor([[[1, 2], [1, 2]]] * 2, dtype=torch.uint8),
        rollout_topk_token_ids=torch.tensor([[1, 2]], dtype=torch.int32),
        rollout_topk_log_probs=torch.tensor([[0.7, 0.3]]).log(),
    )
    setattr(sample, invalid_field, torch.zeros_like(getattr(sample, invalid_field)))
    with pytest.raises(ValueError):
        codec.publish([sample], submission_id="invalid")
    assert not list(tmp_path.rglob("*.pack"))


def test_large_group_bundle_streams_past_scratch_budget_and_preserves_tensor_aliases(tmp_path, monkeypatch):
    store = SharedFilesystemStore(
        tmp_path, "bounded", codecs=CODECS, max_buffer_bytes=32 * 1024, max_record_bytes=16 * 1024
    )
    codec = SampleCodec(store)
    values = []
    for i in range(12):
        tensor = torch.arange(2048, dtype=torch.int32) + i
        values.append({"first": tensor, "same": tensor, "second": tensor + 1})
    publish = store.publish_many
    calls = []

    def tracked(publications, **kwargs):
        size = sum(len(record.payload) for publication in publications for record in publication.records)
        calls.append(size)
        return publish(publications, **kwargs)

    monkeypatch.setattr(store, "publish_many", tracked)
    refs = codec.publish_many(values, submission_ids=[str(i) for i in range(len(values))])
    assert max(calls) > store.max_buffer_bytes
    for expected, ref in zip(values, refs, strict=True):
        actual = codec.load(ref)
        assert actual["first"] is actual["same"]
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])


@pytest.mark.parametrize("status", [Sample.Status.ABORTED, Sample.Status.COMPLETED])
def test_unvalidated_capture_roundtrip_never_narrows_values(codec, status):
    sample = Sample(
        status=status,
        rollout_topk_token_ids=torch.tensor([[2**40]], dtype=torch.int64),
        rollout_topk_log_probs=torch.tensor([[-0.123456789012345]], dtype=torch.float64),
    )
    restored = codec.load(codec.publish(sample, submission_id="unvalidated"))
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
        ref = getattr(restored, key)
        assert not ref.validated
        torch.testing.assert_close(ref.load(), getattr(sample, key), rtol=0, atol=0)


def test_training_projection_reads_only_selected_tensor_records(codec, monkeypatch):
    from slime.data.transport import TrainBatchRef

    ref = codec.publish(
        {
            "tokens": [torch.tensor([index, index + 1]) for index in range(8)],
            "loss_masks": [torch.ones(2, dtype=torch.int) for _ in range(8)],
            "total_lengths": [2] * 8,
            "unused": torch.zeros(10000),
        },
        submission_id="conversion",
    )
    loaded = []
    original = QueueTensorRef.load

    def tracked(value, *args, **kwargs):
        loaded.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(QueueTensorRef, "load", tracked)
    view = TrainBatchRef(
        ref,
        str(codec.store.backend.root),
        "batch",
        0,
        "layout",
        selection={"tokens": [7, 2], "loss_masks": [7, 2], "total_lengths": None},
        schedule={"partition": [7, 2], "num_microbatches": [1]},
    )
    restored = codec.load(codec.publish(view, submission_id="view"))
    data = restored.load()
    assert [tensor.tolist() for tensor in data["tokens"]] == [[7, 8], [2, 3]]
    assert len(loaded) == 4
    assert data["partition"] == [7, 2] and data["total_lengths"] == [2] * 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_batch_reader_rejects_another_storage_pool(codec, tmp_path):
    from straw.errors import InvalidReference

    ref = codec.publish({"value": 7}, submission_id="identity")
    other = SharedFilesystemStore(tmp_path / "other", codec.store.run_id, codecs=CODECS)
    try:
        with other.read_session() as reader:
            with pytest.raises(InvalidReference, match="different storage"):
                codec.load(ref, reader=reader)
    finally:
        other.close()
