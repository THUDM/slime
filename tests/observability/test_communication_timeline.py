import json
from types import SimpleNamespace

import pytest

from slime.observability import communication_timeline as timeline


class _Tensor:
    def __init__(self, elements, element_size):
        self._elements = elements
        self._element_size = element_size

    def numel(self):
        return self._elements

    def element_size(self):
        return self._element_size


@pytest.fixture(autouse=True)
def _reset_timeline(monkeypatch):
    timeline.close_communication_timeline()
    monkeypatch.delenv(timeline.COMMUNICATION_TIMELINE_ENV, raising=False)
    yield
    timeline.close_communication_timeline()


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_timeline_does_not_create_a_file(tmp_path):
    timeline.configure_communication_timeline(None)

    with timeline.communication_phase("train_forward_backward") as phase:
        phase.update(global_step=2)
        phase.mark_consumer()
    timeline.communication_event("weight_sync_complete", weight_version=1)

    assert list(tmp_path.iterdir()) == []


def test_span_and_event_emit_stable_shared_schema(tmp_path):
    path_template = str(tmp_path / "trainer-{rank}.jsonl")
    resolved = tmp_path / "trainer-2.jsonl"
    timeline.configure_communication_timeline(
        path_template,
        rank=2,
        local_rank=0,
        world_size=4,
        run_id="run-1",
    )

    with timeline.communication_context(global_step=7, rollout_id=3, trainer_rank=2):
        with timeline.communication_phase("weight_bucket_send", weight_version=5, bucket_id=1) as phase:
            phase.update(message_bytes=4096, engine_count=2)
            phase.mark_consumer()
        timeline.communication_event("weight_sync_complete", weight_version=5)
    timeline.close_communication_timeline()

    span, event = _read_jsonl(resolved)
    assert span["schema_version"] == timeline.COMMUNICATION_TIMELINE_VERSION
    assert span["framework"] == "slime"
    assert span["gpu_timestamp_semantics"] == "event-bracket"
    assert span["timestamp_domain"] == "process-realtime-projected-cuda-event"
    assert span["clock_sync_error_bound_us"] is None
    assert span["run_id"] == "run-1"
    assert span["operation"] == "weight_bucket_send"
    assert span["logical_operation_id"] == "7/3/5/1/weight_bucket_send"
    assert span["global_step"] == 7
    assert span["rollout_id"] == 3
    assert span["trainer_rank"] == 2
    assert span["message_bytes"] == 4096
    assert span["metadata"]["engine_count"] == 2
    assert span["consumer_timestamp_ns"] is not None
    assert span["completion_timestamp_ns"] >= span["api_launch_timestamp_ns"]
    assert span["duration_ns"] >= 0
    assert event["record_type"] == "event"
    assert event["operation"] == "weight_sync_complete"
    assert event["gpu_timestamp_semantics"] == "event-bracket"
    assert event["clock_sync_error_bound_us"] is None
    assert event["sequence_id"] == span["sequence_id"] + 1


def test_world_size_uses_rank_suffix_when_template_is_shared(tmp_path):
    configured = timeline.configure_communication_timeline(
        str(tmp_path / "timeline.jsonl"),
        rank=3,
        local_rank=1,
        world_size=4,
    )

    assert configured.path == tmp_path / "timeline.rank-3.jsonl"


def test_non_trainer_roles_get_separate_files_without_role_placeholder(tmp_path):
    configured = timeline.configure_communication_timeline(
        str(tmp_path / "timeline-{rank}.jsonl"),
        rank=0,
        world_size=2,
        role="critic",
    )

    assert configured.path == tmp_path / "timeline-0.role-critic.jsonl"


def test_failed_span_is_recorded_and_reraised(tmp_path):
    path = tmp_path / "failure.jsonl"
    timeline.configure_communication_timeline(str(path), rank=0, world_size=1)

    with pytest.raises(RuntimeError, match="boom"):
        with timeline.communication_phase("optimizer_step"):
            raise RuntimeError("boom")
    timeline.close_communication_timeline()

    record = _read_jsonl(path)[0]
    assert record["status"] == "error"
    assert record["metadata"]["error_type"] == "RuntimeError"
    assert record["metadata"]["error_message"] == "boom"


def test_iter_communication_buckets_times_only_produced_buckets(tmp_path):
    path = tmp_path / "buckets.jsonl"
    timeline.configure_communication_timeline(str(path), rank=0, world_size=1)
    chunks = [
        [("a", _Tensor(4, 2))],
        [("b", _Tensor(3, 4)), ("c", _Tensor(1, 1))],
    ]

    yielded = list(timeline.iter_communication_buckets(chunks))
    timeline.close_communication_timeline()

    assert [bucket_id for bucket_id, _ in yielded] == [0, 1]
    records = _read_jsonl(path)
    assert [record["bucket_id"] for record in records] == [0, 1]
    assert [record["message_bytes"] for record in records] == [8, 13]


def test_nested_spans_are_written_in_launch_sequence_order(tmp_path):
    path = tmp_path / "nested.jsonl"
    timeline.configure_communication_timeline(str(path), rank=0, world_size=1)

    with timeline.communication_phase("train_forward_backward"):
        timeline.communication_event("weight_bucket_ready")
        with timeline.communication_phase("grad_sync"):
            pass
    timeline.close_communication_timeline()

    records = _read_jsonl(path)
    assert [record["sequence_id"] for record in records] == [0, 1, 2]
    assert [record["operation"] for record in records] == [
        "train_forward_backward",
        "weight_bucket_ready",
        "grad_sync",
    ]


def test_cuda_events_are_resolved_without_changing_schema(tmp_path, monkeypatch):
    class FakeEvent:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            return None

        def query(self):
            return True

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            assert isinstance(other, FakeEvent)
            return 2.5

    class FakeNvtx:
        pushes = []
        pops = 0

        @classmethod
        def range_push(cls, name):
            cls.pushes.append(name)

        @classmethod
        def range_pop(cls):
            cls.pops += 1

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_stream=lambda: SimpleNamespace(cuda_stream=123),
        Event=FakeEvent,
        nvtx=FakeNvtx,
    )
    monkeypatch.setattr(timeline, "_optional_torch", lambda: SimpleNamespace(cuda=fake_cuda))
    path = tmp_path / "cuda.jsonl"
    timeline.configure_communication_timeline(str(path), rank=0, local_rank=0, world_size=1)

    with timeline.communication_phase("grad_sync"):
        pass
    timeline.close_communication_timeline()

    record = _read_jsonl(path)[0]
    assert record["stream_id"] == 123
    assert record["gpu_elapsed_ns"] == 2_500_000
    assert record["gpu_end_timestamp_ns"] - record["gpu_start_timestamp_ns"] == 2_500_000
    assert FakeNvtx.pushes == ["slime.comm/grad_sync"]
    assert FakeNvtx.pops == 1
