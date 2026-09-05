from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.backends.megatron_utils.update_weight.weight_sync_credit import WeightSyncCreditController

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def test_bucket_and_byte_credits_are_independent() -> None:
    controller = WeightSyncCreditController(max_inflight_buckets=2, max_inflight_bytes=10)
    controller.begin_version(7)

    first = controller.reserve(6)
    second = controller.reserve(4)
    assert first is not None
    assert second is not None
    assert controller.inflight_buckets == 2
    assert controller.inflight_bytes == 10
    assert controller.full
    assert controller.reserve(0) is None

    controller.release(first)
    third = controller.reserve(5)
    assert third is not None
    assert third.bucket_id == 2
    assert controller.inflight_buckets == 2
    assert controller.inflight_bytes == 9

    controller.release(second)
    controller.release(third)
    controller.commit_version(7)


@pytest.mark.parametrize(
    ("max_buckets", "max_bytes", "third_admitted"),
    [
        (2, 0, False),
        (0, 8, False),
        (0, 0, True),
    ],
)
def test_zero_disables_only_its_credit_dimension(max_buckets: int, max_bytes: int, third_admitted: bool) -> None:
    controller = WeightSyncCreditController(max_buckets, max_bytes)
    controller.begin_version(1)

    assert controller.reserve(4) is not None
    assert controller.reserve(4) is not None
    assert (controller.reserve(4) is not None) is third_admitted


def test_one_bucket_must_fit_the_byte_credit() -> None:
    controller = WeightSyncCreditController(max_inflight_bytes=8)
    controller.begin_version(1)

    with pytest.raises(ValueError, match="requires 9 bytes"):
        controller.reserve(9)


def test_release_is_fifo_and_commit_requires_a_drained_version() -> None:
    controller = WeightSyncCreditController(max_inflight_buckets=2)
    controller.begin_version(3)
    first = controller.reserve(4)
    second = controller.reserve(4)
    assert first is not None
    assert second is not None

    with pytest.raises(RuntimeError, match="must complete in order"):
        controller.release(second)
    with pytest.raises(RuntimeError, match=r"bucket\(s\) in flight"):
        controller.commit_version(3)

    controller.release(first)
    controller.release(second)
    controller.commit_version(3)

    with pytest.raises(ValueError, match="must be newer"):
        controller.begin_version(3)


def test_lifecycle_accounting_tracks_distinct_resources_and_duplicate_completion() -> None:
    controller = WeightSyncCreditController(max_inflight_buckets=2, max_inflight_bytes=10)
    controller.begin_version(5)
    first = controller.reserve(6)
    second = controller.reserve(4)
    assert first is not None
    assert second is not None

    controller.mark_launched(first, transport_bytes=6, staging_bytes=12, consumer_objects=2)
    controller.mark_launched(second, transport_bytes=4, staging_bytes=8, consumer_objects=1)
    snapshot = controller.snapshot
    assert snapshot.inflight_bytes == 10
    assert snapshot.transport_outstanding_bytes == 10
    assert snapshot.staging_resident_bytes == 20
    assert snapshot.pending_consumer_objects == 3
    assert snapshot.peak_staging_resident_bytes > snapshot.peak_inflight_bytes

    controller.mark_transport_complete(first)
    controller.mark_transport_complete(first)
    controller.mark_consumers_complete(first)
    controller.mark_consumers_complete(first)
    with pytest.raises(RuntimeError, match="before transport, consumers, and staging"):
        controller.release(first)
    controller.mark_staging_released(first)
    controller.mark_staging_released(first)
    controller.release(first)

    controller.mark_transport_complete(second)
    controller.mark_consumers_complete(second)
    controller.mark_staging_released(second)
    controller.release(second)
    controller.commit_version(5)

    metrics = controller.metrics()
    assert metrics["perf/update_weights_peak_inflight_buckets"] == 2
    assert metrics["perf/update_weights_peak_logical_inflight_bytes"] == 10
    assert metrics["perf/update_weights_peak_transport_outstanding_bytes"] == 10
    assert metrics["perf/update_weights_peak_staging_resident_bytes"] == 20
    assert metrics["perf/update_weights_peak_pending_consumer_objects"] == 3


def test_failed_version_is_poisoned_and_preserves_outstanding_evidence() -> None:
    controller = WeightSyncCreditController(max_inflight_buckets=1)
    controller.begin_version(8)
    reservation = controller.reserve(7)
    assert reservation is not None
    controller.mark_launched(reservation, transport_bytes=7, staging_bytes=11, consumer_objects=1)

    controller.fail_version(8, RuntimeError("consumer load failed"))
    controller.fail_version(8, RuntimeError("later wrapper error"))
    snapshot = controller.snapshot
    assert snapshot.failed_reason == "RuntimeError: consumer load failed"
    assert snapshot.transport_outstanding_bytes == 7
    assert snapshot.staging_resident_bytes == 11
    assert snapshot.pending_consumer_objects == 1
    with pytest.raises(RuntimeError, match="cannot commit failed weight version 8"):
        controller.commit_version(8)
    with pytest.raises(RuntimeError, match="weight version 8 is still active"):
        controller.begin_version(9)


def test_persistent_staging_must_be_released_before_version_commit() -> None:
    controller = WeightSyncCreditController()
    controller.begin_version(2)
    controller.set_persistent_staging_bytes(32)
    controller.set_persistent_staging_bytes(48)
    assert controller.snapshot.staging_resident_bytes == 48
    assert controller.snapshot.peak_staging_resident_bytes == 48
    with pytest.raises(RuntimeError, match="lifecycle resources still resident"):
        controller.commit_version(2)
    controller.set_persistent_staging_bytes(0)
    controller.commit_version(2)


def _load_distributed_update_module(monkeypatch):
    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod

    accelerator_mod = types.ModuleType("slime.utils.accelerator")
    accelerator_mod.current_device = lambda: "cpu"
    accelerator_mod.weight_update_backend = lambda: "gloo"
    distributed_utils_mod = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()
    distributed_utils_mod.init_process_group = lambda **_: object()
    http_utils_mod = types.ModuleType("slime.utils.http_utils")
    http_utils_mod._wrap_ipv6 = lambda address: address
    converter_mod = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    converter_mod.convert_to_hf = lambda *args, **kwargs: []
    common_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common_mod.all_gather_param = lambda _name, param: param
    common_mod.named_params_and_buffers = lambda *_: []

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.accelerator", accelerator_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", distributed_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", http_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.megatron_to_hf", converter_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.update_weight.common", common_mod)

    module_name = "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module_path = (
        REPO_ROOT / "slime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_distributed.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_distributed_bucket_launch_returns_before_collective_wait(monkeypatch) -> None:
    module = _load_distributed_update_module(monkeypatch)
    events = []

    class _Tensor:
        dtype = "float32"
        shape = (4,)
        data = "payload"

    class _Handle:
        def wait(self):
            events.append("wait")

    class _RemoteMethod:
        def remote(self, **kwargs):
            events.append(("metadata", kwargs))
            return "engine-ref"

    engine = types.SimpleNamespace(update_weights_from_distributed=_RemoteMethod())
    monkeypatch.setattr(
        module.dist,
        "broadcast",
        lambda tensor, src, group, async_op: events.append(("broadcast", tensor, src, group, async_op)) or _Handle(),
    )

    refs, handles = module.launch_weights_from_distributed(
        "weights",
        "group",
        9,
        [engine],
        [("layer.weight", _Tensor())],
    )

    assert refs == ["engine-ref"]
    assert len(handles) == 1
    assert [event[0] for event in events] == ["metadata", "broadcast"]
    assert events[0][1]["weight_version"] == "9"
    handles[0].wait()
    assert events[-1] == "wait"


def test_distributed_credit_window_holds_one_lock_and_waits_fifo(monkeypatch) -> None:
    module = _load_distributed_update_module(monkeypatch)
    events = []

    class _Handle:
        def __init__(self, name):
            self.name = name

        def wait(self):
            events.append(("wait", self.name))

    class _RemoteMethod:
        def __init__(self, name, result):
            self.name = name
            self.result = result

        def remote(self):
            events.append((self.name,))
            return self.result

    updater = object.__new__(module.UpdateWeightFromDistributed)
    updater._group_name = "weights"
    updater._model_update_groups = [module.DistributedWeightUpdateGroup((0,), "weights", "group", (object(),))]
    updater.weight_version = 13
    updater.rollout_engines = [object()]
    updater.rollout_engine_lock = types.SimpleNamespace(
        acquire=_RemoteMethod("acquire", True),
        release=_RemoteMethod("release", None),
    )
    updater._weight_sync_credit = WeightSyncCreditController(max_inflight_buckets=2)
    updater._weight_sync_credit.begin_version(13)
    first = updater._weight_sync_credit.reserve(4)
    second = updater._weight_sync_credit.reserve(4)
    assert first is not None
    assert second is not None
    first_bucket = [("b0", object())]
    second_bucket = [("b1", object())]

    def launch(_group_name, _group, _version, _engines, bucket):
        name = bucket[0][0]
        events.append(("launch", name))
        return [f"ref-{name}"], [_Handle(name)]

    def ray_get(value):
        if isinstance(value, list):
            events.append(("ray_get", value[0]))
        return value

    monkeypatch.setattr(module, "launch_weights_from_distributed", launch)
    module.ray.get = ray_get

    updater._flush_weight_bucket_window(
        [(first, first_bucket), (second, second_bucket)],
        pbar=None,
    )
    updater._weight_sync_credit.commit_version(13)

    assert events == [
        ("acquire",),
        ("launch", "b0"),
        ("launch", "b1"),
        ("wait", "b0"),
        ("ray_get", "ref-b0"),
        ("wait", "b1"),
        ("ray_get", "ref-b1"),
        ("release",),
    ]
    assert first_bucket == []
    assert second_bucket == []


def _gloo_credit_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        controller = WeightSyncCreditController(max_inflight_buckets=2, max_inflight_bytes=40)
        controller.begin_version(11)
        pending = []
        received = []
        peak_buckets = 0
        peak_bytes = 0

        for bucket_id, elements in enumerate((4, 6, 2, 8)):
            bucket_bytes = elements * torch.tensor([], dtype=torch.float32).element_size()
            reservation = controller.reserve(bucket_bytes)
            if reservation is None:
                oldest_reservation, oldest_handle = pending.pop(0)
                oldest_handle.wait()
                controller.release(oldest_reservation)
                reservation = controller.reserve(bucket_bytes)
            assert reservation is not None

            tensor = torch.full((elements,), float(bucket_id + 1)) if rank == 0 else torch.zeros(elements)
            handle = dist.broadcast(tensor, src=0, async_op=True)
            pending.append((reservation, handle))
            received.append(tensor)
            peak_buckets = max(peak_buckets, controller.inflight_buckets)
            peak_bytes = max(peak_bytes, controller.inflight_bytes)

        while pending:
            reservation, handle = pending.pop(0)
            handle.wait()
            controller.release(reservation)
        controller.commit_version(11)

        assert peak_buckets == 2
        assert peak_bytes == 40
        for bucket_id, tensor in enumerate(received):
            torch.testing.assert_close(tensor, torch.full_like(tensor, float(bucket_id + 1)))
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_credit_window_drives_real_async_gloo_broadcasts() -> None:
    mp.spawn(_gloo_credit_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_wave_transition_keeps_bucket_credit_and_requires_every_resource_drained() -> None:
    controller = WeightSyncCreditController(max_inflight_buckets=1, max_inflight_bytes=8)
    controller.begin_version(11)
    reservation = controller.reserve(8)
    controller.mark_launched(reservation, transport_bytes=8, staging_bytes=12, consumer_objects=1)
    for release in (
        controller.mark_transport_complete,
        controller.mark_consumers_complete,
        controller.mark_staging_released,
    ):
        with pytest.raises(RuntimeError, match="cannot advance a weight wave"):
            controller.mark_next_wave(reservation, transport_bytes=8, staging_bytes=0, consumer_objects=2)
        release(reservation)
    assert controller.reserve(0) is None
    controller.mark_next_wave(reservation, transport_bytes=16, staging_bytes=0, consumer_objects=2)
    assert controller.snapshot.inflight_bytes == 8
    assert controller.snapshot.transport_outstanding_bytes == 16
    controller.mark_transport_complete(reservation)
    controller.mark_consumers_complete(reservation)
    with pytest.raises(ValueError, match="non-negative"):
        controller.mark_next_wave(reservation, transport_bytes=-1, staging_bytes=0, consumer_objects=0)
    controller.release(reservation)
    controller.commit_version(11)


@pytest.fixture(params=[False, True], ids=["trace-off", "trace-on"])
def capture_updater_trace(request, tmp_path, monkeypatch):
    from slime.observability import communication_timeline as timeline

    timeline.close_communication_timeline()
    monkeypatch.delenv(timeline.COMMUNICATION_TIMELINE_ENV, raising=False)
    monkeypatch.setattr(timeline, "_optional_torch", lambda: None)
    path = tmp_path / "logical-updater.jsonl" if request.param else None
    if path is not None:
        timeline.configure_communication_timeline(str(path), rank=0, world_size=1, run_id="logical-updater-test")
    yield timeline, path
    timeline.close_communication_timeline()


@pytest.mark.parametrize("engine_count", [2, 4])
@pytest.mark.parametrize("fail_load", [False, True])
def test_production_updater_composes_waves_credits_and_publication(
    monkeypatch, engine_count, fail_load, capture_updater_trace
):
    module = _load_distributed_update_module(monkeypatch)
    events = []

    class Remote:
        def __init__(self, name, result=None):
            self.name, self.result = name, result

        def remote(self):
            events.append(self.name)
            return self.result

    engines = [
        types.SimpleNamespace(
            pause_generation=Remote("pause"),
            flush_cache=Remote("flush"),
            continue_generation=Remote("resume"),
        )
        for _ in range(engine_count)
    ]
    updater = module.UpdateWeightFromDistributed(
        types.SimpleNamespace(
            update_weight_max_inflight_buckets=2,
            update_weight_max_inflight_bytes=32,
            update_weight_max_inflight_engine_groups=1,
        ),
        [],
        lambda: {},
        model_name="logical-integration-fixture",
        quantization_config=None,
    )
    updater.rollout_engines = engines
    updater._is_pp_src_rank = False
    updater._model_update_groups = [
        module.DistributedWeightUpdateGroup((i,), f"engine-{i}", i, (engine,)) for i, engine in enumerate(engines)
    ]
    updater.rollout_engine_lock = types.SimpleNamespace(acquire=Remote("lock", True), release=Remote("unlock"))
    updater._iter_non_expert_chunks = lambda: iter([[(f"bucket-{i}", torch.ones(4))] for i in range(3)])
    updater._iter_expert_chunks = lambda: iter(())
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **_: None)

    def launch(_name, group, version, _engines, tensors, **_kwargs):
        assert version == 1
        bucket = tensors[0][0]
        assert updater._weight_sync_credit.snapshot.active_version == 1
        events.append(("launch", bucket, group))
        handle = types.SimpleNamespace(wait=lambda: events.append(("wait", bucket, group)))
        return [(bucket, group)], [handle]

    def ray_get(refs):
        if isinstance(refs, list) and refs and isinstance(refs[0], tuple):
            bucket, group = refs[0]
            snapshot = updater._weight_sync_credit.snapshot
            assert snapshot.transport_outstanding_bytes == 0
            assert snapshot.pending_consumer_objects == 1
            assert snapshot.inflight_buckets > 0
            events.append(("load", bucket, group))
            if fail_load and group == 1:
                raise RuntimeError("injected engine load failure")
        return refs

    monkeypatch.setattr(module, "launch_weights_from_distributed", launch)
    module.ray.get = ray_get
    if fail_load:
        with pytest.raises(RuntimeError, match="injected engine load failure"):
            updater.update_weights()
        snapshot = updater._weight_sync_credit.snapshot
        assert snapshot.active_version == 1
        assert snapshot.failed_reason is not None
        assert snapshot.pending_consumer_objects == 1
        assert updater._active_wave_resources[0][0][0] == "bucket-0"
        assert "resume" not in events
        assert not any(event[:2] == ("launch", "bucket-1") for event in events if isinstance(event, tuple))
    else:
        updater.update_weights()
        snapshot = updater._weight_sync_credit.snapshot
        assert snapshot.active_version is None
        assert snapshot.inflight_buckets == 0
        assert snapshot.peak_inflight_buckets == 2
        assert snapshot.peak_inflight_bytes == 32
        assert snapshot.peak_pending_consumer_objects == 1
        assert snapshot.peak_transport_outstanding_bytes == 16
        assert [event for event in events if isinstance(event, tuple)] == [
            (phase, f"bucket-{bucket}", group)
            for bucket in range(3)
            for group in range(engine_count)
            for phase in ("launch", "wait", "load")
        ]
        assert events.count("resume") == engine_count

    timeline, path = capture_updater_trace
    timeline.close_communication_timeline()
    if path is not None:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        sequence_ids = [row["sequence_id"] for row in records]
        assert sequence_ids == sorted(set(sequence_ids))
        # Each exhausted conversion iterator cancels its terminal span. These
        # are global trace sequence IDs, not communicator collective sequence.
        assert sequence_ids[-1] + 1 - len(records) == (0 if fail_load else 2)
        assert all(row["weight_version"] == 1 for row in records)
        sends = [row for row in records if row["operation"] == "weight_bucket_send"]
        assert len(sends) == (2 if fail_load else 3 * engine_count)
        assert len({row["logical_operation_id"] for row in sends}) == len(sends)
        assert all(row["gpu_start_timestamp_ns"] is None for row in records)
        assert all(
            row["api_launch_timestamp_ns"] <= row["api_return_timestamp_ns"] <= row["completion_timestamp_ns"]
            for row in sends
        )
        if fail_load:
            assert records[-1]["operation"] == "weight_sync_failed"
            assert any(row["status"] == "error" and row["operation"] == "engine_load_weights" for row in records)
            assert not any(row["operation"] in {"weight_sync_complete", "weight_bucket_reusable"} for row in records)
        else:
            assert records[-1]["operation"] == "weight_sync_complete"
            assert sum(row["operation"] == "weight_bucket_reusable" for row in records) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
