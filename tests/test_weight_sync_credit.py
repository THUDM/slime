from __future__ import annotations

import importlib.util
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
    updater._model_update_groups = "group"
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
