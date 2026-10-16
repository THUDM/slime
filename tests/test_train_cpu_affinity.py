import builtins
import ctypes
import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
import torch

from slime.utils import accelerator
from slime.utils.numa_utils import _set_cpu_affinity

NUM_GPUS = 0
GPU_UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(8)]


@pytest.fixture
def devices(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(accelerator, "_ACCELERATOR", accelerator.CUDAAccelerator())
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("SLIME_ENABLE_EXPANDABLE_SEGMENTS", "0")
    state = SimpleNamespace(current=4, cpus={0, 1, 2, 3}, order=list(GPU_UUIDS))
    preferred = {uuid: {0, 1} if i < 4 else {2, 3} for i, uuid in enumerate(GPU_UUIDS)}

    def set_device(index):
        state.current = index

    def properties(index):
        assert index == state.current
        return SimpleNamespace(uuid=state.order[index])

    def get_affinity(pid):
        assert pid == 0
        return state.cpus.copy()

    def set_affinity(pid, cpus):
        assert pid == 0
        state.cpus = set(cpus)

    def cpu_mask(handle, count):
        bits = ctypes.sizeof(ctypes.c_ulonglong) * 8
        return (ctypes.c_ulonglong * count)(
            *(sum(1 << (cpu % bits) for cpu in preferred[handle] if cpu // bits == i) for i in range(count))
        )

    monkeypatch.setattr(torch.cuda, "set_device", Mock(side_effect=set_device))
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=lambda: state.current))
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(side_effect=properties))
    monkeypatch.setattr(os, "sched_getaffinity", Mock(side_effect=get_affinity), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", Mock(side_effect=set_affinity), raising=False)
    nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(side_effect=lambda uuid: uuid),
        nvmlDeviceIsMigDeviceHandle=Mock(return_value=ctypes.c_uint(0)),
        nvmlDeviceGetCpuAffinity=Mock(side_effect=cpu_mask),
        nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda index: GPU_UUIDS[index]),
        nvmlDeviceSetCpuAffinity=Mock(side_effect=lambda handle: set_affinity(0, preferred[handle])),
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    return SimpleNamespace(state=state, preferred=preferred, nvml=nvml)


@pytest.fixture
def actor_module(monkeypatch):
    # The CPU CI image omits W&B; actor startup only needs its import to succeed.
    loaded = set(sys.modules)
    monkeypatch.setitem(sys.modules, "wandb", ModuleType("wandb"))
    from slime.observability import logging_utils
    from slime.ray import train_actor

    monkeypatch.setattr(logging_utils, "_LOGGER_CONFIGURED", False)
    monkeypatch.setattr(logging, "basicConfig", Mock())
    for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(train_actor.dist, "init_process_group", Mock())
    monkeypatch.setattr(train_actor.dist, "new_group", Mock())
    monkeypatch.setattr(train_actor.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(train_actor.dist, "get_world_size", lambda: 1)
    from slime.utils import distributed_utils

    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
    yield train_actor
    for name in ("slime.ray.train_actor", "slime.observability.logging_utils", "slime.observability.wandb_utils"):
        if name not in loaded:
            module = sys.modules.pop(name)
            parent, _, attr = name.rpartition(".")
            if getattr(sys.modules[parent], attr, None) is module:
                delattr(sys.modules[parent], attr)


@pytest.mark.parametrize(
    ("visible", "ray_id", "cuda_order", "ordinal", "gpu"),
    [
        pytest.param(None, 0, list(range(8)), 0, 0, id="contiguous"),
        pytest.param(None, 4, list(range(8)), 4, 4, id="rank-zero-gpu-four"),
        pytest.param("4,6", 4, [4, 6], 0, 4, id="partial-visibility"),
        pytest.param("6,4", 4, [6, 4], 1, 4, id="reordered-visibility"),
        pytest.param(None, 0, [4, 6, 0, 1, 2, 3, 5, 7], 0, 4, id="different-nvml-order"),
    ],
)
def test_actor_affinity_follows_selected_device(
    monkeypatch, devices, actor_module, visible, ray_id, cuda_order, ordinal, gpu
):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    devices.state.order = [GPU_UUIDS[i] for i in cuda_order]
    devices.state.current = None
    monkeypatch.setattr(actor_module.ray, "get_gpu_ids", lambda: [ray_id])
    actor = actor_module.TrainRayActor(1, 0, "127.0.0.1", 29500)
    args = SimpleNamespace(distributed_backend="gloo", distributed_timeout_minutes=1, num_gpus_per_node=8)

    actor.init(args, "actor")

    assert devices.state.current == ordinal
    assert devices.state.cpus == devices.preferred[GPU_UUIDS[gpu]]
    torch.cuda.get_device_properties.assert_called_once_with(ordinal)
    devices.nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(GPU_UUIDS[gpu])
    devices.nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
    devices.nvml.nvmlDeviceSetCpuAffinity.assert_not_called()
    devices.nvml.nvmlInit.assert_called_once_with()
    devices.nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "device_uuid",
    [
        GPU_UUIDS[4],
        GPU_UUIDS[4].encode("ascii"),
        UUID(GPU_UUIDS[4][4:]).bytes,
        SimpleNamespace(bytes=list(UUID(GPU_UUIDS[4][4:]).bytes)),
    ],
    ids=["text", "ascii-bytes", "binary-bytes", "pytorch-cuuuid"],
)
def test_supported_uuid_forms(devices, device_uuid, caplog):
    torch.cuda.get_device_properties.side_effect = None
    torch.cuda.get_device_properties.return_value = SimpleNamespace(uuid=device_uuid)

    _set_cpu_affinity(7)

    devices.nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(GPU_UUIDS[4])
    os.sched_setaffinity.assert_called_once_with(0, {2, 3})
    assert len(caplog.records) == 1
    assert "rank=7 tid=" in caplog.text
    assert f"cuda=4 uuid={GPU_UUIDS[4]} before=0-3 after=2-3 applied" in caplog.text


def test_existing_affinity_is_never_widened(devices):
    devices.state.cpus = {1, 3}

    _set_cpu_affinity(0)

    os.sched_setaffinity.assert_called_once_with(0, {3})
    assert devices.state.cpus == {3}
    devices.nvml.nvmlDeviceSetCpuAffinity.assert_not_called()


def test_empty_intersection_leaves_affinity_unchanged(devices, caplog):
    devices.state.cpus = {0, 1}

    _set_cpu_affinity(0)

    os.sched_setaffinity.assert_not_called()
    assert devices.state.cpus == {0, 1}
    assert "skipped: GPU CPU mask does not intersect existing affinity" in caplog.text
    devices.nvml.nvmlShutdown.assert_called_once_with()


def test_sparse_cpu_ids_size_the_mask(monkeypatch, devices):
    bits = ctypes.sizeof(ctypes.c_ulonglong) * 8
    devices.state.cpus = {1, bits - 1, bits + 2, 3 * bits + 5}
    devices.preferred[GPU_UUIDS[4]] = {0, bits - 1, bits + 2, 3 * bits + 5, 5 * bits}
    monkeypatch.setattr(os, "cpu_count", lambda: 2)

    _set_cpu_affinity(0)

    devices.nvml.nvmlDeviceGetCpuAffinity.assert_called_once_with(GPU_UUIDS[4], 4)
    os.sched_setaffinity.assert_called_once_with(0, {bits - 1, bits + 2, 3 * bits + 5})


def test_logs_effective_mask_after_cpuset_change(devices, caplog):
    os.sched_getaffinity.side_effect = [{0, 1, 2, 3}, {3}]

    _set_cpu_affinity(0)

    os.sched_setaffinity.assert_called_once_with(0, {2, 3})
    assert os.sched_getaffinity.call_count == 2
    assert "before=0-3 after=3 applied" in caplog.text


@pytest.mark.parametrize(
    ("operation", "shutdown", "reason"),
    [
        ("nvmlInit", False, "NVML import/initialization"),
        ("nvmlDeviceGetHandleByUUID", True, "NVML UUID lookup"),
        ("nvmlDeviceIsMigDeviceHandle", True, "NVML MIG check"),
        ("nvmlDeviceGetCpuAffinity", True, "GPU CPU mask query"),
        ("sched_getaffinity", False, "selected device identity"),
        ("get_device_properties", False, "selected device identity"),
        ("sched_setaffinity", True, "sched_setaffinity"),
    ],
)
def test_failures_are_nonfatal(devices, caplog, operation, shutdown, reason):
    owner = devices.nvml if operation.startswith("nvml") else os if operation.startswith("sched") else torch.cuda
    getattr(owner, operation).side_effect = OSError("injected failure")

    _set_cpu_affinity(0)

    assert devices.state.cpus == {0, 1, 2, 3}
    assert devices.nvml.nvmlShutdown.call_count == int(shutdown)
    assert len(caplog.records) == 1
    assert f"skipped: {reason} failed (OSError): injected failure" in caplog.text


def test_missing_nvml_is_nonfatal(monkeypatch, devices, caplog):
    monkeypatch.setitem(sys.modules, "pynvml", None)

    _set_cpu_affinity(0)

    os.sched_setaffinity.assert_not_called()
    devices.nvml.nvmlShutdown.assert_not_called()
    assert "NVML import/initialization failed" in caplog.text


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_shutdown_error_preserves_original_result(devices, caplog, lookup_fails):
    if lookup_fails:
        devices.nvml.nvmlDeviceGetHandleByUUID.side_effect = OSError("lookup failed")
    devices.nvml.nvmlShutdown.side_effect = OSError("shutdown failed")

    _set_cpu_affinity(0)

    assert len(caplog.records) == 1
    assert "NVML shutdown failed" in caplog.text
    assert ("lookup failed" if lookup_fails else "applied: effective mask read back") in caplog.text


def test_readback_failure_does_not_claim_original_mask(devices, caplog):
    os.sched_getaffinity.side_effect = [{0, 1, 2, 3}, OSError("readback failed")]

    _set_cpu_affinity(0)

    assert devices.state.cpus == {2, 3}
    assert "after=unknown applied: affinity readback failed" in caplog.text
    devices.nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "unsupported", ["rocm", "musa", "npu", "supa", "os", "sched_getaffinity", "sched_setaffinity"]
)
def test_unsupported_environment_has_no_gpu_side_effects(monkeypatch, devices, caplog, unsupported):
    if unsupported == "rocm":
        monkeypatch.setattr(torch.version, "hip", "6.4")
    elif unsupported in {"musa", "npu", "supa"}:
        backend = getattr(accelerator, f"{unsupported.upper()}Accelerator")()
        monkeypatch.setattr(accelerator, "_ACCELERATOR", backend)
    elif unsupported == "os":
        monkeypatch.setattr(sys, "platform", "darwin")
    else:
        monkeypatch.delattr(os, unsupported)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pynvml":
            pytest.fail("unsupported environments must not import NVML")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    _set_cpu_affinity(0)

    torch.cuda.current_device.assert_not_called()
    torch.cuda.get_device_properties.assert_not_called()
    assert devices.state.cpus == {0, 1, 2, 3}
    for name in ("sched_getaffinity", "sched_setaffinity"):
        if hasattr(os, name):
            getattr(os, name).assert_not_called()
    devices.nvml.nvmlInit.assert_not_called()
    assert len(caplog.records) == 1
    assert "skipped:" in caplog.text


@pytest.mark.parametrize("device_uuid", [None, object(), "4", "GPU-abcd", b"\xff", SimpleNamespace(bytes=[1, 2])])
def test_unknown_identity_is_not_guessed(devices, caplog, device_uuid):
    torch.cuda.get_device_properties.side_effect = None
    torch.cuda.get_device_properties.return_value = SimpleNamespace(uuid=device_uuid)

    _set_cpu_affinity(0)

    os.sched_setaffinity.assert_not_called()
    devices.nvml.nvmlInit.assert_not_called()
    assert "skipped:" in caplog.text


@pytest.mark.parametrize("uuid_prefix", [True, False])
def test_mig_is_explicitly_skipped(devices, caplog, uuid_prefix):
    if uuid_prefix:
        devices.state.order[4] = "MIG-" + GPU_UUIDS[4][4:]
    else:
        devices.nvml.nvmlDeviceIsMigDeviceHandle.return_value = ctypes.c_uint(1)

    _set_cpu_affinity(0)

    os.sched_setaffinity.assert_not_called()
    devices.nvml.nvmlDeviceGetCpuAffinity.assert_not_called()
    assert devices.nvml.nvmlShutdown.call_count == int(not uuid_prefix)
    assert "skipped: MIG affinity is unsupported" in caplog.text


def test_repeated_startup_uses_fresh_identity_and_nvml_session(devices):
    _set_cpu_affinity(0)
    devices.state.current = 6
    devices.preferred[GPU_UUIDS[6]] = {3}

    _set_cpu_affinity(0)

    assert devices.state.cpus == {3}
    assert [call.args[0] for call in devices.nvml.nvmlDeviceGetHandleByUUID.call_args_list] == GPU_UUIDS[4:7:2]
    assert devices.nvml.nvmlInit.call_count == devices.nvml.nvmlShutdown.call_count == 2


def test_actor_distributed_failure_is_not_hidden(devices, actor_module):
    actor_module.dist.init_process_group.side_effect = RuntimeError("distributed startup failed")
    actor = actor_module.TrainRayActor.__new__(actor_module.TrainRayActor)
    actor._rank = 0
    args = SimpleNamespace(distributed_backend="gloo", distributed_timeout_minutes=1)

    with pytest.raises(RuntimeError, match="distributed startup failed"):
        actor.init(args, "actor")

    devices.nvml.nvmlInit.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
