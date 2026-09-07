import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from slime.utils import accelerator, memory_utils

NUM_GPUS = 0


class FakeAccelerator(accelerator.Accelerator):
    name = "fake"
    device_type = "fake"
    communication_backend_name = "fake"

    def is_available(self):
        return True

    def device(self, index=None):
        return accelerator.torch.device("cpu")

    def device_name(self, index=None):
        return "cpu"

    def set_device(self, index):
        return None

    def current_device(self):
        return "cpu"

    def device_count(self):
        return 0

    def synchronize(self, device=None):
        return None

    def current_stream(self, device=None):
        return None

    def empty_cache(self):
        return None

    def mem_get_info(self, device=None):
        return 0, 0

    def memory_allocated(self, device=None):
        return 0

    def memory_reserved(self, device=None):
        return 0


@pytest.fixture(autouse=True)
def reset_accelerator_selection(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    patch_imported = accelerator._MUSA_PATCH_IMPORTED
    bootstrap_checked = accelerator._MUSA_BOOTSTRAP_CHECKED
    for name in ("SLIME_ACCELERATOR", "MUSA_VISIBLE_DEVICES", "MUSA_PATCH_PATH", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    accelerator._MUSA_PATCH_IMPORTED = False
    accelerator._MUSA_BOOTSTRAP_CHECKED = False
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected
    accelerator._MUSA_PATCH_IMPORTED = patch_imported
    accelerator._MUSA_BOOTSTRAP_CHECKED = bootstrap_checked


@pytest.fixture
def peak_memory_module(monkeypatch):
    module = SimpleNamespace(
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=Mock(return_value=25 * 1024**3),
        max_memory_reserved=Mock(return_value=30 * 1024**3),
    )
    backend = SimpleNamespace(
        supports=lambda capability: capability == "peak_memory",
        accelerator_module=lambda: module,
    )
    monkeypatch.setattr(accelerator, "get_accelerator", lambda: backend)
    monkeypatch.setattr(memory_utils.dist, "get_rank", lambda: 3)
    return module


@pytest.mark.unit
def test_peak_memory_report_uses_selected_accelerator(peak_memory_module, caplog):
    with caplog.at_level(logging.INFO, logger="slime.utils.memory_utils"):
        with memory_utils.report_peak_memory("actor_train"):
            peak_memory_module.reset_peak_memory_stats.assert_called_once()

    assert caplog.messages[-1] == ("[Rank 3] Peak-Memory actor_train: max_allocated_GB=25.0, max_reserved_GB=30.0")


@pytest.mark.unit
def test_peak_memory_report_runs_when_body_raises(peak_memory_module, caplog):
    error = RuntimeError("device out of memory")

    with caplog.at_level(logging.INFO, logger="slime.utils.memory_utils"):
        with pytest.raises(RuntimeError) as exc_info:
            with memory_utils.report_peak_memory("log_probs"):
                raise error

    assert exc_info.value is error
    assert caplog.messages[-1].startswith("[Rank 3] Peak-Memory log_probs:")


@pytest.mark.unit
def test_peak_memory_report_skips_unsupported_accelerator(monkeypatch, caplog):
    backend = SimpleNamespace(supports=lambda _capability: False)
    monkeypatch.setattr(accelerator, "get_accelerator", lambda: backend)

    with caplog.at_level(logging.INFO, logger="slime.utils.memory_utils"):
        with memory_utils.report_peak_memory("actor_train"):
            pass

    assert "Peak-Memory" not in caplog.text


@pytest.mark.unit
def test_cuda_selection_does_not_bootstrap_musa(monkeypatch):
    monkeypatch.setenv("SLIME_ACCELERATOR", "cuda")
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    monkeypatch.setattr(
        accelerator,
        "_import_musa_patch",
        lambda: pytest.fail("CUDA selection must not import musa_patch"),
    )

    assert accelerator.get_accelerator().name == "cuda"
    assert accelerator.process_group_backend() == "nccl"
    assert accelerator.visible_devices_env_key() == "CUDA_VISIBLE_DEVICES"


@pytest.mark.unit
def test_selected_musa_bootstraps_patch_once(monkeypatch):
    imports = []
    fake_musa = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
    )

    def import_musa_patch():
        imports.append("musa_patch")
        monkeypatch.setattr(accelerator.torch, "musa", fake_musa, raising=False)
        return True

    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_import_musa_patch", import_musa_patch)

    assert imports == []
    assert accelerator.initialize_accelerator().name == "musa"
    assert accelerator.initialize_accelerator().name == "musa"
    assert accelerator.initialize_accelerator().supports("peak_memory")
    assert imports == ["musa_patch"]


@pytest.mark.unit
def test_cpu_only_initialization_does_not_require_an_accelerator(monkeypatch):
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)

    assert accelerator.initialize_accelerator() is None


@pytest.mark.unit
def test_musa_backend_maps_devices_and_process_groups(monkeypatch):
    monkeypatch.setattr(accelerator.MUSAAccelerator, "is_available", lambda self: True)
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "2,5")
    accelerator.set_accelerator(accelerator.MUSAAccelerator())

    assert accelerator.visible_devices_env_key() == "MUSA_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1
    assert accelerator.process_group_backend() == "mccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,musa:mccl"
    assert accelerator.process_group_backend("gloo") == "gloo"


@pytest.mark.unit
def test_cuda_visible_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    accelerator.set_accelerator(FakeAccelerator())

    assert accelerator.resolve_visible_device_id(4) == 0
    assert accelerator.resolve_visible_device_id(1) == 1
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=4,6"):
        accelerator.resolve_visible_device_id(7)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")
    assert accelerator.resolve_visible_device_id("GPU-bbbb") == 1


@pytest.mark.unit
def test_registered_backend_can_be_selected(monkeypatch):
    class RegisteredAccelerator(FakeAccelerator):
        name = "registered"

    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    accelerator.register_accelerator("registered", RegisteredAccelerator, lambda: True, priority=300)

    assert accelerator.get_accelerator().name == "registered"


@pytest.mark.unit
def test_cuda_backend_uses_torch_cuda_namespace(monkeypatch):
    monkeypatch.setattr(accelerator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(accelerator.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(accelerator.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(accelerator.torch.cuda, "memory_allocated", lambda device=None: 123)
    backend = accelerator.CUDAAccelerator()

    assert backend.is_available()
    assert backend.device_name() == "cuda:1"
    assert backend.memory_allocated() == 123
    assert backend.supports("peak_memory")


@pytest.mark.unit
def test_routing_replay_uses_selected_backend_current_device(monkeypatch):
    from slime.utils import routing_replay

    transfers = []

    class FakeTopIndices:
        def is_pinned(self):
            return False

        def to(self, device, *, dtype, non_blocking):
            transfers.append((device, dtype, non_blocking))
            return self

    accelerator.set_accelerator(FakeAccelerator())
    monkeypatch.setattr(routing_replay.RoutingReplay, "all_routing_replays", [])
    replay = routing_replay.RoutingReplay()
    replay.top_indices_list.append(FakeTopIndices())

    replay.pop_forward()
    replay.pop_backward()
    assert transfers == [
        ("cpu", accelerator.torch.int32, False),
        ("cpu", accelerator.torch.int32, False),
    ]


@pytest.mark.unit
def test_musa_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(accelerator.torch, "musa", raising=False)

    assert accelerator.is_musa_available() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
