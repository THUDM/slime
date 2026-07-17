import importlib
from types import SimpleNamespace

import pytest

from slime.utils import accelerator

NUM_GPUS = 0


@pytest.mark.unit
def test_resolve_cuda_visible_device_id(monkeypatch):
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)

    assert accelerator.resolve_visible_device_id(4) == 0
    assert accelerator.resolve_visible_device_id(6.0) == 1
    assert accelerator.resolve_visible_device_id(1) == 1


@pytest.mark.unit
def test_resolve_musa_visible_device_id(monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert accelerator.visible_devices_env_key() == "MUSA_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1


@pytest.mark.unit
def test_invalid_visible_device_id_reports_active_environment(monkeypatch):
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=4,6"):
        accelerator.resolve_visible_device_id(7)


@pytest.mark.unit
def test_musa_process_group_backends(monkeypatch):
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: True)

    assert accelerator.process_group_backend() == "mccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,musa:mccl"
    assert accelerator.process_group_backend("gloo") == "gloo"


@pytest.mark.unit
def test_requested_musa_runtime_does_not_fall_back_to_nccl(monkeypatch):
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch.musa is unavailable"):
        accelerator.process_group_backend()


@pytest.mark.unit
def test_musa_patch_import_is_idempotent(monkeypatch):
    import_count = 0

    def import_patch_once():
        nonlocal import_count
        import_count += 1
        return True

    monkeypatch.setattr(accelerator, "_MUSA_PATCH_IMPORTED", False)
    monkeypatch.setattr(accelerator, "_import_musa_patch", import_patch_once)
    monkeypatch.setattr(accelerator, "is_musa_environment", lambda: True)

    assert accelerator._try_import_musa_patch()
    assert accelerator._try_import_musa_patch()
    assert import_count == 1


@pytest.mark.unit
def test_import_does_not_reapply_patch_after_import(monkeypatch):
    reapplied = False

    def patch_after_import_torch():
        nonlocal reapplied
        reapplied = True

    monkeypatch.setattr(
        accelerator.importlib,
        "import_module",
        lambda name: SimpleNamespace(patch_after_import_torch=patch_after_import_torch),
    )

    assert accelerator._import_musa_patch()
    assert not reapplied


@pytest.mark.unit
def test_cuda_environment_does_not_import_musa_patch(monkeypatch):
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUSA_PATCH_PATH", raising=False)
    monkeypatch.setattr(accelerator, "_MUSA_PATCH_IMPORTED", False)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(
        accelerator,
        "_import_musa_patch",
        lambda: (_ for _ in ()).throw(AssertionError("musa_patch must not be imported in a CUDA environment")),
    )

    assert not accelerator._try_import_musa_patch()


@pytest.mark.unit
def test_accelerator_import_bootstraps_musa_patch(monkeypatch):
    imported_modules = []

    with monkeypatch.context() as patch:
        patch.setenv("MUSA_VISIBLE_DEVICES", "0")
        patch.setattr(
            importlib,
            "import_module",
            lambda name: imported_modules.append(name) or SimpleNamespace(),
        )

        importlib.reload(accelerator)

    assert imported_modules == ["musa_patch"]
    importlib.reload(accelerator)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
