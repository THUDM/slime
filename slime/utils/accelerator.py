import importlib
import logging
import os
import sys
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_MUSA_PATCH_IMPORTED = False


def _append_musa_patch_path() -> None:
    patch_path = os.environ.get("MUSA_PATCH_PATH")
    if patch_path and patch_path not in sys.path:
        sys.path.append(patch_path)


def _import_musa_patch() -> bool:
    _append_musa_patch_path()
    try:
        importlib.import_module("musa_patch")
    except ImportError:
        return False
    except Exception as exc:
        logger.warning("Failed to import musa_patch: %s", exc)
        return False
    return True


import torch


def _musa() -> ModuleType | None:
    return getattr(torch, "musa", None)


def is_musa_available() -> bool:
    musa = _musa()
    return bool(musa is not None and getattr(musa, "is_available", lambda: False)())


def is_musa_environment() -> bool:
    return is_musa_available() or "MUSA_VISIBLE_DEVICES" in os.environ or bool(os.environ.get("MUSA_PATCH_PATH"))


def _require_musa_available() -> None:
    if is_musa_environment() and not is_musa_available():
        raise RuntimeError(
            "MUSA environment was requested via MUSA_VISIBLE_DEVICES or MUSA_PATCH_PATH, "
            "but torch.musa is unavailable. Check the MUSA runtime and musa_patch import order."
        )


def device_type() -> str:
    if is_musa_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_name(index: int | None = None) -> str:
    current_type = device_type()
    if current_type == "cpu":
        return "cpu"
    if index is None:
        index = current_device()
    return f"{current_type}:{index}"


def device(index: int | None = None) -> torch.device:
    return torch.device(device_name(index))


def accelerator_module() -> Any:
    if is_musa_available():
        return _musa()
    return torch.cuda


def set_device(index: int | str | torch.device) -> None:
    if device_type() != "cpu":
        accelerator_module().set_device(index)


def current_device() -> int | str:
    if device_type() == "cpu":
        return "cpu"
    return accelerator_module().current_device()


def synchronize(device_arg: int | str | torch.device | None = None) -> None:
    if device_type() == "cpu":
        return
    module = accelerator_module()
    if device_arg is None:
        module.synchronize()
    else:
        module.synchronize(device_arg)


def empty_cache() -> None:
    if device_type() != "cpu":
        accelerator_module().empty_cache()


def ipc_collect() -> None:
    if device_type() == "cpu":
        return
    ipc_collect_fn = getattr(accelerator_module(), "ipc_collect", None)
    if ipc_collect_fn is not None:
        ipc_collect_fn()


def mem_get_info(device_arg: int | str | torch.device | None = None) -> tuple[int, int]:
    if device_type() == "cpu":
        raise RuntimeError("Accelerator memory info is unavailable on CPU")
    if device_arg is None:
        device_arg = current_device()
    return accelerator_module().mem_get_info(device_arg)


def memory_allocated(device_arg: int | str | torch.device | None = None) -> int:
    if device_type() == "cpu":
        return 0
    return accelerator_module().memory_allocated(device_arg)


def memory_reserved(device_arg: int | str | torch.device | None = None) -> int:
    if device_type() == "cpu":
        return 0
    return accelerator_module().memory_reserved(device_arg)


def get_device_properties(device_arg: int | str | torch.device | None = None) -> Any:
    if device_type() == "cpu":
        return None
    if device_arg is None:
        device_arg = current_device()
    return accelerator_module().get_device_properties(device_arg)


def visible_devices_env_key() -> str:
    if "MUSA_VISIBLE_DEVICES" in os.environ or is_musa_available():
        return "MUSA_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def resolve_visible_device_id(physical_device_id: int | float | str) -> int:
    env_key = visible_devices_env_key()
    visible_devices = os.environ.get(env_key)
    physical_device_id = int(float(physical_device_id))
    if not visible_devices:
        return physical_device_id

    visible = [int(device_id) for device_id in visible_devices.split(",") if device_id.strip()]
    if physical_device_id in visible:
        return visible.index(physical_device_id)
    if 0 <= physical_device_id < len(visible):
        return physical_device_id
    raise RuntimeError(
        f"Device id {physical_device_id} is not valid under {env_key}={visible_devices}. "
        f"Expected one of {visible} (physical) or 0..{len(visible) - 1} (local)."
    )


def process_group_backend(default: str = "nccl") -> str:
    if default == "nccl":
        if is_musa_available():
            return "mccl"
        _require_musa_available()
    return default


def weight_update_backend(default: str = "nccl") -> str:
    if default == "nccl":
        if is_musa_available():
            return "cpu:gloo,musa:mccl"
        _require_musa_available()
    return default


def current_stream():
    if device_type() == "cpu":
        return None
    return accelerator_module().current_stream()


def _try_import_musa_patch() -> bool:
    global _MUSA_PATCH_IMPORTED
    if _MUSA_PATCH_IMPORTED:
        return True
    if not is_musa_environment():
        return False

    _MUSA_PATCH_IMPORTED = _import_musa_patch()
    if not _MUSA_PATCH_IMPORTED and is_musa_environment():
        logger.warning("musa_patch is not importable; continuing without it")
    return _MUSA_PATCH_IMPORTED


_try_import_musa_patch()
