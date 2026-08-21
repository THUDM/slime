"""Runtime-selectable, backend-neutral accelerator API for Slime.

The module-level functions are compatibility shims for the historical
``slime.utils.accelerator`` API. New code can use :func:`get_accelerator`
when it needs capability inspection or dependency injection.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .base import Accelerator
from .cuda import CUDAAccelerator
from .musa import MUSAAccelerator
from .musa import is_musa_available as _is_musa_available

logger = logging.getLogger(__name__)

_MUSA_PATCH_IMPORTED = False
_MUSA_BOOTSTRAP_CHECKED = False
_ACCELERATOR: Accelerator | None = None
_SELECTION_LOCK = threading.RLock()


@dataclass(frozen=True)
class _BackendRegistration:
    factory: Callable[[], Accelerator]
    is_available: Callable[[], bool]
    priority: int
    communication_backends: tuple[str, ...]


_REGISTRY: dict[str, _BackendRegistration] = {}


def register_accelerator(
    name: str,
    factory: Callable[[], Accelerator],
    is_available: Callable[[], bool] | None = None,
    priority: int = 0,
    communication_backends: tuple[str, ...] = (),
) -> None:
    """Register a lazily constructed backend without changing Slime core."""
    normalized = name.strip().lower()
    if not normalized or normalized in {"auto", "none"}:
        raise ValueError("Accelerator name must be a non-empty backend name")
    with _SELECTION_LOCK:
        _REGISTRY[normalized] = _BackendRegistration(
            factory=factory,
            is_available=is_available or (lambda: factory().is_available()),
            priority=priority,
            communication_backends=tuple(name.lower() for name in communication_backends),
        )


def _append_musa_patch_path() -> None:
    patch_path = os.environ.get("MUSA_PATCH_PATH")
    if patch_path and patch_path not in sys.path:
        sys.path.append(patch_path)


def _import_musa_patch() -> bool:
    _append_musa_patch_path()
    try:
        importlib.import_module("musa_patch")
    except ModuleNotFoundError as exc:
        if exc.name == "musa_patch":
            return False
        raise RuntimeError(f"musa_patch failed because dependency {exc.name!r} is missing") from exc
    except Exception as exc:
        raise RuntimeError(f"musa_patch initialization failed: {exc}") from exc
    return True


def is_musa_available() -> bool:
    return _is_musa_available()


def is_musa_environment() -> bool:
    return (
        is_musa_available()
        or os.environ.get("SLIME_ACCELERATOR", "").lower() == "musa"
        or "MUSA_VISIBLE_DEVICES" in os.environ
        or bool(os.environ.get("MUSA_PATCH_PATH"))
    )


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


def _musa_requested() -> bool:
    configured = os.environ.get("SLIME_ACCELERATOR", "").lower()
    if configured and configured != "auto":
        return configured == "musa"
    return "MUSA_VISIBLE_DEVICES" in os.environ or bool(os.environ.get("MUSA_PATCH_PATH"))


def _bootstrap_musa_patch_if_needed() -> bool:
    """Bootstrap the patch for an already chosen MUSA backend at most once."""
    global _MUSA_BOOTSTRAP_CHECKED
    if _MUSA_BOOTSTRAP_CHECKED:
        return _MUSA_PATCH_IMPORTED
    _MUSA_BOOTSTRAP_CHECKED = True
    return _try_import_musa_patch()


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except (ImportError, RuntimeError):
        return False


def _register_builtin_backends() -> None:
    if "cuda" not in _REGISTRY:
        register_accelerator("cuda", CUDAAccelerator, _cuda_available, priority=100, communication_backends=("nccl",))
    if "musa" not in _REGISTRY:
        register_accelerator(
            "musa", MUSAAccelerator, is_musa_available, priority=200, communication_backends=("mccl",)
        )


def _requested_name() -> str | None:
    value = os.environ.get("SLIME_ACCELERATOR")
    if value and value.lower() != "auto":
        return value.strip().lower()
    if _musa_requested():
        return "musa"
    return None


def _make_selected(name: str, explicit: bool) -> Accelerator:
    _register_builtin_backends()
    entry = _REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown accelerator {name!r}; registered backends: {available}")
    if name == "musa":
        # musa_patch may expose torch.musa, so bootstrap after MUSA has been
        # chosen but before validating and constructing its backend.
        _bootstrap_musa_patch_if_needed()
    if explicit and not entry.is_available():
        if name == "musa":
            detail = (
                "torch.musa is unavailable; install a MUSA-enabled PyTorch runtime and set MUSA_PATCH_PATH if required"
            )
        elif name == "cuda":
            detail = "torch.cuda.is_available() is false or no CUDA device is visible"
        else:
            detail = "the backend availability check returned false"
        raise RuntimeError(f"Requested accelerator {name!r} is unavailable: {detail}")
    backend = entry.factory()
    if not backend.is_available():
        raise RuntimeError(f"Accelerator backend {name!r} was selected but is unavailable at runtime")
    return backend


def get_accelerator() -> Accelerator:
    global _ACCELERATOR
    if _ACCELERATOR is not None:
        return _ACCELERATOR
    with _SELECTION_LOCK:
        if _ACCELERATOR is not None:
            return _ACCELERATOR
        _register_builtin_backends()
        requested = _requested_name()
        if requested is not None:
            _ACCELERATOR = _make_selected(requested, explicit=True)
            logger.info("Selected accelerator %s (explicit)", _ACCELERATOR.name)
            return _ACCELERATOR

        # Highest priority wins; names break priority ties deterministically.
        candidates = sorted(_REGISTRY.items(), key=lambda item: (-item[1].priority, item[0]))
        for name, registration in candidates:
            if registration.is_available():
                _ACCELERATOR = _make_selected(name, explicit=False)
                logger.info("Selected accelerator %s (auto)", _ACCELERATOR.name)
                return _ACCELERATOR
        registered = ", ".join(sorted(_REGISTRY))
        raise RuntimeError(
            "No usable accelerator was detected. "
            f"Registered backends: {registered}. "
            "Set SLIME_ACCELERATOR explicitly or install a supported accelerator runtime."
        )


def initialize_accelerator() -> Accelerator | None:
    """Finalize runtime selection when a backend is requested or available.

    Explicit requests retain ``get_accelerator``'s fail-fast behavior. An
    environment without accelerator hardware remains importable for CPU-only
    tooling and documentation.
    """
    if _ACCELERATOR is not None:
        return _ACCELERATOR
    with _SELECTION_LOCK:
        _register_builtin_backends()
        if _requested_name() is not None or any(entry.is_available() for entry in _REGISTRY.values()):
            return get_accelerator()
        return None


def set_accelerator(accelerator: Accelerator) -> None:
    global _ACCELERATOR
    if not isinstance(accelerator, Accelerator):
        raise TypeError(f"Expected Accelerator, got {type(accelerator).__name__}")
    if not accelerator.is_available():
        raise RuntimeError(f"Cannot install unavailable accelerator backend {accelerator.name!r}")
    with _SELECTION_LOCK:
        _ACCELERATOR = accelerator


def reset_accelerator() -> None:
    """Reset the singleton; intended for tests and process initialization."""
    global _ACCELERATOR
    with _SELECTION_LOCK:
        _ACCELERATOR = None


def _backend() -> Accelerator:
    return get_accelerator()


def device_type() -> str:
    return _backend().device_type


def accelerator_module() -> Any:
    return _backend().accelerator_module()


def device(index: int | str | torch.device | None = None) -> torch.device:
    return _backend().device(index)


def device_name(index: int | str | torch.device | None = None) -> str:
    return _backend().device_name(index)


def set_device(index: int | str | torch.device) -> None:
    return _backend().set_device(index)


def current_device() -> int | str:
    return _backend().current_device()


def device_count() -> int:
    return _backend().device_count()


def synchronize(device_arg: int | str | torch.device | None = None) -> None:
    return _backend().synchronize(device_arg)


def current_stream(device_arg: int | str | torch.device | None = None) -> Any:
    return _backend().current_stream(device_arg)


def default_stream(device_arg: int | str | torch.device | None = None) -> Any:
    return _backend().default_stream(device_arg)


def stream(stream_arg: Any):
    return _backend().stream(stream_arg)


def new_stream(*args, **kwargs) -> Any:
    stream_type = _backend().Stream
    if stream_type is None:
        raise NotImplementedError(f"Accelerator {_backend().name!r} does not support streams")
    return stream_type(*args, **kwargs)


def new_event(*args, **kwargs) -> Any:
    event_type = _backend().Event
    if event_type is None:
        raise NotImplementedError(f"Accelerator {_backend().name!r} does not support events")
    return event_type(*args, **kwargs)


def empty_cache() -> None:
    return _backend().empty_cache()


def ipc_collect() -> None:
    return _backend().ipc_collect()


def set_allocator_expandable_segments() -> bool:
    return _backend().set_allocator_expandable_segments()


def mem_get_info(device_arg: int | str | torch.device | None = None) -> tuple[int, int]:
    return _backend().mem_get_info(device_arg)


def memory_allocated(device_arg: int | str | torch.device | None = None) -> int:
    return _backend().memory_allocated(device_arg)


def memory_reserved(device_arg: int | str | torch.device | None = None) -> int:
    return _backend().memory_reserved(device_arg)


def get_device_properties(device_arg: int | str | torch.device | None = None) -> Any:
    return _backend().get_device_properties(device_arg)


def memory_module() -> Any:
    return _backend().memory_module()


def attach_oom_observer(callback) -> bool:
    return _backend().attach_oom_observer(callback)


def supports(capability: str) -> bool:
    return _backend().supports(capability)


def autocast(*args, **kwargs):
    return _backend().autocast(*args, **kwargs)


def manual_seed(seed: int) -> None:
    return _backend().manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    return _backend().manual_seed_all(seed)


def get_rng_state(device_arg: int | str | torch.device | None = None) -> torch.Tensor:
    return _backend().get_rng_state(device_arg)


def set_rng_state(state: torch.Tensor, device_arg: int | str | torch.device | None = None) -> None:
    return _backend().set_rng_state(state, device_arg)


def initial_seed() -> int:
    return _backend().initial_seed()


def distributed_device_id(index: int | str | torch.device | None = None) -> torch.device | None:
    return _backend().distributed_device_id(index)


def post_import_torch() -> None:
    return _backend().post_import_torch()


def is_accelerator_backend(backend: str) -> bool:
    """Return whether a distributed backend belongs to a registered device accelerator."""
    _register_builtin_backends()
    normalized = backend.lower()
    return any(
        name in normalized for registration in _REGISTRY.values() for name in registration.communication_backends
    )


def process_group_backend(default: str = "nccl") -> str:
    return _backend().communication_backend(default)


def weight_update_backend(default: str = "nccl") -> str:
    return _backend().weight_update_backend(default)


def visible_devices_env_key() -> str:
    return _backend().visible_devices_env


def resolve_visible_device_id(physical_device_id: int | float | str) -> int:
    return _backend().resolve_visible_device_id(physical_device_id)
