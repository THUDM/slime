"""Shared adapter for PyTorch accelerator namespaces."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from .base import Accelerator

logger = logging.getLogger(__name__)


class TorchAccelerator(Accelerator):
    """Delegate common CUDA-like APIs to a vendor torch namespace."""

    def _module(self) -> Any:
        raise NotImplementedError

    def accelerator_module(self) -> Any:
        return self._module()

    def is_available(self) -> bool:
        module = self._module()
        checker = getattr(module, "is_available", None)
        return bool(module is not None and checker is not None and checker())

    def device(self, index: int | str | torch.device | None = None) -> torch.device:
        return torch.device(self.device_name(index))

    def device_name(self, index: int | str | torch.device | None = None) -> str:
        if isinstance(index, torch.device):
            return str(index)
        if isinstance(index, str):
            return index if ":" in index else f"{self.device_type}:{index}"
        if index is None:
            index = self.current_device()
        return f"{self.device_type}:{index}"

    def set_device(self, index: int | str | torch.device) -> None:
        self._module().set_device(index)

    def current_device(self) -> int:
        return int(self._module().current_device())

    def device_count(self) -> int:
        return int(self._module().device_count())

    def synchronize(self, device: int | str | torch.device | None = None) -> None:
        if device is None:
            self._module().synchronize()
        else:
            self._module().synchronize(device)

    def current_stream(self, device: int | str | torch.device | None = None) -> Any:
        if device is None:
            return self._module().current_stream()
        return self._module().current_stream(device)

    def default_stream(self, device: int | str | torch.device | None = None) -> Any:
        default_stream = getattr(self._module(), "default_stream", None)
        if default_stream is None:
            raise NotImplementedError(f"Accelerator {self.name!r} does not expose a default stream")
        if device is None:
            return default_stream()
        return default_stream(device)

    def stream(self, stream: Any):
        stream_context = getattr(self._module(), "stream", None)
        if stream_context is None:
            raise NotImplementedError(f"Accelerator {self.name!r} does not expose stream contexts")
        return stream_context(stream)

    @property
    def Stream(self) -> Any:
        return getattr(self._module(), "Stream", None)

    @property
    def Event(self) -> Any:
        return getattr(self._module(), "Event", None)

    def empty_cache(self) -> None:
        self._module().empty_cache()

    def ipc_collect(self) -> None:
        collect = getattr(self._module(), "ipc_collect", None)
        if collect is not None:
            collect()

    def set_allocator_expandable_segments(self) -> bool:
        value = os.getenv("SLIME_ENABLE_EXPANDABLE_SEGMENTS", "0")
        if value not in {"0", "1"}:
            raise ValueError(f"SLIME_ENABLE_EXPANDABLE_SEGMENTS must be 0 or 1, got {value!r}")
        if value == "0":
            return False

        memory = self.memory_module()
        setter = getattr(memory, "_set_allocator_settings", None)
        if setter is None:
            logger.warning(
                "%s memory allocator settings API is unavailable; skip expandable_segments:True",
                self.name.upper(),
            )
            return False
        setter("expandable_segments:True")
        return True

    def mem_get_info(self, device: int | str | torch.device | None = None) -> tuple[int, int]:
        if device is None:
            device = self.current_device()
        free, total = self._module().mem_get_info(device)
        return int(free), int(total)

    def memory_allocated(self, device: int | str | torch.device | None = None) -> int:
        return int(self._module().memory_allocated(device))

    def memory_reserved(self, device: int | str | torch.device | None = None) -> int:
        return int(self._module().memory_reserved(device))

    def get_device_properties(self, device: int | str | torch.device | None = None) -> Any:
        if device is None:
            device = self.current_device()
        return self._module().get_device_properties(device)

    def memory_module(self) -> Any:
        return getattr(self._module(), "memory", None)

    def autocast(self, *args, **kwargs):
        return torch.autocast(self.device_type, *args, **kwargs)

    def manual_seed(self, seed: int) -> None:
        self._module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self._module().manual_seed_all(seed)

    def get_rng_state(self, device: int | str | torch.device | None = None) -> torch.Tensor:
        if device is None:
            return self._module().get_rng_state()
        return self._module().get_rng_state(device)

    def set_rng_state(self, state: torch.Tensor, device: int | str | torch.device | None = None) -> None:
        if device is None:
            self._module().set_rng_state(state)
        else:
            self._module().set_rng_state(state, device)

    def initial_seed(self) -> int:
        return int(self._module().initial_seed())

    def supports(self, capability: str) -> bool:
        module = self._module()
        if capability == "device_memory":
            return all(hasattr(module, name) for name in ("empty_cache", "mem_get_info", "memory_allocated"))
        if capability == "events":
            return hasattr(module, "Event")
        if capability == "rng":
            return all(hasattr(module, name) for name in ("get_rng_state", "set_rng_state", "manual_seed"))
        if capability == "streams":
            return all(hasattr(module, name) for name in ("Stream", "current_stream", "stream"))
        return capability in {"amp", "fp16"}
