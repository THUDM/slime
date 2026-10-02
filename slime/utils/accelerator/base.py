"""Small, backend-neutral accelerator contract used by Slime.

The contract intentionally contains only operations that Slime uses in its
runtime.  Vendor modules are supplied by concrete implementations and are
never imported by this module.
"""

from __future__ import annotations

import abc
import os
from typing import Any

import torch


class Accelerator(abc.ABC):
    """Common device/runtime surface exposed to Slime code."""

    name: str
    device_type: str
    communication_backend_name: str

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return whether this backend can actually execute on this host."""

    @abc.abstractmethod
    def device(self, index: int | str | torch.device | None = None) -> torch.device:
        """Return a :class:`torch.device` for a local device index."""

    @abc.abstractmethod
    def device_name(self, index: int | str | torch.device | None = None) -> str:
        """Return the canonical device string used by PyTorch APIs."""

    @abc.abstractmethod
    def set_device(self, index: int | str | torch.device) -> None:
        """Select the current local device."""

    @abc.abstractmethod
    def current_device(self) -> int | str:
        """Return the current local device index, or ``cpu`` for CPU."""

    @abc.abstractmethod
    def device_count(self) -> int:
        """Return the number of visible devices."""

    @abc.abstractmethod
    def synchronize(self, device: int | str | torch.device | None = None) -> None:
        """Synchronize work on one device or the current device."""

    @abc.abstractmethod
    def current_stream(self, device: int | str | torch.device | None = None) -> Any:
        """Return the current stream, or ``None`` when streams are unsupported."""

    def default_stream(self, device: int | str | torch.device | None = None) -> Any:
        """Return the default stream, or ``None`` when streams are unsupported."""
        return None

    def stream(self, stream: Any):
        """Return a context manager for a stream when the backend supports it."""
        raise NotImplementedError(f"Accelerator {self.name!r} does not support streams")

    @property
    def Stream(self) -> Any:
        return None

    @property
    def Event(self) -> Any:
        return None

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Release allocator-held, currently unused memory."""

    def ipc_collect(self) -> None:
        """Collect inter-process allocator state when supported."""
        return None

    def set_allocator_expandable_segments(self) -> bool:
        """Configure expandable allocator segments when supported."""
        return False

    @abc.abstractmethod
    def mem_get_info(self, device: int | str | torch.device | None = None) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)`` for the selected device."""

    @abc.abstractmethod
    def memory_allocated(self, device: int | str | torch.device | None = None) -> int:
        """Return currently allocated device memory in bytes."""

    @abc.abstractmethod
    def memory_reserved(self, device: int | str | torch.device | None = None) -> int:
        """Return allocator-reserved device memory in bytes."""

    def get_device_properties(self, device: int | str | torch.device | None = None) -> Any:
        return None

    def memory_module(self) -> Any:
        """Return the backend memory namespace, if it exposes one."""
        return None

    def attach_oom_observer(self, callback) -> bool:
        """Attach an OOM callback; return ``False`` when unsupported."""
        return False

    def supports(self, capability: str) -> bool:
        """Return whether a named optional capability is implemented."""
        return False

    def autocast(self, *args, **kwargs):
        """Return an autocast context for this backend."""
        raise NotImplementedError(f"Accelerator {self.name!r} does not support autocast")

    def manual_seed(self, seed: int) -> None:
        """Seed the current device generator when supported."""
        return None

    def manual_seed_all(self, seed: int) -> None:
        """Seed all device generators when supported."""
        return None

    def get_rng_state(self, device: int | str | torch.device | None = None) -> torch.Tensor:
        """Return the current generator state."""
        return torch.get_rng_state()

    def set_rng_state(self, state: torch.Tensor, device: int | str | torch.device | None = None) -> None:
        """Restore the current generator state."""
        torch.set_rng_state(state)

    def initial_seed(self) -> int:
        return int(torch.initial_seed())

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> torch.device | None:
        """Return the device id accepted by ``dist.init_process_group``."""
        return self.device(index)

    def post_import_torch(self) -> None:
        """Apply an optional backend hook after third-party torch imports."""
        return None

    def communication_backend(self, default: str = "nccl") -> str:
        """Map a logical default backend to this accelerator's transport."""
        return self.communication_backend_name if default == "nccl" else default

    def weight_update_backend(self, default: str = "nccl") -> str:
        return self.communication_backend(default)

    @property
    def visible_devices_env(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    def resolve_visible_device_id(self, physical_device_id: int | float | str) -> int:
        """Map a physical id to a local id under this backend's visibility env."""
        raw_value = str(physical_device_id).strip()
        visible = os.environ.get(self.visible_devices_env)
        if not visible:
            return int(float(raw_value))

        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if raw_value in ids:
            return ids.index(raw_value)

        try:
            value = int(float(raw_value))
        except ValueError:
            value = None
        if value is not None and str(value) in ids:
            return ids.index(str(value))
        if value is not None and 0 <= value < len(ids):
            return value
        raise RuntimeError(
            f"Device id {raw_value} is not valid under {self.visible_devices_env}={visible}. "
            f"Expected one of {ids} (physical) or 0..{len(ids) - 1} (local)."
        )

    def accelerator_module(self) -> Any:
        """Return the torch backend namespace, or ``None`` for CPU."""
        return None
