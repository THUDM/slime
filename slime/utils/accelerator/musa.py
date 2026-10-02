"""MUSA accelerator implementation.

Importing this module never imports ``torch_musa``.  A MUSA runtime or the
optional ``musa_patch`` bootstrap may attach ``torch.musa`` before selection.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


def musa_module() -> Any:
    return getattr(torch, "musa", None)


def is_musa_available() -> bool:
    module = musa_module()
    checker = getattr(module, "is_available", None)
    return bool(module is not None and checker is not None and checker())


class MUSAAccelerator(TorchAccelerator):
    name = "musa"
    device_type = "musa"
    communication_backend_name = "mccl"

    @property
    def visible_devices_env(self) -> str:
        return "MUSA_VISIBLE_DEVICES"

    def _module(self) -> Any:
        module = musa_module()
        if module is None:
            raise RuntimeError("MUSA backend requires a runtime that exposes torch.musa")
        return module

    def is_available(self) -> bool:
        return is_musa_available()

    def weight_update_backend(self, default: str = "nccl") -> str:
        return "cpu:gloo,musa:mccl" if default == "nccl" else default

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> None:
        return None

    def post_import_torch(self) -> None:
        try:
            module = importlib.import_module("musa_patch")
        except ModuleNotFoundError as exc:
            if exc.name == "musa_patch":
                return
            raise RuntimeError(f"musa_patch failed because dependency {exc.name!r} is missing") from exc
        callback = getattr(module, "patch_after_import_torch", None)
        if callback is not None:
            callback()

    def attach_oom_observer(self, callback) -> bool:
        musa_c = getattr(self._module(), "_MUSAC", None)
        attach = getattr(musa_c, "_musa_attach_out_of_memory_observer", None)
        if attach is None:
            return False
        attach(callback)
        return True

    def autocast(self, *args, **kwargs):
        amp = getattr(self._module(), "amp", None)
        autocast = getattr(amp, "autocast", None)
        if autocast is None:
            raise NotImplementedError("MUSA runtime does not expose torch.musa.amp.autocast")
        return autocast(*args, **kwargs)

    def supports(self, capability: str) -> bool:
        if capability in {"nvml_affinity", "sglang_fp8_utils", "strict_fp32_logits"}:
            return False
        if capability == "requires_cpu_initialization":
            return True
        if capability == "amp":
            amp = getattr(self._module(), "amp", None)
            return callable(getattr(amp, "autocast", None))
        if capability == "bf16":
            checker = getattr(self._module(), "is_bf16_supported", None)
            return bool(checker and checker())
        return super().supports(capability)
