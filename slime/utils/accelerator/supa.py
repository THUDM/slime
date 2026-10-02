"""SUPA accelerator implementation for Biren GPUs.

Importing this module never imports ``torch_supa``.  A SUPA runtime attaches
``torch.supa`` and registers the ``supa`` device type with PyTorch, so this
module only inspects what is already present on ``torch``.

The Biren stack names its collectives ``bccl`` and accepts both the plain
``"bccl"`` form and the device-qualified ``"supa:bccl"`` form, which is why
:meth:`SUPAAccelerator.weight_update_backend` can hand out a composite string.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


def supa_module() -> Any:
    return getattr(torch, "supa", None)


def is_supa_available() -> bool:
    module = supa_module()
    checker = getattr(module, "is_available", None)
    return bool(module is not None and checker is not None and checker())


class SUPAAccelerator(TorchAccelerator):
    name = "supa"
    device_type = "supa"
    communication_backend_name = "bccl"

    @property
    def visible_devices_env(self) -> str:
        return "SUPA_VISIBLE_DEVICES"

    def _module(self) -> Any:
        module = supa_module()
        if module is None:
            raise RuntimeError("SUPA backend requires a runtime that exposes torch.supa")
        return module

    def is_available(self) -> bool:
        return is_supa_available()

    def weight_update_backend(self, default: str = "nccl") -> str:
        return "cpu:gloo,supa:bccl" if default == "nccl" else default

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> None:
        # Skip the ``device_id`` argument of ``init_process_group`` and let the
        # SUPA runtime bind the device itself.
        return None

    def post_import_torch(self) -> None:
        try:
            importlib.import_module("torch_supa")
        except ModuleNotFoundError as exc:
            if exc.name == "torch_supa":
                return
            raise RuntimeError(f"torch_supa failed because dependency {exc.name!r} is missing") from exc

    def attach_oom_observer(self, callback) -> bool:
        supa_c = getattr(self._module(), "_SUPAC", None)
        attach = getattr(supa_c, "_supa_attach_out_of_memory_observer", None)
        if attach is None:
            return False
        attach(callback)
        return True

    def autocast(self, *args, **kwargs):
        amp = getattr(self._module(), "amp", None)
        autocast = getattr(amp, "autocast", None)
        if autocast is None:
            raise NotImplementedError("SUPA runtime does not expose torch.supa.amp.autocast")
        return autocast(*args, **kwargs)

    def supports(self, capability: str) -> bool:
        # NVML is NVIDIA-only; the fp8/int4 helpers and the strict fp32 logits
        # path all depend on CUDA-specific kernels.
        if capability in {"nvml_affinity", "sglang_fp8_utils", "strict_fp32_logits", "cuda_int4_extension"}:
            return False
        # Inductor codegen is not usable on the SUPA runtime yet.
        if capability == "triton_kernels":
            return False
        # Parameters are built on CPU and then moved onto the device.
        if capability == "requires_cpu_initialization":
            return True
        if capability == "amp":
            amp = getattr(self._module(), "amp", None)
            return callable(getattr(amp, "autocast", None))
        if capability == "bf16":
            checker = getattr(self._module(), "is_bf16_supported", None)
            return bool(checker and checker())
        return super().supports(capability)
