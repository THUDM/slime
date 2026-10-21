"""Ascend NPU accelerator implementation.

Importing this module never imports ``torch_npu``. The runtime bootstrap in
``slime.utils.accelerator`` may attach ``torch.npu`` before selection.
"""

from __future__ import annotations

from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


def npu_module() -> Any:
    return getattr(torch, "npu", None)


def is_npu_available() -> bool:
    module = npu_module()
    checker = getattr(module, "is_available", None)
    return bool(module is not None and checker is not None and checker())


class NPUAccelerator(TorchAccelerator):
    name = "npu"
    device_type = "npu"
    communication_backend_name = "hccl"

    @property
    def visible_devices_env(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def _module(self) -> Any:
        module = npu_module()
        if module is None:
            raise RuntimeError("Ascend NPU backend requires torch_npu and a runtime that exposes torch.npu")
        return module

    def is_available(self) -> bool:
        return is_npu_available()

    def weight_update_backend(self, default: str = "nccl") -> str:
        return "cpu:gloo,npu:hccl" if default == "nccl" else default

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> None:
        # HCCL selects the local device through torch.npu.set_device/LOCAL_RANK.
        return None

    def empty_cache(self) -> None:
        module = self._module()
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is not None and is_capturing():
            return
        module.empty_cache()

    def supports(self, capability: str) -> bool:
        if capability in {"cuda_int4_extension", "nvml_affinity", "sglang_fp8_utils", "triton_kernels"}:
            return False
        if capability == "requires_cpu_initialization":
            return True
        if capability == "bf16":
            checker = getattr(self._module(), "is_bf16_supported", None)
            return bool(checker and checker())
        return super().supports(capability)
