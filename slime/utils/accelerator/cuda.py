"""CUDA/ROCm accelerator implementation."""

from __future__ import annotations

from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


class CUDAAccelerator(TorchAccelerator):
    name = "cuda"
    device_type = "cuda"
    communication_backend_name = "nccl"

    def _module(self) -> Any:
        return torch.cuda

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)

    def attach_oom_observer(self, callback) -> bool:
        attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
        if attach is None:
            return False
        attach(callback)
        return True

    def supports(self, capability: str) -> bool:
        if capability == "nvml_affinity":
            return torch.version.hip is None
        if capability == "bf16":
            checker = getattr(torch.cuda, "is_bf16_supported", None)
            return bool(checker and checker())
        if capability in {"cuda_int4_extension", "sglang_fp8_utils", "strict_fp32_logits", "triton_kernels"}:
            return True
        if capability == "requires_cpu_initialization":
            return torch.version.hip is not None
        return super().supports(capability)
