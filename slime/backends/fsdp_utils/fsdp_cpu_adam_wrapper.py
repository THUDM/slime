from typing import Any, Dict, List

import torch
import torch.nn as nn
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.distributed.tensor import DTensor


class FSDPCPUAdamWrapper:
    """
    Wrapper for DeepSpeedCPUAdam to work with FSDP models where parameters are on GPU.

    DeepSpeedCPUAdam requires both parameters and gradients to be on CPU. This wrapper:
    1. Maintains CPU shadow copies of GPU parameters (contiguous, proper dtype)
    2. Copies gradients from GPU to CPU before optimizer step (contiguous)
    3. Runs optimizer update on CPU
    4. Copies updated parameters back to GPU

    Following the parameter copy pattern from update_weight_utils.py
    """

    def __init__(self, optimizer_config: Dict[str, Any], model: nn.Module) -> None:
        self.model: nn.Module = model
        self.gpu_params: List[nn.Parameter] = list(model.parameters())
        self.optimizer_config: Dict[str, Any] = optimizer_config
        self.cpu_params: List[torch.Tensor] = []
        self.cpu_optimizer: DeepSpeedCPUAdam

        # Create CPU shadow copies of parameters using the pattern from update_weight_utils.py
        # Store only the LOCAL SHARD for each rank, not the full tensor
        for gpu_param in self.gpu_params:
            param_data = gpu_param.detach()
            if isinstance(param_data, DTensor):
                param_data = param_data.to_local()

            cpu_param = param_data.contiguous().to(device="cpu", dtype=torch.float32, non_blocking=True)
            cpu_param.requires_grad_(True)

            assert cpu_param.is_contiguous(), f"CPU param must be contiguous for AVX"
            assert cpu_param.dtype == torch.float32, f"CPU param must be FP32 for DeepSpeed"

            self.cpu_params.append(cpu_param)

        torch.cuda.synchronize()

        self.cpu_optimizer = DeepSpeedCPUAdam(
            self.cpu_params,
            lr=self.optimizer_config["lr"],
            betas=self.optimizer_config["betas"],
            eps=self.optimizer_config["eps"],
            weight_decay=self.optimizer_config["weight_decay"],
            adamw_mode=self.optimizer_config["adamw_mode"],
            fp32_optimizer_states=self.optimizer_config["fp32_optimizer_states"],
        )

        self.param_groups = self.cpu_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on GPU parameters.

        Args:
            set_to_none: If True, set gradients to None; otherwise zero them.
        """
        for param in self.gpu_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        """Perform optimizer step.

        Steps:
            1. Copy gradients from GPU to CPU (handling DTensor, ensuring contiguous FP32)
            2. Run optimizer update on CPU
            3. Copy updated parameters back to GPU

        Uses the same .to() pattern as update_weight_utils.py for proper memory layout.
        """
        # Copy gradients from GPU to CPU - handle DTensor and ensure FP32 for DeepSpeed AVX
        for gpu_param, cpu_param in zip(self.gpu_params, self.cpu_params):
            if gpu_param.grad is not None:

                grad_data = gpu_param.grad.detach()
                if isinstance(grad_data, DTensor):
                    grad_data = grad_data.to_local()

                # DeepSpeed's AVX operations expect FP32 gradients to match FP32 params
                cpu_grad = grad_data.contiguous().to(device="cpu", dtype=torch.float32, non_blocking=True)

                # Verify gradient properties for DeepSpeed AVX
                assert cpu_grad.is_contiguous(), "CPU gradient must be contiguous for AVX"
                assert cpu_grad.dtype == torch.float32, "CPU gradient must be FP32 for DeepSpeed"

                cpu_param.grad = cpu_grad
            else:
                cpu_param.grad = None

        torch.cuda.synchronize()

        # Run optimizer step on CPU
        self.cpu_optimizer.step()

        for gpu_param, cpu_param in zip(self.gpu_params, self.cpu_params):
            updated_param = cpu_param.data.to(
                device=torch.cuda.current_device(), dtype=gpu_param.dtype, non_blocking=True
            )

            if isinstance(gpu_param.data, DTensor):
                gpu_param.data.to_local().copy_(updated_param, non_blocking=True)
            else:
                gpu_param.data.copy_(updated_param, non_blocking=True)

        torch.cuda.synchronize()
