"""TMS allocation policy for Megatron's parameter and gradient buffers."""

from contextlib import nullcontext
from functools import wraps


def configure_buffer_allocation(args):
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    # Older Megatron patches implement these options in the buffer constructor.
    if not hasattr(_ParamAndGradBuffer, "_allocate_buffer"):
        return

    allocator = _ParamAndGradBuffer._allocate_buffer
    allocator = getattr(allocator, "__wrapped__", allocator)

    @wraps(allocator)
    def allocate(self, buffer_type, numel, dtype):
        disable_param_backup = (
            getattr(args, "disable_param_buffers_cpu_backup", False) and self.ddp_config.use_distributed_optimizer
        )
        disable_grad_backup = getattr(args, "disable_grad_buffers_cpu_backup", False)
        if buffer_type == "shared":
            buffer_type = "param" if disable_param_backup else "grad"
        disable_backup = disable_param_backup if buffer_type == "param" else disable_grad_backup
        context = nullcontext()
        if disable_backup:
            assert not self.nccl_ub, "Disabling CPU buffer backup is not supported with nccl_ub=True"
            from torch_memory_saver import torch_memory_saver

            context = torch_memory_saver.region(tag=f"{buffer_type}_buffer", enable_cpu_backup=False)
        with context:
            return allocator(self, buffer_type, numel, dtype)

    _ParamAndGradBuffer._allocate_buffer = allocate
