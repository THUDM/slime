"""Slime extensions to Megatron's Transformer Engine layers."""

import inspect
import os
from functools import wraps

import torch


def patch_precision_aware_optimizer_checkpointing(optimizer):
    from transformer_engine.pytorch.optimizers import FusedAdam

    for distributed_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        adam = distributed_optimizer.optimizer
        if not isinstance(adam, FusedAdam) or not getattr(adam, "store_param_remainders", False):
            continue
        get_states = distributed_optimizer._get_main_param_and_optimizer_states
        set_states = distributed_optimizer._set_main_param_and_optimizer_states

        def get_checkpoint_states(model_param, get_states=get_states, distributed_optimizer=distributed_optimizer):
            states = get_states(model_param)
            master = states["param"]
            if master.dtype == torch.int16:
                group, index = distributed_optimizer.model_param_group_index_map[model_param]
                param = distributed_optimizer.optimizer.param_groups[group]["params"][index]
                # TE stores the low FP32 bits separately from rounded BF16
                # weights. Megatron's reshardable checkpoint expects full FP32.
                bits = (param.detach().view(torch.int16).to(torch.int32) << 16) + master.to(torch.int32)
                states["param"] = bits.view(torch.float32)
            return states

        def set_checkpoint_states(
            model_param, states, set_states=set_states, distributed_optimizer=distributed_optimizer
        ):
            group, index = distributed_optimizer.model_param_group_index_map[model_param]
            param = distributed_optimizer.optimizer.param_groups[group]["params"][index]
            master = states["param"]
            if param.dtype == torch.bfloat16 and master.dtype == torch.float32:
                bits = master.contiguous().view(torch.int32)
                # Match TE's rounding, including the signed low-bit remainder.
                rounded = ((bits >> 16) + ((bits & 0x8000) != 0)).to(torch.int16).view(torch.bfloat16)
                with torch.no_grad():
                    param.copy_(rounded)
                states = {**states, "param": bits.to(torch.int16)}
            set_states(model_param, states)

        distributed_optimizer._get_main_param_and_optimizer_states = get_checkpoint_states
        distributed_optimizer._set_main_param_and_optimizer_states = set_checkpoint_states


class _FakeInt4QuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, group_size):
        rows, columns = weight.shape
        padded_columns = ((columns + group_size - 1) // group_size) * group_size
        padded = torch.nn.functional.pad(weight, (0, padded_columns - columns))
        grouped = padded.view(rows, -1, group_size)
        scale = (grouped.float().abs().amax(dim=-1, keepdim=True) / 7).clamp(min=1e-5)
        quantized = (grouped / scale).round().clamp(-7, 7) * scale
        return quantized.view(rows, padded_columns)[:, :columns].contiguous().to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_int4_quantization_ste(weight, group_size):
    quantized = _FakeInt4QuantizationSTE.apply(weight, group_size)
    if hasattr(weight, "main_grad"):
        quantized.main_grad = weight.main_grad
    return quantized


def install_transformer_engine_extensions():
    from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELinear

    if getattr(TELinear.__init__, "_slime_extensions", False):
        return

    original_init = TELinear.__init__
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_init(self, *args, **kwargs)
        for parameter in self.parameters():
            parameter.parallel_mode = bound.arguments["parallel_mode"]

    init._slime_extensions = True
    TELinear.__init__ = init

    original_weights = TEGroupedLinear._get_weight_tensors
    # Older slime Megatron patches already implement QAT inside this method.
    if "fake_int4_quantization_ste" in original_weights.__code__.co_names:
        return

    @wraps(original_weights)
    def get_weight_tensors(self):
        weights = original_weights(self)
        if os.environ.get("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0") == "1":
            group_size = int(os.environ.get("OPEN_TRAINING_INT4_GROUP_SIZE", "128"))
            weights = [fake_int4_quantization_ste(weight, group_size) for weight in weights]
        return weights

    TEGroupedLinear._get_weight_tensors = get_weight_tensors
