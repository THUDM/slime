"""Megatron adapter for the logit-free linear cross entropy kernel."""

from types import MethodType

import torch
from megatron.core import mpu, tensor_parallel

from .linear_cross_entropy import linear_cross_entropy


def _fused_postprocess(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if self.config.sequence_parallel:
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states)
    else:
        hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region(hidden_states)

    if self.share_embeddings_and_output_weights:
        weight = self.shared_embedding_or_output_weight()
    else:
        weight = self.output_layer.weight
    if getattr(self.output_layer, "bias", None) is not None:
        raise NotImplementedError("triton log probs require a bias-free output layer")

    labels = labels.reshape(-1).contiguous()
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    if hidden_states.shape[0] != labels.numel():
        raise RuntimeError(f"hidden/label token mismatch: {hidden_states.shape[0]} != {labels.numel()}")

    args = getattr(self, "_slime_triton_log_probs_args")
    temperature = float(args.rollout_temperature)
    tp_group = mpu.get_tensor_model_parallel_group()
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        weight,
        labels,
        temperature,
        "none",
        tp_group,
    )
    return torch.stack((log_probs, entropy), dim=-1).unsqueeze(0)


def install_triton_log_probs(model_chunks, args) -> None:
    """Patch only the final pipeline stage's GPT postprocess method."""
    if getattr(args, "log_probs_backend", "torch") != "triton":
        return

    for wrapped in model_chunks:
        module = wrapped
        while hasattr(module, "module"):
            module = module.module
        if not getattr(module, "post_process", False):
            continue
        if not hasattr(module, "_postprocess"):
            raise TypeError(f"{type(module).__name__} does not expose Megatron _postprocess")

        original = module._postprocess

        def patched(this, *positional, __original=original, **kwargs):
            labels = kwargs.get("labels")
            if labels is None:
                return __original(*positional, **kwargs)
            hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else positional[0]
            return _fused_postprocess(this, hidden_states, labels)

        module._slime_triton_log_probs_args = args
        module._postprocess = MethodType(patched, module)
