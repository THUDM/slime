"""Use SGLang's block-FP8 DeepGEMM result in selected Megatron linears.

This is an opt-in numerical-alignment hook.  It replaces the selected
Transformer Engine linear with a custom autograd function:

* forward: SGLang-style block-FP8 DeepGEMM;
* backward: explicit BF16 GEMMs for dgrad/wgrad plus analytic norm gradients
  for fused LayerNorm/RMSNorm linears.

The first implementation requires tensor parallel size one.  This makes each
target a full matrix, matching SGLang's dense-TP1 execution and avoiding
row-parallel partial-sum rounding as a confounder.
"""

from __future__ import annotations

import logging
import os
import re
import types
from collections.abc import Iterable

import torch
from megatron.core import parallel_state

logger = logging.getLogger(__name__)

try:
    from megatron.core.extensions.transformer_engine import te_general_gemm
except ImportError:
    te_general_gemm = None


def router_gating_linear_backward(inp, weight, grad_output, router_dtype):
    """Compute router-linear dgrad/wgrad without evaluating its forward.

    Re-homed into slime so the GLM-5 alignment path does not depend on this
    helper being present in Megatron ``moe_utils`` (it only exists on newer
    Megatron commits).  Behaviour matches that upstream helper.
    """
    input_dtype = inp.dtype
    weight_dtype = weight.dtype
    inp_shape = inp.shape
    inp_2d = inp.reshape(-1, inp_shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

    if (
        te_general_gemm is not None
        and router_dtype != torch.float64
        and inp_2d.is_cuda
        and weight.is_cuda
        and grad_2d.is_cuda
    ):
        grad_input = te_general_gemm(weight.to(router_dtype), grad_2d, router_dtype, layout="NN", grad=True)[0].to(
            input_dtype
        )
        grad_weight = te_general_gemm(inp_2d.to(router_dtype), grad_2d, router_dtype, layout="NT", grad=True)[0].to(
            weight_dtype
        )
    else:
        grad_input = torch.mm(grad_2d, weight.to(router_dtype)).to(input_dtype)
        grad_weight = torch.mm(grad_2d.t(), inp_2d.to(router_dtype)).to(weight_dtype)

    return grad_input.reshape(*inp_shape), grad_weight


_LAYER_PATH_RE = re.compile(r"^(?P<layer_path>(?:.*\.)?decoder\.layers\.(?P<local_layer_index>\d+))\.")
_DEFAULT_TARGET_SUFFIXES = (
    "self_attention.linear_q_down_proj",
    "self_attention.linear_q_up_proj",
    "self_attention.linear_kv_down_proj",
    "self_attention.linear_kv_up_proj",
    "self_attention.linear_proj",
    "self_attention.wq_b",
    "self_attention.wk",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "mlp.shared_experts.linear_fc1",
    "mlp.shared_experts.linear_fc2",
)
_SUPPORTED_TE_CLASS_NAMES = {
    "TELinear",
    "TEColumnParallelLinear",
    "TERowParallelLinear",
    "TELayerNormLinear",
    "TELayerNormColumnParallelLinear",
}


def _should_log_deepgemm_summary() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _format_int_ranges(values: Iterable) -> str:
    sorted_values = sorted({int(value) for value in values})
    if not sorted_values:
        return "[]"

    ranges = []
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _deepgemm_bf16_gemm_nn(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Compute ``lhs @ rhs`` using DeepGEMM BF16 when available."""
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise RuntimeError(f"BF16 NN GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if lhs.is_cuda and rhs.is_cuda and lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        import deep_gemm

        out = torch.empty((lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nn(lhs.contiguous(), rhs.contiguous(), out)
        return out
    return lhs.matmul(rhs).to(lhs.dtype)


def _deepgemm_bf16_gemm_nt(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Compute ``lhs @ rhs.T`` using DeepGEMM BF16 when available."""
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
        raise RuntimeError(f"BF16 NT GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if lhs.is_cuda and rhs.is_cuda and lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        import deep_gemm

        out = torch.empty((lhs.shape[0], rhs.shape[0]), device=lhs.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(lhs.contiguous(), rhs.contiguous(), out)
        return out
    return lhs.matmul(rhs.transpose(0, 1)).to(lhs.dtype)


def _deepgemm_bf16_gemm_tn(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Compute ``lhs.T @ rhs`` using DeepGEMM BF16 when available."""
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] != rhs.shape[0]:
        raise RuntimeError(f"BF16 TN GEMM shape mismatch: {tuple(lhs.shape)} x {tuple(rhs.shape)}")
    if lhs.is_cuda and rhs.is_cuda and lhs.dtype == torch.bfloat16 and rhs.dtype == torch.bfloat16:
        import deep_gemm

        out = torch.empty((lhs.shape[1], rhs.shape[1]), device=lhs.device, dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_tn(lhs.contiguous(), rhs.contiguous(), out)
        return out
    return lhs.transpose(0, 1).matmul(rhs).to(lhs.dtype)


def _sum_to_parameter_dtype(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(dtype=reference.dtype)


def _effective_norm_weight(norm_weight: torch.Tensor, zero_centered_gamma: bool) -> torch.Tensor:
    weight = norm_weight.float()
    if zero_centered_gamma:
        weight = weight + 1.0
    return weight


def _norm_forward(
    input_: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    *,
    normalization: str,
    eps: float,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    if normalization == "RMSNorm" and os.environ.get("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "0") == "1":
        if norm_bias is not None:
            raise RuntimeError("SGLang RMSNorm alignment does not support a norm bias")
        from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant

        weight = norm_weight
        if zero_centered_gamma:
            weight = (norm_weight.float() + 1.0).to(dtype=norm_weight.dtype)
        return rms_norm_batch_invariant(input_, weight, eps)

    x = input_.float()
    weight = _effective_norm_weight(norm_weight, zero_centered_gamma)
    if normalization == "RMSNorm":
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        output = x * rstd * weight
    elif normalization == "LayerNorm":
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        rstd = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + eps)
        output = centered * rstd * weight
        if norm_bias is not None:
            output = output + norm_bias.float()
    else:
        raise RuntimeError(f"Unsupported fused norm type for DeepGEMM wrapper: {normalization}")
    return output.to(input_.dtype)


def _norm_backward(
    grad_output: torch.Tensor,
    input_: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    *,
    normalization: str,
    eps: float,
    zero_centered_gamma: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = input_.float()
    grad = grad_output.float()
    weight = _effective_norm_weight(norm_weight, zero_centered_gamma)
    reduce_dims = tuple(range(grad.ndim - 1))

    if normalization == "RMSNorm":
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        xhat = x * rstd
        grad_norm_weight = (grad * xhat).sum(dim=reduce_dims)
        scaled_grad = grad * weight
        mean_scaled_grad_x = (scaled_grad * x).mean(dim=-1, keepdim=True)
        grad_input = scaled_grad * rstd - x * mean_scaled_grad_x * rstd.pow(3)
        grad_norm_bias = None
    elif normalization == "LayerNorm":
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        rstd = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + eps)
        xhat = centered * rstd
        grad_norm_weight = (grad * xhat).sum(dim=reduce_dims)
        scaled_grad = grad * weight
        hidden = x.shape[-1]
        grad_input = (
            scaled_grad
            - scaled_grad.mean(dim=-1, keepdim=True)
            - xhat * (scaled_grad * xhat).mean(dim=-1, keepdim=True)
        ) * rstd
        if hidden <= 0:
            raise RuntimeError("LayerNorm hidden size must be positive")
        grad_norm_bias = grad.sum(dim=reduce_dims) if norm_bias is not None else None
    else:
        raise RuntimeError(f"Unsupported fused norm type for DeepGEMM wrapper: {normalization}")

    return (
        grad_input.to(dtype=input_.dtype),
        _sum_to_parameter_dtype(grad_norm_weight, norm_weight),
        None if grad_norm_bias is None else _sum_to_parameter_dtype(grad_norm_bias, norm_bias),
    )


class _SGLangRMSNormWithAnalyticBackward(torch.autograd.Function):
    """SGLang RMSNorm forward with a single-pass analytic RMSNorm backward."""

    @staticmethod
    def forward(
        ctx,
        visible_input: torch.Tensor,
        backward_input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_fp32_sum: bool,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.eps = eps
        ctx.save_for_backward(backward_input, weight)
        if use_fp32_sum:
            return _sglang_rmsnorm_from_fp32_sum(
                visible_input,
                weight,
                eps,
                output_dtype,
            )
        return _sglang_batch_invariant_rmsnorm(visible_input, weight, eps)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        backward_input, weight = ctx.saved_tensors
        grad_input, grad_weight, _ = _norm_backward(
            grad_output,
            backward_input,
            weight,
            None,
            normalization="RMSNorm",
            eps=ctx.eps,
            zero_centered_gamma=False,
        )
        if not ctx.needs_input_grad[1]:
            grad_input = None
        if not ctx.needs_input_grad[2]:
            grad_weight = None
        return None, grad_input, grad_weight, None, None, None


class _SGLangRouterGEMMWithMegatronBackward(torch.autograd.Function):
    """SGLang batch-invariant FP32 router forward with Megatron backward."""

    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.batch_invariant_ops import router_gemm_batch_invariant

        original_shape = value.shape
        output = router_gemm_batch_invariant(
            value.reshape(-1, original_shape[-1]),
            weight,
        ).view(*original_shape[:-1], -1)
        ctx.save_for_backward(value, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        value, weight = ctx.saved_tensors
        grad_input, grad_weight = router_gating_linear_backward(
            value,
            weight,
            grad_output,
            torch.float32,
        )
        if not ctx.needs_input_grad[0]:
            grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_weight = None
        return grad_input, grad_weight


class _SGLangSwiGLUWithAnalyticBackward(torch.autograd.Function):
    """SGLang fused SwiGLU forward with an analytic FP32 backward."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        from sgl_kernel import silu_and_mul

        output_shape = (*value.shape[:-1], value.shape[-1] // 2)
        output = torch.empty(output_shape, dtype=value.dtype, device=value.device)
        silu_and_mul(
            value.contiguous().view(-1, value.shape[-1]),
            output.view(-1, output.shape[-1]),
        )
        ctx.save_for_backward(value)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None
        (value,) = ctx.saved_tensors
        gate, up = value.chunk(2, dim=-1)
        gate_fp32 = gate.float()
        up_fp32 = up.float()
        grad_fp32 = grad_output.float()
        sigmoid = torch.sigmoid(gate_fp32)
        silu_gate = gate_fp32 * sigmoid
        grad_gate = grad_fp32 * up_fp32 * sigmoid * (1.0 + gate_fp32 * (1.0 - sigmoid))
        grad_up = grad_fp32 * silu_gate
        return torch.cat((grad_gate, grad_up), dim=-1).to(value.dtype)


class _DeepGEMMLinearWithBF16Backward(torch.autograd.Function):
    """FP8 DeepGEMM forward with BF16 GEMM dgrad/wgrad."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        output = _deepgemm_linear(input_, weight)
        ctx.input_shape = tuple(input_.shape)
        ctx.save_for_backward(input_, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view(-1, grad_output.shape[-1]).to(dtype=input_.dtype)
        input_2d = input_.contiguous().view(-1, input_.shape[-1])

        grad_input = _deepgemm_bf16_gemm_nn(grad_2d, weight).view(ctx.input_shape) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = _sum_to_parameter_dtype(
                _deepgemm_bf16_gemm_tn(grad_2d, input_2d),
                weight,
            )
        return grad_input, grad_weight


class _DeepGEMMLayerNormLinearWithBF16Backward(torch.autograd.Function):
    """FP8 DeepGEMM fused norm+linear forward with BF16 GEMM backward."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor | None,
        normalization: str,
        eps: float,
        zero_centered_gamma: bool,
    ) -> torch.Tensor:
        normalized = _norm_forward(
            input_,
            norm_weight,
            norm_bias,
            normalization=normalization,
            eps=eps,
            zero_centered_gamma=zero_centered_gamma,
        )
        output = _deepgemm_linear(normalized, weight)
        ctx.input_shape = tuple(input_.shape)
        ctx.normalization = normalization
        ctx.eps = eps
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.has_norm_bias = norm_bias is not None
        tensors = (input_, normalized, weight, norm_weight)
        if norm_bias is not None:
            tensors = (*tensors, norm_bias)
        ctx.save_for_backward(*tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        input_, normalized, weight, norm_weight = saved[:4]
        norm_bias = saved[4] if ctx.has_norm_bias else None

        grad_2d = grad_output.contiguous().view(-1, grad_output.shape[-1]).to(dtype=normalized.dtype)
        normalized_2d = normalized.contiguous().view(-1, normalized.shape[-1])
        grad_normalized = _deepgemm_bf16_gemm_nn(grad_2d, weight).view(ctx.input_shape)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = _sum_to_parameter_dtype(
                _deepgemm_bf16_gemm_tn(grad_2d, normalized_2d),
                weight,
            )
        grad_input, grad_norm_weight, grad_norm_bias = _norm_backward(
            grad_normalized,
            input_,
            norm_weight,
            norm_bias,
            normalization=ctx.normalization,
            eps=ctx.eps,
            zero_centered_gamma=ctx.zero_centered_gamma,
        )
        return (
            grad_input if ctx.needs_input_grad[0] else None,
            grad_weight,
            grad_norm_weight if ctx.needs_input_grad[2] else None,
            grad_norm_bias if ctx.needs_input_grad[3] else None,
            None,
            None,
            None,
        )


def _deepgemm_linear(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    import deep_gemm
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback

    from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import _configure_batch_invariant
    from slime.backends.megatron_utils.kernels.fp8_kernel import blockwise_cast_to_fp8_triton

    # Megatron actors are separate processes from SGLang workers.  Environment
    # propagation alone does not call DeepGEMM's process-local setter.
    _configure_batch_invariant(deep_gemm, deep_gemm_wrapper)

    if input_.dtype != torch.bfloat16:
        raise RuntimeError(f"DeepGEMM alignment forward requires BF16 input, got {input_.dtype}")
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(f"DeepGEMM alignment forward requires BF16 weight, got {weight.dtype}")
    if input_.shape[-1] != weight.shape[-1]:
        raise RuntimeError(f"DeepGEMM input/weight K mismatch: {input_.shape[-1]} != {weight.shape[-1]}")

    with torch.no_grad():
        if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            # Blackwell (sm100/sm103): the block-FP8 GEMM requires UE8M0
            # (power-of-two) weight scales. The SGLang rollout produces these by
            # quantizing the BF16 weight to FP8 (slime quantizer_fp8 ->
            # quant_weight_ue8m0) and then *requantizing* in
            # process_weights_after_loading (requant_weight_ue8m0). That requant
            # is lossy (recomputes per-block scales on the rounded values), so a
            # single quant_weight_ue8m0 here would NOT bit-match the rollout.
            # Replicate the same quant -> requant chain to align to ~e-7.
            from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, requant_weight_ue8m0

            qweight, weight_scale = quant_weight_ue8m0(weight.detach().contiguous(), [128, 128])
            qweight, weight_scale = requant_weight_ue8m0(qweight, weight_scale, [128, 128])
        else:
            # Hopper (sm90): FP32 block scales. Unchanged.
            qweight, weight_scale = blockwise_cast_to_fp8_triton(weight.detach().contiguous(), (128, 128))
        return deepgemm_w8a8_block_fp8_linear_with_fallback(
            input_.detach().contiguous(),
            qweight,
            [128, 128],
            weight_scale,
            bias=None,
        )


def _validate_custom_backward(module: torch.nn.Module) -> None:
    config = getattr(module, "config", None)
    if config is not None and getattr(config, "delay_wgrad_compute", False):
        raise RuntimeError(
            "DeepGEMM custom backward does not support Megatron delay_wgrad_compute; " "disable delay_wgrad_compute."
        )


def _wrap_te_linear(module: torch.nn.Module, module_name: str) -> bool:
    if getattr(module, "_slime_deepgemm_forward_wrapped", False):
        return False

    class_name = type(module).__name__
    if class_name not in _SUPPORTED_TE_CLASS_NAMES:
        raise RuntimeError(f"DeepGEMM target {module_name} has unsupported module class {class_name}")
    if getattr(module, "use_bias", False):
        raise RuntimeError(f"DeepGEMM alignment probe currently requires bias-free linears: {module_name}")

    _validate_custom_backward(module)
    is_layernorm_linear = "LayerNorm" in class_name

    def deepgemm_forward(self, input_):
        input_already_normalized = bool(getattr(self, "_deepgemm_input_already_normalized", False))
        if is_layernorm_linear and not input_already_normalized:
            config = getattr(self, "config", None)
            if config is None:
                raise RuntimeError(f"DeepGEMM custom backward for {module_name} requires module.config")
            norm_weight = getattr(self, "layer_norm_weight", None)
            if not isinstance(norm_weight, torch.Tensor):
                raise RuntimeError(f"{module_name} is missing layer_norm_weight")
            norm_bias = getattr(self, "layer_norm_bias", None)
            if norm_bias is not None and not isinstance(norm_bias, torch.Tensor):
                raise RuntimeError(f"{module_name}.layer_norm_bias is not a Tensor")
            output = _DeepGEMMLayerNormLinearWithBF16Backward.apply(
                input_,
                self.weight,
                norm_weight,
                norm_bias,
                getattr(config, "normalization", "RMSNorm"),
                float(getattr(config, "layernorm_epsilon", 1e-5)),
                bool(getattr(config, "layernorm_zero_centered_gamma", False)),
            )
            return output, None

        output = _DeepGEMMLinearWithBF16Backward.apply(
            input_,
            self.weight,
        )
        return output, None

    module.forward = types.MethodType(deepgemm_forward, module)
    module._slime_deepgemm_forward_wrapped = True
    module._slime_deepgemm_module_name = module_name
    return True


def _get_global_layer_index(model_chunk: torch.nn.Module, module_name: str) -> int | None:
    match = _LAYER_PATH_RE.search(module_name)
    if match is None:
        return None
    layer_path = match.group("layer_path")
    layer = model_chunk.get_submodule(layer_path)
    layer_number = getattr(layer, "layer_number", None)
    if layer_number is None:
        raise RuntimeError(
            "DeepGEMM requires TransformerLayer.layer_number to select global pipeline "
            f"layers, but {layer_path} (from {module_name}) has no layer_number"
        )
    return int(layer_number) - 1


def _as_set(values: Iterable | None, default: Iterable) -> set:
    return set(default if values is None else values)


def enable_deepgemm_forward(args, model, store_prefix: str) -> None:
    """Install the forward-value replacement on selected full-matrix TE linears."""
    del store_prefix
    if parallel_state.get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError("The initial DeepGEMM alignment probe requires tensor model parallel size 1")

    target_layers = _as_set(args.megatron_deepgemm_forward_layers, ())
    if not target_layers:
        raise RuntimeError("--megatron-deepgemm-forward-layers must select at least one layer")
    target_suffixes = _as_set(
        args.megatron_deepgemm_forward_modules,
        _DEFAULT_TARGET_SUFFIXES,
    )
    if isinstance(model, torch.nn.Module):
        model_chunks = [model]
    else:
        try:
            model_chunks = list(model)
        except TypeError:
            # Hook-composition tests may use an opaque sentinel while
            # monkeypatching the concrete installers.
            return

    wrapped = []
    for model_chunk in model_chunks:
        for name, module in model_chunk.named_modules():
            if not any(name.endswith(suffix) for suffix in target_suffixes):
                continue
            global_layer_index = _get_global_layer_index(model_chunk, name)
            if global_layer_index not in target_layers:
                continue
            if _wrap_te_linear(module, name):
                wrapped.append(name)

    if wrapped and _should_log_deepgemm_summary():
        logger.info(
            "Enabled SGLang DeepGEMM forward+BF16-backward on %d Megatron linears (layers=%s)",
            len(wrapped),
            _format_int_ranges(target_layers),
        )
        logger.debug("DeepGEMM wrapped Megatron linears: %s", ", ".join(wrapped))


def enable_sglang_router_gemm(args, model, store_prefix: str) -> None:
    """Use one SGLang persistent FP32 forward with Megatron GEMM backward."""

    del store_prefix
    saved_rollout_replay = bool(
        getattr(args, "debug_train_only", False) and getattr(args, "load_debug_rollout_data", None)
    )
    if not (getattr(args, "sglang_enable_fp32_moe_router", False) or saved_rollout_replay):
        raise RuntimeError(
            "MEGATRON_USE_SGLANG_ROUTER_GEMM=1 requires "
            "--sglang-enable-fp32-moe-router so Megatron and SGLang "
            "use the same FP32 router forward"
        )
    target_layers = _as_set(
        getattr(args, "megatron_deepgemm_moe_forward_layers", None),
        (),
    )
    if not target_layers:
        return

    wrapped = []
    for model_chunk in model:
        for name, module in model_chunk.named_modules():
            if not name.endswith("mlp.router"):
                continue
            global_layer_index = _get_global_layer_index(model_chunk, name)
            if global_layer_index not in target_layers:
                continue
            if getattr(module, "_slime_sglang_router_gemm_wrapped", False):
                continue

            original_gating = module.gating

            def gating(
                self,
                value: torch.Tensor,
            ) -> torch.Tensor:
                if self.weight.device.type == "cpu" and value.is_cuda:
                    self.weight.data = self.weight.data.to(device=value.device)
                router_dtype = getattr(
                    getattr(self, "config", None),
                    "moe_router_dtype",
                    "fp32",
                )
                if router_dtype != "fp32":
                    raise RuntimeError(
                        "SGLang persistent router alignment requires " f"moe_router_dtype=fp32, got {router_dtype}"
                    )
                return _SGLangRouterGEMMWithMegatronBackward.apply(
                    value,
                    self.weight,
                )

            module.gating = types.MethodType(gating, module)
            module._slime_sglang_router_gemm_wrapped = True
            module._slime_sglang_router_gemm_original = original_gating
            wrapped.append(name)

    if wrapped and _should_log_deepgemm_summary():
        logger.info(
            "Enabled SGLang batch-invariant FP32 router forward on %d routers "
            "(layers=%s); Megatron backward retained",
            len(wrapped),
            _format_int_ranges(target_layers),
        )


def enable_sglang_global_batch_invariant_ops() -> None:
    """Enable the same process-global deterministic operators as SGLang.

    SGLang turns these operators on from ``ModelRunner`` when deterministic
    inference is requested.  Megatron actors are separate processes, so
    DeepGEMM's process-local batch-invariant switch alone is insufficient:
    RMS reductions, BMMs, FP32 matmuls, and log-softmax would otherwise still
    use Megatron's normal batch-shaped kernels.
    """
    if os.environ.get("SGLANG_DEEPGEMM_BATCH_INVARIANT", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode, is_batch_invariant_mode_enabled

    enable_batch_invariant_mode()
    if not is_batch_invariant_mode_enabled():
        raise RuntimeError("SGLang global batch-invariant mode did not enable")


def _sglang_batch_invariant_rmsnorm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant

    return rms_norm_batch_invariant(value, weight, eps)


def _sglang_rmsnorm_from_fp32_sum(
    value_fp32: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Match SGLang's native RMSNorm on an unrounded FP32 residual sum."""
    variance = value_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = value_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight).to(output_dtype)


def enable_sglang_layer0_input_rmsnorm(
    args,
    model,
    store_prefix: str,
) -> None:
    """Match standalone replay's independent RMSNorm at every PP boundary.

    Later Transformer layers consume the explicit FP32 residual sum and use
    Megatron's SGLang-aligned fused residual/RMSNorm path.  The first local
    layer on each pipeline stage has no local preceding residual sum, so it
    still calls its standalone TE RMSNorm module.  The standalone accuracy
    replay replaces all such modules with SGLang's batch-invariant RMSNorm; do
    the same here.  Checking for global layer zero is insufficient with PP>1:
    stage 1 in the canonical PP8 layout starts at global layer 2.

    Training uses an analytic RMSNorm backward, so the original TE forward is
    not evaluated a second time.
    """
    del args, store_prefix
    if os.environ.get("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "0") != "1":
        return

    if isinstance(model, torch.nn.Module):
        model_chunks = [model]
    else:
        try:
            model_chunks = list(model)
        except TypeError:
            return

    patched = []
    for model_chunk in model_chunks:
        local_layers = []
        for layer_name, layer in model_chunk.named_modules():
            match = re.match(
                r"^(?:.*\.)?decoder\.layers\.(\d+)$",
                layer_name,
            )
            if match is not None:
                local_layers.append((int(match.group(1)), layer_name, layer))
        if not local_layers:
            continue

        # Each model chunk owns one contiguous local decoder.  Its first layer
        # is the independent RMS boundary even when its global layer number is
        # greater than one because it follows a pipeline send/recv.
        _, layer_name, layer = min(local_layers, key=lambda item: item[0])
        module = getattr(layer, "input_layernorm", None)
        if module is None or getattr(module, "_slime_sglang_pipeline_input_rmsnorm_wrapped", False):
            continue
        if not hasattr(module, "weight") or not hasattr(module, "eps"):
            raise RuntimeError(f"{layer_name}.input_layernorm is missing RMSNorm weight/eps")

        original_forward = module.forward

        def forward(
            patched_module: torch.nn.Module,
            value: torch.Tensor,
        ) -> torch.Tensor:
            return _SGLangRMSNormWithAnalyticBackward.apply(
                value.detach(),
                value,
                patched_module.weight,
                float(patched_module.eps),
                False,
                value.dtype,
            )

        module.forward = types.MethodType(forward, module)
        module._slime_sglang_pipeline_input_rmsnorm_wrapped = True
        module._slime_sglang_pipeline_input_rmsnorm_original = original_forward
        patched.append(f"{layer_name}.input_layernorm")

    if patched and _should_log_deepgemm_summary():
        logger.info(
            "Enabled single-pass SGLang batch-invariant pipeline-stage input "
            "RMSNorm with analytic backward on %d module(s)",
            len(patched),
        )


def enable_sglang_absorbed_kv_rmsnorm(
    args,
    model,
    store_prefix: str,
) -> None:
    """Match SGLang's RMSNorm for the absorbed MLA KV latent.

    MCore invokes ``torch.nn.functional.rms_norm`` directly for this value, so
    neither the standalone TE RMSNorm replacement nor the fused-linear wrapper
    reaches it.  The standalone accuracy harness scopes a replacement to each
    attention forward.  Reproduce that behavior with a direct analytic
    backward instead of evaluating a native RMSNorm surrogate.
    """
    del store_prefix
    if os.environ.get("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "0") != "1":
        return

    target_layers = _as_set(
        getattr(args, "megatron_deepgemm_forward_layers", None),
        (),
    )
    if isinstance(model, torch.nn.Module):
        model_chunks = [model]
    else:
        try:
            model_chunks = list(model)
        except TypeError:
            return

    wrapped = []
    for model_chunk in model_chunks:
        for layer_name, layer in model_chunk.named_modules():
            if re.match(r"^(?:.*\.)?decoder\.layers\.\d+$", layer_name) is None:
                continue
            global_layer = int(getattr(layer, "layer_number", 0)) - 1
            if target_layers and global_layer not in target_layers:
                continue
            attention = getattr(layer, "self_attention", None)
            kv_up = getattr(attention, "linear_kv_up_proj", None)
            target_weight = getattr(kv_up, "layer_norm_weight", None)
            if attention is None or target_weight is None:
                continue
            if getattr(attention, "_slime_sglang_absorbed_kv_rmsnorm_wrapped", False):
                continue

            original_forward = attention.forward

            def forward(
                patched_attention: torch.nn.Module,
                *forward_args,
                original_forward=original_forward,
                target_weight=target_weight,
                **forward_kwargs,
            ):
                original_rms_norm = torch.nn.functional.rms_norm

                def matched_rms_norm(
                    value: torch.Tensor,
                    normalized_shape,
                    weight: torch.Tensor | None = None,
                    eps: float | None = None,
                ) -> torch.Tensor:
                    expected_shape = (target_weight.numel(),)
                    if value.shape[-1] != target_weight.numel() or tuple(normalized_shape) != expected_shape:
                        return original_rms_norm(
                            value,
                            normalized_shape,
                            weight=weight,
                            eps=eps,
                        )

                    effective_eps = float(patched_attention.config.layernorm_epsilon if eps is None else eps)
                    return _SGLangRMSNormWithAnalyticBackward.apply(
                        value.to(dtype=target_weight.dtype).detach(),
                        value,
                        target_weight,
                        effective_eps,
                        False,
                        target_weight.dtype,
                    )

                torch.nn.functional.rms_norm = matched_rms_norm
                try:
                    return original_forward(*forward_args, **forward_kwargs)
                finally:
                    torch.nn.functional.rms_norm = original_rms_norm

            attention.forward = types.MethodType(forward, attention)
            attention._slime_sglang_absorbed_kv_rmsnorm_wrapped = True
            attention._slime_sglang_absorbed_kv_rmsnorm_original = original_forward
            wrapped.append(f"{layer_name}.self_attention")

    if wrapped and _should_log_deepgemm_summary():
        logger.info(
            "Enabled SGLang batch-invariant absorbed-KV RMSNorm forward "
            "with analytic backward on %d attention module(s) (layers=%s)",
            len(wrapped),
            _format_int_ranges(target_layers),
        )


def enable_sglang_final_rmsnorm(
    args,
    model,
    store_prefix: str,
) -> None:
    """Match the standalone replay's final RMSNorm visible forward.

    Compact layer outputs are captured before ``decoder.final_layernorm``.
    Consequently, all Transformer layers can be bitwise identical while the
    final logprobs still differ if the training path keeps Transformer
    Engine's final RMS kernel.  The standalone replay uses SGLang's RMSNorm
    and, when available, normalizes the unrounded FP32 residual sum retained
    by the last Transformer layer.

    Use an analytic RMSNorm backward evaluated from the same input that the
    original final norm consumed.  When an exact FP32 residual is available,
    it remains visible-forward-only to preserve the established gradient
    topology without executing the original TE forward.
    """
    del args, store_prefix
    if os.environ.get("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "0") != "1":
        return

    if isinstance(model, torch.nn.Module):
        model_chunks = [model]
    else:
        try:
            model_chunks = list(model)
        except TypeError:
            return

    patched = []
    for model_chunk in model_chunks:
        named_modules = dict(model_chunk.named_modules())
        for module_name, module in named_modules.items():
            if not module_name.endswith("decoder.final_layernorm"):
                continue
            if getattr(module, "_slime_sglang_final_rmsnorm_wrapped", False):
                continue
            if not hasattr(module, "weight") or not hasattr(module, "eps"):
                raise RuntimeError(f"{module_name} is missing RMSNorm weight/eps")

            decoder_name = module_name.removesuffix(".final_layernorm")
            decoder = named_modules.get(decoder_name)
            layers = getattr(decoder, "layers", None)
            residual_source = layers[-1] if layers else None
            original_forward = module.forward

            def forward(
                patched_module: torch.nn.Module,
                value: torch.Tensor,
                *,
                residual_source=residual_source,
            ) -> torch.Tensor:
                exact_residual_sum = getattr(
                    value,
                    "_sglang_residual_sum_fp32",
                    None,
                )
                if exact_residual_sum is None and residual_source is not None:
                    exact_residual_sum = getattr(
                        residual_source,
                        "_sglang_residual_sum_fp32",
                        None,
                    )

                visible_input = value if exact_residual_sum is None else exact_residual_sum
                return _SGLangRMSNormWithAnalyticBackward.apply(
                    visible_input.detach(),
                    value,
                    patched_module.weight,
                    float(patched_module.eps),
                    exact_residual_sum is not None,
                    value.dtype,
                )

            module.forward = types.MethodType(forward, module)
            module._slime_sglang_final_rmsnorm_wrapped = True
            module._slime_sglang_final_rmsnorm_original = original_forward
            patched.append(module_name)

    if patched and _should_log_deepgemm_summary():
        logger.info(
            "Enabled single-pass SGLang-aligned final RMSNorm with analytic " "backward on %d module(s)",
            len(patched),
        )


def _sglang_swiglu_with_megatron_backward(value: torch.Tensor) -> torch.Tensor:
    """Use one SGLang BF16 SwiGLU forward with an analytic backward."""
    if value.shape[-1] % 2:
        raise RuntimeError(f"SwiGLU input width must be even, got {value.shape[-1]}")
    return _SGLangSwiGLUWithAnalyticBackward.apply(value)


def _wrap_sglang_swiglu_mlp(
    module: torch.nn.Module,
    module_name: str,
    *,
    return_tuple: bool,
) -> bool:
    if getattr(module, "_slime_sglang_swiglu_wrapped", False):
        return False
    if not hasattr(module, "linear_fc1") or not hasattr(module, "linear_fc2"):
        raise RuntimeError(f"SwiGLU target {module_name} is not an MLP")

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_token_scale: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        # Current Megatron passes this keyword to both dense MLPs and MoE
        # layers.  Its native dense MLP accepts and ignores it; padding tokens
        # are excluded by the loss mask rather than altered inside the MLP.
        del padding_mask
        if per_token_scale is not None:
            raise RuntimeError(f"SGLang SwiGLU alignment does not support per_token_scale: {module_name}")
        gate_up, bias = self.linear_fc1(hidden_states)
        if bias is not None:
            raise RuntimeError(f"SGLang SwiGLU alignment requires a bias-free MLP: {module_name}")
        down_input = _sglang_swiglu_with_megatron_backward(gate_up)
        output, output_bias = self.linear_fc2(down_input)
        if output_bias is not None:
            raise RuntimeError(f"SGLang SwiGLU alignment requires a bias-free MLP: {module_name}")

        if not return_tuple:
            if getattr(self, "use_shared_expert_gate", False):
                logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
                output = output * torch.sigmoid(logits)
            return output
        return output, None

    module.forward = types.MethodType(forward, module)
    module._slime_sglang_swiglu_wrapped = True
    module._slime_sglang_swiglu_module_name = module_name
    return True


def enable_sglang_swiglu_forward(args, model, store_prefix: str) -> None:
    """Match SGLang's fused SwiGLU on dense and shared-expert MLPs."""
    del store_prefix
    target_layers = _as_set(
        getattr(args, "megatron_deepgemm_forward_layers", None),
        (),
    )
    if not target_layers:
        return

    if isinstance(model, torch.nn.Module):
        model_chunks = [model]
    else:
        try:
            model_chunks = list(model)
        except TypeError:
            return

    wrapped = []
    for model_chunk in model_chunks:
        for layer_name, layer in model_chunk.named_modules():
            match = re.match(r"^(?:.*\.)?decoder\.layers\.\d+$", layer_name)
            if match is None:
                continue
            layer_number = getattr(layer, "layer_number", None)
            if layer_number is None or int(layer_number) - 1 not in target_layers:
                continue
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "experts"):
                shared = getattr(mlp, "shared_experts", None)
                if shared is not None and _wrap_sglang_swiglu_mlp(
                    shared,
                    f"{layer_name}.mlp.shared_experts",
                    return_tuple=False,
                ):
                    wrapped.append(f"{layer_name}.mlp.shared_experts")
            elif _wrap_sglang_swiglu_mlp(
                mlp,
                f"{layer_name}.mlp",
                return_tuple=True,
            ):
                wrapped.append(f"{layer_name}.mlp")

    if wrapped and _should_log_deepgemm_summary():
        logger.info(
            "Enabled single-pass SGLang SwiGLU forward with analytic backward on %d MLPs " "(layers=%s)",
            len(wrapped),
            _format_int_ranges(target_layers),
        )


def _enable_deepgemm_all_forward(
    args,
    model,
    store_prefix: str,
) -> None:
    """Install selected dense/indexer/shared and routed-MoE forward paths."""
    linear_layers = getattr(args, "megatron_deepgemm_forward_layers", None)
    moe_layers = getattr(args, "megatron_deepgemm_moe_forward_layers", None)
    if not linear_layers and not moe_layers:
        raise RuntimeError(
            "The combined DeepGEMM hook requires at least one of "
            "--megatron-deepgemm-forward-layers or "
            "--megatron-deepgemm-moe-forward-layers"
        )

    enable_sglang_global_batch_invariant_ops()
    enable_sglang_layer0_input_rmsnorm(args, model, store_prefix)
    enable_sglang_absorbed_kv_rmsnorm(args, model, store_prefix)
    enable_sglang_final_rmsnorm(args, model, store_prefix)

    if linear_layers:
        enable_deepgemm_forward(args, model, store_prefix)
        enable_sglang_swiglu_forward(args, model, store_prefix)
    if moe_layers:
        from slime.backends.megatron_utils.alignment.deepgemm_moe_forward import (
            enable_deepgemm_moe_forward,
            enable_sglang_deepep_moe_alignment,
        )

        enable_deepgemm_moe_forward(args, model, store_prefix)
        if os.environ.get("MEGATRON_USE_SGLANG_ROUTER_GEMM", "0") == "1":
            enable_sglang_router_gemm(args, model, store_prefix)
        enable_sglang_deepep_moe_alignment(args, model, store_prefix)

    if os.getenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR"):
        from slime.backends.megatron_utils.alignment.layerwise_alignment import enable_megatron_layerwise_dump

        enable_megatron_layerwise_dump(args, model, store_prefix)


def enable_deepgemm_all_forward(args, model, store_prefix: str) -> None:
    """Install every selected alignment path (native LM head)."""
    _enable_deepgemm_all_forward(args, model, store_prefix)


def enable_deepgemm_all_forward_before_train_step(
    args,
    rollout_id: int,
    step_id: int,
    model,
    optimizer,
    opt_param_scheduler,
) -> None:
    """Install alignment with the native LM head before the train step."""
    del rollout_id, step_id, opt_param_scheduler
    enable_deepgemm_all_forward(args, model, store_prefix="")
    del optimizer
