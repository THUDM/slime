import re
import torch
import torch.nn.functional as F

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .qwen2 import convert_qwen2_to_hf
from .qwen3moe import convert_qwen3moe_to_hf
import triton
import triton.language as tl


def ceildiv(a, b):
    return -(-a // b)


@triton.jit
def per_block_fp8_quant_kernel(
    weight_ptr,
    qweight_ptr,
    scale_ptr,
    n_tiles_n,
    n_tiles_k,
    n,
    k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_n = offs_n < n
    mask_k = offs_k < k
    mask = mask_n[:, None] & mask_k[None, :]

    weight_block = tl.load(weight_ptr + offs_n[:, None] * k + offs_k[None, :], mask=mask, other=0.0)

    block_max = tl.max(tl.abs(weight_block))

    scale = tl.where(block_max == 0.0, 1.0, block_max / FP8_MIN)

    tl.store(scale_ptr + pid_n * n_tiles_k + pid_k, scale)

    qweight = tl.where(mask, tl.clamp(weight_block / scale, FP8_MIN, FP8_MAX), 0.0)
    tl.store(qweight_ptr + offs_n[:, None] * k + offs_k[None, :], qweight, mask=mask)


def per_block_fp8_quant_triton(name, weight, weight_block_size):
    """Triton-based quantize_param function with same interface as PyTorch version"""
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"

    n, k = weight.shape
    block_n, block_k = weight_block_size
    n_tiles_n = (n + block_n - 1) // block_n
    n_tiles_k = (k + block_k - 1) // block_k

    # Allocate output tensors
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((n_tiles_n, n_tiles_k), dtype=torch.float32, device=weight.device)

    # Launch Triton kernel
    grid = (n_tiles_n, n_tiles_k)
    FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
    FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

    per_block_fp8_quant_kernel[grid](
        weight, qweight, scales, n_tiles_n, n_tiles_k, n, k, block_n, block_k, FP8_MIN, FP8_MAX
    )

    # Return in same format as PyTorch version
    scale_name = name.replace(".weight", ".weight_scale_inv")
    return [(name, qweight), (scale_name, scales.squeeze())]


def quantize_param_torch(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        # per block quant
        block_n, block_k = weight_block_size[0], weight_block_size[1]

        shape_0, shape_1 = weight.shape

        n_tiles = ceildiv(shape_0, block_n)
        k_tiles = ceildiv(shape_1, block_k)

        q_weight = F.pad(
            weight,
            (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
            mode="constant",
            value=0.0,
        )

        qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
        block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
        block_max = torch.max(block_max, dim=3, keepdim=True)[0]

        scale = block_max.to(torch.float32) / FP8_MAX
        qweight = (
            (qweight / scale)
            .clamp(min=FP8_MIN, max=FP8_MAX)
            .reshape((n_tiles * block_n, k_tiles * block_k))
            .to(torch.float8_e4m3fn)
        )
        qweight = qweight[:shape_0, :shape_1]
        scale = scale.squeeze()
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # per tensor quant
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]


def quantize_params(args, megatron_name, converted_named_params, quantization_config, use_triton_kernel=True):
    if quantization_config is None:
        return converted_named_params
    assert quantization_config["quant_method"] == "fp8"
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

    quantize_param_func = per_block_fp8_quant_triton if use_triton_kernel else quantize_param_torch

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        return converted_named_params

    layer_idx, rest = match.groups()
    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(quantize_param_func(converted_name, param, weight_block_size))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(quantize_param_func(converted_name, param, weight_block_size))

            return quantize_named_params

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(quantize_param_func(converted_name, param, weight_block_size))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


cached_tensors = {}


def convert_to_hf(args, model_name, name, param, quantization_config=None):
    if "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
        # to compatible with sglang implementation
        if args.q_lora_rank is not None:
            old_converted_named_tensors = converted_named_tensors
            converted_named_tensors = []
            for converted_name, converted_param in old_converted_named_tensors:
                if "q_a_proj" in converted_name:
                    pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                    if pair_name in cached_tensors:
                        converted_named_tensors += [
                            (converted_name, converted_param),
                            (pair_name, cached_tensors[pair_name]),
                        ]
                        del cached_tensors[pair_name]
                    else:
                        cached_tensors[converted_name] = converted_param
                elif "kv_a_proj_with_mqa" in converted_name:
                    pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                    if pair_name in cached_tensors:
                        converted_named_tensors += [
                            (converted_name, converted_param),
                            (pair_name, cached_tensors[pair_name]),
                        ]
                        del cached_tensors[pair_name]
                    else:
                        cached_tensors[converted_name] = converted_param
                else:
                    converted_named_tensors.append((converted_name, converted_param))

    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not quantization_config:
        return converted_named_tensors

    return quantize_params(args, name, converted_named_tensors, quantization_config)
