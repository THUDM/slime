import re
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .qwen2 import convert_qwen2_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


def ceildiv(a, b):
    return -(-a // b)


# Original implementation reference:
# https://github.com/pytorch/ao/blob/main/torchao/prototype/blockwise_fp8/blockwise_quantization.py
@triton.jit
def fp8_blockwise_weight_quant_kernel(
    x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factors in `s_ptr`.

    Args:
        x_ptr (tl.pointer): Pointer to the input tensor.
        y_ptr (tl.pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (tl.pointer): Pointer to the output tensor where scaling factors will be stored.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


# Original implementation reference:
# https://github.com/pytorch/ao/blob/main/torchao/prototype/blockwise_fp8/blockwise_quantization.py
def fp8_blockwise_weight_quant(
    x: torch.Tensor, block_size: int = 128, dtype=torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the given weight tensor using block-wise quantization with block size being BLOCK_SIZExBLOCK_SIZE.

    Args:
        x (torch.Tensor): The weight tensor to be quantized.
        block_size (int, optional): The block size to use for quantization. Defaults to 128.
        dtype (torch.dtype, optional): The dtype to use for the quantized tensor. Defaults to `torch.float8_e4m3fn`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized weight tensor with dtype `dtype`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    assert x.size(0) % block_size == 0 and x.size(1) % block_size == 0, (
        f"Both dimensions of x must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(M // block_size, N // block_size, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    fp8_blockwise_weight_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s

def _pad_to_tile(x, tile=128):
    M, K = x.shape
    Mp = triton.cdiv(M, tile) * tile
    Kp = triton.cdiv(K, tile) * tile
    if Mp == M and Kp == K:
        return x, 0, 0
    x_pad = F.pad(x, (0, Kp - K, 0, Mp - M), value=0.0)
    return x_pad, Mp - M, Kp - K


def quantize_param(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size == (128, 128):
        # triton per block quant
            tile = 128
            w_pad, _, _ = _pad_to_tile(weight, tile)
            q_pad, scale = fp8_blockwise_weight_quant(w_pad, block_size=tile, dtype=torch.float8_e4m3fn)

            M, K = weight.shape
            qweight = q_pad[:M, :K]
            n_tiles_m = triton.cdiv(M, tile)
            n_tiles_k = triton.cdiv(K, tile)
            scale = scale[:n_tiles_m, :n_tiles_k].contiguous()
            scale_name = name.replace(".weight", ".weight_scale_inv")
    elif weight_block_size is not None:
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


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    assert quantization_config["quant_method"] == "fp8"
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

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
                quantize_named_params.extend(quantize_param(converted_name, param, weight_block_size))

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
                quantize_named_params.extend(quantize_param(converted_name, param, weight_block_size))

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
            quantize_named_params.extend(quantize_param(converted_name, param, weight_block_size))

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
