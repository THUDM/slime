import re
import torch
from typing import Dict, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import math  # 引入 math 模块

try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    should_deepgemm_weight_requant_ue8m0 = None
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None

try:
    from slime.utils.fp8_kernel import blockwise_cast_to_fp8_triton
except ImportError:
    blockwise_cast_to_fp8_triton = None


def pack_to_int32(
        value: torch.Tensor,
        num_bits: int,
        packed_dim: Union[Literal[0], Literal[1]] = 1,
) -> torch.Tensor:
    """
    Packs a tensor of quantized weights stored in int8 into int32s with padding

    Pseudocode:
     1. Shift wrt num_bits to convert to unsigned. num_bits=8
        [1,2] -> [129, 130]
     2. Pad to fill in 32 bits
        [129, 130] -> [129, 130, 0, 0]
     3. convert to binary align in order
        [129, 130, 0, 0] -> 00000000 00000000 10000010 10000001
     4. convert aligned binary to number
        00000000000000001000001010000001 -> 33409
     5. covert back to uint32
        33409 -> 33409

    :param value: tensor to pack
    :param num_bits: number of bits used to store underlying data, must be at least 1
    :returns: packed int32 tensor
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def pack_int4_to_int32(q_weight: torch.Tensor) -> torch.Tensor:
    """
    将 Int4 数据 (范围 [0, 15] 或 [-8, 7]) 打包成 Int32 格式 (GPTQ Style)。
    每 8 个 Int4 数值打包成 1 个 Int32。

    Args:
        q_weight: [N, K] tensor, dtype=int8 or uint8
    Returns:
        packed: [N, K // 8] tensor, dtype=int32
    """
    return pack_to_int32(q_weight, 4, -1)

    N, K = q_weight.shape
    assert K % 8 == 0, f"K ({K}) must be divisible by 8 for Int32 packing."

    # 1. 准备容器
    # 将最后维度变成 (K//8, 8)，方便并行处理 8 个数
    w_view = q_weight.view(N, K // 8, 8).to(torch.int32)

    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=q_weight.device)

    # 2. 循环移位打包 (GPTQ order: first weight is LSB)
    # 第 0 个数 -> bits 0-3
    # 第 1 个数 -> bits 4-7
    # ...
    # 第 7 个数 -> bits 28-31
    for i in range(8):
        # 取出第 i 列，限制在 4bit (mask 0xF)，然后左移 i*4 位
        val = (w_view[..., i] & 0xF) << (i * 4)
        packed |= val

    return packed


def int4_block_quantize(
        x: torch.Tensor,
        group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    纯对称量化 (Symmetric Quantization) 转换函数。
    逻辑：Signed Int4 Representation (-7 to 7)
    公式：De-quantized = Scale * Quantized (Zero Point is always 0)
    """
    N, K = x.shape
    if group_size == -1:
        group_size = K

    # Padding (如果 K 不能整除 group_size)
    if K % group_size != 0:
        import torch.nn.functional as F
        x = F.pad(x, (0, group_size - (K % group_size)))
        N, K = x.shape

    num_groups = K // group_size
    x_reshaped = x.float().view(N, num_groups, group_size)

    # =========================================================
    # 1. Scale 计算
    #    对称量化基于绝对值最大值
    #    Range: [-7, 7] -> dividing by 7.0
    # =========================================================
    x_abs_max = x_reshaped.abs().amax(dim=-1, keepdim=True)
    scale = x_abs_max / 7
    scale = scale.clamp(min=1e-5)  # 防止除以0

    # =========================================================
    # 2. 量化 (Quantize)
    #    直接保留有符号整数 [-7, 7]
    # =========================================================
    # 注意：这里不再加 8，也不再强制转为 unsigned
    x_int_sym = (x_reshaped / scale).round().clamp(-8, 7)

    out = x_int_sym.to(torch.int8)

    # =========================================================
    # 3. Zero Point
    #    对称量化中，零点恒为 0
    # =========================================================
    zero_point = torch.zeros_like(scale)

    # 4. 格式化输出
    out = out.view(N, K)

    scale_out = scale.squeeze(-1).contiguous()
    zero_out = zero_point.squeeze(-1).contiguous()

    return out, scale_out, zero_out


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params

    quant_method = quantization_config.get("quant_method")

    if quant_method == "fp8":
        return _quantize_params_fp8(megatron_name, converted_named_params, quantization_config)
    elif quant_method == "compressed-tensors":
        # 把过滤逻辑交给具体的函数去处理
        return _quantize_params_int4(converted_named_params, quantization_config)
    else:
        return converted_named_params


def _quantize_params_int4(converted_named_params, quantization_config):
    # 1. 解析配置
    try:
        w_cfg = quantization_config["config_groups"]["group_0"]["weights"]
        group_size = w_cfg["group_size"]
        is_symmetric = w_cfg["symmetric"]
        ignore_rules = quantization_config.get("ignore", [])
    except KeyError as e:
        raise ValueError(f"Invalid quantization_config: missing {e}")

    results = []
    # print(f"\n====== Quantization Start (GroupSize={group_size}) ======")

    for name, param in converted_named_params:
        # 2. 检查是否需跳过 (Ignore规则匹配)
        # 使用 any() + 生成器表达式简化逻辑
        # print(f'[Debug quant slime 0] \n'
        #     f'name is {name}',
        #     f'input_tensor shape is {param.shape}')

        is_ignored = any(
            (r.startswith("re:") and re.match(r[3:], name)) or r == name
            for r in ignore_rules
        )

        # 3. 统一处理跳过的情况
        # 条件：被忽略 OR 不是权重后缀 OR 是一维张量(Bias/Norm)
        if is_ignored or not name.endswith(".weight") or param.dim() < 2:
            if is_ignored:
                print(f"IGNORING: {name}")
            results.append((name, param))
            continue

        # 4. 形状预处理 (兼容 3D MoE: [E, N, K] -> [E*N, K])
        # 如果 dim > 2 则展平，否则保持原样
        input_tensor = param.view(-1, param.shape[-1]) if param.dim() > 2 else param

        # 5. 尺寸安全检查
        if group_size != -1 and input_tensor.shape[-1] < group_size:
            print(f"WARNING: Skipping {name}, K-dim {input_tensor.shape[-1]} < group_size")
            results.append((name, param))
            continue

        # print(f'[Debug quant slime 1] \n'
        #      f'name is {name}',
        #      f'input_tensor shape is {param.shape}')
        # 6. 执行量化
        results.extend(_quantize_param_int4(
            name,
            input_tensor,
            group_size,
            param.shape,  # 传入原始形状用于后续恢复
            is_symmetric
        ))

    # print("====== Quantization End ======\n")
    return results


def _quantize_param_int4(name: str, weight: torch.Tensor, group_size: int, shape: torch.Tensor, is_symmetric: bool):
    """
    Wraps the quantization function, handles renaming and packing. 统一使用非对称逻辑。
    """

    # --- Renaming Logic (保持不变) ---

    base_name = name.replace(".weight", "")

    new_base_name = base_name

    # print(f"[Debug the old name ] is  {name} \n\
    #         the new name is {new_base_name}")

    original_dtype = weight.dtype

    if group_size == -1:
        group_size = weight.shape[1]
    elif weight.shape[1] % group_size != 0:
        print(
            f"Warning: Weight {name} with shape {weight.shape} K dim is not divisible by group_size {group_size}. Skipping.")
        return [(name, weight.to(original_dtype))]

    q_weight, scales, zeros = int4_block_quantize(weight, group_size)

    packed_q_weight = pack_int4_to_int32(q_weight)

    qweight_name = f"{new_base_name}.weight_packed"
    scales_name = f"{new_base_name}.weight_scale"
    qweight_shape = f"{new_base_name}.weight_shape"

    q_shape = torch.tensor(shape, dtype=torch.int32, device='cuda')

    # print(f'[Debug quant slime over] \n'
    #         f'name is {qweight_name} \n'
    #         f'packed_q_weight shape is {packed_q_weight.shape}', flush=True)
    # print(f"[Debug the {qweight_name} shape] is  {packed_q_weight.shape} \n\
    #         the {scales_name} shape is {scales.shape}")

    return [
        (qweight_name, packed_q_weight),
        (scales_name, scales.to(original_dtype)),
        (qweight_shape, q_shape)
    ]


def _quantize_params_fp8(megatron_name, converted_named_params, quantization_config):
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

    quantize_named_params = []

    quant_targets = [
        "linear_proj.weight", "linear_qkv.weight", "linear_fc1.weight", "linear_fc2.weight"
    ]

    should_quantize = False
    for converted_name, param in converted_named_params:
        if converted_name.endswith(".weight") and any(target in converted_name for target in quant_targets):
            should_quantize = True
            break

    if should_quantize:
        for converted_name, param in converted_named_params:
            if converted_name.endswith(".weight") and any(target in converted_name for target in quant_targets):
                quantize_named_params.extend(_quantize_param_fp8(converted_name, param, weight_block_size))
            else:
                quantize_named_params.append((converted_name, param))
        return quantize_named_params

    return converted_named_params


def _quantize_param_fp8(name, weight, weight_block_size):
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"

    if blockwise_cast_to_fp8_triton is None:
        print("Warning: FP8 kernel dependencies are missing. Using per-tensor quant.")

    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

    if weight_block_size is not None and blockwise_cast_to_fp8_triton is not None:
        if should_deepgemm_weight_requant_ue8m0 and should_deepgemm_weight_requant_ue8m0(
                weight_block_size=weight_block_size
        ):
            qweight, scale = quant_weight_ue8m0(weight, weight_block_size=weight_block_size)
            scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        else:
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")

    return [(name, qweight), (scale_name, scale)]
