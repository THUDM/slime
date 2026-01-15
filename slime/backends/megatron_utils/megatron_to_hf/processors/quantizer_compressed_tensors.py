import logging
import math
import re
from typing import Literal

import torch

logger = logging.getLogger(__name__)


__all__ = ["quantize_params_compressed_tensors"]


def pack_to_int32(
    value: torch.Tensor,
    num_bits: int,
    packed_dim: Literal[0] | Literal[1] = 1,
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
    pack int4 to int32
    Args:
        q_weight: [N, K] tensor, dtype=int8 or uint8
    Returns:
        packed: [N, K // 8] tensor, dtype=int32
    """
    return pack_to_int32(q_weight, 4, -1)


def int4_block_quantize(x: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    De-quantized = Scale * Quantized (Zero Point is always 0)
    """
    N, K = x.shape
    if group_size == -1:
        group_size = K

    # Padding
    if K % group_size != 0:
        import torch.nn.functional as F

        x = F.pad(x, (0, group_size - (K % group_size)))
        N, K = x.shape

    num_groups = K // group_size
    x_reshaped = x.float().view(N, num_groups, group_size)

    # =========================================================
    # 1. Scale
    #    Range: [-7, 7] -> dividing by 7.0
    # =========================================================
    x_abs_max = x_reshaped.abs().amax(dim=-1, keepdim=True)
    scale = x_abs_max / 7.0
    scale = scale.clamp(min=1e-5)

    # =========================================================
    # 2. Quantize
    # =========================================================
    x_int_sym = (x_reshaped / scale).round().clamp(-8, 7)

    out = x_int_sym.to(torch.int8)

    # =========================================================
    # 3. Zero Point
    # =========================================================
    zero_point = torch.zeros_like(scale)
    out = out.view(N, K)

    scale_out = scale.squeeze(-1).contiguous()
    zero_out = zero_point.squeeze(-1).contiguous()

    return out, scale_out, zero_out


def quantize_params_compressed_tensors(converted_named_params, quantization_config):
    w_cfg = quantization_config["config_groups"]["group_0"]["weights"]
    group_size = w_cfg["group_size"]
    is_symmetric = w_cfg["symmetric"]
    ignore_rules = quantization_config.get("ignore", [])

    results = []

    for name, param in converted_named_params:
        is_ignored = any(
            (r.startswith("re:") and re.match(r[3:], name)) or r == name or name.startswith(r) for r in ignore_rules
        )

        if is_ignored or not name.endswith(".weight") or param.dim() < 2:
            results.append((name, param))
            continue

        input_tensor = param.view(-1, param.shape[-1]) if param.dim() > 2 else param

        if group_size != -1 and input_tensor.shape[-1] < group_size:
            logger.warning(f"Skipping {name}, K-dim {input_tensor.shape[-1]} < group_size")
            results.append((name, param))
            continue

        results.extend(_quantize_param_int4(name, input_tensor, group_size, param.shape, is_symmetric))  # origin shape

    return results


def _quantize_param_int4(name: str, weight: torch.Tensor, group_size: int, shape: torch.Tensor, is_symmetric: bool):
    """
    Wraps the quantization function, handles renaming and packing.
    """
    base_name = name.replace(".weight", "")

    new_base_name = base_name

    original_dtype = weight.dtype

    if group_size == -1:
        group_size = weight.shape[1]
    elif weight.shape[1] % group_size != 0:
        logger.warning(
            f"Weight {name} with shape {weight.shape} has K-dimension "
            f"not divisible by group_size {group_size}. Skipping."
        )
        return [(name, weight.to(original_dtype))]

    q_weight, scales, zeros = int4_block_quantize(weight, group_size)

    packed_q_weight = pack_int4_to_int32(q_weight)

    qweight_name = f"{new_base_name}.weight_packed"
    scales_name = f"{new_base_name}.weight_scale"
    qweight_shape = f"{new_base_name}.weight_shape"

    q_shape = torch.tensor(shape, dtype=torch.int32, device="cuda")

    return [(qweight_name, packed_q_weight), (scales_name, scales.to(original_dtype)), (qweight_shape, q_shape)]
