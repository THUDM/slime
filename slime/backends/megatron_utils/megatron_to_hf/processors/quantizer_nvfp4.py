import logging
import math
import re

import torch
import torch.nn as nn

try:
    import fake_int4_quant_cuda
except ImportError:
    fake_int4_quant_cuda = None

logger = logging.getLogger(__name__)


__all__ = ["quantize_params_nvfp4"]


def round_to_quantized_type_dtype(
    tensor,
    dtype,
    cast_to_original_dtype=False,
):
    original_dtype = tensor.dtype
    iinfo = torch.iinfo(dtype)
    rounded = torch.round(torch.clamp(tensor, iinfo.min, iinfo.max)).to(dtype)
    if cast_to_original_dtype:
        return rounded.to(original_dtype)
    return rounded


def cast_to_fp4x2(x):
    """Quantize a tensor to FP4 E2M1 and store in a byte tensor"""

    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16

def quantize_blockwise_reference(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    tile_len_x: int,
    tile_len_y: int,
    pow_2_scales: bool,
    with_2d_quantization: bool,
    eps: float,  # pylint: disable=unused-argument
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert x.ndim == 2
    m, n = x.shape
    # Compute vec_max based on the original x (before reshape)
    # For 1D quantization: amax over each row chunk of 16
    # For 2D quantization: amax over each 16x16 block, but output shape is still (128, 8, 1), filled with block amax
    if with_2d_quantization:
        # x shape: (128, 128)
        x_blocks = (
            x.unfold(0, tile_len_y, tile_len_y)
            .unfold(1, tile_len_x, tile_len_x)
            .to(torch.float32)
        )  # (8, 8, 16, 16)
        block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))  # (8, 8)
        # Now, expand to (128, 8, 1) by repeating each block_amax for 16 rows
        vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)  # (128, 8, 1)
    else:
        # x shape: (128, 128)
        x_reshaped = x.view(m, n // tile_len_x, tile_len_x)  # (128, 8, 16)
        vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(
            torch.float32
        )  # (128, 8, 1)
    x = x.view(m, n // tile_len_x, tile_len_x)
    FLOAT4_E2M1_MAX = torch.tensor(6.0, device=x.device, dtype=torch.float32)
    FLOAT8_E4M3_MAX = torch.tensor(448.0, device=x.device, dtype=torch.float32)
    decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)

    if pow_2_scales:
        decode_scale = cast_to_e8(decode_scale)
        encode_scale = torch.div(
            torch.tensor(1.0, device=x.device, dtype=torch.float32),
            decode_scale.to(torch.float32),
        )
    else:
        global_encode_scale = torch.div(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, global_amax)
        global_encode_scale = torch.min(
            global_encode_scale,
            torch.tensor(
                torch.finfo(torch.float32).max,
                device=global_encode_scale.device,
                dtype=torch.float32,
            ),
        )
        if global_encode_scale == torch.tensor(0.0, device=x.device, dtype=torch.float32):
            global_encode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        global_decode_scale = torch.div(1.0, global_encode_scale)

        decode_scale = decode_scale * global_encode_scale
        decode_scale = torch.min(
            decode_scale,
            torch.tensor(
                torch.finfo(torch.float32).max,
                device=decode_scale.device,
                dtype=torch.float32,
            ),
        )
        decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
        decode_scale = decode_scale.to(torch.float8_e4m3fn)

        encode_scale = torch.min(
            torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
            torch.tensor(
                torch.finfo(torch.float32).max,
                device=decode_scale.device,
                dtype=torch.float32,
            ),
        )

    scaled_x = x.to(torch.float32) * encode_scale

    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)

    return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)


def pack_layer(weight, group_size):
    global_amax = torch.max(torch.abs(weight)).to(torch.float32).view(1)
    return quantize_blockwise_reference(
        weight, global_amax, group_size, group_size, False, True, 0.0
    ), global_amax


def quantize_params_nvfp4(converted_named_params, quantization_config):
    w_cfg = quantization_config["config_groups"]["group_0"]["weights"]
    group_size = w_cfg["group_size"]
    ignore_rules = quantization_config.get("ignore", [])

    results = []

    for name, param in converted_named_params:
        is_ignored = any(
            (r.startswith("re:") and re.match(r[3:], name)) or r == name or name.startswith(r) for r in ignore_rules
        )

        if is_ignored or not name.endswith(".weight") or param.dim() < 2:
            results.append((name, param))
            continue

        (qw, scale), amax = pack_layer(param, group_size)
        scale_name = name.replace(".weight", ".weight_scale")
        scale_2_name = name.replace(".weight", ".weight_scale_2")
        scale_2 = amax / (448.0 * 6.0)

        input_scale_name = name.replace(".weight", ".input_scale")
        input_scale = torch.tensor([1.0])

        results.append((name, qw))
        results.append((scale_name, scale))
        results.append((scale_2_name, scale_2))
        results.append((input_scale_name, input_scale))

    return results
