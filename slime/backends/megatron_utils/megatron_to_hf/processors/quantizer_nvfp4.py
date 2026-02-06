import re

import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16

GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config is not None
    assert quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4"
    group_size = _resolve_group_size(quantization_config)

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        _, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        _, rest = match.groups()

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, _ = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            return _quantize_moe_params(converted_named_params, group_size)

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            return _quantize_moe_params(converted_named_params, group_size)

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _resolve_group_size(quantization_config):
    group_size = quantization_config.get("group_size", NVFP4_GROUP_SIZE)
    if group_size != NVFP4_GROUP_SIZE:
        raise ValueError(f"NVFP4 group_size must be {NVFP4_GROUP_SIZE}, got {group_size}.")
    return group_size


def _quantize_moe_params(converted_named_params, group_size):
    shared_global_amax = {}
    gated_candidates = {}
    for converted_name, param in converted_named_params:
        base, role = _split_gated_pair_name(converted_name)
        if base is None or role is None:
            continue
        if _should_quantize_param(converted_name, param, group_size):
            gated_candidates.setdefault(base, {})[role] = param

    for base, roles in gated_candidates.items():
        if "gate" in roles and "up" in roles:
            gate_amax = roles["gate"].abs().max().to(torch.float32)
            up_amax = roles["up"].abs().max().to(torch.float32)
            shared_global_amax[base] = torch.max(gate_amax, up_amax)

    quantize_named_params = []
    for converted_name, param in converted_named_params:
        if not _should_quantize_param(converted_name, param, group_size):
            quantize_named_params.append((converted_name, param))
            continue
        base, _role = _split_gated_pair_name(converted_name)
        global_amax = shared_global_amax.get(base) if base else None
        qweight, block_scale, weight_scale_2 = quantize_nvfp4(param, global_amax=global_amax, group_size=group_size)
        quantize_named_params.append((converted_name, qweight))
        quantize_named_params.append((converted_name.replace(".weight", ".weight_scale"), block_scale))
        quantize_named_params.append((converted_name.replace(".weight", ".weight_scale_2"), weight_scale_2))
        quantize_named_params.append(
            (converted_name.replace(".weight", ".input_scale"), torch.ones_like(weight_scale_2, dtype=torch.float32))
        )

    return quantize_named_params


def _should_quantize_param(name, weight, group_size):
    if not name.endswith(".weight"):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % group_size != 0:
        raise ValueError(f"Last dim {weight.shape[-1]} must be divisible by {group_size} for NVFP4 ({name}).")
    return True


def _split_gated_pair_name(name: str):
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP4 E2M1 and pack two values per byte."""
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


def _quantize_nvfp4(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    pow_2_scales: bool = False,
    with_2d_quantization: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    tile_len_x = 16
    tile_len_y = 16

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

    return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1), global_amax


def quantize_nvfp4(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    group_size: int = NVFP4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() == 2:
        return _quantize_nvfp4(weight, global_amax=global_amax)
    if weight.dim() == 3:
        if global_amax is not None:
            raise ValueError("global_amax override is only supported for 2D weights.")
        qweights = []
        block_scales = []
        global_scales = []
        for idx in range(weight.shape[0]):
            qweight, block_scale, global_scale = _quantize_nvfp4(weight[idx], global_amax=global_amax)
            qweights.append(qweight)
            block_scales.append(block_scale)
            global_scales.append(global_scale)
        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )
    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")