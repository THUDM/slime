import re
from typing import Tuple

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

W2_SUFFIXES = {
    ".down_proj.weight": "w2",
    ".w2.weight": "w2",
}

QKV_TRIPLET_SUFFIXES = {
    ".q_proj.weight": "q",
    ".k_proj.weight": "k",
    ".v_proj.weight": "v",
}


def _get_ignore_list(quantization_config) -> list[str]:
    ignore = quantization_config.get("ignore", [])
    if ignore is None:
        return []
    if not isinstance(ignore, list):
        raise TypeError(f"quantization_config['ignore'] must be a list, got {type(ignore)}")
    return [x for x in ignore if isinstance(x, str) and x]


def _is_ignored_module_or_param(name: str, ignore_list: list[str]) -> bool:
    for ignored in ignore_list:
        if name == ignored or name.startswith(ignored + "."):
            return True
    return False


def _is_expert_weight_name(name: str) -> bool:
    # Qwen/DeepSeek-style MoE expert (and shared-expert) weights are quantized;
    # modelopt ignore lists typically cover attn/router/norms only.
    return (
        ".experts." in name
        or ".shared_expert." in name
        or ".shared_experts." in name
    )


def is_nvfp4_quant_config(quantization_config) -> bool:
    if not quantization_config:
        return False
    if quantization_config.get("quant_algo") == "NVFP4":
        return True
    return quantization_config.get("quant_method") == "nvfp4"


def quantize_converted_hf_params_nvfp4(converted_named_params, quantization_config):
    """Batch-quantize already-converted HF named params (no megatron-name routing).

    Intended for update-weight buckets where many expert tensors are available at once,
    so MoE gate/up/down can share grouped miniTE launches.
    """
    assert quantization_config is not None
    group_size = _resolve_group_size(quantization_config)
    ignore_list = _get_ignore_list(quantization_config)

    moe_params = []
    qkv_params = []
    mlp_fc1_params = []
    dense_params = []
    passthrough = []

    for item in converted_named_params:
        name = item[0]
        if _is_expert_weight_name(name):
            moe_params.append(item)
        elif _split_qkv_triplet_name(name)[0] is not None:
            qkv_params.append(item)
        elif _split_gated_pair_name(name)[0] is not None and ".mlp." in name:
            mlp_fc1_params.append(item)
        elif _split_w2_name(name) is not None or name.endswith(".o_proj.weight"):
            dense_params.append(item)
        else:
            passthrough.append(item)

    out = []
    if moe_params:
        out.extend(_quantize_moe_params(moe_params, group_size, ignore_list))
    if qkv_params:
        out.extend(_quantize_qkv_params(qkv_params, group_size, ignore_list))
    if mlp_fc1_params:
        out.extend(_quantize_mlp_params(mlp_fc1_params, group_size, ignore_list))
    if dense_params:
        out.extend(_quantize_dense_single_params(dense_params, group_size, ignore_list))
    out.extend(passthrough)
    return out


def quantize_params_nvfp4(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config is not None
    assert quantization_config.get("quant_algo") == "NVFP4" or quantization_config.get("quant_method") == "nvfp4"

    group_size = _resolve_group_size(quantization_config)
    ignore_list = _get_ignore_list(quantization_config)

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        _, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        _, rest = match.groups()

    # self attention: qkv / o_proj
    qkv_pattern = r"self_attention\.(.+)"
    match = re.match(qkv_pattern, rest)
    if match:
        attn_rest = match.groups()[0]
        if attn_rest in ["linear_qkv.weight"]:
            return _quantize_qkv_params(converted_named_params, group_size, ignore_list)
        if attn_rest in ["linear_proj.weight"]:
            return _quantize_dense_single_params(converted_named_params, group_size, ignore_list)

    # IMPORTANT:
    # MoE experts must be checked before the generic MLP branch.
    # Otherwise "mlp.experts.linear_fc1.weight6" would be truncated to
    # "experts.linear_fc1.weight6" and miss the expert branch.
    expert_pattern = r"mlp\.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        expert_rest, _expert_id = match.groups()
        if expert_rest in ["linear_fc1", "linear_fc2"]:
            return _quantize_moe_params(converted_named_params, group_size, ignore_list)

    # shared expert should also be checked before generic MLP.
    shared_expert_pattern = r"mlp\.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        shared_rest = match.groups()[0]
        if shared_rest in ["linear_fc1.weight", "linear_fc2.weight"]:
            return _quantize_moe_params(converted_named_params, group_size, ignore_list)

    # ordinary dense MLP
    mlp_pattern = r"mlp\.(.+)"
    match = re.match(mlp_pattern, rest)
    if match:
        mlp_rest = match.groups()[0]
        if mlp_rest in ["linear_fc1.weight"]:
            return _quantize_mlp_params(converted_named_params, group_size, ignore_list)
        if mlp_rest in ["linear_fc2.weight"]:
            return _quantize_dense_single_params(converted_named_params, group_size, ignore_list)

    return converted_named_params


def _resolve_group_size(quantization_config):
    group_size = quantization_config.get("group_size", NVFP4_GROUP_SIZE)
    if group_size != NVFP4_GROUP_SIZE:
        raise ValueError(f"NVFP4 group_size must be {NVFP4_GROUP_SIZE}, got {group_size}.")
    return group_size


_gated_pair_pending: dict = {}


def _emit_quantized(result_list, name, param, global_amax, group_size):
    qweight, block_scale, weight_scale_2 = quantize_nvfp4(
        param, global_amax=global_amax, group_size=group_size,
    )
    result_list.append((name, qweight))
    result_list.append((name.replace(".weight", ".weight_scale"), block_scale))
    result_list.append((name.replace(".weight", ".weight_scale_2"), weight_scale_2))
    result_list.append(
        (name.replace(".weight", ".input_scale"), torch.ones_like(weight_scale_2, dtype=torch.float32))
    )


def _append_quantized_outputs(result_list, name, qweight, block_scale, weight_scale_2):
    result_list.append((name, qweight))
    result_list.append((name.replace(".weight", ".weight_scale"), block_scale))
    result_list.append((name.replace(".weight", ".weight_scale_2"), weight_scale_2))
    result_list.append(
        (name.replace(".weight", ".input_scale"), torch.ones_like(weight_scale_2, dtype=torch.float32))
    )


def _emit_quantized_grouped(result_list, names, params, global_amaxs):
    """Quantize same-shaped 2D weights in one grouped miniTE launch."""
    if not names:
        return
    if len(names) == 1:
        amax = None if global_amaxs is None else global_amaxs[0]
        _emit_quantized(result_list, names[0], params[0], amax, NVFP4_GROUP_SIZE)
        return

    if isinstance(params, torch.Tensor):
        # Already stacked [E, M, K].
        qweights, block_scales, weight_scale_2s = _quantize_nvfp4_miniTE_grouped_stacked(
            params, global_amaxs
        )
    else:
        qweights, block_scales, weight_scale_2s = _quantize_nvfp4_miniTE_grouped(params, global_amaxs)
    for i, name in enumerate(names):
        _append_quantized_outputs(result_list, name, qweights[i], block_scales[i], weight_scale_2s[i])


def _quantize_moe_params(converted_named_params, group_size, ignore_list):
    """
    Quantize MoE expert weights for SGLang's HF-style MoE weight-loader path.

    SGLang receives gate_proj/up_proj/down_proj keys and routes them into
    packed internal w13/w2 storage. Therefore this function keeps the original
    HF names, but converts tensor payloads from BF16/FP16/FP32 to NVFP4 packed
    uint8 and appends corresponding scale tensors.

    Gate/up pairs share a global amax (matching training-side behaviour where
    the fused w13 = [gate, up] is quantized with a single per-expert amax).
    This works both when gate+up arrive in the same call *and* when they
    arrive in separate consecutive calls (streaming weight-sync path).

    When many experts are present in one call, gate/up/down are quantized with
    grouped miniTE launches (one launch per role) instead of per-tensor launches.
    """
    quantize_named_params = []
    passthrough = []
    gate_by_base = {}
    up_by_base = {}
    down_items = []
    other_items = []

    for converted_name, param in converted_named_params:
        # Expert weights are not in the modelopt ignore list; skip O(N_ignore) scans.
        if _is_expert_weight_name(converted_name):
            if not _should_quantize_param(converted_name, param, group_size, ignore_list=None):
                passthrough.append((converted_name, param))
                continue
        elif not _should_quantize_param(converted_name, param, group_size, ignore_list):
            passthrough.append((converted_name, param))
            continue

        base, role = _split_gated_pair_name(converted_name)
        if base is not None and role == "gate":
            gate_by_base[base] = (converted_name, param)
            continue
        if base is not None and role == "up":
            up_by_base[base] = (converted_name, param)
            continue
        if _split_w2_name(converted_name) is not None:
            down_items.append((converted_name, param))
            continue
        other_items.append((converted_name, param))

    # Resolve gate/up pairs (same-call) and streaming pending pairs.
    # Keep deterministic order so stacked amax/quant stay stable across runs.
    paired_gate_names, paired_gate_params = [], []
    paired_up_names, paired_up_params = [], []

    all_bases = sorted(set(gate_by_base) | set(up_by_base))
    for base in all_bases:
        gate_item = gate_by_base.get(base)
        up_item = up_by_base.get(base)

        if gate_item is not None and up_item is not None:
            gate_name, gate_param = gate_item
            up_name, up_param = up_item
            paired_gate_names.append(gate_name)
            paired_gate_params.append(gate_param)
            paired_up_names.append(up_name)
            paired_up_params.append(up_param)
            continue

        # Streaming mode: only one side present in this call.
        name, param = gate_item if gate_item is not None else up_item
        current_amax = param.reshape(-1).abs().max().to(torch.float32)
        if base in _gated_pair_pending:
            cached_name, cached_param, cached_amax = _gated_pair_pending.pop(base)
            global_amax = torch.maximum(cached_amax, current_amax)
            # Emit cached + current with shared amax (usually E==1 streaming).
            _emit_quantized_grouped(
                quantize_named_params,
                [cached_name, name],
                [cached_param, param],
                [global_amax, global_amax],
            )
        else:
            _gated_pair_pending[base] = (name, param, current_amax)

    if paired_gate_names:
        # Fused w13 path: pack gate/up as [E, 2M, K], one grouped launch, kernel-side amax.
        # Matches training-style fused w13 = cat(gate, up) with a single per-expert amax.
        gate_w = torch.stack(paired_gate_params, dim=0)  # [E, M, K]
        up_w = torch.stack(paired_up_params, dim=0)
        if gate_w.shape != up_w.shape:
            raise ValueError(
                f"gate/up shape mismatch for fused NVFP4 w13: {tuple(gate_w.shape)} vs {tuple(up_w.shape)}"
            )
        m = gate_w.shape[1]
        w13 = torch.cat([gate_w, up_w], dim=1)  # [E, 2M, K]
        qweights, block_scales, weight_scale_2s = _quantize_nvfp4_miniTE_grouped_stacked(w13, None)
        for i, name in enumerate(paired_gate_names):
            _append_quantized_outputs(
                quantize_named_params,
                name,
                qweights[i, :m],
                block_scales[i, :m],
                weight_scale_2s[i],
            )
        for i, name in enumerate(paired_up_names):
            _append_quantized_outputs(
                quantize_named_params,
                name,
                qweights[i, m:],
                block_scales[i, m:],
                weight_scale_2s[i],
            )

    if down_items:
        # Group downs by shape so mixed shared-expert / expert sizes stay valid.
        by_shape = {}
        for name, param in down_items:
            by_shape.setdefault(tuple(param.shape), []).append((name, param))
        for items in by_shape.values():
            names = [n for n, _ in items]
            params = [p for _, p in items]
            if len(params) == 1:
                _emit_quantized_grouped(quantize_named_params, names, params, None)
            else:
                stacked = torch.stack(params, dim=0)
                _emit_quantized_grouped(quantize_named_params, names, stacked, None)

    for name, param in other_items:
        _emit_quantized(quantize_named_params, name, param, None, group_size)

    # Preserve a stable-ish order: quantized experts first (gate batch, up batch,
    # down batch, others), then passthrough. Callers key by name.
    quantize_named_params.extend(passthrough)
    return quantize_named_params


def _quantize_mlp_params(converted_named_params, group_size, ignore_list):
    shared_global_amax = {}
    gated_candidates = {}

    for converted_name, param in converted_named_params:
        base, role = _split_gated_pair_name(converted_name)
        if base is None or role is None:
            continue
        if _should_quantize_param(converted_name, param, group_size, ignore_list):
            gated_candidates.setdefault(base, {})[role] = param

    for base, roles in gated_candidates.items():
        if "gate" in roles and "up" in roles:
            gate_amax = roles["gate"].abs().max().to(torch.float32)
            up_amax = roles["up"].abs().max().to(torch.float32)
            shared_global_amax[base] = torch.max(gate_amax, up_amax)

    quantize_named_params = []
    for converted_name, param in converted_named_params:
        if not _should_quantize_param(converted_name, param, group_size, ignore_list):
            quantize_named_params.append((converted_name, param))
            continue

        base, _role = _split_gated_pair_name(converted_name)
        global_amax = shared_global_amax.get(base) if base else None

        if global_amax is None and base is not None and _role in ("gate", "up"):
            current_amax = param.abs().max().to(torch.float32)
            if base in _gated_pair_pending:
                cached_name, cached_param, cached_amax = _gated_pair_pending.pop(base)
                global_amax = torch.max(cached_amax, current_amax)
                _emit_quantized(quantize_named_params, cached_name, cached_param, global_amax, group_size)
            else:
                _gated_pair_pending[base] = (converted_name, param, current_amax)
                continue

        _emit_quantized(quantize_named_params, converted_name, param, global_amax, group_size)

    return quantize_named_params


def _quantize_dense_single_params(converted_named_params, group_size, ignore_list):
    quantize_named_params = []

    for converted_name, param in converted_named_params:
        if not _should_quantize_param(converted_name, param, group_size, ignore_list):
            quantize_named_params.append((converted_name, param))
            continue

        _emit_quantized(quantize_named_params, converted_name, param, None, group_size)

    return quantize_named_params


_qkv_triplet_pending: dict = {}


def _quantize_qkv_params(converted_named_params, group_size, ignore_list):
    shared_global_amax = {}
    qkv_candidates = {}

    for converted_name, param in converted_named_params:
        base, role = _split_qkv_triplet_name(converted_name)
        if base is None or role is None:
            continue
        if _should_quantize_param(converted_name, param, group_size, ignore_list):
            qkv_candidates.setdefault(base, {})[role] = param

    for base, roles in qkv_candidates.items():
        if "q" in roles and "k" in roles and "v" in roles:
            q_amax = roles["q"].abs().max().to(torch.float32)
            k_amax = roles["k"].abs().max().to(torch.float32)
            v_amax = roles["v"].abs().max().to(torch.float32)
            shared_global_amax[base] = torch.max(torch.max(q_amax, k_amax), v_amax)

    quantize_named_params = []
    for converted_name, param in converted_named_params:
        if not _should_quantize_param(converted_name, param, group_size, ignore_list):
            quantize_named_params.append((converted_name, param))
            continue

        base, _role = _split_qkv_triplet_name(converted_name)
        global_amax = shared_global_amax.get(base) if base else None

        if global_amax is None and base is not None and _role in ("q", "k", "v"):
            current_amax = param.abs().max().to(torch.float32)
            pending = _qkv_triplet_pending.setdefault(base, [])
            pending.append((converted_name, param, current_amax))

            if len(pending) < 3:
                continue

            parts = _qkv_triplet_pending.pop(base)
            global_amax = torch.max(torch.max(parts[0][2], parts[1][2]), parts[2][2])
            for p_name, p_param, _ in parts:
                _emit_quantized(quantize_named_params, p_name, p_param, global_amax, group_size)
            continue

        _emit_quantized(quantize_named_params, converted_name, param, global_amax, group_size)

    return quantize_named_params


def _split_qkv_triplet_name(name: str):
    for suffix, role in QKV_TRIPLET_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def _should_quantize_param(name, weight, group_size, ignore_list=None):
    if ignore_list and _is_ignored_module_or_param(name, ignore_list):
        return False
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


def _split_w2_name(name: str):
    for suffix in W2_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


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


# Cache (device, E, rows, cols) -> (row_offsets, scale_offsets, n_padded, sf_stride)
_GROUPED_OFFSET_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, int]] = {}


def _get_grouped_offsets(device, E: int, rows: int, cols: int):
    key = (str(device), E, rows, cols)
    cached = _GROUPED_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    T = 128
    n_padded = ((rows + T - 1) // T) * T
    sf_stride = n_padded * (cols // 16)
    row_offsets = torch.arange(0, (E + 1) * rows, rows, dtype=torch.int32, device=device)
    scale_offsets = torch.arange(0, (E + 1) * sf_stride, sf_stride, dtype=torch.int64, device=device)
    cached = (row_offsets, scale_offsets, n_padded, sf_stride)
    _GROUPED_OFFSET_CACHE[key] = cached
    return cached


def _quantize_nvfp4_miniTE(
    x: torch.Tensor,
    global_amax: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 quantization using miniTE C++ kernel (bit-identical to training).

    Returns (fp4_data, block_scale_rowmajor, weight_scale_2) matching the
    Python _quantize_nvfp4 interface.
    """
    qweights, block_scales, weight_scale_2s = _quantize_nvfp4_miniTE_grouped_stacked(
        x.unsqueeze(0), None if global_amax is None else global_amax.reshape(1)
    )
    return qweights[0], block_scales[0], weight_scale_2s[0]


def _quantize_nvfp4_miniTE_grouped(
    weights: list[torch.Tensor],
    global_amaxs: torch.Tensor | list[torch.Tensor] | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack same-shaped 2D weights then call grouped miniTE quant."""
    if not weights:
        empty = torch.empty(0)
        return empty, empty, empty
    if len(weights) == 1:
        stacked = weights[0].unsqueeze(0)
    else:
        stacked = torch.stack(weights, dim=0)
    if isinstance(global_amaxs, list):
        global_amaxs = torch.stack(
            [a.to(dtype=torch.float32).reshape(()) for a in global_amaxs]
        )
    return _quantize_nvfp4_miniTE_grouped_stacked(stacked, global_amaxs)


def _quantize_nvfp4_miniTE_grouped_stacked(
    weights_emk: torch.Tensor,
    global_amaxs: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped NVFP4 quant for a packed ``[E, M, K]`` weight tensor.

    Returns:
      qweights: ``[E, M, K/2]`` uint8
      block_scales: ``[E, M, K/16]`` float8_e4m3fn (row-major, unpadded rows only)
      weight_scale_2s: ``[E]`` float32
    """
    import fp4_gemm

    if weights_emk.ndim != 3:
        raise ValueError(f"Expected [E,M,K] weights, got shape {tuple(weights_emk.shape)}")

    weights_emk = weights_emk.contiguous()
    E, rows, cols = weights_emk.shape
    device = weights_emk.device
    x = weights_emk.reshape(E * rows, cols)

    row_offsets, scale_offsets, n_padded, sf_stride = _get_grouped_offsets(device, E, rows, cols)

    data = torch.empty((E * rows, cols // 2), dtype=torch.uint8, device=device)
    gs_out = torch.empty(E, dtype=torch.float32, device=device)
    scale_flat = torch.empty(E * sf_stride, dtype=torch.float8_e4m3fn, device=device)

    if global_amaxs is None:
        amax = x.view(E, -1).abs().amax(dim=1).to(torch.float32).contiguous()
    else:
        amax = global_amaxs.to(device=device, dtype=torch.float32).reshape(E).contiguous()

    fp4_gemm.rtn_fp4_group_tensor_global(
        data,
        scale_flat,
        gs_out,
        x,
        row_offsets,
        scale_offsets,
        amax,
        1.0,
        swizzle=False,
    )

    qweights = data.view(E, rows, cols // 2)
    block_scales = scale_flat.view(E, n_padded, cols // 16)[:, :rows, :].contiguous()
    return qweights, block_scales, gs_out


def _quantize_nvfp4(
    x: torch.Tensor,
    global_amax: torch.Tensor | None,
    pow_2_scales: bool = False,
    with_2d_quantization: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tile_len_x = 16
    tile_len_y = 16

    x = x.contiguous()
    x_f = x.to(torch.float32)

    if global_amax is None:
        global_amax = torch.amax(torch.abs(x_f))
    else:
        global_amax = global_amax.to(device=x.device, dtype=torch.float32)

    assert x.ndim == 2
    m, n = x.shape

    if with_2d_quantization:
        x_blocks = (
            x.unfold(0, tile_len_y, tile_len_y)
            .unfold(1, tile_len_x, tile_len_x)
            .to(torch.float32)
        )
        block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))
        vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)
    else:
        x_reshaped = x.view(m, n // tile_len_x, tile_len_x)
        vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(torch.float32)

    x = x.view(m, n // tile_len_x, tile_len_x)

    float4_e2m1_max = torch.tensor(FP4_E2M1_MAX, device=x.device, dtype=torch.float32)
    float8_e4m3_max = torch.tensor(FP8_E4M3_MAX, device=x.device, dtype=torch.float32)

    decode_scale = torch.div(vec_max, float4_e2m1_max)

    if pow_2_scales:
        decode_scale = cast_to_e8(decode_scale)
        encode_scale = torch.div(
            torch.tensor(1.0, device=x.device, dtype=torch.float32),
            decode_scale.to(torch.float32),
        )
        global_decode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    else:
        global_encode_scale = torch.div(float8_e4m3_max * float4_e2m1_max, global_amax)
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
        decode_scale = torch.clamp(decode_scale, min=-float8_e4m3_max, max=float8_e4m3_max)
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
    clipped_x = torch.clamp(scaled_x, -float4_e2m1_max, float4_e2m1_max).reshape(m, n)

    return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1), global_decode_scale


def quantize_nvfp4(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    group_size: int = NVFP4_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() == 2:
        return _quantize_nvfp4_miniTE(weight, global_amax=global_amax)

    if weight.dim() == 3:
        if global_amax is not None:
            raise ValueError("global_amax override is only supported for 2D weights.")

        qweights = []
        block_scales = []
        global_scales = []

        for idx in range(weight.shape[0]):
            qweight, block_scale, global_scale = _quantize_nvfp4_miniTE(weight[idx], global_amax=None)
            qweights.append(qweight)
            block_scales.append(block_scale)
            global_scales.append(global_scale)

        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )

    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")