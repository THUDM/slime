import argparse
import gc
import json
import os
import shutil

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm
from typing import List, Tuple

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
DEFAULT_KV_CACHE_SCHEME = {"dynamic": False, "num_bits": 8, "type": "float"}
DEFAULT_KV_CACHE_QUANT_ALGO = "FP8"

EXPERT_WEIGHT_SUFFIXES = (
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
)

EXPERT_NAME_MARKERS = (
    ".experts.",
    ".shared_experts.",
    "block_sparse_moe.experts.",
    ".moe.experts.",
)

FUSED_QKV_SUFFIXES = (".q_proj", ".k_proj", ".v_proj")
GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}


def _is_moe_expert_weight_name(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if not any(marker in name for marker in EXPERT_NAME_MARKERS):
        return False
    return any(name.endswith(suffix) for suffix in EXPERT_WEIGHT_SUFFIXES)


def _extract_layer_id(name: str) -> int | None:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            layer_id = parts[idx + 1]
            if layer_id.isdigit():
                return int(layer_id)
    return None


def _get_num_hidden_layers(model_dir: str) -> int:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError("config.json is required to use --keep-last-n.")
    cfg = json.load(open(config_path))
    num_layers = cfg.get("num_hidden_layers")
    if num_layers is None and isinstance(cfg.get("text_config"), dict):
        num_layers = cfg["text_config"].get("num_hidden_layers")
    if num_layers is None:
        raise ValueError("num_hidden_layers not found in config.json.")
    return int(num_layers)


def _get_last_n_layer_ids(num_layers: int, keep_last_n: int) -> set[int]:
    if keep_last_n <= 0:
        return set()
    start = max(0, num_layers - keep_last_n)
    return set(range(start, num_layers))


def _build_keep_last_n_ignore_list(num_layers: int, keep_last_n: int) -> list[str]:
    if keep_last_n <= 0:
        return []
    start = max(0, num_layers - keep_last_n)
    ignore_list = []
    for layer_id in range(start, num_layers):
        prefix = f"model.layers.{layer_id}"
        ignore_list.extend(
            [
                f"{prefix}.self_attn.qkv_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp",
                f"{prefix}.mlp.experts",
            ]
        )
    return ignore_list


def should_quantize(
    name: str,
    weight: torch.Tensor,
    skip_layers: set[int] | None = None,
) -> bool:
    if skip_layers:
        layer_id = _extract_layer_id(name)
        if layer_id is not None and layer_id in skip_layers:
            return False
    if not _is_moe_expert_weight_name(name):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"Last dim {weight.shape[-1]} must be divisible by {NVFP4_GROUP_SIZE} " f"for NVFP4 quantization ({name})."
        )
    return True


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
    with_2d_quantization: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    tile_len_x = 16
    tile_len_y = 16

    x = x.contiguous()
    x_f = x.to(torch.float32)

    if global_amax is None:
        global_amax = torch.amax(torch.abs(x_f))          # scalar tensor (fp32, on device)
    else:
        global_amax = global_amax.to(device=x.device, dtype=torch.float32)

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

    return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1), global_decode_scale


def quantize_nvfp4(
    weight: torch.Tensor,
    global_amax: torch.Tensor | None = None,
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
            qweight, block_scale, global_scale = _quantize_nvfp4(weight[idx], global_amax)
            qweights.append(qweight)
            block_scales.append(block_scale)
            global_scales.append(global_scale)
        return (
            torch.stack(qweights, dim=0),
            torch.stack(block_scales, dim=0),
            torch.stack(global_scales, dim=0),
        )
    raise ValueError(f"Unsupported weight rank {weight.dim()} for NVFP4 quantization.")


class ConversionResult:
    def __init__(self) -> None:
        self.weight_map: dict[str, str] = {}
        self.total_size: int = 0
        self.modules_to_not_convert: list[str] = []

    def add_result(self, filename: str, q_weights: dict[str, torch.Tensor], module_names: list[str]) -> None:
        for key, tensor in q_weights.items():
            self.weight_map[key] = filename
            self.total_size += tensor.numel() * tensor.element_size()
        self.modules_to_not_convert.extend(module_names)


def _update_quantization_config(cfg: dict, ignore_list: list[str]) -> None:
    quant_cfg = cfg.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}

    quant_cfg["quant_algo"] = "NVFP4"
    quant_cfg["quant_method"] = "modelopt"
    quant_cfg["group_size"] = NVFP4_GROUP_SIZE
    quant_cfg["ignore"] = ignore_list
    quant_cfg.setdefault("kv_cache_scheme", DEFAULT_KV_CACHE_SCHEME)

    config_groups = quant_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        for group in config_groups.values():
            if not isinstance(group, dict):
                continue
            group.setdefault("targets", ["Linear"])
            for key in ("input_activations", "weights"):
                section = group.get(key)
                if not isinstance(section, dict):
                    continue
                section.setdefault("dynamic", False)
                section.setdefault("num_bits", 4)
                section.setdefault("type", "float")
                section["group_size"] = NVFP4_GROUP_SIZE

    cfg["quantization_config"] = quant_cfg


def _write_hf_quant_config(output_path: str, ignore_list: list[str], input_path: str) -> None:
    hf_quant_path = os.path.join(input_path, "hf_quant_config.json")
    if os.path.exists(hf_quant_path):
        with open(hf_quant_path) as f:
            hf_quant_cfg = json.load(f)
    else:
        hf_quant_cfg = {"producer": {"name": "modelopt"}}

    quant_section = hf_quant_cfg.get("quantization")
    if not isinstance(quant_section, dict):
        quant_section = {}

    quant_section["quant_algo"] = "NVFP4"
    quant_section["kv_cache_quant_algo"] = DEFAULT_KV_CACHE_QUANT_ALGO
    quant_section["group_size"] = NVFP4_GROUP_SIZE
    quant_section["exclude_modules"] = ignore_list
    hf_quant_cfg["quantization"] = quant_section

    with open(os.path.join(output_path, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant_cfg, f, indent=2)


def _augment_ignore_list(ignore_list: list[str]) -> list[str]:
    ignore_set = set(ignore_list)
    extra = set()
    for name in ignore_list:
        if name.endswith(FUSED_QKV_SUFFIXES):
            for suffix in FUSED_QKV_SUFFIXES:
                if name.endswith(suffix):
                    extra.add(name[: -len(suffix)] + ".qkv_proj")
                    break
    ignore_set.update(extra)
    return sorted(ignore_set)


def _split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def process_file(
    input_path: str,
    output_path: str,
    filename: str,
    result_collector: ConversionResult,
    device: str,
    skip_layers: set[int],
) -> None:
    if not filename.endswith(".safetensors"):
        return

    weights: dict[str, torch.Tensor] = {}
    q_weights: dict[str, torch.Tensor] = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device=device) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    modules_to_not_convert: list[str] = []
    shared_global_amax: dict[str, torch.Tensor] = {}
    gated_candidates: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in weights.items():
        base, role = _split_gated_pair_name(key)
        if base is None or role is None:
            continue
        if should_quantize(key, tensor, skip_layers):
            gated_candidates.setdefault(base, {})[role] = tensor
    for base, roles in gated_candidates.items():
        if "gate" in roles and "up" in roles:
            gate_amax = roles["gate"].abs().max().to(torch.float32)
            up_amax = roles["up"].abs().max().to(torch.float32)
            shared_global_amax[base] = torch.max(gate_amax, up_amax)
    for key, tensor in weights.items():
        if should_quantize(key, tensor, skip_layers):
            base, role = _split_gated_pair_name(key)
            global_amax = shared_global_amax.get(base) if base else None
            qweight, block_scale, weight_scale_2 = quantize_nvfp4(tensor, global_amax=global_amax)
            q_weights[key] = qweight
            q_weights[key.replace(".weight", ".weight_scale")] = block_scale
            q_weights[key.replace(".weight", ".weight_scale_2")] = weight_scale_2.detach().clone()
            q_weights[key.replace(".weight", ".input_scale")] = torch.ones_like(weight_scale_2, dtype=torch.float32).clone()
        else:
            if key.endswith(".weight"):
                modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = tensor

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_nvfp4(model_dir: str, save_dir: str, device: str, keep_last_n: int) -> None:
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    num_layers = _get_num_hidden_layers(input_path) if keep_last_n > 0 else 0
    skip_layers = _get_last_n_layer_ids(num_layers, keep_last_n)
    keep_last_ignore = _build_keep_last_n_ignore_list(num_layers, keep_last_n)

    result_collector = ConversionResult()
    for filename in tqdm(safetensors_files, desc="Processing files"):
        process_file(
            input_path,
            output_path,
            filename,
            result_collector,
            device,
            skip_layers,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ignore_list = _augment_ignore_list(result_collector.modules_to_not_convert + keep_last_ignore)

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        _update_quantization_config(cfg, ignore_list)
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    _write_hf_quant_config(output_path, ignore_list, input_path)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.total_size},
    }
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HF safetensors model.")
    parser.add_argument("--save-dir", type=str, required=True, help="Path to save converted model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run quantization on (default: cuda).",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=0,
        help="Keep the last N transformer layers unquantized (BF16/FP16).",
    )
    args = parser.parse_args()

    if isinstance(args.device, str) and args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot run NVFP4 quantization.")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_nvfp4(args.model_dir, args.save_dir, str(device), args.keep_last_n)


if __name__ == "__main__":
    main()
