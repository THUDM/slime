"""
Usage:
------
python convert_k2_thinking_int4_to_bf16.py [-h] --model-dir MODEL_DIR [--output-dir OUTPUT_DIR]
                           [--files FILE [FILE ...]] [--config-path CONFIG_PATH]
                           [--overwrite]
options:
  -h, --help            Show this help message and exit.
  --model-dir MODEL_DIR Path to the directory of the HF safetensors quantized model.
  --output-dir OUTPUT_DIR
                        Path to the directory to save the converted BF16 model.
                        Default: <model-dir>_bf16
  --files FILE [FILE ...]
                        Specific safetensors filenames to convert (relative to model-dir).
                        Convert all if omitted.
  --config-path CONFIG_PATH
                        Path to config.json to extract group_size (default: model-dir/config.json).
  --overwrite           Rewrite output files even if they already exist.


Example:
--------
python convert_k2_thinking_int4_to_bf16.py --model-dir /Kimi-K2-Thinking --output-dir /Kimi-K2-Thinking-bf16
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import torch
from compressed_tensors.compressors import unpack_from_int32
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


def _load_config(model_dir: str, config_path: str | None) -> tuple[int, int, int]:
    """Read config.json and return hidden_size, inter_size, and group_size."""
    cfg_path = config_path or os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    hidden_size = int(cfg.get("hidden_size"))
    inter_size = int(cfg.get("moe_intermediate_size"))
    group_size = int(
        cfg.get("quantization_config", {})
        .get("config_groups", {})
        .get("group_0", {})
        .get("weights", {})
        .get("group_size", 128)
    )
    return hidden_size, inter_size, group_size


def _dequantize_tensor(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Unpack int32 quantized tensor and multiply with scales to create BF16 tensor."""
    if isinstance(weight_shape, torch.Tensor):
        shape = tuple(int(v) for v in weight_shape.view(-1).tolist())
    else:
        shape = tuple(weight_shape)

    weight = unpack_from_int32(weight_packed, 4, shape)

    if group_size > 0:
        scale = weight_scale.to(torch.float32)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        scales = torch.repeat_interleave(scale, repeats=group_size, dim=1)
    else:
        scales = weight_scale.to(torch.float32)

    if scales.shape != weight.shape:
        if scales.numel() == weight.numel():
            scales = scales.reshape_as(weight)
        else:
            raise ValueError(f"Scale shape {scales.shape} incompatible with weight shape {weight.shape}")

    bf16 = (weight.to(torch.float32) * scales).to(torch.bfloat16)
    return bf16.contiguous()


def _is_quantized_weight_key(key: str) -> bool:
    """Check if the key is a quantized MoE expert weight key."""
    if ".mlp.experts." not in key or ".shared_experts." in key:
        return False
    suffixes = ("weight_packed", "weight_scale", "weight_shape")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in suffixes:
            if key.endswith(f".{proj}.{suffix}"):
                return True
    return False


def convert_file(
    input_path: str,
    output_path: str,
    group_size: int,
    skip_existing: bool = True,
):
    """Convert a single safetensors file from quantized format to BF16 (GPU accelerated)."""
    if skip_existing and os.path.exists(output_path):
        return

    tensors = {}
    expert_buffers = defaultdict(lambda: defaultdict(dict))

    # Load weights directly on GPU
    with safe_open(input_path, framework="pt", device="cuda") as reader:
        keys = list(reader.keys())
        for key in keys:
            tensor = reader.get_tensor(key)
            if not _is_quantized_weight_key(key):
                tensors[key] = tensor
                continue
            parts = key.split(".")
            try:
                expert_idx = parts.index("experts")
            except ValueError:
                tensors[key] = tensor
                continue
            prefix = ".".join(parts[: expert_idx + 2])
            project = parts[-2]
            suffix = parts[-1]
            expert_buffers[prefix][project][suffix] = tensor

    # Convert quantized weights
    for prefix, components in expert_buffers.items():
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj_data = components.get(proj_name, {})
            required = {"weight_packed", "weight_scale", "weight_shape"}

            if not required.issubset(proj_data.keys()):
                # Keep quantized tensors if incomplete
                for suffix, value in proj_data.items():
                    tensors[f"{prefix}.{proj_name}.{suffix}"] = value
                continue

            # Dequantize to BF16
            bf16_weight = _dequantize_tensor(
                proj_data["weight_packed"].to(torch.int32),
                proj_data["weight_scale"].to(torch.float32),
                proj_data["weight_shape"],
                group_size,
            )
            tensors[f"{prefix}.{proj_name}.weight"] = bf16_weight.to(torch.bfloat16)

    # Save converted file (moved to CPU for compatibility)
    cpu_tensors = {k: v.cpu() for k, v in tensors.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(cpu_tensors, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GPTQ MoE experts to BF16 weights.")
    parser.add_argument("--model-dir", required=True, help="Directory containing safetensors checkpoints.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination BF16 model directory (default: <model-dir>_bf16).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Optional specific safetensor files to convert.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to config.json if not in model-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing BF16 files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir or f"{model_dir}_bf16")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    _, _, group_size = _load_config(model_dir, args.config_path)

    # Collect target files
    if args.files:
        targets = [os.path.join(model_dir, fname) for fname in args.files]
    else:
        targets = [
            os.path.join(model_dir, name) for name in sorted(os.listdir(model_dir)) if name.endswith(".safetensors")
        ]

    if not targets:
        print("No safetensors checkpoints found.")
        return

    # Convert with progress bar
    for path in tqdm(targets, desc="Converting weights", unit="file"):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, model_dir)
        output_path = os.path.join(output_dir, rel)
        convert_file(path, output_path, group_size, skip_existing=not args.overwrite)

    # Copy config/json/py/tokenizer
    for fname in os.listdir(model_dir):
        src_path = os.path.join(model_dir, fname)
        dst_path = os.path.join(output_dir, fname)
        if fname == "model.safetensors.index.json":
            continue
        if fname.endswith(".json") or fname.endswith(".py") or fname.startswith("tokenizer"):
            shutil.copy2(src_path, dst_path)

    # Generate new index
    new_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    weight_map = {}
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".safetensors"):
            continue
        safetensor_path = os.path.join(output_dir, fname)
        with safe_open(safetensor_path, framework="pt") as reader:
            for key in reader.keys():
                weight_map[key] = fname

    with open(new_index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    print(f"\nSuccessful! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
