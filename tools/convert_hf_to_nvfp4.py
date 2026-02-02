      
      
"""
python tools/convert_hf_to_fp8.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR] [--strategy {block,channel,tensor}] [--block-size [BLOCK_SIZE ...]]
                           [--max-workers MAX_WORKERS]

python tools/convert_hf_to_nvfp4.py --model-dir

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to the directory of the HF safetensors model.
  --save-dir SAVE_DIR   Path to the directory to save the converted model.
  --strategy {block,channel,tensor}
  --block-size [BLOCK_SIZE ...]
                        eg. --block-size 32 32
  --max-workers MAX_WORKERS
                        Number of worker threads for parallel processing
"""

import argparse
import gc
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

from typing import List, Optional, Tuple, Union
import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F

from tqdm import tqdm


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


def quant_nvfp4(weight):
    global_amax = torch.max(torch.abs(weight)).to(torch.float32).view(1)
    return quantize_blockwise_reference(
        weight, global_amax, 16, 16, False, True, 0.0
    ), global_amax


class ConversionResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0
        self.modules_to_not_convert = []

    def add_result(self, filename, q_weights, module_names):
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.param_count += len(v)
            self.modules_to_not_convert.extend(module_names)


def process_file(input_path, output_path, filename, result_collector):
    if not filename.endswith(".safetensors"):
        return

    print(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    weights = {}
    q_weights = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    modules_to_not_convert = []
    for key in weights.keys():
        if (
            "weight" in key
            and "expert" in key
            and "layernorm" not in key
            and "embed" not in key
            and "router" not in key
            and "mlp.gate." not in key
            and "norm" not in key
            and "lm_head" not in key
            and "eh_proj" not in key
        ):
            (qw, s), amax = quant_nvfp4(weights[key])
            q_weights[key] = qw
            scale_name = key.replace(".weight", ".weight_scale")
            q_weights[scale_name] = s
            scale_2_name = key.replace(".weight", ".weight_scale_2")
            q_weights[scale_2_name] = amax / (448.0 * 6.0)

            input_scale = key.replace(".weight", ".input_scale")
            q_weights[input_scale] = torch.tensor([1.0])
        else:
            if ((".self_attn.q_proj.weight" in key or
                 ".self_attn.k_proj.weight" in key or
                 ".self_attn.v_proj.weight" in key) and
                 "model.layers.*.self_attn.qkv_proj" not in modules_to_not_convert):
                modules_to_not_convert.append("model.layers.*.self_attn.qkv_proj")
                modules_to_not_convert.append(key.replace(".weight", ""))
            else:
                modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = weights[key]

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_nvfp4(input_path, output_path, max_workers=4):
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                process_file, input_path, output_path, filename, result_collector
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    quant_group = {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "type": "float",
            },
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "type": "float",
            },
        },
    }
    quantization_config = {
        "config_groups": quant_group,
        "quant_algo": "NVFP4",
        "kv_cache_scheme": {
            "dynamic": False,
            "num_bits": 8,
            "type": "float"
        },
        "group_size": 16,
        "ignore": list(set(result_collector.modules_to_not_convert)),
        "quant_method": "modelopt",
    }

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {"weight_map": result_collector.weight_map, "metadata": {"total_size": result_collector.param_count}}
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    # 1) Ensure config.json has complete quantization_config for SGLang
    cfg_path_out = os.path.join(output_path, "config.json")
    if os.path.exists(cfg_path_out):
        cfg_out = json.load(open(cfg_path_out))

        qc = cfg_out.get("quantization_config", {})
        qc["group_size"] = 16

        # also fill group-level weights.group_size to avoid ambiguity
        cg = qc.get("config_groups", {})
        if "group_0" in cg and isinstance(cg["group_0"], dict):
            cg["group_0"].setdefault("weights", {})
            cg["group_0"]["weights"]["group_size"] = 16
        qc["config_groups"] = cg

        cfg_out["quantization_config"] = qc
        json.dump(cfg_out, open(cfg_path_out, "w"), indent=2)

    # 2) Write legacy hf_quant_config.json so has_hf_quant_config() returns True
    # SGLang ModelOptFp4Config.from_config() supports this nested format.
    hf_quant_cfg = {
        "group_size": 16,
        "quantization": {
            "quant_algo": "NVFP4",
            # kv_cache_quant_algo: your quantization_config uses kv_cache_scheme {type=float, num_bits=8}
            # SGLang will interpret that as FP8 kv-cache
            "kv_cache_quant_algo": "FP8",
            # exclude_modules corresponds to quantization_config.ignore
            "exclude_modules": list(set(result_collector.modules_to_not_convert)),
        },
    }
    json.dump(
        hf_quant_cfg,
        open(os.path.join(output_path, "hf_quant_config.json"), "w"),
        indent=2,
    )


    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Path to the directory of the HF safetensors model.")
    parser.add_argument("--save-dir", type=str, help="Path to the directory to save the converted model.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_nvfp4(args.model_dir, args.save_dir, args.max_workers)
