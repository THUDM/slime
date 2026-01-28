"""
python tools/convert_hf_to_int4_direct.py [-h] [--model-dir MODEL_DIR] [--save-dir SAVE_DIR]
                           [--group-size GROUP_SIZE] [--is-symmetric IS_SYMMETRIC] [--ignore-rules IGNORE_RULES]
                           [--max-workers MAX_WORKERS]
options:
  -h, --help            show this help message and exit
"""

import argparse
import gc
import json
import math
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

try:
    import fake_int4_quant_cuda
except ImportError:
    fake_int4_quant_cuda = None


def pack_to_int32(
    value,
    num_bits,
    packed_dim=1,
    sym=False,
):
    # if value.dtype is not torch.int8:
    #     raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    if sym:
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


@torch.no_grad()
def quantize(
    x,
    scale,
    zero_point,
    dtype=torch.int8,
):
    group_size = x.shape[-1] // scale.shape[-1]
    output_dtype = dtype
    output = torch.zeros_like(x).to(output_dtype)

    reshaped_dims = (
        math.ceil(x.shape[-1] / group_size),
        group_size,
    )
    x = x.unflatten(-1, reshaped_dims)

    scaled = x / scale.unsqueeze(-1)

    if zero_point is not None:
        zero_point = zero_point.unsqueeze(-1)
        scaled += zero_point.to(x.dtype)

    # clamp and round
    output = round_to_quantized_type_dtype(tensor=scaled, dtype=dtype)

    output = output.flatten(start_dim=-2)
    output = output.to(output_dtype)

    return output


def pack_layer(weight, group_size, sym=True):
    w, scale, zp = fake_int4_quant_cuda.fake_int4_quant_cuda(weight, (1, group_size), sym)
    w = w.view(weight.shape[0], 1, weight.shape[1] // group_size, group_size)
    scale = scale.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    zp = zp.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    if sym:
        w = w * scale
    else:
        w = (w - zp) * scale
    w = w.view(weight.shape)
    scale = scale.view(weight.shape[0], -1).contiguous()
    if not sym:
        zp = zp.view(weight.shape[0], -1)
        zeros = zp.t().contiguous().to(torch.float32)
        zeros = zeros.to(dtype=torch.int32, device=w.device)
        zeros = zeros.reshape(-1, zeros.shape[1] // 8, 8)
        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=zeros.device) * 4
        zeros = zeros << new_order_map
        packed_zp = torch.sum(zeros, dim=-1).to(torch.int32)
    else:
        zp = None
        packed_zp = None

    quantized_weight = quantize(
        x=w,
        scale=scale,
        zero_point=zp,
        dtype=torch.int8 if sym else torch.uint8,
    )
    packed_weight = pack_to_int32(quantized_weight, 4, sym=sym)
    return packed_weight, scale, packed_zp


class ConversionResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0

    def add_result(self, filename, q_weights):
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.param_count += len(v)


def process_file(input_path, output_path, filename, group_size, is_symmetric, ignore_rules, result_collector):

    print(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    weights = {}
    q_weights = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    for name, weight in weights.items():
        is_ignored = any(
            (r.startswith("re:") and re.match(r[3:], name)) or r == name or name.startswith(r) for r in ignore_rules
        )

        if is_ignored or not name.endswith(".weight") or weight.dim() < 2:
            print(f"Ignoring {name}, memory usage: {torch.cuda.memory_allocated()}")
            q_weights[name] = weight
            continue

        print(f"Packing {name}, memory usage: {torch.cuda.memory_allocated()}")
        qw, s, zp = pack_layer(weight, group_size, is_symmetric)
        qweight_name = name.replace(".weight", ".weight_packed")
        scale_name = name.replace(".weight", ".weight_scale")
        weight_shape = torch.tensor(weight.shape, dtype=torch.int32, device="cuda")
        weight_shape_name = name.replace(".weight", ".weight_shape")
        if zp is not None:
            zp_name = name.replace(".weight", ".weight_zero_point")
            q_weights[zp_name] = zp
        q_weights[qweight_name] = qw
        q_weights[scale_name] = s
        q_weights[weight_shape_name] = weight_shape

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, q_weights)


def convert_int4(input_path, output_path, group_size, is_symmetric, ignore_rules, max_workers):
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)
    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()
    # debug in single thread
    # for filename in safetensors_files:
    #     process_file(input_path, output_path, filename, group_size, is_symmetric, ignore_rules, result_collector)

    # multi thread
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                process_file,
                input_path,
                output_path,
                filename,
                group_size,
                is_symmetric,
                ignore_rules,
                result_collector,
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    quant_group = {
        "group_0": {
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": group_size,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "group",
                "symmetric": is_symmetric,
                "type": "int",
            },
        },
    }
    quantization_config = {
        "config_groups": quant_group,
        "format": "pack-quantized",
        "ignore": ignore_rules,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {"weight_map": result_collector.weight_map, "metadata": {"total_size": result_collector.param_count}}
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    torch.cuda.empty_cache()

    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="local BF16 path")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=32, help="Group Size")
    parser.add_argument("--is-symmetric", type=bool, default=True, help="Is Symmetric")
    parser.add_argument(
        "--ignore-rules",
        nargs="+",
        default=[
            "re:.*lm_head.*",
            "re:.*norm.*",
            "re:.*embed.*",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            "re:.*mlp\\.gate\\.*",
        ],
        help="Ignore Rules",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_int4(
        args.model_dir, args.save_dir, args.group_size, args.is_symmetric, args.ignore_rules, args.max_workers
    )
    print(f"Conversion complete, output saved to {args.save_dir}")


if __name__ == "__main__":
    main()
