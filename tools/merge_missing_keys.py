#!/usr/bin/env python3
"""Merge missing keys (e.g., visual encoder weights) from the original HF model
into a converted Megatron-to-HF checkpoint.

This is needed for VLM models like Qwen3.5-397B-A17B where the Megatron checkpoint
only contains the language model weights, and the visual encoder weights must be
copied from the original HF checkpoint.

Usage:
    python merge_missing_keys.py \
        --origin-hf-dir /path/to/original/Qwen3.5-397B-A17B \
        --converted-dir /path/to/converted/checkpoint \
        [--dry-run]
"""

import argparse
import json
import os
import shutil

import safetensors.torch
from safetensors import safe_open
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Merge missing keys from original HF model into converted checkpoint")
    parser.add_argument(
        "--origin-hf-dir", type=str, required=True, help="Path to the original HuggingFace model directory"
    )
    parser.add_argument("--converted-dir", type=str, required=True, help="Path to the converted checkpoint directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print missing keys without merging")
    parser.add_argument(
        "--chunk-size", type=int, default=5 * 1024**3, help="Chunk size for safetensors files (default 5GB)"
    )
    args = parser.parse_args()

    # Load both index files
    origin_index_path = os.path.join(args.origin_hf_dir, "model.safetensors.index.json")
    converted_index_path = os.path.join(args.converted_dir, "model.safetensors.index.json")

    if not os.path.exists(origin_index_path):
        raise FileNotFoundError(f"Origin index not found: {origin_index_path}")
    if not os.path.exists(converted_index_path):
        raise FileNotFoundError(f"Converted index not found: {converted_index_path}")

    with open(origin_index_path) as f:
        origin_index = json.load(f)
    with open(converted_index_path) as f:
        converted_index = json.load(f)

    origin_keys = set(origin_index["weight_map"].keys())
    converted_keys = set(converted_index["weight_map"].keys())
    missing_keys = sorted(origin_keys - converted_keys)

    if not missing_keys:
        print("No missing keys detected. The converted checkpoint is complete.")
        return

    print(f"Found {len(missing_keys)} missing keys (present in origin but not in converted checkpoint):")

    # Categorize missing keys
    from collections import Counter

    prefix_patterns = Counter()
    for key in missing_keys:
        parts = key.split(".")
        # Group by first 3 parts (e.g., model.visual.blocks.0)
        prefix = ".".join(parts[:4]) if len(parts) >= 4 else key
        prefix_patterns[prefix] += 1

    print("\nMissing key categories:")
    for prefix, count in prefix_patterns.most_common():
        print(f"  {prefix}.*: {count} keys")

    for key in missing_keys[:5]:
        print(f"  - {key}")
    if len(missing_keys) > 5:
        print(f"  ... and {len(missing_keys) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without making changes.")
        return

    # Group missing keys by their source file in origin
    missing_by_file = {}
    for key in missing_keys:
        src_file = origin_index["weight_map"][key]
        if src_file not in missing_by_file:
            missing_by_file[src_file] = []
        missing_by_file[src_file].append(key)

    # Load missing tensors from origin HF safetensors
    print(f"\nLoading {len(missing_keys)} missing tensors from origin HF model...")
    missing_tensors = {}
    for src_file, keys in tqdm(missing_by_file.items(), desc="Reading origin safetensors"):
        src_path = os.path.join(args.origin_hf_dir, src_file)
        if not os.path.exists(src_path):
            print(f"WARNING: {src_path} not found. Skipping keys: {keys}")
            continue
        with safe_open(src_path, framework="pt", device="cpu") as f:
            for key in keys:
                missing_tensors[key] = f.get_tensor(key)

    # Determine current file count
    current_files = set(converted_index["weight_map"].values())
    total_files = len(current_files)

    # Calculate missing size
    missing_size = sum(t.numel() * t.element_size() for t in missing_tensors.values())
    print(f"Missing tensors total size: {missing_size / 1e9:.2f} GB")

    # Add missing tensors to a new shard
    new_total = total_files + 1
    new_shard_name = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"
    new_shard_path = os.path.join(args.converted_dir, new_shard_name)

    print(f"Writing {len(missing_tensors)} tensors to new shard: {new_shard_name}")
    safetensors.torch.save_file(missing_tensors, new_shard_path)

    # Update weight map: add new entries and update file numbering
    weight_map = converted_index["weight_map"]

    # Add missing keys pointing to the new shard
    for key in missing_tensors:
        weight_map[key] = new_shard_name

    # Rename existing files to update total count
    print(f"Renaming {total_files} existing shards to update total count from {total_files} to {new_total}...")
    for i in range(1, total_files + 1):
        old_name = f"model-{i:05d}-of-{total_files:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{new_total:05d}.safetensors"
        old_path = os.path.join(args.converted_dir, old_name)
        new_path = os.path.join(args.converted_dir, new_name)
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
        # Update weight map references
        for k, v in weight_map.items():
            if v == old_name:
                weight_map[k] = new_name

    # Update and save index
    converted_index["metadata"]["total_size"] = converted_index["metadata"].get("total_size", 0) + missing_size
    converted_index["weight_map"] = weight_map

    with open(converted_index_path, "w") as f:
        json.dump(converted_index, f, indent=2)

    print(f"\nDone! Merged {len(missing_tensors)} missing keys into the converted checkpoint.")
    print(f"New total size: {converted_index['metadata']['total_size'] / 1e9:.2f} GB")
    print(f"Total shards: {new_total}")
    print(f"Total keys: {len(weight_map)}")


if __name__ == "__main__":
    main()
