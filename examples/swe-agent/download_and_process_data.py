#!/usr/bin/env python3
"""Download and process data to Slime format."""

import argparse
import json
import tempfile
from pathlib import Path
from datasets import load_dataset


def convert_to_slime_format(input_path: str, output_path: str, limit: int = None, split: str = "train"):
    """Convert JSONL to Slime format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file in Slime format
        limit: Optional limit on number of samples
        split: Dataset split name (used in metadata)
    """
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if limit and count >= limit:
                break

            instance = json.loads(line)

            # Add subset and split to metadata for Gym API
            metadata = dict(instance)
            metadata["subset"] = "gym"
            metadata["split"] = split

            slime_sample = {
                "prompt": instance.get("problem_statement", ""),
                "metadata": metadata,
            }

            fout.write(json.dumps(slime_sample) + "\n")
            count += 1

    print(f"Converted {count} samples: {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset and convert to Slime format")
    parser.add_argument("--input", type=str, required=True, help="HuggingFace dataset path or local JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: train, only for HF datasets)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of samples")

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.exists() and input_path.suffix == ".jsonl":
        print(f"Processing local file: {args.input}")
        convert_to_slime_format(args.input, args.output, args.limit, args.split)
    else:
        print(f"Loading HuggingFace dataset: {args.input} (split={args.split})")
        ds = load_dataset(args.input, split=args.split)

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                tmp_path = tmp.name

            print(f"Downloading to temporary file: {tmp_path}")
            ds.to_json(tmp_path)

            print(f"Converting to Slime format: {args.output}")
            convert_to_slime_format(tmp_path, args.output, split=args.split)
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    print("Done.")


if __name__ == "__main__":
    main()
