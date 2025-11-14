#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__:
    from .data_utils import ensure_metadata_dict, normalize_prompt, normalize_reward_model
else:  # pragma: no cover - support execution via `python path/to/script.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from examples.simpletir.data_utils import ensure_metadata_dict, normalize_prompt, normalize_reward_model


SYSTEM_PROMPT = textwrap.dedent(
    """
    Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (after "Code execution result: ") is returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.
    Code Format:
    Each code snippet is wrapped between ```. You need to use `print()` to output intermediate results.
    Answer Format:
    You can use the `final_answer()` function in the code to return your final answer. For example, to answer the User Question: What is the result of the 5 + 3 + 1294.678?, you can write:
    ```py
    answer = 5 + 3 + 1294.678
    final_answer(answer)
    ```
    You can also use \\boxed to return your answer. The last part of your response should be:
    \\boxed{'The final answer goes here.'}
    User Question:
    """
).strip()


TRAIN_SPLITS_WITH_SYSTEM = {("deepscaler", "train"), ("simplelr_math_35", "train")}


def ensure_system_prompt(messages: list[dict[str, str]], *, enabled: bool = True) -> list[dict[str, str]]:
    if not enabled:
        return messages
    if messages and (messages[0].get("role") == "system"):
        content = (messages[0].get("content") or "").strip()
        if content == SYSTEM_PROMPT:
            return messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def _stringify_ground_truth(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def discover_parquet_files(source_dir: Path, include: set[str] | None) -> Iterable[tuple[str, Path]]:
    for dataset_dir in sorted(source_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for parquet_file in sorted(dataset_dir.glob("*.parquet")):
            spec = f"{dataset_dir.name}/{parquet_file.stem}"
            if include and spec not in include:
                continue
            yield spec, parquet_file


def convert_file(dataset_name: str, parquet_path: Path, output_path: Path):
    df = pd.read_parquet(parquet_path)
    split_name = parquet_path.stem
    inject_system_prompt = (dataset_name, split_name) in TRAIN_SPLITS_WITH_SYSTEM
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row_id, row in enumerate(df.to_dict(orient="records")):
            reward_model = normalize_reward_model(row.get("reward_model"))
            ability = row.get("ability") or ""
            extra_info = ensure_metadata_dict(row.get("extra_info"))
            extra_info.setdefault("index", row_id)
            extra_info.setdefault("split", split_name)
            extra_info.setdefault("dataset", dataset_name)
            extra_info["simpletir_dataset_spec"] = extra_info.get(
                "simpletir_dataset_spec", f"{dataset_name}/{split_name}"
            )
            extra_info["simpletir_dataset_path"] = str(output_path)
            prompt = normalize_prompt(row.get("prompt"))
            prompt = ensure_system_prompt(prompt, enabled=inject_system_prompt)
            record = {
                "prompt": prompt,
                "label": _stringify_ground_truth(reward_model.get("ground_truth")),
                "extra_info": {
                    **extra_info,
                    "ability": ability,
                    "reward_style": reward_model.get("style"),
                    "data_source": row.get("data_source"),
                },
                "ability": ability,
                "data_source": row.get("data_source"),
                "reward_model": reward_model,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert SimpleTIR parquet releases into slime-friendly JSONL files.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing the official SimpleTIR datasets (e.g. path/to/SimpleTIR/datasets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/datasets/simpletir"),
        help="Destination directory for converted jsonl files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help='Optional whitelist like "deepscaler/train simplelr_math_35/train". Defaults to all datasets found.',
    )

    args = parser.parse_args()
    include = set(args.datasets) if args.datasets else None
    specs = list(discover_parquet_files(args.source, include))
    if not specs:
        raise SystemExit(f"No parquet files found in {args.source}")

    for spec, parquet_file in specs:
        dataset_name, split = spec.split("/")
        output_path = args.output / f"{dataset_name}_{split}.jsonl"
        print(f"[SimpleTIR] Converting {parquet_file} -> {output_path}")
        convert_file(dataset_name, parquet_file, output_path)

    print(f"Finished converting {len(specs)} splits into {args.output}")


if __name__ == "__main__":
    main()
