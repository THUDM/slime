#!/usr/bin/env python3
"""Generate MOPD eval config and materialized eval JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from common.dataset_selection import resolve_eval_datasets
from common.eval_prep_utils import (
    build_code_eval_row as _build_code_eval_row,
    build_math_eval_row as _build_math_eval_row,
    materialize_eval_dataset as materialize_eval_dataset_shared,
    preprocess_pool_eval_jsonl,
)
from multidomain_shared import GENERIC_EVAL_DATASETS, OFFICIAL_EVAL_DATASETS


def _preprocess_eval_jsonl(src: Path, dst: Path, _domain: str | None = None, row_filter=None) -> int:
    return preprocess_pool_eval_jsonl(src, dst, row_filter=row_filter)


def _bfcl_single_turn_filter(row: dict) -> bool:
    """Keep only BFCL samples that can be evaluated in single-turn mode."""
    meta = row.get("metadata") or {}
    test_cat = str(meta.get("test_category", "")).lower()
    return "multi_turn" not in test_cat


EVAL_DATASETS = GENERIC_EVAL_DATASETS

# Per-dataset row filters applied during preprocessing
_DATASET_FILTERS: dict[str, callable] = {
    "bfcl_v3": _bfcl_single_turn_filter,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-manifest-output", default="")
    parser.add_argument("--max-response-len", type=int, default=8192)
    parser.add_argument("--profile", default="mopd")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--dataset-extra", action="append", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--source-extra", action="append", default=None)
    args = parser.parse_args()

    pool = Path(args.pool_root)
    output_dir = Path(args.output).parent
    eval_data_dir = output_dir / "eval"

    datasets = []
    resolved_datasets = resolve_eval_datasets(
        pool_root=pool,
        profile=args.profile,
        datasets=args.dataset,
        dataset_extras=args.dataset_extra,
        paths=args.source,
        path_extras=args.source_extra,
        default_dataset_names=[name for name, _, _ in EVAL_DATASETS],
    )
    for resolved in resolved_datasets:
        name = resolved.name
        src_path = resolved.source
        if not src_path.exists():
            print(f"  SKIP {name}: {src_path} not found")
            continue

        domain = resolved.runtime_mode
        dst_path = eval_data_dir / f"{name}.jsonl"

        if domain == "pool":
            row_filter = _DATASET_FILTERS.get(name)
            count = _preprocess_eval_jsonl(src_path, dst_path, row_filter=row_filter)
            if count == 0:
                print(f"  SKIP {name}: no valid samples after preprocessing")
                continue
            data_path = str(dst_path)
        elif domain in {"math", "code"}:
            count = materialize_eval_dataset_shared(src_path, dst_path, runtime_mode=domain)
            if count == 0:
                print(f"  SKIP {name}: no valid {domain} samples after runtime conversion")
                continue
            data_path = str(dst_path)
        else:
            data_path = str(src_path)

        datasets.append({"name": name, "path": data_path, "n_samples_per_eval_prompt": resolved.n_samples_per_eval_prompt})

    config = {
        "eval": {
            "defaults": {
                "input_key": "prompt",
                "label_key": "label",
                "metadata_key": "metadata",
                "tool_key": "tools",
                "max_response_len": args.max_response_len,
                "top_k": 1,
            },
            "datasets": datasets,
        }
    }

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote eval config with {len(datasets)} datasets to {dest}")
    for d in datasets:
        print(f"  {d['name']} (n={d['n_samples_per_eval_prompt']})")

    if args.official_manifest_output:
        official_datasets = []
        official_specs = resolve_eval_datasets(
            pool_root=pool,
            paths=None,
            datasets=[name for name, _ in OFFICIAL_EVAL_DATASETS],
        )
        for resolved in official_specs:
            name = resolved.name
            src_path = resolved.source
            if not src_path.exists():
                print(f"  SKIP official {name}: {src_path} not found")
                continue

            dst_path = eval_data_dir / f"{name}.jsonl"
            row_filter = _DATASET_FILTERS.get(name)
            count = _preprocess_eval_jsonl(src_path, dst_path, "tool", row_filter=row_filter)
            if count == 0:
                print(f"  SKIP official {name}: no valid samples after preprocessing")
                continue

            official_datasets.append({"name": name, "path": str(dst_path), "n_samples": count})

        manifest_path = Path(args.official_manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump({"datasets": official_datasets}, handle, ensure_ascii=False, indent=2)

        print(f"Wrote official eval manifest with {len(official_datasets)} datasets to {manifest_path}")
        for dataset in official_datasets:
            print(f"  official {dataset['name']} (n={dataset['n_samples']})")


if __name__ == "__main__":
    main()
