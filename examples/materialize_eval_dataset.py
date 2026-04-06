#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from multidomain_shared import GENERIC_EVAL_DATASETS, OFFICIAL_EVAL_DATASETS
from pool_runtime_semantics import materialize_runtime_pool_row


GENERIC_EVAL_MAP = {name: rel for name, rel, _ in GENERIC_EVAL_DATASETS}
OFFICIAL_EVAL_MAP = {name: rel for name, rel in OFFICIAL_EVAL_DATASETS}


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def materialize_dataset(pool_root: Path, dataset_name: str, dest: Path, max_samples: int | None = None) -> int:
    if dataset_name in GENERIC_EVAL_MAP:
        src = pool_root / GENERIC_EVAL_MAP[dataset_name]
    elif dataset_name in OFFICIAL_EVAL_MAP:
        src = pool_root / OFFICIAL_EVAL_MAP[dataset_name]
    else:
        supported = ", ".join(sorted(set(GENERIC_EVAL_MAP) | set(OFFICIAL_EVAL_MAP)))
        raise ValueError(f"Unsupported eval dataset '{dataset_name}'. Supported datasets: {supported}")

    if not src.is_file():
        raise FileNotFoundError(f"Eval dataset source does not exist: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in _load_rows(src):
            if max_samples is not None and written >= max_samples:
                break
            payload = materialize_runtime_pool_row(row)
            if payload is None:
                continue
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one eval dataset from pool into runtime JSONL.")
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = materialize_dataset(
        pool_root=Path(args.pool_root),
        dataset_name=args.dataset,
        dest=Path(args.dest),
        max_samples=args.max_samples,
    )
    print(f"Materialized {count} rows for {args.dataset} to {args.dest}")


if __name__ == "__main__":
    main()

