#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from dataset_selection import resolve_eval_datasets
from eval_prep_utils import materialize_eval_dataset as materialize_eval_dataset_shared


def materialize_dataset(
    pool_root: Path,
    dataset_name: str | None,
    dest: Path,
    max_samples: int | None = None,
    source: Path | None = None,
) -> int:
    resolved = resolve_eval_datasets(
        pool_root=pool_root,
        datasets=[dataset_name] if dataset_name else None,
        paths=[source] if source is not None else None,
    )
    if len(resolved) != 1:
        raise ValueError("Expected exactly one eval dataset to resolve.")
    spec = resolved[0]
    return materialize_eval_dataset_shared(
        spec.source,
        dest,
        runtime_mode=spec.runtime_mode,
        max_samples=max_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one eval dataset into runtime JSONL.")
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset and not args.source:
        raise SystemExit("Provide either --dataset or --source.")
    count = materialize_dataset(
        pool_root=Path(args.pool_root),
        dataset_name=args.dataset,
        dest=Path(args.dest),
        max_samples=args.max_samples,
        source=Path(args.source) if args.source else None,
    )
    label = args.dataset or args.source
    print(f"Materialized {count} rows for {label} to {args.dest}")


if __name__ == "__main__":
    main()
