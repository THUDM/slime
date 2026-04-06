#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from dataset_registry import TRAIN_DATASET_SOURCE_MAP
from dataset_selection import (
    discover_train_sources as discover_train_sources_shared,
    resolve_train_sources,
    resolve_named_datasets as resolve_named_datasets_from_registry,
)
from pool_data_utils import iter_json_object_rows, write_source_manifest as write_source_manifest_shared
from pool_runtime_semantics import materialize_runtime_pool_row
DATASET_SOURCE_MAP = TRAIN_DATASET_SOURCE_MAP


def discover_sources(pool_root: Path) -> list[Path]:
    return discover_train_sources_shared(pool_root)


def iter_rows(path: Path) -> Iterator[dict]:
    yield from iter_json_object_rows(path)


def align_row_to_v1_normalized_shape(row: dict[str, Any]) -> dict[str, Any] | None:
    return materialize_runtime_pool_row(row)


def resolve_named_datasets(pool_root: Path, dataset_names: Sequence[str]) -> list[Path]:
    return resolve_named_datasets_from_registry(pool_root, list(dataset_names))


def iter_selected_rows(
    sources: Sequence[Path],
    skip_samples: int = 0,
    max_samples: int | None = None,
) -> Iterator[dict]:
    seen = 0
    yielded = 0
    for source in sources:
        for row in iter_rows(source):
            if seen < skip_samples:
                seen += 1
                continue
            if max_samples is not None and yielded >= max_samples:
                return
            yield row
            yielded += 1


def write_dataset(
    sources: Sequence[Path],
    dest: Path,
    skip_samples: int = 0,
    max_samples: int | None = None,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in iter_selected_rows(sources, skip_samples=skip_samples, max_samples=max_samples):
            row = align_row_to_v1_normalized_shape(row)
            if row is None:
                continue
            import json

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def write_source_manifest(sources: Sequence[Path], dest: Path) -> int:
    return write_source_manifest_shared(sources, dest)


def _resolve_sources(args: argparse.Namespace) -> list[Path]:
    pool_root = Path(args.pool_root) if args.pool_root else None
    if pool_root is None:
        if args.source:
            return [Path(source) for source in args.source]
        raise ValueError("Provide --pool-root when using dataset-based resolution.")
    return resolve_train_sources(
        pool_root=pool_root,
        profile=args.profile,
        datasets=args.dataset,
        dataset_extras=args.dataset_extra,
        paths=args.source,
        path_extras=args.source_extra,
        manifest=args.manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve or concatenate pre-normalized multidomain v2 pool data.")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--source-extra", action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--dataset-extra", action="append", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pool-root", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--print-sources", action="store_true", default=False)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = _resolve_sources(args)
    if not sources:
        raise SystemExit("No normalized pool sources were found.")
    if args.print_sources:
        for source in sources:
            print(source)
        return
    if not args.dest:
        raise SystemExit("--dest is required unless --print-sources is set.")
    write_dataset(
        sources,
        Path(args.dest),
        skip_samples=args.skip_samples,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
