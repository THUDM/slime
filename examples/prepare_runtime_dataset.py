#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from common.dataset_selection import (
    ResolvedEvalDataset,
    discover_train_sources,
    resolve_eval_datasets,
    resolve_named_datasets as resolve_named_datasets_from_registry,
    resolve_train_sources,
)
from common.eval_prep_utils import materialize_eval_dataset as materialize_eval_dataset_shared
from common.pool_data_utils import iter_json_object_rows, write_source_manifest as write_source_manifest_shared
from common.pool_runtime_semantics import materialize_runtime_pool_row


def discover_sources(pool_root: Path) -> list[Path]:
    return discover_train_sources(pool_root)


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


def resolve_train_data_sources(args: argparse.Namespace) -> list[Path]:
    pool_root = Path(args.pool_root) if args.pool_root else None
    if pool_root is None:
        if args.source:
            return [Path(source) for source in args.source]
        raise ValueError("Provide --pool-root when using dataset-based train resolution.")
    return resolve_train_sources(
        pool_root=pool_root,
        profile=args.profile,
        datasets=args.dataset,
        dataset_extras=args.dataset_extra,
        paths=args.source,
        path_extras=args.source_extra,
        manifest=args.manifest,
    )


def resolve_eval_data_source(args: argparse.Namespace) -> ResolvedEvalDataset:
    resolved = resolve_eval_datasets(
        pool_root=Path(args.pool_root),
        profile=args.profile,
        datasets=[args.dataset] if args.dataset else None,
        paths=[args.source] if args.source else None,
    )
    if len(resolved) != 1:
        raise ValueError("Expected exactly one eval dataset to resolve.")
    return resolved[0]


def materialize_eval_data(
    pool_root: Path,
    dest: Path,
    max_samples: int | None = None,
    dataset_name: str | None = None,
    source: Path | None = None,
    profile: str | None = None,
) -> int:
    args = argparse.Namespace(
        pool_root=str(pool_root),
        dataset=dataset_name,
        source=str(source) if source is not None else None,
        profile=profile,
    )
    resolved = resolve_eval_data_source(args)
    return materialize_eval_dataset_shared(
        resolved.source,
        dest,
        runtime_mode=resolved.runtime_mode,
        max_samples=max_samples,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared runtime-data preparation CLI for multidomain examples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Resolve and materialize runtime training data.")
    train_parser.add_argument("--source", action="append", default=None)
    train_parser.add_argument("--source-extra", action="append", default=None)
    train_parser.add_argument("--dataset", action="append", default=None)
    train_parser.add_argument("--dataset-extra", action="append", default=None)
    train_parser.add_argument("--manifest", default=None)
    train_parser.add_argument("--pool-root", default=None)
    train_parser.add_argument("--profile", default=None)
    train_parser.add_argument("--dest", default=None)
    train_parser.add_argument("--print-sources", action="store_true", default=False)
    train_parser.add_argument("--skip-samples", type=int, default=0)
    train_parser.add_argument("--max-samples", type=int, default=None)

    eval_parser = subparsers.add_parser("eval", help="Materialize one eval dataset into runtime JSONL.")
    eval_parser.add_argument("--pool-root", required=True)
    eval_parser.add_argument("--profile", default=None)
    eval_parser.add_argument("--dataset", default=None)
    eval_parser.add_argument("--source", default=None)
    eval_parser.add_argument("--dest", required=True)
    eval_parser.add_argument("--max-samples", type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        sources = resolve_train_data_sources(args)
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
        return

    if not args.dataset and not args.source:
        raise SystemExit("Provide either --dataset or --source.")
    count = materialize_eval_data(
        pool_root=Path(args.pool_root),
        dest=Path(args.dest),
        max_samples=args.max_samples,
        dataset_name=args.dataset,
        source=Path(args.source) if args.source else None,
        profile=args.profile,
    )
    label = args.dataset or args.source
    print(f"Materialized {count} rows for {label} to {args.dest}")


if __name__ == "__main__":
    main()
