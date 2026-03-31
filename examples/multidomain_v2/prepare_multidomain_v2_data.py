#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence


POOL_SUBDIRS = ("structured", "stem", "tool")


def discover_sources(pool_root: Path) -> list[Path]:
    if not pool_root.exists():
        raise FileNotFoundError(f"Pool root does not exist: {pool_root}")
    sources: list[Path] = []
    for subdir in POOL_SUBDIRS:
        candidate = pool_root / subdir
        if not candidate.is_dir():
            continue
        sources.extend(path for path in candidate.rglob("*.jsonl") if path.is_file())
    return sorted(sources)


def iter_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def align_row_to_v1_normalized_shape(row: dict[str, Any]) -> dict[str, Any]:
    aligned = dict(row)
    aligned["tools"] = list(row.get("tools") or [])
    return aligned


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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def _resolve_sources(args: argparse.Namespace) -> list[Path]:
    explicit_sources = [Path(source) for source in args.source or []]
    if explicit_sources:
        return explicit_sources
    if args.pool_root:
        return discover_sources(Path(args.pool_root))
    raise ValueError("Provide at least one --source or set --pool-root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate pre-normalized multidomain v2 jsonl data.")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--pool-root", default=None)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = _resolve_sources(args)
    if not sources:
        raise SystemExit("No normalized pool sources were found.")
    write_dataset(
        sources,
        Path(args.dest),
        skip_samples=args.skip_samples,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
