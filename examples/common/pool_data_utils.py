from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


def iter_json_object_rows(
    path: Path,
    *,
    skip_invalid_json: bool = False,
    open_errors: str = "strict",
) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors=open_errors) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if skip_invalid_json:
                    continue
                raise
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def transform_jsonl(
    src: Path,
    dst: Path,
    *,
    row_builder: Callable[[dict[str, Any]], dict[str, Any] | None],
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
    skip_invalid_json: bool = True,
    open_errors: str = "strict",
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst.open("w", encoding="utf-8") as fout:
        for row in iter_json_object_rows(
            src,
            skip_invalid_json=skip_invalid_json,
            open_errors=open_errors,
        ):
            transformed = row_builder(row)
            if transformed is None:
                continue
            if row_filter and not row_filter(transformed):
                continue
            fout.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_source_manifest(sources: Sequence[Path], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for source in sources:
            handle.write(f"{source}\n")
    return len(sources)
