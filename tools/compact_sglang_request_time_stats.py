#!/usr/bin/env python3
"""Compact SGLang ReqTimeStats logs into one structured JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.profiling.request_time_stats import (
    PARSER_VERSION,
    iter_request_time_stats_files,
    parse_request_time_stats_line,
    parse_request_time_stats_record_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="ReqTimeStats log file or directory, e.g. $RUN_DIR/request_time_stats/sglang")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path. The file is written atomically via a temporary sibling file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    files = iter_request_time_stats_files(input_path, exclude={output_path, tmp_path})

    record_count = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for file_path in files:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parsed = parse_request_time_stats_record_line(line, source=str(file_path))
                    if parsed is None:
                        parsed = parse_request_time_stats_line(line, source=str(file_path))
                    if parsed is None:
                        if "ReqTimeStats(" not in line:
                            continue
                        record = {
                            "schema_version": "slime.request_time_stats.parse_error.v1",
                            "parse_error": True,
                            "parser_version": PARSER_VERSION,
                            "source": str(file_path),
                            "raw_line": line.rstrip("\n"),
                        }
                    else:
                        rid, attrs = parsed
                        record = {"rid": rid, **attrs}
                    out.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")))
                    out.write("\n")
                    record_count += 1

    os.replace(tmp_path, output_path)
    print(f"files: {len(files)}")
    print(f"records: {record_count}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
