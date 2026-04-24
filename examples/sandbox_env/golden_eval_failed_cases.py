#!/usr/bin/env python3
"""Prepare or run official SWE-rebench-V2 golden eval for selected cases."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
DATASET_ROOT = (
    AVALANCHE_ROOT / "data" / "raw_data" / "single" / "swe_rebench_v2"
)
TASKS_JSON = DATASET_ROOT / "data" / "train-00000-of-00001.json"
REBENCH_REPO = DATASET_ROOT / "SWE-rebench-V2"
EVAL_SCRIPT = REBENCH_REPO / "scripts" / "eval.py"
OUT_DIR = Path(__file__).resolve().parent / "data_output"

# These are the instances we repeatedly saw fail in our local replay eval,
# especially at test_patch application time.
DEFAULT_INSTANCE_IDS = [
    "rust-bitcoin__rust-bitcoin-2255",
    "gradleup__shadow-1448",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run official SWE-rebench-V2 golden eval."
    )
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=TASKS_JSON,
        help="Full dataset JSON file.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Instance id to include. Repeat this flag to add more.",
    )
    parser.add_argument(
        "--instance-ids",
        default="",
        help="Comma-separated instance ids to include.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUT_DIR / "rebench_golden_eval_failed_cases.json",
        help="Subset JSON to write for official eval.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=OUT_DIR / "rebench_golden_eval_failed_cases.report.json",
        help="Official eval report output path.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallelism to pass to official eval.py.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the official golden eval command after preparing files.",
    )
    return parser.parse_args()


def _collect_instance_ids(args: argparse.Namespace) -> list[str]:
    values = list(DEFAULT_INSTANCE_IDS)
    values.extend(str(x).strip() for x in args.instance_id if str(x).strip())
    if args.instance_ids:
        values.extend(
            item.strip() for item in str(args.instance_ids).split(",") if item.strip()
        )

    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _load_tasks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _select_tasks(tasks: list[dict], instance_ids: list[str]) -> tuple[list[dict], list[str]]:
    wanted = set(instance_ids)
    selected = [task for task in tasks if str(task.get("instance_id") or "") in wanted]
    found = {str(task.get("instance_id") or "") for task in selected}
    missing = [instance_id for instance_id in instance_ids if instance_id not in found]
    return selected, missing


def _build_command(
    subset_json: Path,
    report_json: Path,
    max_workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--json",
        str(subset_json),
        "--golden-eval",
        "--max-workers",
        str(max_workers),
        "--report-json",
        str(report_json),
    ]


def main() -> int:
    args = _parse_args()
    instance_ids = _collect_instance_ids(args)
    if not instance_ids:
        print("No instance ids selected.", file=sys.stderr)
        return 1
    if args.max_workers < 1:
        print("--max-workers must be >= 1.", file=sys.stderr)
        return 1

    tasks = _load_tasks(args.tasks_json)
    selected, missing = _select_tasks(tasks, instance_ids)
    if not selected:
        print("No tasks matched the selected instance ids.", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cmd = _build_command(
        subset_json=args.output_json.resolve(),
        report_json=args.report_json.resolve(),
        max_workers=args.max_workers,
    )
    quoted_cmd = " ".join(shlex.quote(part) for part in cmd)

    summary = {
        "tasks_json": str(args.tasks_json.resolve()),
        "subset_json": str(args.output_json.resolve()),
        "report_json": str(args.report_json.resolve()),
        "selected_instance_ids": [str(task.get("instance_id") or "") for task in selected],
        "missing_instance_ids": missing,
        "official_eval_cwd": str(REBENCH_REPO.resolve()),
        "official_eval_command": quoted_cmd,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.run:
        return 0

    result = subprocess.run(
        cmd,
        cwd=str(REBENCH_REPO),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
