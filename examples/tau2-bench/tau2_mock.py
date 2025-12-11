#!/usr/bin/env python3
"""
Export tau2 tasks into JSONL files for slime training (tau1_mock style).

Each file looks like domain_split_tasks.jsonl with lines:
{"index": 0, "task_id": "0", "task_set": "airline", "task_split": "train", "metadata": {...}}
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional


def iter_tasks(task_set: str, task_split: Optional[str]):
    from tau2.registry import registry

    task_loader = registry.get_tasks_loader(task_set)
    try:
        return task_loader(task_split)
    except TypeError:
        return task_loader()


def write_tasks(tasks: Iterable, task_set: str, task_split: Optional[str], output_dir: Path) -> None:
    suffix = f"_{task_split}" if task_split else ""
    output_path = output_dir / f"{task_set}{suffix}_tasks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for idx, task in enumerate(tasks):
            row = {
                "index": idx,
                "task_id": str(task.id),
                "task_set": task_set,
                "task_split": task_split,
                "metadata": task.model_dump(),
            }
            f.write(json.dumps(row) + "\n")
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="tau2 mock data generator (tau1_mock style).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write JSONL files")
    args = parser.parse_args()

    from tau2.registry import registry

    task_sets = registry.get_task_sets()
    for task_set in task_sets:
        split_loader = registry.get_task_splits_loader(task_set)
        if split_loader is not None:
            splits = list(split_loader().keys())
        else:
            splits = [None]
        for split in splits:
            try:
                tasks = iter_tasks(task_set, split)
            except ValueError as e:
                print(f"[skip] {task_set} split={split}: {e}")
                continue
            write_tasks(tasks, task_set, split, args.output_dir)


if __name__ == "__main__":
    main()
