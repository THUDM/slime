#!/usr/bin/env python3
"""Preprocess tau2-bench tasks into JSONL index files for slime."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


DEFAULT_DOMAINS = ("retail", "airline", "telecom")
DEFAULT_SPLITS = ("train", "test", "base")


def _parse_csv(value: str) -> list[str]:
    items = [x.strip() for x in value.split(",")]
    return [x for x in items if x]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess tau2-bench tasks to JSONL for slime")
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Output directory for `{domain}_{split}_tasks.jsonl` files",
    )
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help=f"Comma-separated list of domains (default: {','.join(DEFAULT_DOMAINS)})",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help=f"Comma-separated list of task splits (default: {','.join(DEFAULT_SPLITS)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max tasks per (domain, split) for smoke testing",
    )
    args = parser.parse_args()

    from tau2.registry import registry

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    domains = _parse_csv(args.domains)
    splits = _parse_csv(args.splits)

    for split in splits:
        all_rows: list[dict[str, Any]] = []
        for domain in domains:
            tasks_loader = registry.get_tasks_loader(domain)
            tasks = tasks_loader(split)
            if args.limit is not None:
                tasks = tasks[: args.limit]

            output_path = os.path.join(local_dir, f"{domain}_{split}_tasks.jsonl")
            with open(output_path, "w") as f:
                for i, task in enumerate(tasks):
                    row: dict[str, Any] = {
                        "text": [{"role": "user", "content": "task"}],
                        "metadata": {
                            "domain": domain,
                            "split": split,
                            "task_id": task.id,
                            "task_index": i,
                            "task": task.model_dump(),
                        },
                    }
                    f.write(json.dumps(row) + "\n")
                    all_rows.append(row)

            print(f"Saved {len(tasks)} tasks to {output_path}")

        merged_path = os.path.join(local_dir, f"tau2_{split}_all_tasks.jsonl")
        with open(merged_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {len(all_rows)} tasks to {merged_path}")


if __name__ == "__main__":
    main()
