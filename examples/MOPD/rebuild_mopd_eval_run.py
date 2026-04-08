#!/usr/bin/env python3
"""Rebuild a clean eval-only wandb run from the messy MOPD training run.

Extracts all eval rows from the source run, deduplicates by eval/step
(keeping the row with the most metrics for each step), and logs them
into a fresh run.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_eval_rows(
    api: wandb.Api, entity: str, project: str, run_id: str,
) -> dict[int, dict[str, Any]]:
    """Scan run history, return deduplicated eval rows keyed by eval/step.

    For duplicate eval/step values, the row with the most non-None eval keys wins.
    """
    run = api.run(f"{entity}/{project}/{run_id}")
    # Collect ALL rows for each eval/step, then pick the best
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in run.scan_history(page_size=1000):
        eval_step = row.get("eval/step")
        if eval_step is None:
            continue
        step = int(eval_step)
        # Strip internal keys
        cleaned = {
            k: v for k, v in row.items()
            if not k.startswith("_") and v is not None
        }
        # Only keep if it has actual eval metrics (not just eval/step)
        eval_keys = [
            k for k in cleaned
            if k.startswith("eval/") and k != "eval/step"
        ]
        if eval_keys:
            candidates[step].append(cleaned)

    # For each step, pick the row with the most eval keys
    best: dict[int, dict[str, Any]] = {}
    for step, rows in candidates.items():
        winner = max(rows, key=lambda r: len([
            k for k in r if k.startswith("eval/") and k != "eval/step"
        ]))
        n_keys = len([k for k in winner if k.startswith("eval/") and k != "eval/step"])
        best[step] = winner
        if len(rows) > 1:
            logger.info("  eval/step=%d: %d candidates, picked row with %d eval keys",
                        step, len(rows), n_keys)

    logger.info("Collected %d unique eval steps from %s: %s",
                len(best), run_id, sorted(best.keys()))
    return best


def create_clean_run(
    entity: str,
    project: str,
    run_name: str,
    group: str,
    eval_data: dict[int, dict[str, Any]],
    source_run_id: str,
) -> str:
    """Create a clean eval-only run with proper metric definitions."""
    tags = ["rebuilt-eval", f"source:{source_run_id}"]

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group or None,
        tags=tags,
        config={"_rebuild": {"source_run": source_run_id}},
    )

    # Define eval/step as the x-axis for all eval metrics
    wandb.define_metric("eval/step", overwrite=True)
    wandb.define_metric("eval/*", step_metric="eval/step", overwrite=True)

    for eval_step in sorted(eval_data.keys()):
        row = dict(eval_data[eval_step])
        row["eval/step"] = eval_step
        wandb.log(row)

    logger.info("Logged %d eval steps", len(eval_data))

    run_url = run.url or ""
    run_id = run.id or ""
    wandb.finish(exit_code=0)
    return run_url or f"{entity}/{project}/{run_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-host", default="https://wandb2.sii.edu.cn")
    parser.add_argument("--wandb-key", default="local-c6d3e5712d547834724d8d98094f340b3c2a869c")
    parser.add_argument("--wandb-entity", default="gzy")
    parser.add_argument("--source-project", default="slime-mopd")
    parser.add_argument("--source-run-id", default="0vrz32be")
    parser.add_argument("--target-project", default="slime-mopd")
    parser.add_argument("--run-name", default="mopd-eval-rebuilt")
    parser.add_argument("--group", default="mopd-3node-h200-liteeval-noeval0-dist-h200-noroutingreplay-retry-0407-0633")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wandb.login(key=args.wandb_key, host=args.wandb_host)

    api_kwargs: dict[str, Any] = {"timeout": 60}
    if args.wandb_host:
        api_kwargs["overrides"] = {"base_url": args.wandb_host}
    api = wandb.Api(**api_kwargs)

    entity = args.wandb_entity

    logger.info("Scanning source run: %s/%s/%s", entity, args.source_project, args.source_run_id)
    eval_data = collect_eval_rows(api, entity, args.source_project, args.source_run_id)

    if not eval_data:
        logger.error("No eval data found!")
        return

    if args.dry_run:
        for step in sorted(eval_data.keys()):
            keys = sorted(k for k in eval_data[step] if k.startswith("eval/") and k != "eval/step")
            logger.info("  step=%d: %d eval keys", step, len(keys))
        logger.info("[DRY RUN] Would create run '%s' with %d steps", args.run_name, len(eval_data))
        return

    url = create_clean_run(
        entity=entity,
        project=args.target_project,
        run_name=args.run_name,
        group=args.group,
        eval_data=eval_data,
        source_run_id=args.source_run_id,
    )
    logger.info("Created: %s", url)


if __name__ == "__main__":
    main()
