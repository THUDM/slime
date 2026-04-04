#!/usr/bin/env python3
"""Consolidate messy wandb runs into clean experiment runs.

For each experiment (data mix), merges training/rollout/eval data from multiple
source runs (initial + resumes) into a single clean run with:
  - Proper metric axis definitions (eval/* vs eval/step, etc.)
  - Deduplicated eval rows merged by eval/step
  - Clean short names like "tool45-struct25-stem30"
  - Continuous training curves

Usage:
    # Dry-run: see what would be consolidated
    python wandb_consolidate.py --dry-run

    # Consolidate all experiments
    python wandb_consolidate.py

    # Consolidate a single experiment
    python wandb_consolidate.py --experiment tool45-struct25-stem30

    # Custom target project
    python wandb_consolidate.py --target-project slime-mopd-clean
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Experiment definitions ────────────────────────────────────────────
# Each experiment groups runs that belong to the same training trajectory.
# Runs are listed in chronological order; later runs override earlier data
# at overlapping steps.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "tool15-struct55-stem30": {
        "runs": ["irgu6dsr", "zebi2wqu", "ez3stpua"],
        "description": "mdv1 3-node: 15% tool, 55% structured, 30% stem",
    },
    "tool45-struct25-stem30": {
        "runs": ["0hgesrzt", "2kb49tbm"],
        "description": "mdv1 3-node: 45% tool, 25% structured, 30% stem",
    },
    "tool50-struct35-stem15": {
        "runs": ["akzm0qu9"],
        "description": "mdv1 3-node: 50% tool, 35% structured, 15% stem",
    },
}

SOURCE_PROJECT = "slime-multidomain-v1"


# ── Helpers ───────────────────────────────────────────────────────────

def _define_wandb_metrics() -> None:
    """Define custom x-axes so each metric family plots against its own step."""
    wandb.define_metric("train/step", overwrite=True)
    wandb.define_metric("train/*", step_metric="train/step", overwrite=True)
    wandb.define_metric("rollout/step", overwrite=True)
    wandb.define_metric("rollout/*", step_metric="rollout/step", overwrite=True)
    wandb.define_metric("rollout_by_domain/*", step_metric="rollout/step", overwrite=True)
    wandb.define_metric("rollout_by_source/*", step_metric="rollout/step", overwrite=True)
    wandb.define_metric("perf/*", step_metric="rollout/step", overwrite=True)
    wandb.define_metric("eval/step", overwrite=True)
    wandb.define_metric("eval/*", step_metric="eval/step", overwrite=True)
    wandb.define_metric("eval_by_domain/*", step_metric="eval/step", overwrite=True)
    wandb.define_metric("eval_by_source/*", step_metric="eval/step", overwrite=True)


def _classify_row(row: dict[str, Any]) -> tuple[str, int | None]:
    """Classify a history row and return (category, step).

    Returns one of:
      ("rollout", rollout_step) — rollout/perf/rollout_by_* metrics
      ("train", train_step)     — train/* metrics
      ("eval", eval_step)       — eval/eval_by_* metrics
      ("skip", None)            — no useful data
    """
    eval_step = row.get("eval/step")
    if eval_step is not None:
        has_eval = any(
            k.startswith("eval/") and k != "eval/step" for k, v in row.items() if v is not None
        ) or any(
            k.startswith("eval_by_") for k, v in row.items() if v is not None
        )
        if has_eval:
            return "eval", int(eval_step)

    train_step = row.get("train/step")
    if train_step is not None:
        has_train = any(
            k.startswith("train/") and k != "train/step" for k, v in row.items() if v is not None
        )
        if has_train:
            return "train", int(train_step)

    rollout_step = row.get("rollout/step")
    if rollout_step is not None:
        return "rollout", int(rollout_step)

    return "skip", None


def _strip_internal_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Remove wandb internal keys (_step, _runtime, _timestamp, etc.)."""
    return {k: v for k, v in row.items() if not k.startswith("_") and v is not None}


def collect_run_data(
    api: wandb.Api, entity: str, project: str, run_id: str,
) -> tuple[
    dict[int, dict[str, Any]],  # rollout rows by step
    dict[int, dict[str, Any]],  # train rows by step
    dict[int, dict[str, Any]],  # eval rows by eval/step
    dict[str, Any],             # run config
]:
    """Scan a single run and return classified rows keyed by their step."""
    run = api.run(f"{entity}/{project}/{run_id}")
    rollout: dict[int, dict[str, Any]] = {}
    train: dict[int, dict[str, Any]] = {}
    eval_data: dict[int, dict[str, Any]] = {}

    for row in run.scan_history(page_size=500):
        cat, step = _classify_row(row)
        cleaned = _strip_internal_keys(row)
        if cat == "rollout" and step is not None:
            rollout.setdefault(step, {}).update(cleaned)
        elif cat == "train" and step is not None:
            train.setdefault(step, {}).update(cleaned)
        elif cat == "eval" and step is not None:
            eval_data.setdefault(step, {}).update(cleaned)

    config = dict(run.config or {})
    logger.info(
        "  %s (%s): rollout=%d train=%d eval=%d steps",
        run_id, run.state, len(rollout), len(train), len(eval_data),
    )
    return rollout, train, eval_data, config


def merge_experiment_data(
    api: wandb.Api, entity: str, project: str, run_ids: list[str],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    """Merge data from multiple runs. Later runs override earlier at same step."""
    merged_rollout: dict[int, dict[str, Any]] = {}
    merged_train: dict[int, dict[str, Any]] = {}
    merged_eval: dict[int, dict[str, Any]] = {}
    merged_config: dict[str, Any] = {}

    for run_id in run_ids:
        rollout, train, eval_data, config = collect_run_data(api, entity, project, run_id)
        for step, data in rollout.items():
            merged_rollout.setdefault(step, {}).update(data)
        for step, data in train.items():
            merged_train.setdefault(step, {}).update(data)
        for step, data in eval_data.items():
            merged_eval.setdefault(step, {}).update(data)
        merged_config.update(config)

    return merged_rollout, merged_train, merged_eval, merged_config


def create_clean_run(
    entity: str,
    target_project: str,
    experiment_name: str,
    description: str,
    rollout_data: dict[int, dict[str, Any]],
    train_data: dict[int, dict[str, Any]],
    eval_data: dict[int, dict[str, Any]],
    config: dict[str, Any],
    source_run_ids: list[str],
) -> str:
    """Create a clean wandb run with properly ordered data."""
    tags = ["consolidated", f"sources:{','.join(source_run_ids)}"]
    config["_consolidation"] = {
        "source_runs": source_run_ids,
        "source_project": SOURCE_PROJECT,
        "description": description,
    }

    run = wandb.init(
        project=target_project,
        entity=entity,
        name=experiment_name,
        config=config,
        tags=tags,
        reinit=True,
        settings=wandb.Settings(mode="shared"),
    )
    _define_wandb_metrics()

    # Log rollout + train data interleaved by step
    # Merge rollout and train at the same step into one log call
    all_steps = sorted(set(list(rollout_data.keys()) + list(train_data.keys())))
    for step in all_steps:
        row = {}
        if step in rollout_data:
            row.update(rollout_data[step])
        if step in train_data:
            row.update(train_data[step])
        if row:
            wandb.log(row)

    # Log eval data separately, ordered by eval/step
    for eval_step in sorted(eval_data.keys()):
        row = eval_data[eval_step]
        row["eval/step"] = eval_step
        wandb.log(row)

    run_url = getattr(run, "url", "")
    run_id = getattr(run, "id", "")
    wandb.finish()
    return run_url or f"{entity}/{target_project}/{run_id}"


def consolidate_experiment(
    api: wandb.Api,
    entity: str,
    target_project: str,
    experiment_name: str,
    experiment_def: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Consolidate one experiment's runs into a single clean run."""
    run_ids = experiment_def["runs"]
    description = experiment_def["description"]

    logger.info("Consolidating experiment: %s", experiment_name)
    logger.info("  Source runs: %s", run_ids)

    rollout, train, eval_data, config = merge_experiment_data(
        api, entity, SOURCE_PROJECT, run_ids,
    )

    logger.info("  Merged: rollout=%d train=%d eval=%d steps", len(rollout), len(train), len(eval_data))
    logger.info("  Rollout step range: %s-%s", min(rollout) if rollout else "N/A", max(rollout) if rollout else "N/A")
    logger.info("  Train step range: %s-%s", min(train) if train else "N/A", max(train) if train else "N/A")
    logger.info("  Eval steps: %s", sorted(eval_data.keys()))

    if dry_run:
        logger.info("  [DRY RUN] Would create run: %s in project %s", experiment_name, target_project)
        return

    url = create_clean_run(
        entity=entity,
        target_project=target_project,
        experiment_name=experiment_name,
        description=description,
        rollout_data=rollout,
        train_data=train,
        eval_data=eval_data,
        config=config,
        source_run_ids=run_ids,
    )
    logger.info("  Created clean run: %s", url)


def main():
    parser = argparse.ArgumentParser(description="Consolidate wandb runs into clean experiments")
    parser.add_argument("--wandb-host", default="https://wandb2.sii.edu.cn")
    parser.add_argument("--wandb-key", default="")
    parser.add_argument("--wandb-entity", default="gzy")
    parser.add_argument("--target-project", default="slime-mdv1-clean",
                        help="Target project for clean runs")
    parser.add_argument("--experiment", default="", help="Only consolidate this experiment")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't create runs")
    args = parser.parse_args()

    if args.wandb_key:
        wandb.login(key=args.wandb_key, host=args.wandb_host)
    else:
        wandb.login(host=args.wandb_host)

    api_kwargs: dict[str, Any] = {"timeout": 60}
    if args.wandb_host:
        api_kwargs["overrides"] = {"base_url": args.wandb_host}
    api = wandb.Api(**api_kwargs)

    entity = args.wandb_entity or getattr(api.viewer, "entity", None)
    if not entity:
        raise RuntimeError("Cannot determine wandb entity. Pass --wandb-entity.")

    experiments = EXPERIMENTS
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            raise RuntimeError(f"Unknown experiment: {args.experiment}. Available: {list(EXPERIMENTS.keys())}")
        experiments = {args.experiment: EXPERIMENTS[args.experiment]}

    for name, exp_def in experiments.items():
        consolidate_experiment(api, entity, args.target_project, name, exp_def, dry_run=args.dry_run)

    logger.info("Done.")


if __name__ == "__main__":
    main()
