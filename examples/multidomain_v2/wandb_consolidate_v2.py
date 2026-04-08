#!/usr/bin/env python3
"""Consolidate multidomain-v2 eval runs into clean experiment runs.

Each experiment has multiple per-checkpoint wandb runs sharing the same run_id.
This script merges them into a single clean EVAL-ONLY run per experiment with:
  - run name = data/setting name (e.g. "toolbench+jsbench+xlam")
  - same group as the original training run
  - ONLY eval metrics logged, using iter number as the x-axis step directly
    (no training data mixed in — avoids x-axis confusion)
  - deduplicated eval rows merged by eval/step

Usage:
    # Dry-run: see what would be consolidated
    python wandb_consolidate_v2.py --dry-run

    # Consolidate all experiments
    python wandb_consolidate_v2.py

    # Consolidate a single experiment
    python wandb_consolidate_v2.py --experiment toolbench+jsbench+xlam

    # Custom target project
    python wandb_consolidate_v2.py --target-project slime-mdv2-clean
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Experiment definitions ────────────────────────────────────────────
# run_id   = the wandb run ID shared across all checkpoint backfill sessions.
# group    = the wandb group from submit_eval_backfill.sh (same as original training).
# run_name = descriptive name based on the actual training data configuration:
#   toolbench+jsbench+xlam : toolbench_v1 + apigen + apibench + agent + jsonschemabench
#                            + nemotron_structured + xlam_60k
#   toolbench+jsbench      : same as above but WITHOUT xlam (the "main" baseline)
#   nk+ns                  : nemotron_knowledge_mcqa + nemotron_structured (stem-only)
#   toolbench+ns           : toolbench_v1 + apigen + apibench + agent + nemotron_structured
#                            (removes jsonschemabench vs toolbench+jsbench)
EXPERIMENTS: dict[str, dict[str, Any]] = {
    # ── Original v2 experiments ───────────────────────────────────────
    "toolbench+jsbench+xlam": {
        "runs": ["xcphh05q"],
        "group": "mdv2-main-plus-xlam-toolcall",
        "description": "toolbench_v1 + apigen + apibench + agent + jsonschemabench + nemotron_structured + xlam_60k",
    },
    "toolbench+jsbench": {
        "runs": ["k2dtse1o"],
        "group": "mdv2-main-retry-0331-2159",
        "description": "toolbench_v1 + apigen + apibench + agent + jsonschemabench + nemotron_structured (no xlam)",
    },
    "nk+ns": {
        "runs": ["m4blb0d9"],
        "group": "mdv2-nk-ns-retry-0331-2242",
        "description": "nemotron_knowledge_mcqa + nemotron_structured_outputs (stem-only mix)",
    },
    "toolbench+ns": {
        "runs": ["22lh8kda"],
        "group": "mdv2-no-jsonschema-toolcall",
        "description": "toolbench_v1 + apigen + apibench + agent + nemotron_structured (no jsonschemabench)",
    },
    # ── r3 reward variant experiments (0401) ─────────────────────────
    # r3 = reward version 3; light-stem adds medmcqa+nemotron_knowledge_mcqa
    "r3+light-stem": {
        "runs": ["bdz3g6oa"],
        "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench+medmcqa+nemotron_knowledge_mcqa",
        "description": "r3 reward; toolbench_v1+apibench+apigen+agent+jsonschemabench+medmcqa+nemotron_knowledge_mcqa (mdv2_r3_light_stem_0401_2130)",
    },
    "r3+main": {
        "runs": ["5yh11qu7"],
        "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench",
        "description": "r3 reward; toolbench_v1+apibench+apigen+agent+jsonschemabench (mdv2_r3_main_0401_2130)",
    },
    "r3+struct-anchor": {
        "runs": ["eio7vlud"],
        "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench+nemotron_structured_outputs",
        "description": "r3 reward; toolbench_v1+apibench+apigen+agent+jsonschemabench+nemotron_structured_outputs (mdv2_r3_struct_anchor_0401_2130)",
    },
    # ── rsb (reward scaling baseline) on H100, struct mix (0403) ─────
    "rsb+h100+struct": {
        "runs": ["oh1av3xh"],
        "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench+nemotron_structured_outputs",
        "description": "rsb reward on H100; toolbench_v1+apibench+apigen+agent+jsonschemabench+nemotron_structured_outputs (mdv2-3node-rsb-h100-0403-1559)",
    },
    # ── tb+jsb variants: default vs rsb reward (0401/0403) ───────────
    "tb+jsb": {
        "runs": ["8igvvv92"],
        "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench",
        "description": "default reward; toolbench_v1+apibench+apigen+agent+jsonschemabench (mdv2-tb-api-apigen-agent-jsb-0401-1356)",
    },
    "tb+jsb+rsb-h100": {
        "runs": ["2l9ml01d"],
        "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench",
        "description": "rsb reward on H100; toolbench_v1+apibench+apigen+agent+jsonschemabench (mdv2-tb-api-apigen-agent-jsb-rsb-h100-0403-1614)",
    },
}

SOURCE_PROJECT = "slime-multidomain-v2"


# ── Helpers ───────────────────────────────────────────────────────────

def _define_wandb_metrics() -> None:
    """Eval-only run: all metrics plot against the iter (= eval/step = wandb step)."""
    # The wandb internal step IS the iter number (we log with step=eval_step).
    # eval/step is redundant but kept as a metric for reference.
    wandb.define_metric("eval/*", overwrite=True)
    wandb.define_metric("eval_by_domain/*", overwrite=True)
    wandb.define_metric("eval_by_source/*", overwrite=True)


def _classify_row(row: dict[str, Any]) -> tuple[str, int | None]:
    """Classify a history row and return (category, step)."""
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


def collect_eval_data(
    api: wandb.Api, entity: str, project: str, run_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Scan a single run and return ONLY eval rows keyed by eval/step."""
    run = api.run(f"{entity}/{project}/{run_id}")
    eval_data: dict[int, dict[str, Any]] = {}

    for row in run.scan_history(page_size=1000):
        cat, step = _classify_row(row)
        if cat == "eval" and step is not None:
            cleaned = _strip_internal_keys(row)
            eval_data.setdefault(step, {}).update(cleaned)

    config = dict(run.config or {})
    logger.info("  %s (%s): eval=%d steps: %s",
                run_id, run.state, len(eval_data), sorted(eval_data.keys()))
    return eval_data, config


def merge_eval_data(
    api: wandb.Api, entity: str, project: str, run_ids: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Merge eval data from multiple runs. Later runs override earlier at same step."""
    merged_eval: dict[int, dict[str, Any]] = {}
    merged_config: dict[str, Any] = {}

    for run_id in run_ids:
        eval_data, config = collect_eval_data(api, entity, project, run_id)
        for step, data in eval_data.items():
            merged_eval.setdefault(step, {}).update(data)
        merged_config.update(config)

    return merged_eval, merged_config


def create_clean_run(
    entity: str,
    target_project: str,
    experiment_name: str,
    group: str,
    description: str,
    eval_data: dict[int, dict[str, Any]],
    config: dict[str, Any],
    source_run_ids: list[str],
) -> str:
    """Create a clean eval-only wandb run.

    Uses eval_step as the wandb step directly so the x-axis shows iter numbers
    (19, 39, ...) without any train/rollout rows mixed in.
    """
    tags = ["consolidated-v2", "eval-only", f"sources:{','.join(source_run_ids)}"]
    config["_consolidation"] = {
        "source_runs": source_run_ids,
        "source_project": SOURCE_PROJECT,
        "description": description,
    }

    run = wandb.init(
        project=target_project,
        entity=entity,
        name=experiment_name,
        group=group,
        config=config,
        tags=tags,
        reinit=True,
    )
    _define_wandb_metrics()

    # Log eval data with step = iter number so x-axis is the iteration.
    # wandb requires step to be monotonically increasing, which holds since
    # eval steps are sorted ascending (19, 39, 59, ...).
    for eval_step in sorted(eval_data.keys()):
        row = {k: v for k, v in eval_data[eval_step].items()
               if not k.startswith("_") and v is not None}
        row["eval/step"] = eval_step
        wandb.log(row, step=eval_step)

    logger.info("  Logged %d eval steps, finishing run...", len(eval_data))
    run_url = getattr(run, "url", "")
    run_id = getattr(run, "id", "")
    wandb.finish(exit_code=0, quiet=False)
    return run_url or f"{entity}/{target_project}/{run_id}"


def consolidate_experiment(
    api: wandb.Api,
    entity: str,
    target_project: str,
    experiment_name: str,
    experiment_def: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Consolidate one experiment's eval data into a single clean run."""
    run_ids = experiment_def["runs"]
    group = experiment_def["group"]
    description = experiment_def["description"]

    logger.info("Consolidating: %s", experiment_name)
    logger.info("  source runs: %s  group: %s", run_ids, group)

    eval_data, config = merge_eval_data(api, entity, SOURCE_PROJECT, run_ids)
    logger.info("  eval steps: %s", sorted(eval_data.keys()))

    if dry_run:
        logger.info("  [DRY RUN] Would create: '%s' (group=%s) → %s", experiment_name, group, target_project)
        return

    url = create_clean_run(
        entity=entity,
        target_project=target_project,
        experiment_name=experiment_name,
        group=group,
        description=description,
        eval_data=eval_data,
        config=config,
        source_run_ids=run_ids,
    )
    logger.info("  Created: %s", url)


def main():
    parser = argparse.ArgumentParser(description="Consolidate multidomain-v2 wandb runs")
    parser.add_argument("--wandb-host", default="https://wandb2.sii.edu.cn")
    parser.add_argument("--wandb-key", default="local-c6d3e5712d547834724d8d98094f340b3c2a869c")
    parser.add_argument("--wandb-entity", default="gzy")
    parser.add_argument("--target-project", default="slime-mdv2-clean",
                        help="Target project for clean runs")
    parser.add_argument("--experiment", default="",
                        help=f"Only consolidate this experiment (leave empty for all). "
                             f"Choices: {list(EXPERIMENTS.keys())}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't create runs")
    args = parser.parse_args()

    wandb.login(key=args.wandb_key, host=args.wandb_host)

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
            raise RuntimeError(
                f"Unknown experiment: {args.experiment}. "
                f"Available: {list(EXPERIMENTS.keys())}"
            )
        experiments = {args.experiment: EXPERIMENTS[args.experiment]}

    for name, exp_def in experiments.items():
        consolidate_experiment(api, entity, args.target_project, name, exp_def, dry_run=args.dry_run)

    logger.info("Done.")


if __name__ == "__main__":
    main()
