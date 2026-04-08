#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

SLIME_ROOT = Path(__file__).resolve().parents[3]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from examples.scripts.maintenance.wandb_consolidate_base import (
    collect_run_data,
    create_clean_run,
    init_wandb_api,
    make_argparser,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES: dict[str, dict[str, Any]] = {
    "mopd": {
        "description": "Consolidate wandb runs into clean MOPD/mdv1 experiments",
        "source_project": "slime-multidomain-v1",
        "target_project": "slime-mdv1-clean",
        "data_modes": ("rollout", "train", "eval"),
        "eval_step_as_wandb_step": False,
        "experiments": {
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
        },
    },
    "mdv2": {
        "description": "Consolidate multidomain-v2 eval runs",
        "source_project": "slime-multidomain-v2",
        "target_project": "slime-mdv2-clean",
        "data_modes": ("eval",),
        "eval_step_as_wandb_step": True,
        "experiments": {
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
            "rsb+h100+struct": {
                "runs": ["oh1av3xh"],
                "group": "toolbench_v1+apibench+apigen+agent+jsonschemabench+nemotron_structured_outputs",
                "description": "rsb reward on H100; toolbench_v1+apibench+apigen+agent+jsonschemabench+nemotron_structured_outputs (mdv2-3node-rsb-h100-0403-1559)",
            },
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
        },
    },
}


def _resolve_mode(mode_name: str) -> dict[str, Any]:
    try:
        return MODES[mode_name]
    except KeyError as exc:
        raise RuntimeError(f"Unknown consolidation mode '{mode_name}'. Supported modes: {', '.join(sorted(MODES))}") from exc


def consolidate_experiment(
    *,
    api,
    entity: str,
    mode_name: str,
    target_project: str,
    experiment_name: str,
    experiment_def: dict[str, Any],
    dry_run: bool,
) -> None:
    mode = _resolve_mode(mode_name)
    logger.info("Consolidating %s/%s", mode_name, experiment_name)
    logger.info("  Source runs: %s", experiment_def["runs"])

    data_by_mode, config = collect_run_data(
        api,
        experiment_def["runs"],
        entity=entity,
        project=mode["source_project"],
        modes=mode["data_modes"],
    )

    for data_mode in mode["data_modes"]:
        logger.info("  %s=%d steps", data_mode, len(data_by_mode[data_mode]))

    if dry_run:
        logger.info("  [DRY RUN] Would create '%s' in %s", experiment_name, target_project)
        return

    config["_consolidation"] = {
        "mode": mode_name,
        "source_runs": experiment_def["runs"],
        "source_project": mode["source_project"],
        "description": experiment_def["description"],
    }
    tags = [f"consolidated-{mode_name}", f"sources:{','.join(experiment_def['runs'])}"]
    if mode_name == "mdv2":
        tags.append("eval-only")

    url = create_clean_run(
        project=target_project,
        entity=entity,
        name=experiment_name,
        group=experiment_def.get("group"),
        config=config,
        tags=tags,
        data_by_mode=data_by_mode,
        eval_step_as_wandb_step=mode["eval_step_as_wandb_step"],
    )
    logger.info("  Created: %s", url)


def main(argv: list[str] | None = None, *, default_mode: str | None = None) -> int:
    parser = make_argparser("Consolidate wandb experiments")
    if default_mode is None:
        parser.add_argument("--mode", default="mopd", choices=sorted(MODES))
    args = parser.parse_args(argv)
    if default_mode is not None:
        args.mode = default_mode

    mode = _resolve_mode(args.mode)
    if not args.target_project:
        args.target_project = mode["target_project"]

    api, entity = init_wandb_api(args)
    experiments = mode["experiments"]
    if args.experiment:
        if args.experiment not in experiments:
            raise RuntimeError(f"Unknown experiment: {args.experiment}. Available: {list(experiments)}")
        experiments = {args.experiment: experiments[args.experiment]}

    for name, experiment_def in experiments.items():
        consolidate_experiment(
            api=api,
            entity=entity,
            mode_name=args.mode,
            target_project=args.target_project,
            experiment_name=name,
            experiment_def=experiment_def,
            dry_run=args.dry_run,
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
