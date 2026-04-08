from __future__ import annotations

import argparse
import logging
from typing import Any, Iterable

import wandb

logger = logging.getLogger(__name__)


def classify_row(row: dict[str, Any]) -> tuple[str, int | None, dict[str, Any]]:
    cleaned = strip_internal_keys(row)

    eval_step = cleaned.get("eval/step")
    if eval_step is not None:
        has_eval = any(k.startswith("eval/") and k != "eval/step" for k in cleaned) or any(
            k.startswith("eval_by_") for k in cleaned
        )
        if has_eval:
            return "eval", int(eval_step), cleaned

    train_step = cleaned.get("train/step")
    if train_step is not None:
        has_train = any(k.startswith("train/") and k != "train/step" for k in cleaned)
        if has_train:
            return "train", int(train_step), cleaned

    rollout_step = cleaned.get("rollout/step")
    if rollout_step is not None:
        return "rollout", int(rollout_step), cleaned

    return "skip", None, cleaned


def strip_internal_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_") and v is not None}


def define_wandb_metrics(modes: Iterable[str], *, eval_step_as_wandb_step: bool = False) -> None:
    enabled = set(modes)
    if "train" in enabled:
        wandb.define_metric("train/step", overwrite=True)
        wandb.define_metric("train/*", step_metric="train/step", overwrite=True)
    if "rollout" in enabled:
        wandb.define_metric("rollout/step", overwrite=True)
        wandb.define_metric("rollout/*", step_metric="rollout/step", overwrite=True)
        wandb.define_metric("rollout_by_domain/*", step_metric="rollout/step", overwrite=True)
        wandb.define_metric("rollout_by_source/*", step_metric="rollout/step", overwrite=True)
        wandb.define_metric("perf/*", step_metric="rollout/step", overwrite=True)
    if "eval" in enabled:
        if not eval_step_as_wandb_step:
            wandb.define_metric("eval/step", overwrite=True)
            wandb.define_metric("eval/*", step_metric="eval/step", overwrite=True)
            wandb.define_metric("eval_by_domain/*", step_metric="eval/step", overwrite=True)
            wandb.define_metric("eval_by_source/*", step_metric="eval/step", overwrite=True)
        else:
            wandb.define_metric("eval/*", overwrite=True)
            wandb.define_metric("eval_by_domain/*", overwrite=True)
            wandb.define_metric("eval_by_source/*", overwrite=True)


def collect_run_data(
    api: wandb.Api,
    run_ids: Iterable[str],
    *,
    entity: str,
    project: str,
    modes: Iterable[str],
    page_size: int = 1000,
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    enabled = set(modes)
    merged = {mode: {} for mode in enabled}
    merged_config: dict[str, Any] = {}

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        per_run = {mode: {} for mode in enabled}
        for row in run.scan_history(page_size=page_size):
            category, step, cleaned = classify_row(row)
            if category not in enabled or step is None:
                continue
            per_run[category].setdefault(step, {}).update(cleaned)

        merged_config.update(dict(run.config or {}))
        for mode in enabled:
            for step, payload in per_run[mode].items():
                merged[mode].setdefault(step, {}).update(payload)

        logger.info(
            "  %s (%s): %s",
            run_id,
            run.state,
            " ".join(f"{mode}={len(per_run[mode])}" for mode in sorted(enabled)),
        )

    return merged, merged_config


def create_clean_run(
    *,
    project: str,
    entity: str,
    name: str,
    config: dict[str, Any],
    tags: list[str],
    data_by_mode: dict[str, dict[int, dict[str, Any]]],
    group: str | None = None,
    eval_step_as_wandb_step: bool = False,
) -> str:
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        config=config,
        tags=tags,
        reinit=True,
        settings=wandb.Settings(mode="shared"),
    )
    define_wandb_metrics(data_by_mode, eval_step_as_wandb_step=eval_step_as_wandb_step)

    if eval_step_as_wandb_step and set(data_by_mode) == {"eval"}:
        for eval_step in sorted(data_by_mode["eval"]):
            row = dict(data_by_mode["eval"][eval_step])
            row["eval/step"] = eval_step
            wandb.log(row, step=eval_step)
    else:
        combined_steps = sorted(set(data_by_mode.get("rollout", {})) | set(data_by_mode.get("train", {})))
        for step in combined_steps:
            row: dict[str, Any] = {}
            if step in data_by_mode.get("rollout", {}):
                row.update(data_by_mode["rollout"][step])
            if step in data_by_mode.get("train", {}):
                row.update(data_by_mode["train"][step])
            if row:
                wandb.log(row)

        for eval_step in sorted(data_by_mode.get("eval", {})):
            row = dict(data_by_mode["eval"][eval_step])
            row["eval/step"] = eval_step
            wandb.log(row)

    run_url = getattr(run, "url", "")
    run_id = getattr(run, "id", "")
    wandb.finish(exit_code=0, quiet=False)
    return run_url or f"{entity}/{project}/{run_id}"


def make_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--wandb-host", default="https://wandb2.sii.edu.cn")
    parser.add_argument("--wandb-key", default="")
    parser.add_argument("--wandb-entity", default="gzy")
    parser.add_argument("--target-project", default="")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def init_wandb_api(args: argparse.Namespace) -> tuple[wandb.Api, str]:
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
    return api, entity
