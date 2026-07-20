"""MLflow tracking utilities for SLIME RL training."""

import logging
import numbers
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

_mlflow_initialized = False
_mlflow_is_primary = False


def _sanitize_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def _flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)[:500]))
    return dict(items)


def init_mlflow_primary(args):
    global _mlflow_initialized, _mlflow_is_primary

    if not args.use_mlflow:
        logger.info("MLflow disabled (--use-mlflow not set), skipping initialization")
        return

    import mlflow

    tracking_uri = args.mlflow_tracking_uri
    logger.info(f"MLflow primary init: connecting to tracking server at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = args.mlflow_experiment_name or "default"
    logger.info(f"MLflow primary init: setting experiment={experiment_name}")
    experiment = mlflow.set_experiment(experiment_name)

    run_name = args.mlflow_run_name
    logger.info(f"MLflow primary init: starting run={run_name} " f"(experiment_id={experiment.experiment_id})")

    mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=run_name,
        log_system_metrics=True,
    )

    try:
        raw = deepcopy(args.__dict__)
        allowed_env_vars = ["SLURM_JOB_ID"]
        raw["env_vars"] = {k: v for k, v in os.environ.items() if k in allowed_env_vars}
        params = _flatten_dict(raw)
        logger.info(f"MLflow primary init: logging {len(params)} params")
        mlflow.log_params(params)
    except Exception as e:
        logger.warning(f"Failed to log params to MLflow: {e}")

    args.mlflow_run_id = mlflow.active_run().info.run_id

    _mlflow_initialized = True
    _mlflow_is_primary = True
    logger.info(
        f"MLflow primary init complete: experiment={experiment_name}, " f"run={run_name}, run_id={args.mlflow_run_id}"
    )

    import atexit

    atexit.register(finish_mlflow)


def init_mlflow_secondary(args):
    global _mlflow_initialized

    if not getattr(args, "use_mlflow", False):
        return

    mlflow_run_id = getattr(args, "mlflow_run_id", None)
    if mlflow_run_id is None:
        return

    import mlflow

    tracking_uri = args.mlflow_tracking_uri
    logger.info(f"MLflow secondary init: connecting to {tracking_uri}, " f"joining run_id={mlflow_run_id}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.start_run(run_id=mlflow_run_id, log_system_metrics=False)

    _mlflow_initialized = True
    logger.info(f"MLflow secondary init complete: joined run_id={mlflow_run_id}")


def log_metrics(metrics: dict, step: int):
    global _mlflow_initialized
    if not _mlflow_initialized:
        return

    import mlflow

    sanitized = {}
    for k, v in metrics.items():
        if isinstance(v, numbers.Number) and not isinstance(v, bool):
            sanitized[_sanitize_metric_name(k)] = float(v)

    if sanitized:
        try:
            mlflow.log_metrics(metrics=sanitized, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")


def finish_mlflow():
    global _mlflow_initialized, _mlflow_is_primary
    if not _mlflow_initialized:
        return

    if _mlflow_is_primary:
        import mlflow

        logger.info("MLflow finishing: ending primary run")
        mlflow.end_run()
        logger.info("MLflow run ended successfully")

    _mlflow_initialized = False
    _mlflow_is_primary = False
