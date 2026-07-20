import logging

import wandb

from . import mlflow_utils, wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        mlflow_utils.init_mlflow_primary(args)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
        mlflow_utils.init_mlflow_secondary(args)


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not args.use_wandb:
        pass
    else:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            logging.getLogger(__name__).exception("Failed to finish wandb run")

    mlflow_utils.finish_mlflow()


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_mlflow:
        step = metrics.get(step_key, 0)
        mlflow_utils.log_metrics(metrics, step=int(step))

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
