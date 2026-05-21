import logging
import os
from copy import deepcopy
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_INITIALIZED = False


def _import_trackio():
    try:
        import trackio
    except ImportError as exc:
        raise ImportError("Trackio logging requires the `trackio` package. Install it with `pip install trackio` or remove `--use-trackio`.") from exc
    return trackio


def _trackio_project(args) -> str:
    return args.trackio_project or args.wandb_project or "slime"


def _trackio_run_name(args) -> str:
    return args.trackio_run_name or args.wandb_group or f"slime-rank-{getattr(args, 'rank', 0)}"


def init_trackio(args, primary: bool = True):
    global _INITIALIZED
    if not getattr(args, "use_trackio", False):
        return
    if _INITIALIZED:
        return

    trackio = _import_trackio()
    trackio.init(
        project=_trackio_project(args),
        name=_trackio_run_name(args),
        config=_compute_config_for_logging(args),
    )
    _INITIALIZED = True


def finish_trackio(args):
    global _INITIALIZED
    if not getattr(args, "use_trackio", False):
        return
    trackio = _import_trackio()
    try:
        trackio.finish()
        _INITIALIZED = False
    except Exception:
        logger.exception("Failed to finish Trackio run")


def log_metrics(args, metrics: dict[str, Any], step: int | None = None):
    if not getattr(args, "use_trackio", False):
        return
    init_trackio(args)
    trackio = _import_trackio()
    trackio.log(metrics, step=step)


def log_rollout_traces(args, rollout_id: int, samples: list[Sample], *, split: str, step: int):
    if not getattr(args, "use_trackio", False):
        return
    init_trackio(args)

    max_traces = getattr(args, "trackio_max_traces_per_rollout", 32)
    if max_traces is not None and max_traces > 0:
        samples = samples[:max_traces]

    traces = [_sample_to_trackio_trace(args, rollout_id, sample, split=split, step=step) for sample in samples]
    traces = [trace for trace in traces if trace is not None]
    if not traces:
        return

    trackio = _import_trackio()
    trackio.log({f"{split}/trajectories": traces}, step=step)


def _sample_to_trackio_trace(args, rollout_id: int, sample: Sample, *, split: str, step: int):
    trackio = _import_trackio()
    messages = _sample_messages(sample)
    if not messages:
        return None
    return trackio.Trace(
        messages=messages,
        metadata={
            "split": split,
            "rollout_id": rollout_id,
            "step": step,
            "sample_index": sample.index,
            "group_index": sample.group_index,
            "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
            "reward": sample.reward,
            "response_length": sample.response_length,
            "effective_response_length": sample.effective_response_length,
            "metadata": sample.metadata,
            "trace": getattr(sample, "trace", None),
            "trackio_max_traces_per_rollout": getattr(args, "trackio_max_traces_per_rollout", None),
        },
    )


def _sample_messages(sample: Sample) -> list[dict[str, Any]]:
    messages = _prompt_messages(sample.prompt)
    if sample.response:
        messages.append({"role": "assistant", "content": sample.response})
    return messages


def _prompt_messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return [dict(message) for message in prompt if isinstance(message, dict)]
    return [{"role": "user", "content": str(prompt)}]


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    whitelist_env_vars = [
        "SLURM_JOB_ID",
    ]
    output["env_vars"] = {key: value for key, value in os.environ.items() if key in whitelist_env_vars}
    return output
