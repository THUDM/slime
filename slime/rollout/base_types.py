from dataclasses import dataclass
from typing import Any

from slime.utils.misc import load_function
from slime.utils.types import Sample


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def apply_rollout_sample_filter(args, samples: list[Any]) -> None:
    """Apply `--rollout-sample-filter-path` in place on grouped rollout samples."""
    if args is None or getattr(args, "rollout_sample_filter_path", None) is None:
        return

    filter_func = load_function(args.rollout_sample_filter_path)
    filter_func(args, samples)


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    if not evaluation:
        apply_rollout_sample_filter(args[0] if args else None, output.samples)

    return output
