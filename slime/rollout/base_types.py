from dataclasses import dataclass
from typing import Any, Dict, Optional

from slime.utils.types import Sample


# TODO may make input dataclass too to allow extensibility
@dataclass
class RolloutFnCallOutput:
    samples: Optional[list[list[Sample]]] = None
    metrics: Optional[Dict[str, Any]] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, RolloutFnCallOutput):
        output = RolloutFnCallOutput(metrics=output) if evaluation else RolloutFnCallOutput(samples=output)

    return output
