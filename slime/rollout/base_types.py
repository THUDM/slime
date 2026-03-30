from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample


@dataclass
class RolloutFnTrainOutput:
    samples: list[Sample]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None
