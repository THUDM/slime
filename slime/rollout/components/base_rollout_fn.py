from dataclasses import dataclass
from typing import Protocol, Any

from slime.utils.types import Sample


@dataclass
class RolloutFnInitParams:
    args: Any
    evaluation: bool


@dataclass
class RolloutFnCallParams:
    rollout_id: int


@dataclass
class RolloutFnCallOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any]  # TODO what is the type


class BaseRolloutFn(Protocol):
    def __init__(self, params: RolloutFnInitParams):
        ...

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        ...
