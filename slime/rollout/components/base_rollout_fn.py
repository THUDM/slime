from dataclasses import dataclass
from typing import Protocol

from slime.utils.types import Sample


@dataclass
class RolloutFnInitParams:
    evaluation: bool
    TODO


@dataclass
class RolloutFnCallParams:
    TODO


@dataclass
class RolloutFnCallOutput:
    samples: list[list[Sample]]


class BaseRolloutFn(Protocol):
    def __init__(self, params: RolloutFnInitParams):
        ...

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        ...
