from dataclasses import dataclass
from typing import Protocol


@dataclass
class RolloutFnInitParams:
    evaluation: bool
    TODO


@dataclass
class RolloutFnCallParams:
    TODO


@dataclass
class RolloutFnCallOutput:
    TODO


class BaseRolloutFn(Protocol):
    def __init__(self, params: RolloutFnInitParams):
        ...

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        ...
