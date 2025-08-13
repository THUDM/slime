from dataclasses import dataclass
from typing import Protocol


@dataclass
class RolloutFnInitParams:
    pass


@dataclass
class RolloutFnCallParams:
    pass


@dataclass
class RolloutFnCallOutput:
    pass


class BaseRolloutFn(Protocol):
    def __init__(self, params: RolloutFnInitParams):
        ...

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        ...
