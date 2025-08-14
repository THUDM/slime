from dataclasses import dataclass
from typing import Protocol, Any, Optional, TYPE_CHECKING

from slime.utils.types import Sample

if TYPE_CHECKING:
    from slime.ray.rollout_data_source import RolloutDataSource


@dataclass
class RolloutFnInitParams:
    args: Any
    evaluation: bool
    data_source: "RolloutDataSource"


@dataclass
class RolloutFnCallParams:
    rollout_id: int


@dataclass
class RolloutFnCallOutput:
    samples: Optional[list[list[Sample]]]
    metrics: Optional[dict[str, Any]]  # TODO what is the type


class BaseRolloutFn(Protocol):
    def __init__(self, params: RolloutFnInitParams): ...

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput: ...
