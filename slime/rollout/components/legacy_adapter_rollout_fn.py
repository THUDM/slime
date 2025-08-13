from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput


class LegacyAdapterRolloutFn:
    def __init__(self, params: RolloutFnInitParams, original_fn: Callable):
        TODO

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        return TODO
