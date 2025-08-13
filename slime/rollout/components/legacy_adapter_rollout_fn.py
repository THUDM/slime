from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput


class LegacyAdapterRolloutFn:
    def __init__(self, params: RolloutFnInitParams, original_fn: Callable):
        self.original_fn = original_fn
        self.init_params = params

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        samples = self.original_fn(self.init_params.args, params.rollout_id, TODO, evaluation=self.init_params.evaluation)
        return RolloutFnCallOutput(samples=samples)
