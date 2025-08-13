from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput


class LegacyAdapterRolloutFn:
    def __init__(self, params: RolloutFnInitParams, original_fn: Callable):
        print("Using legacy format for rollout fn. Please switch to the new format.")
        self.original_fn = original_fn
        self.init_params = params

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        raw_output = self.original_fn(
            self.init_params.args,
            params.rollout_id,
            self.init_params.buffer,
            evaluation=self.init_params.evaluation,
        )

        if self.init_params.evaluation:
            return RolloutFnCallOutput(samples=None, metrics=raw_output)
        else:
            return RolloutFnCallOutput(samples=raw_output, metrics=None)
