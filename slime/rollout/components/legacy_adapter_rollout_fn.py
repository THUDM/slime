from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput


class LegacyAdapterRolloutFn:
    def __init__(self, params: RolloutFnInitParams, original_fn: Callable):
        print("Using legacy format for rollout fn.")
        self.original_fn = original_fn
        self.init_params = params
        self._legacy_data_buffer_adapter = _LegacyDataBufferAdapter()

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        raw_output = self.original_fn(
            self.init_params.args,
            params.rollout_id,
            self._legacy_data_buffer_adapter,
            evaluation=self.init_params.evaluation,
        )

        if self.init_params.evaluation:
            return RolloutFnCallOutput(samples=None, metrics=raw_output)
        else:
            return RolloutFnCallOutput(samples=raw_output, metrics=None)

class _LegacyDataBufferAdapter:
    TODO
