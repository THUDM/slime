from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput
from slime.utils.types import Sample


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
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        TODO

    def add_samples(self, samples: list[list[Sample]]):
        TODO

    def update_metadata(self, metadata: dict):
        TODO

    def get_metadata(self):
        TODO

    def get_buffer_length(self):
        TODO
