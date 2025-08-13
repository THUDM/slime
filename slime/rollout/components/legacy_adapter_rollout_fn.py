"""
This file deliberately contains some code duplication, since it is to support the legacy rollout format,
and it is not unified with the main code to avoid making the main code abstraction worse.
"""

from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput
from slime.utils.misc import load_function
from slime.utils.types import Sample


class LegacyAdapterRolloutFn:
    def __init__(self, params: RolloutFnInitParams, original_fn: Callable):
        print("Using legacy format for rollout fn. Please switch to the new format.")

        self.original_fn = original_fn
        self.init_params = params
        self.args = params.args

        # a list of sample group.
        # each group has n_samples_per_prompt samples, all of them has the same prompt.
        self.buffer: list[list[Sample]] = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        raw_output = self.original_fn(
            self.init_params.args,
            params.rollout_id,
            self,
            evaluation=self.init_params.evaluation,
        )

        if self.init_params.evaluation:
            return RolloutFnCallOutput(samples=None, metrics=raw_output)
        else:
            return RolloutFnCallOutput(samples=raw_output, metrics=None)
