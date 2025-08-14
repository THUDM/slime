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
        self.data_source = params.data_source

        # a list of sample group.
        # each group has n_samples_per_prompt samples, all of them has the same prompt.
        self.buffer: list[list[Sample]] = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        self.rollout_id = params.rollout_id
        raw_output = self.original_fn(
            self.init_params.args,
            params.rollout_id,
            self,
            evaluation=self.init_params.evaluation,
        )
        del self.rollout_id

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

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += self.data_source.get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, self.rollout_id, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    def get_buffer_length(self):
        return len(self.buffer)

    def update_metadata(self, metadata: dict):
        self.data_source.metadata.update(metadata)

    def get_metadata(self):
        return self.data_source.metadata


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
