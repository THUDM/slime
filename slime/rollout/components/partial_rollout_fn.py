from functools import partial
from typing import Callable

from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput
from slime.utils.misc import load_function
from slime.utils.types import Sample


class PartialRolloutFn:
    """A rollout fn to support partial rollout.

    The user only needs to provide `generate_one_step` to do arbitrary generation for one step (the
    "rollout worker" green box in paper), and this class maintains an aborted_samples_buffer (the
    "Replay Buffer" red box in paper) for partial rollout.

    For more details, please visit https://arxiv.org/abs/2501.12599
    """

    def __init__(
        self,
        params: RolloutFnInitParams,
        generate_one_step: Callable,
    ):
        self.args = params.args
        self.data_source = params.data_source
        self.generate_one_step = generate_one_step

        # a list of sample group.
        # each group has n_samples_per_prompt samples, all of them has the same prompt.
        self.aborted_samples_buffer: list[list[Sample]] = []

        if (p := self.args.buffer_filter_path) is not None:
            self.buffer_filter = load_function(p)
        else:
            self.buffer_filter = _buffer_filter_pop_first

    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        output, aborted_samples = self.generate_one_step(
            params=params,
            get_samples=partial(self._get_samples, rollout_id=params.rollout_id),
        )
        self._add_samples_to_buffer(aborted_samples)
        return output

    # TODO simplify
    def _get_samples(self, num_samples: int, rollout_id: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples, rollout_id=rollout_id)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += self._get_samples_from_data_source(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int, rollout_id: int) -> list[list[Sample]]:
        if len(self.aborted_samples_buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, rollout_id, self.aborted_samples_buffer, num_samples)
        return samples

    def _get_samples_from_data_source(self, num_samples: int) -> list[list[Sample]]:
        return self.data_source.get_samples(num_samples=num_samples)

    def _add_samples_to_buffer(self, samples: list[list[Sample]]):
        if not samples:
            return

        # TODO improve code, e.g. separate assertion and addition
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.aborted_samples_buffer.append(group)


def _buffer_filter_pop_first(
    args, rollout_id, aborted_samples_buffer: list[list[Sample]], num_samples: int
) -> list[list[Sample]]:
    num_to_pop = min(len(aborted_samples_buffer), num_samples)
    samples = aborted_samples_buffer[:num_to_pop]
    del aborted_samples_buffer[:num_to_pop]
    return samples
