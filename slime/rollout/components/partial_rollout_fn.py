from slime.rollout.components.base_rollout_fn import RolloutFnInitParams, RolloutFnCallParams, RolloutFnCallOutput
from slime.utils.misc import load_function
from slime.utils.types import Sample


class PartialRolloutFn:
    def __init__(self, params: RolloutFnInitParams):
        # a list of sample group.
        # each group has n_samples_per_prompt samples, all of them has the same prompt.
        self.aborted_samples_buffer: list[list[Sample]] = []
        if params.args.buffer_filter_path is None:
            self.buffer_filter = _buffer_filter_pop_first
        else:
            self.buffer_filter = load_function(params.args.buffer_filter_path)


    def __call__(self, params: RolloutFnCallParams) -> RolloutFnCallOutput:
        return TODO

def _buffer_filter_pop_first(args, rollout_id, aborted_samples_buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(aborted_samples_buffer), num_samples)
    samples = aborted_samples_buffer[:num_to_pop]
    del aborted_samples_buffer[:num_to_pop]
    return samples
