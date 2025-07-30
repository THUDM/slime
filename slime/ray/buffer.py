import logging
from pathlib import Path
from typing import Union

import ray
import torch

from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.ray.rollout_data_source import RolloutDataSource
from slime.utils.ray_utils import Box
from slime.utils.wandb_utils import init_wandb_secondary

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


@ray.remote
class Buffer:
    def __init__(self, args, wandb_run_id):
        self.args = args
        init_wandb_secondary(args, wandb_run_id)

        self.data_source = RolloutDataSource(args)

        # a list of sample group.
        # each group has n_samples_per_prompt samples, all of them has the same prompt.
        self.buffer: list[list[Sample]] = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    # TODO simplify remaining logic
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

        assert len(samples) % self.args.n_samples_per_prompt == 0
        for i in range(0, len(samples), self.args.n_samples_per_prompt):
            group = samples[i : i + self.args.n_samples_per_prompt]
            self.buffer.append(group)

    def generate(self, rollout_id, evaluation=False):
        self.rollout_id = rollout_id
        if self.args.debug_train_only and evaluation:
            # if debug train only, we don't generate evaluation data
            return

        if not evaluation and self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            generate_rollout = self.eval_generate_rollout if evaluation else self.generate_rollout
            data = generate_rollout(self.args, rollout_id, self, evaluation=evaluation)
            # flatten the data if it is a list of lists
            if not evaluation and isinstance(data[0], list):
                data = sum(data, [])

        # TODO to be refactored (originally Buffer._set_data)
        if not evaluation:
            # TODO extract to a function during refactor
            if (path_template := self.args.save_debug_rollout_data) is not None:
                path = Path(path_template.format(rollout_id=self.rollout_id))
                print(f"Save debug rollout data to {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    dict(
                        rollout_id=self.rollout_id,
                        samples=[sample.to_dict() for sample in data],
                    ),
                    path,
                )
            data = self._convert_samples_to_train_data(data)

        return Box(ray.put(data))

    def get_data(self, rollout_id, evaluation=False):
        data_pool = self.train_data_pool if not evaluation else self.eval_data_pool
        assert rollout_id in data_pool
        data = data_pool[rollout_id]
        del data_pool[rollout_id]
        return data

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Convert inference generated samples to training data.
        """
        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": [sample.get_reward_value(self.args) for sample in samples],
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]
        return train_data

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.data_source.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.data_source.metadata

    def get_buffer_length(self):
        return len(self.buffer)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)
