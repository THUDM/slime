import copy
import logging
import os
import pickle
from typing import Any, Union

import ray
import torch
from transformers import AutoTokenizer

import wandb
from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


@ray.remote
class Buffer:
    def __init__(self, args):
        self.args = args

        # a list of sample group.
        # each group has n_samples_per_prompt samples, all of them has the same prompt.
        self.buffer: list[list[Sample]] = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

        self.train_data_pool = {}
        self.eval_data_pool = {}
        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.dataset) // self.args.rollout_batch_size

    def update_wandb_run_id(self, run_id):
        """Update wandb run_id and initialize wandb"""
        self.args.wandb_run_id = run_id
        self._init_wandb()  # Now initialize wandb with the correct run_id
        return True

    def _init_wandb(self):
        """Initialize wandb for buffer process if use_wandb is enabled"""
        if not hasattr(self.args, "use_wandb") or not self.args.use_wandb:
            return

        # Check if wandb is already initialized in this process
        if wandb.run is not None:
            print("Wandb already initialized in buffer process")
            return

        # Use the same wandb configuration as main training process
        wandb_config = {
            "entity": getattr(self.args, "wandb_team", None),
            "project": getattr(self.args, "wandb_project", "slime"),
            "group": getattr(self.args, "wandb_group", None),
            "config": self.args.__dict__,
            "reinit": True,  # Allow reinit in same process
        }

        # If wandb_run_id is available, join the existing run
        if hasattr(self.args, "wandb_run_id") and self.args.wandb_run_id:
            wandb_config["id"] = self.args.wandb_run_id
            wandb_config["resume"] = "allow"
            print("=" * 100)
            print(f"Buffer process joining existing wandb run: {self.args.wandb_run_id}")
            print("=" * 100)
        else:
            # Fallback: create a separate run for buffer process
            wandb_config["name"] = f"buffer-{os.getpid()}"
            print("Buffer process creating separate wandb run")

        # Remove None values
        wandb_config = {k: v for k, v in wandb_config.items() if v is not None}

        wandb.init(**wandb_config, settings=wandb.Settings(mode="shared"))

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
            for prompt_sample in prompt_samples:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = copy.deepcopy(prompt_sample)
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)
        else:
            for _ in range(num_samples):
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = Sample(
                        index=self.sample_index,
                    )
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)
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
            data = pickle.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )
            data = [Sample.from_dict(sample) for sample in data]
        else:
            generate_rollout = self.eval_generate_rollout if evaluation else self.generate_rollout
            data = generate_rollout(self.args, rollout_id, self, evaluation=evaluation)
            # flatten the data if it is a list of lists
            if not evaluation and isinstance(data[0], list):
                data = sum(data, [])

        self._set_data(data, evaluation=evaluation)

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
            "rewards": [
                sample.reward if not self.args.reward_key else sample.rewards[self.args.reward_key]
                for sample in samples
            ],
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
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

    def _set_data(self, data: Union[list[Sample], Any], evaluation=False):
        data_pool = self.eval_data_pool if evaluation else self.train_data_pool
        if not evaluation:
            if self.args.save_debug_rollout_data:
                pickle.dump(
                    [sample.to_dict() for sample in data],
                    open(self.args.save_debug_rollout_data.format(rollout_id=self.rollout_id), "wb"),
                )
            data = self._convert_samples_to_train_data(data)
        data_pool[self.rollout_id] = data

    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist.")
            return

        print(f"load metadata from {path}")
        print(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)
