import logging
from pathlib import Path
from typing import Union
from time import time
import wandb

import ray
import torch

from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.utils.ray_utils import Box
from slime.utils.wandb_utils import init_wandb_secondary

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@ray.remote
class RolloutController:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, wandb_run_id):
        self.args = args
        init_wandb_secondary(args, wandb_run_id)

        self.data_source = RolloutDataSourceWithBuffer(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        self.rollout_id = rollout_id

        start_time = time()
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
        else:
            data = self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False)
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = sum(data, [])

            if len(data) % self.args.global_batch_size != 0:
                trim_len = (len(data) // self.args.global_batch_size) * self.args.global_batch_size
                origin_data_length = len(data)
                data = data[:trim_len]
                print(f"trim number of samples from {origin_data_length} to {trim_len}")

        # TODO to be refactored (originally Buffer._set_data)
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
        log_rollout_data(rollout_id, self.args, data, time() - start_time)
        data = self._convert_samples_to_train_data(data)
        return Box(ray.put(data))

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return

        data = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
        log_eval_data(rollout_id, self.args, data)

    def post_process_rewards(self, samples: Union[list[Sample], list[list[Sample]]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Convert inference generated samples to training data.
        """
        raw_rewards, rewards = self.post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
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

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        return train_data

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)


def log_eval_data(rollout_id, args, data):
    log_dict = {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)

    print(f"eval {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["eval/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)


def log_rollout_data(rollout_id, args, samples, rollout_time):
    if args.load_debug_rollout_data:
        return

    log_dict = {}
    response_lengths = [
        sum(sample.loss_mask) if sample.loss_mask is not None else sample.response_length for sample in samples
    ]
    log_dict["perf/rollout_time"] = rollout_time
    if args.rollout_num_gpus is not None:
        log_dict["perf/tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
    log_dict["perf/longest_sample_tokens_per_sec"] = max(response_lengths) / rollout_time
    print(f"perf {rollout_id}: {log_dict}")
    if args.use_wandb:
        log_dict["rollout/step"] = (
            rollout_id
            if not args.wandb_always_use_train_step
            else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        )
        wandb.log(log_dict)
