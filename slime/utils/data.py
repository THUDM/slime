import json
import random
import numpy as np
import pandas as pd
import ray
import torch.distributed as dist

from slime.utils.types import Sample
from .seqlen_balancing import get_seqlen_balanced_partitions
from .timer import Timer

__all__ = ["Dataset"]


# TODO: don't read the whole file into memory.
def read_file(path):
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, dtype={"label": str})
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")
    for _, row in df.iterrows():
        yield row.to_dict()


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
    ):
        self.origin_samples = []
        for data in read_file(path):
            if multimodal_keys:
                prompt_content = []
                if prompt_key in data:
                    prompt_content.append({"type": "text", "text": data[prompt_key]})
                for media_type, data_key in multimodal_keys.items():
                    if data_key in data:
                        media_path = data[data_key]
                        prompt_content.append({"type": media_type, "path": media_path})
            else:
                prompt_content = data.get(prompt_key)

            if apply_chat_template:
                if tool_key is not None:
                    tools = data[tool_key]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    elif isinstance(tools, np.ndarray):
                        tools = tools.tolist()
                    assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                else:
                    tools = None
                template_input = [{"role": "user", "content": prompt_content}] if multimodal_keys else prompt_content
                prompt = tokenizer.apply_chat_template(
                    template_input, tools, tokenize=False, add_generation_prompt=True
                )

            else:
                prompt = prompt_content

            # TODO: this is slow.
            if max_length is not None:
                if not multimodal_keys:
                    if len(prompt) > max_length:
                        continue

            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=data.get(metadata_key) or {},
                )
            )

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for l in total_lengths:
        for i in range(len(batches)):
            if batches[i] + l <= max_tokens_per_gpu:
                batches[i] += l
                break
        else:
            batches.append(l)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    rollout_data = {}

    rank = dist.get_rank()
    if rank == 0:
        data = ray.get(rollout_data_ref.inner)
        dist.broadcast_object_list([data], src=0)
    else:
        data = [None]
        dist.broadcast_object_list(data, src=0)
        data = data[0]

    # save the unprocessed reward for logging
    rollout_data["raw_reward"] = data["raw_reward"]

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths

    if args.balance_data:
        # Group-aware partitioning to keep each group together
        n_samples_per_prompt = getattr(args, "n_samples_per_prompt", 1)
        # Calculate group-level lengths (sum of lengths for each group)
        num_groups = len(total_lengths) // n_samples_per_prompt
        group_lengths = []
        for i in range(num_groups):
            start_idx = i * n_samples_per_prompt
            end_idx = start_idx + n_samples_per_prompt
            group_total_length = sum(total_lengths[start_idx:end_idx])
            group_lengths.append(group_total_length)

        # Get partitions at group level
        group_partitions = get_seqlen_balanced_partitions(group_lengths, dp_size, equal_size=True)

        # Expand group partitions to trajectory level
        parititions = []
        for dp_rank_groups in group_partitions:
            trajectory_indices = []
            for group_idx in dp_rank_groups:
                # Add all trajectories in this group
                start_idx = group_idx * n_samples_per_prompt
                end_idx = start_idx + n_samples_per_prompt
                trajectory_indices.extend(range(start_idx, end_idx))
            parititions.append(trajectory_indices)

    def get_partition(val):
        if args.balance_data:
            return [val[i] for i in parititions[dp_rank]]
        else:
            return val[dp_rank::dp_size]

    for key in [
        "tokens",
        "total_lengths",
        "response_lengths",
        "rewards",
        "truncated",
        "loss_masks",
        "round_number",
        "sample_indices",
        "rollout_log_probs",
        "prompt",
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        rollout_data[key] = val

    return rollout_data
