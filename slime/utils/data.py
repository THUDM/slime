import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd
import ray
import torch.distributed as dist

from slime.utils.types import MultimodalTypes, Sample

from .seqlen_balancing import get_seqlen_balanced_partitions
from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


# TODO: don't read the whole file into memory.
def read_file(path):
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True, dtype={"label": str})
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:
        logger.info(f"read_file path={path} slice {len(df)=} rows into {row_slice=}")
        df = df.iloc[row_slice]

    for _, row in df.iterrows():
        yield row.to_dict()


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
    if max_length is None:
        return False

    from slime.utils.processing_utils import prepare_model_inputs

    input_ids, _ = prepare_model_inputs(prompt, tokenizer, processor, None, apply_chat_template_kwargs)
    return len(input_ids) > max_length


def _build_messages(data: dict, prompt_key: str, multimodal_keys: dict = None):
    messages = data.get(prompt_key)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, data.get(data_key).tolist())

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in messages:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return messages


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        self.origin_samples = []
        for data in read_file(path):
            prompt = _build_messages(data, prompt_key, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            # TODO: this is slow.
            if _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
                continue

            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
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
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

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

    # save the unprocessed reward for logging (optional for forward-only passes)
    if "raw_reward" in data:
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
        partitions = []
        for dp_rank_groups in group_partitions:
            trajectory_indices = []
            for group_idx in dp_rank_groups:
                # Add all trajectories in this group
                start_idx = group_idx * n_samples_per_prompt
                end_idx = start_idx + n_samples_per_prompt
                trajectory_indices.extend(range(start_idx, end_idx))
            partitions.append(trajectory_indices)

    def get_partition(val):
        if args.balance_data:
            return [val[i] for i in partitions[dp_rank]]
        else:
            return val[dp_rank::dp_size]

    for key in [
        "tokens",
        "multimodal_inputs",
        "total_lengths",
        "response_lengths",
        "rewards",
        "truncated",
        "loss_masks",
        "round_number",
        "sample_indices",
        "rollout_log_probs",
        "rollout_routed_experts",
        "prompt",
        "teacher_log_probs",
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        rollout_data[key] = val

    return rollout_data
