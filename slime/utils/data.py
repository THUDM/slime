import itertools
import json
import logging
import os
import random
import re

import numpy as np
import ray

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _should_skip_prompt(formatted_prompt: str, tokenizer, processor, max_length, multimodal_inputs=None):
    if max_length is None:
        return False

    if processor:
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        input_ids = processor_output["input_ids"][0]
    else:
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    return len(input_ids) > max_length


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        assert as_conversation, "as_conversation must be True when multimodal_keys is not None"
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
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

    return prompt


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
            as_conversation = apply_chat_template
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                formatted_prompt = prompt

            if processor:
                from slime.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            # TODO: this is slow.
            if _should_skip_prompt(formatted_prompt, tokenizer, processor, max_length, multimodal_inputs):
                continue

            self.origin_samples.append(
                Sample(
                    prompt=formatted_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
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
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
