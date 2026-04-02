import itertools
import json
import logging
import os
import random
import re
from hashlib import sha1
from pathlib import PurePosixPath

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


MATH_SYSTEM_PROMPTS = [
    "Solve the following math problem carefully. Show your reasoning, and put the final answer in \\boxed{} on the last line.",
    "Work through the following mathematics problem step by step. Make the derivation clear and place the final answer in \\boxed{} on the last line.",
]

CODE_STDIN_SYSTEM_PROMPTS = [
    "Solve the following programming problem in Python 3. Return only one Python code block. Your program must read from standard input and write to standard output.",
    "Write a correct Python 3 solution for the following programming problem. Return a single Python code block, read from standard input, and write to standard output.",
]

CODE_FUNCTION_SYSTEM_PROMPTS = [
    "Solve the following programming task in Python 3. Return one Python code block containing a complete implementation.",
    "Write a complete Python 3 implementation for the following task. Return exactly one Python code block.",
]

STEM_SYSTEM_PROMPTS = [
    "You are a careful STEM reasoning assistant. Solve the problem step by step. If options are provided, end with the best option and a concise justification.",
    "You are an expert science and quantitative reasoning assistant. Reason carefully, avoid unsupported assumptions, and give the final answer clearly. If this is multiple choice, state the correct option explicitly.",
]

TOOL_SYSTEM_PROMPTS = [
    "You are a tool-using assistant. Use the provided tools precisely when they are useful, keep tool calls valid, and ground the final answer in the observed results.",
    "You are an assistant with access to external tools. Plan briefly, call tools only when needed, and produce a final answer that is consistent with the tool outputs.",
]

STRUCTURED_SYSTEM_PROMPTS = [
    "You are a structured output assistant. Produce an answer that exactly matches the requested schema or structure. Do not add extra keys or explanatory text.",
    "You are an information extraction assistant. Return only content that satisfies the requested structure, using valid JSON when structured output is required.",
]

IFRL_SYSTEM_PROMPTS = [
    "You are a helpful assistant. Follow the user's instruction carefully and answer clearly.",
    "You are a careful assistant. Respond directly to the user's request and keep the answer accurate and concise.",
]


def _stable_pick(options: list[str], payload: dict, domain: str, suffix: str = "") -> str:
    if len(options) == 1:
        return options[0]

    metadata = payload.get("metadata") or {}
    key_parts = [
        domain,
        suffix,
        str(metadata.get("dataset_name") or ""),
        str(metadata.get("record_id") or payload.get("id") or ""),
        str(payload.get("prompt_mode") or ""),
        str(payload.get("question") or payload.get("prompt") or ""),
    ]
    digest = sha1("||".join(key_parts).encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _make_messages(system_prompt: str, user_content: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content.strip()},
    ]


def _prepend_system_if_missing(prompt, system_prompt: str):
    if isinstance(prompt, str):
        return _make_messages(system_prompt, prompt)
    if not isinstance(prompt, list):
        return prompt
    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
        return prompt
    return [{"role": "system", "content": system_prompt.strip()}, *prompt]


def _build_code_prompt(payload: dict) -> list[dict]:
    prompt_mode = str(payload.get("prompt_mode") or "").strip()
    question = str(payload.get("question") or "").strip()
    if not prompt_mode or not question:
        return payload.get("prompt")

    extra_notes = [str(item).strip() for item in (payload.get("extra_notes") or []) if str(item).strip()]
    starter_code = str(payload.get("starter_code") or "").strip()
    fn_name = str(payload.get("fn_name") or "").strip()
    signature_hint = str(payload.get("signature_hint") or "").strip()

    if prompt_mode == "stdin":
        system_prompt = _stable_pick(CODE_STDIN_SYSTEM_PROMPTS, payload, "code", suffix="stdin")
        sections = []
        if extra_notes:
            sections.append("\n".join(extra_notes))
        sections.append(f"Problem:\n{question}")
        if starter_code:
            sections.append(f"Starter code:\n```python\n{starter_code}\n```")
    elif prompt_mode in {"call", "function", "function_test"}:
        system_prompt = _stable_pick(CODE_FUNCTION_SYSTEM_PROMPTS, payload, "code", suffix=prompt_mode)
        sections = []
        if fn_name:
            sections.append(f"Your solution must define a callable named `{fn_name}` with the required signature.")
        if extra_notes:
            sections.append("\n".join(extra_notes))
        sections.append(f"Task:\n{question}")
        if signature_hint:
            sections.append(f"Required signature / scaffold:\n```python\n{signature_hint}\n```")
        if starter_code:
            sections.append(f"Starter code:\n```python\n{starter_code}\n```")
    else:
        return payload.get("prompt")

    return _make_messages(system_prompt, "\n\n".join(sections))


def _apply_pool_prompt_template(payload: dict, domain: str):
    normalized = dict(payload)

    if domain == "code":
        prompt = _build_code_prompt(normalized)
        if prompt is not None:
            normalized["prompt"] = prompt
        return normalized

    if domain == "math":
        question = str(normalized.get("question") or "").strip()
        if question:
            normalized["prompt"] = _make_messages(_stable_pick(MATH_SYSTEM_PROMPTS, normalized, "math"), question)
        return normalized

    if domain == "stem":
        normalized["prompt"] = _prepend_system_if_missing(
            normalized.get("prompt"),
            _stable_pick(STEM_SYSTEM_PROMPTS, normalized, "stem"),
        )
        return normalized

    if domain == "tool":
        normalized["prompt"] = _prepend_system_if_missing(
            normalized.get("prompt"),
            _stable_pick(TOOL_SYSTEM_PROMPTS, normalized, "tool"),
        )
        return normalized

    if domain == "structured":
        normalized["prompt"] = _prepend_system_if_missing(
            normalized.get("prompt"),
            _stable_pick(STRUCTURED_SYSTEM_PROMPTS, normalized, "structured"),
        )
        return normalized

    if domain == "ifrl":
        normalized["prompt"] = _prepend_system_if_missing(
            normalized.get("prompt"),
            _stable_pick(IFRL_SYSTEM_PROMPTS, normalized, "ifrl"),
        )
        return normalized

    return normalized


def _normalize_pool_row(path: str, payload: dict):
    posix_path = PurePosixPath(path.replace("\\", "/"))
    parts = posix_path.parts
    if "pool" not in parts:
        return payload

    try:
        pool_idx = parts.index("pool")
        domain = parts[pool_idx + 1]
    except (ValueError, IndexError):
        return payload

    normalized = dict(payload)
    metadata = dict(normalized.get("metadata") or {})
    metadata.setdefault("domain", domain)
    normalized["metadata"] = metadata
    return _apply_pool_prompt_template(normalized, domain)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".list"):

        def list_reader(p):
            missing_count = 0
            with open(p, encoding="utf-8") as f:
                for line in f:
                    source_path = line.strip()
                    if not source_path or source_path.startswith("#"):
                        continue
                    if not os.path.exists(source_path):
                        missing_count += 1
                        logger.warning("Skipping missing prompt dataset path from list: %s", source_path)
                        continue
                    yield from read_file(source_path)
            if missing_count:
                logger.warning("Skipped %d missing prompt dataset paths listed in %s", missing_count, p)

        reader = list_reader(path)

    elif path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield _normalize_pool_row(p, json.loads(line))
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
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl, .parquet, and .list.")

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


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        if multimodal:
            from slime.utils.processing_utils import process_vision_info

            for sample in multimodal:
                multimodal_inputs = process_vision_info(sample.prompt, processor)
                processor_output = processor(text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
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

        for placeholder, (mt, remaining) in multimodals.items():
            assert len(remaining) == 0, (
                f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                f"than '{placeholder}' placeholders in prompt"
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
        origin_samples = []
        for data in read_file(path):
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
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
                output_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor:
                from slime.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

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
