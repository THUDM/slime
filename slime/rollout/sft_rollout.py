import logging
import json
import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import build_processor_kwargs, load_processor, load_tokenizer
from slime.utils.types import Sample

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


class OverlongSFTSampleError(ValueError):
    def __init__(self, length: int, max_context_len: int):
        super().__init__(f"SFT sample length {length} exceeds --rollout-max-context-len {max_context_len}.")
        self.length = length
        self.max_context_len = max_context_len


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _to_list_ids(input_ids: Any) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        return input_ids.tolist()
    if isinstance(input_ids, np.ndarray):
        return input_ids.tolist()
    return list(input_ids)


def _first_sequence_ids(input_ids: Any) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        return _to_list_ids(input_ids[0] if input_ids.ndim > 1 else input_ids)
    if isinstance(input_ids, np.ndarray):
        return _to_list_ids(input_ids[0] if input_ids.ndim > 1 else input_ids)
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return _to_list_ids(input_ids[0])
    return _to_list_ids(input_ids)


def _to_tensor_or_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _messages_from_sample(sample: Sample) -> list[dict[str, Any]]:
    prompt = deepcopy(sample.prompt)

    if isinstance(prompt, str):
        if sample.label is None:
            raise ValueError("SFT sample has a string prompt but no label to form the assistant response.")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sample.label},
        ]
    elif isinstance(prompt, list):
        messages = prompt
        has_assistant = any(message.get("role") == "assistant" for message in messages)
        if not has_assistant:
            if sample.label is None:
                raise ValueError("SFT sample has no assistant turn and no label.")
            messages = messages + [{"role": "assistant", "content": sample.label}]
    else:
        raise TypeError(f"Unsupported SFT prompt type: {type(prompt)}")

    return _normalize_tool_call_arguments(messages)


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI tool-call arguments to the mapping expected by Qwen templates."""
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed_arguments, dict):
                    function["arguments"] = parsed_arguments
    return messages


def _render_messages(messages: list[dict[str, Any]], tools: list[dict] | None) -> str:
    return TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        return_dict=False,
    )


def _build_sft_sample(args, sample: Sample) -> Sample:
    messages = _messages_from_sample(sample)
    tools = sample.metadata.get("tools", None) if sample.metadata else None

    if args.multimodal_keys is not None:
        if PROCESSOR is None:
            raise RuntimeError("--multimodal-keys is set, but no HuggingFace processor was loaded.")

        rendered_text = _render_messages(messages, tools)
        processor_output = PROCESSOR(
            text=[rendered_text],
            **build_processor_kwargs(sample.multimodal_inputs or {}),
        )
        input_ids = _first_sequence_ids(processor_output["input_ids"])
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask_with_multimodal_alignment(
            messages,
            input_ids,
            tools=tools,
            rendered_text=rendered_text,
            image_grid_thw=processor_output.get("image_grid_thw"),
            image_token=getattr(PROCESSOR, "image_token", "<|image_pad|>"),
            image_merge_size=getattr(PROCESSOR.image_processor, "merge_size", 2),
        )
        sample.multimodal_train_inputs = {
            key: _to_tensor_or_value(value)
            for key, value in processor_output.items()
            if key not in ("input_ids", "attention_mask")
        } or None
    else:
        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
        sample.multimodal_train_inputs = None

    if len(token_ids) != len(loss_mask):
        raise ValueError(
            f"SFT rollout produced mismatched token_ids/loss_mask lengths: {len(token_ids)=}, {len(loss_mask)=}"
        )

    max_context_len = getattr(args, "rollout_max_context_len", None)
    if max_context_len is not None and len(token_ids) > max_context_len:
        raise OverlongSFTSampleError(len(token_ids), max_context_len)

    response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        raise ValueError(f"SFT sample has no supervised assistant tokens: {messages=}")

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.reward = 0.0
    sample.loss_mask = loss_mask[-response_length:]
    sample.status = Sample.Status.COMPLETED
    return sample


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[Sample]: a list of samples generated by the rollout
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    skip_overlong = _env_flag("SLIME_SFT_SKIP_OVERLONG", "0")
    max_fetch_factor = max(1, int(os.environ.get("SLIME_SFT_SKIP_OVERLONG_MAX_FACTOR", "8")))
    max_attempts = args.rollout_batch_size * max_fetch_factor
    sample_groups = []
    attempted_samples = 0
    skipped_overlong = 0
    pending_groups = data_buffer.get_samples(args.rollout_batch_size)

    while pending_groups and len(sample_groups) < args.rollout_batch_size:
        for group in pending_groups:
            (sample,) = group
            attempted_samples += 1
            try:
                _build_sft_sample(args, sample)
            except OverlongSFTSampleError as exc:
                if not skip_overlong:
                    raise
                skipped_overlong += 1
                if skipped_overlong <= 8:
                    logger.warning(
                        "Skipping overlong SFT sample: length=%s max_context_len=%s",
                        exc.length,
                        exc.max_context_len,
                    )
                continue

            sample_groups.append([sample])
            if len(sample_groups) >= args.rollout_batch_size:
                break

        remaining = args.rollout_batch_size - len(sample_groups)
        if not skip_overlong or remaining <= 0:
            break
        if attempted_samples >= max_attempts:
            break
        pending_groups = data_buffer.get_samples(remaining)

    if len(sample_groups) < args.rollout_batch_size:
        raise RuntimeError(
            f"Only built {len(sample_groups)}/{args.rollout_batch_size} SFT samples after "
            f"{attempted_samples} attempts; skipped_overlong={skipped_overlong}. "
            "Increase SLIME_SFT_SKIP_OVERLONG_MAX_FACTOR or clean the dataset."
        )

    if skipped_overlong:
        logger.warning(
            "Skipped %s overlong SFT samples while building rollout batch of %s "
            "(attempted_samples=%s, max_attempts=%s).",
            skipped_overlong,
            args.rollout_batch_size,
            attempted_samples,
            max_attempts,
        )

    if not SAMPLE_PRINTED:
        (sample,) = sample_groups[0]
        mm_keys = sorted(sample.multimodal_train_inputs.keys()) if sample.multimodal_train_inputs else []
        logger.info(
            "sft_rollout example: "
            f"tokens={len(sample.tokens)}, response_length={sample.response_length}, "
            f"loss_tokens={sum(sample.loss_mask or [])}, multimodal_train_input_keys={mm_keys}"
        )
        SAMPLE_PRINTED = True

    return sample_groups
