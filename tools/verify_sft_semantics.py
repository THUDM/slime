#!/usr/bin/env python3
"""Verify SFT token/mask and CP label alignment for Qwen-VL debug training.

This script is intentionally read-only and CPU friendly.  It compares the
current Slime SFT path against Relax's mask generator, then checks the Slime
VLM packed/CP shifted-label construction without launching Megatron training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from relax.utils.data.mask_utils import MultiTurnLossMaskGenerator as RelaxMaskGenerator
from slime.rollout.sft_rollout import _first_sequence_ids, _normalize_tool_call_arguments
from slime.utils.data import _build_messages
from slime.utils.mask_utils import MultiTurnLossMaskGenerator as SlimeMaskGenerator
from slime.utils.processing_utils import build_processor_kwargs, load_processor, process_vision_info


def _read_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _parse_json_dict(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected JSON object, got {type(parsed)}")
    return parsed


def _get_tools(row: dict[str, Any], tool_key: str | None) -> list[dict[str, Any]] | None:
    if tool_key and tool_key in row:
        tools = row[tool_key]
    else:
        metadata = row.get("metadata") or {}
        tools = metadata.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    return tools


def _row_messages(
    row: dict[str, Any],
    *,
    input_key: str,
    label_key: str | None,
    tool_key: str | None,
    multimodal_keys: dict[str, str] | None,
    normalize_tool_arguments: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    messages = _build_messages(
        row,
        input_key,
        as_conversation=True,
        multimodal_keys=multimodal_keys,
    )
    if not isinstance(messages, list):
        raise TypeError(f"Expected conversation list from {input_key}, got {type(messages)}")
    messages = copy.deepcopy(messages)

    if not any(message.get("role") == "assistant" for message in messages):
        if not label_key or label_key not in row:
            raise ValueError(f"Row has no assistant turn and no label key {label_key!r}")
        messages.append({"role": "assistant", "content": row[label_key]})

    tools = _get_tools(row, tool_key)
    if normalize_tool_arguments:
        messages = _normalize_tool_call_arguments(messages)
    return messages, tools


def _assert_same_tokens_and_mask(
    tokenizer,
    slime_generator: SlimeMaskGenerator,
    relax_generator: RelaxMaskGenerator,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    context: str,
) -> tuple[list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, return_dict=False)
    rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    template_ids = tokenizer.apply_chat_template(messages, tokenize=True, tools=tools, return_dict=False)
    if rendered_ids != template_ids:
        raise AssertionError(f"{context}: rendered-token ids differ from apply_chat_template(tokenize=True)")

    slime_ids, slime_mask = slime_generator.get_loss_mask(copy.deepcopy(messages), tools=tools)
    relax_ids, relax_mask = relax_generator.get_loss_mask(copy.deepcopy(messages), tools=tools)
    if slime_ids != relax_ids:
        raise AssertionError(f"{context}: Slime/Relax token ids differ: {len(slime_ids)} vs {len(relax_ids)}")
    if slime_mask != relax_mask:
        raise AssertionError(
            f"{context}: Slime/Relax loss masks differ: sums {sum(slime_mask)} vs {sum(relax_mask)}"
        )
    selected_slime = slime_generator.get_text_from_loss_mask(slime_ids, slime_mask)
    selected_relax = relax_generator.get_text_from_loss_mask(relax_ids, relax_mask)
    if selected_slime != selected_relax:
        raise AssertionError(f"{context}: supervised text differs")
    return slime_ids, slime_mask


def check_text_template_and_mask(
    args,
    tokenizer,
    slime_generator,
    relax_generator,
) -> list[tuple[list[int], list[int]]]:
    rows = _read_jsonl(args.text_data, args.max_text_samples)
    raw_relax_failures = 0
    checked = 0
    checked_samples = []
    multimodal_keys = _parse_json_dict(args.text_multimodal_keys)
    for idx, row in enumerate(rows):
        raw_messages, tools = _row_messages(
            row,
            input_key=args.text_input_key,
            label_key=args.text_label_key,
            tool_key=args.text_tool_key,
            multimodal_keys=multimodal_keys,
            normalize_tool_arguments=False,
        )
        try:
            relax_generator.get_loss_mask(copy.deepcopy(raw_messages), tools=tools)
        except Exception:
            raw_relax_failures += 1

        messages, tools = _row_messages(
            row,
            input_key=args.text_input_key,
            label_key=args.text_label_key,
            tool_key=args.text_tool_key,
            multimodal_keys=multimodal_keys,
            normalize_tool_arguments=True,
        )
        token_ids, loss_mask = _assert_same_tokens_and_mask(
            tokenizer,
            slime_generator,
            relax_generator,
            messages,
            tools,
            context=f"text row {idx}",
        )
        if sum(loss_mask) <= 0:
            raise AssertionError(f"text row {idx}: no supervised tokens")
        response_length = slime_generator.get_response_lengths([loss_mask])[0]
        checked_samples.append((token_ids, loss_mask[-response_length:]))
        checked += 1

    print(
        "PASS text template/mask:",
        f"checked={checked}",
        f"raw_relax_template_failures={raw_relax_failures}",
    )
    return checked_samples


def check_multimodal_template_and_mask(
    args,
    tokenizer,
    processor,
    slime_generator,
    relax_generator,
) -> list[tuple[list[int], list[int]]]:
    rows = _read_jsonl(args.mm_data)
    multimodal_keys = _parse_json_dict(args.mm_multimodal_keys)
    if not multimodal_keys:
        raise ValueError("--mm-multimodal-keys is required for multimodal checks")

    checked = 0
    checked_samples = []
    for line_idx, row in enumerate(rows):
        if not any(row.get(data_key) for data_key in multimodal_keys.values()):
            continue
        messages, tools = _row_messages(
            row,
            input_key=args.mm_input_key,
            label_key=args.mm_label_key,
            tool_key=args.mm_tool_key,
            multimodal_keys=multimodal_keys,
            normalize_tool_arguments=True,
        )
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, return_dict=False)
        multimodal_inputs = process_vision_info(messages, processor)
        slime_output = processor(text=[rendered], **build_processor_kwargs(multimodal_inputs))
        relax_output = processor(
            text=rendered,
            use_audio_in_video=False,
            return_mm_token_type_ids=False,
            **multimodal_inputs,
        )
        slime_ids = _first_sequence_ids(slime_output["input_ids"])
        relax_ids = _first_sequence_ids(relax_output["input_ids"])
        if slime_ids != relax_ids:
            raise AssertionError(f"mm row {line_idx}: Slime/Relax processor input_ids differ")

        slime_tokens, slime_mask = slime_generator.get_loss_mask_with_multimodal_alignment(
            copy.deepcopy(messages),
            slime_ids,
            tools=tools,
        )
        relax_tokens, relax_mask = relax_generator.get_loss_mask_with_multimodal_alignment(
            copy.deepcopy(messages),
            relax_ids,
            tools=tools,
        )
        if slime_tokens != relax_tokens:
            raise AssertionError(f"mm row {line_idx}: multimodal token ids differ")
        if slime_mask != relax_mask:
            raise AssertionError(
                f"mm row {line_idx}: multimodal loss masks differ: {sum(slime_mask)} vs {sum(relax_mask)}"
            )
        mm_keys = sorted(key for key in slime_output.keys() if key not in ("input_ids", "attention_mask"))
        if not {"pixel_values", "image_grid_thw"}.issubset(mm_keys):
            raise AssertionError(f"mm row {line_idx}: missing expected image tensors, got {mm_keys}")
        response_length = slime_generator.get_response_lengths([slime_mask])[0]
        checked_samples.append((slime_tokens, slime_mask[-response_length:]))
        checked += 1
        if checked >= args.max_mm_samples:
            break

    if checked == 0:
        raise AssertionError("No multimodal rows were checked")
    print("PASS multimodal processor/mask:", f"checked={checked}")
    return checked_samples


def check_synthetic_prompt_image(args, tokenizer, processor, slime_generator, relax_generator) -> None:
    if not args.synthetic_image:
        return
    if not Path(args.synthetic_image).exists():
        raise FileNotFoundError(args.synthetic_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "image": args.synthetic_image},
                {"type": "text", "text": "after"},
            ],
        },
        {"role": "assistant", "content": "answer"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, return_dict=False)
    multimodal_inputs = process_vision_info(messages, processor)
    output = processor(text=[rendered], **build_processor_kwargs(multimodal_inputs))
    input_ids = _first_sequence_ids(output["input_ids"])
    slime_tokens, slime_mask = slime_generator.get_loss_mask_with_multimodal_alignment(messages, input_ids)
    relax_tokens, relax_mask = relax_generator.get_loss_mask_with_multimodal_alignment(messages, input_ids)
    if slime_tokens != relax_tokens or slime_mask != relax_mask:
        raise AssertionError("synthetic text-image-text prompt changed Slime/Relax token or mask alignment")
    print("PASS synthetic prompt image alignment:", f"tokens={len(input_ids)}", f"loss_tokens={sum(slime_mask)}")


class _FakeMPU:
    def __init__(self, *, cp_rank: int, cp_size: int, tp_size: int):
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.tp_size = tp_size

    def get_context_parallel_rank(self):
        return self.cp_rank

    def get_context_parallel_world_size(self):
        return self.cp_size

    def get_tensor_model_parallel_world_size(self):
        return self.tp_size


def _patch_mpu(cp_rank: int, cp_size: int, tp_size: int):
    from slime.backends.megatron_utils import cp_utils, loss

    fake = _FakeMPU(cp_rank=cp_rank, cp_size=cp_size, tp_size=tp_size)
    cp_utils.mpu = fake
    loss.mpu = fake
    return cp_utils, loss


def _make_response_mask(response_length: int, zero_every: int | None = None) -> torch.Tensor:
    mask = torch.ones(response_length, dtype=torch.float32)
    if zero_every is not None:
        mask[zero_every - 1 :: zero_every] = 0
    return mask


def _verify_cp_samples(
    *,
    name: str,
    samples: list[tuple[list[int], list[int] | torch.Tensor]],
    cp_size: int,
    tp_size: int,
    use_padded_total_lengths: bool,
    pad_multiplier: int,
) -> None:
    if not samples:
        raise AssertionError(f"{name}: no samples to verify")
    total_lengths = [len(token_ids) for token_ids, _ in samples]
    response_masks = [
        torch.tensor(loss_mask, dtype=torch.float32) if not torch.is_tensor(loss_mask) else loss_mask.float()
        for _, loss_mask in samples
    ]
    response_lengths = [int(mask.numel()) for mask in response_masks]
    unconcat_tokens = [torch.tensor(token_ids, dtype=torch.long) for token_ids, _ in samples]
    align = tp_size * cp_size * 2
    padded_total_lengths = (
        [(total_length + align - 1) // align * align for total_length in total_lengths]
        if use_padded_total_lengths
        else None
    )

    total_supervised = 0
    for cp_rank in range(cp_size):
        cp_utils, loss_mod = _patch_mpu(cp_rank, cp_size, tp_size)
        local_tokens = []
        local_masks = []
        expected_labels = []
        for tokens, total_length, response_length, response_mask in zip(
            unconcat_tokens,
            total_lengths,
            response_lengths,
            response_masks,
            strict=True,
        ):
            prompt_length = total_length - response_length
            full_mask = F.pad(response_mask, (prompt_length - 1, 1), value=0)
            positions = torch.arange(total_length, dtype=torch.long)
            local_positions = cp_utils.slice_with_cp(positions, -1, "thd")
            local_tokens.append(cp_utils.slice_with_cp(tokens, 0, "thd"))
            local_masks.append(cp_utils.slice_with_cp(full_mask, 0, "thd"))

            expected = torch.zeros_like(local_positions)
            valid = (local_positions >= 0) & (local_positions < total_length - 1)
            expected[valid] = tokens[local_positions[valid] + 1]
            expected_labels.append(expected)

        cat_tokens = torch.cat(local_tokens)
        full_loss_masks = torch.cat(local_masks)
        expected_labels = torch.cat(expected_labels)
        pad_size = max(tp_size * pad_multiplier, 1)
        pad = (pad_size - cat_tokens.numel() % pad_size) % pad_size
        if pad:
            cat_tokens = F.pad(cat_tokens, (0, pad), value=0)
            full_loss_masks = F.pad(full_loss_masks, (0, pad), value=0)
            expected_labels = F.pad(expected_labels, (0, pad), value=0)

        shifted_labels = loss_mod._build_shifted_tokens(
            cat_tokens.numel(),
            cat_tokens.device,
            unconcat_tokens,
            total_lengths,
            response_lengths,
            "thd",
            None,
            False,
            padded_total_lengths,
        )
        if shifted_labels.shape != cat_tokens.shape:
            raise AssertionError(f"rank {cp_rank}: shifted_labels shape mismatch")
        train_positions = full_loss_masks > 0
        if not torch.equal(shifted_labels[train_positions], expected_labels[train_positions]):
            raise AssertionError(f"rank {cp_rank}: shifted labels do not match next-token targets under mask")
        masked_labels = shifted_labels.masked_fill(full_loss_masks.le(0), -100)
        if not torch.all(masked_labels[full_loss_masks <= 0].eq(-100)):
            raise AssertionError(f"rank {cp_rank}: non-training labels were not masked to -100")
        if int(masked_labels.ne(-100).sum().item()) != int(full_loss_masks.gt(0).sum().item()):
            raise AssertionError(f"rank {cp_rank}: label/mask supervised counts differ")
        total_supervised += int(full_loss_masks.gt(0).sum().item())

    expected_supervised = sum(int(mask.sum().item()) for mask in response_masks)
    if total_supervised != expected_supervised:
        raise AssertionError(f"all CP ranks: supervised count {total_supervised} != {expected_supervised}")
    print(
        "PASS CP shifted-label/mask alignment:",
        f"name={name}",
        f"samples={len(samples)}",
        f"cp={cp_size}",
        f"tp={tp_size}",
        f"padded={use_padded_total_lengths}",
        f"loss_tokens={total_supervised}",
    )


def _verify_cp_case(*, cp_size: int, tp_size: int, use_padded_total_lengths: bool) -> None:
    samples = []
    for sample_idx, (total_length, response_length, response_mask) in enumerate(
        zip(
            [37, 50, 19],
            [11, 17, 7],
            [
                _make_response_mask(11, zero_every=4),
                _make_response_mask(17, zero_every=5),
                _make_response_mask(7, zero_every=None),
            ],
            strict=True,
        )
    ):
        token_ids = list(range(1000 + sample_idx * 1000, 1000 + sample_idx * 1000 + total_length))
        if response_length != int(response_mask.numel()):
            raise AssertionError("internal synthetic response length mismatch")
        samples.append((token_ids, response_mask))
    _verify_cp_samples(
        name="synthetic",
        samples=samples,
        cp_size=cp_size,
        tp_size=tp_size,
        use_padded_total_lengths=use_padded_total_lengths,
        pad_multiplier=16,
    )


def check_cp_label_alignment() -> None:
    _verify_cp_case(cp_size=1, tp_size=1, use_padded_total_lengths=False)
    _verify_cp_case(cp_size=8, tp_size=1, use_padded_total_lengths=False)
    _verify_cp_case(cp_size=8, tp_size=1, use_padded_total_lengths=True)


def check_real_sample_cp_label_alignment(
    text_samples: list[tuple[list[int], list[int]]],
    multimodal_samples: list[tuple[list[int], list[int]]],
    *,
    cp_size: int,
    tp_size: int,
    pad_multiplier: int,
) -> None:
    _verify_cp_samples(
        name="real-text",
        samples=text_samples[: min(3, len(text_samples))],
        cp_size=cp_size,
        tp_size=tp_size,
        use_padded_total_lengths=True,
        pad_multiplier=pad_multiplier,
    )
    _verify_cp_samples(
        name="real-multimodal",
        samples=multimodal_samples,
        cp_size=cp_size,
        tp_size=tp_size,
        use_padded_total_lengths=True,
        pad_multiplier=pad_multiplier,
    )
    mixed_samples = []
    if text_samples:
        mixed_samples.append(text_samples[0])
    if multimodal_samples:
        mixed_samples.append(multimodal_samples[0])
    if len(mixed_samples) >= 2:
        _verify_cp_samples(
            name="real-mixed",
            samples=mixed_samples,
            cp_size=cp_size,
            tp_size=tp_size,
            use_padded_total_lengths=True,
            pad_multiplier=pad_multiplier,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--text-data", required=True)
    parser.add_argument("--text-input-key", default="prompt")
    parser.add_argument("--text-label-key", default="label")
    parser.add_argument("--text-tool-key", default="tools")
    parser.add_argument("--text-multimodal-keys", default=None)
    parser.add_argument("--max-text-samples", type=int, default=10)
    parser.add_argument("--mm-data", required=True)
    parser.add_argument("--mm-input-key", default="prompt")
    parser.add_argument("--mm-label-key", default="label")
    parser.add_argument("--mm-tool-key", default="tools")
    parser.add_argument("--mm-multimodal-keys", required=True)
    parser.add_argument("--max-mm-samples", type=int, default=3)
    parser.add_argument("--synthetic-image", default=None)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--data-pad-size-multiplier", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
    if processor is None:
        raise RuntimeError(f"Failed to load processor from {args.hf_checkpoint}")
    slime_generator = SlimeMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    relax_generator = RelaxMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    text_samples = check_text_template_and_mask(args, tokenizer, slime_generator, relax_generator)
    multimodal_samples = check_multimodal_template_and_mask(
        args,
        tokenizer,
        processor,
        slime_generator,
        relax_generator,
    )
    check_synthetic_prompt_image(args, tokenizer, processor, slime_generator, relax_generator)
    check_cp_label_alignment()
    check_real_sample_cp_label_alignment(
        text_samples,
        multimodal_samples,
        cp_size=args.cp_size,
        tp_size=args.tp_size,
        pad_multiplier=args.data_pad_size_multiplier,
    )


if __name__ == "__main__":
    main()
