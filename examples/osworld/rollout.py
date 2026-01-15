"""OSWorld multi-turn VLM rollout for GSPO training.

The generate() function:
1. Creates OSWorld environment from task config
2. Runs multi-turn VLM interaction loop
3. Collects tokens and loss masks for training
4. Computes hybrid reward signal

Usage:
    Set --custom-generate-function-path examples.osworld.rollout.generate
    in train_grpo.sh
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from argparse import Namespace
from typing import Any

import httpx
import yaml
from PIL import Image

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import (
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
    process_vision_info,
    prepare_model_inputs,
)
from slime.utils.types import Sample

from .env import (
    OSWORLD_SYSTEM_PROMPT,
    QWEN3_MAX_PIXELS,
    OSWorldEnvWrapper,
    build_env,
    finalize_episode,
    on_reset,
    resize_screenshot_for_vlm,
)
from .replay_buffer import get_replay_buffer
from .reward import compute_reward_from_metadata

logger = logging.getLogger(__name__)

# Use synchronous HTTP to bypass async event loop issues in Ray actors
import concurrent.futures
import time

_sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def _vlm_post_sync(url: str, payload: dict, max_retries: int = 60) -> dict:
    """Synchronous POST with retries."""
    with httpx.Client(timeout=300.0) as client:
        for attempt in range(max_retries):
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt >= max_retries - 1:
                    logger.error(f"VLM request failed after {max_retries} retries: {e}")
                    raise
                time.sleep(1)
    raise RuntimeError(f"VLM request failed after {max_retries} retries")


async def _vlm_post(url: str, payload: dict, max_retries: int = 60) -> dict:
    """Blocking HTTP POST via thread pool (works in Ray actors)."""
    future = _sync_executor.submit(_vlm_post_sync, url, payload, max_retries)
    return future.result()


class OSWorldRolloutState:
    """Singleton state for OSWorld rollout generation.

    Holds tokenizer and processor loaded once per training run.
    """

    _instance = None

    def __new__(cls, args: Namespace):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, args: Namespace):
        if self._initialized:
            return
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        self.custom_config = _load_custom_config(getattr(args, "custom_config_path", None))
        self.env_defaults = dict(self.custom_config.get("env_defaults", {}) or {})
        self.reward_config = dict(self.custom_config.get("reward", {}) or {})
        env_max_turns = os.environ.get("OSWORLD_MAX_TURNS")
        if env_max_turns:
            try:
                self.max_turns = int(env_max_turns)
            except ValueError:
                logger.warning("OSWORLD_MAX_TURNS must be an int, got: %s", env_max_turns)
                self.max_turns = int(self.custom_config.get("max_turns", getattr(args, "max_turns", 8)))
        else:
            self.max_turns = int(self.custom_config.get("max_turns", getattr(args, "max_turns", 8)))
        self._initialized = True


async def generate_vlm_response(
    args: Namespace,
    state: OSWorldRolloutState,
    messages: list[dict],
    images: list,
    sampling_params: dict[str, Any],
) -> tuple[str, list[int], list[float]]:
    """Generate VLM response via SGLang router."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    processor_messages = messages
    if state.processor is not None and hasattr(state.processor, "apply_chat_template"):
        processor_messages = _normalize_messages_for_processor(messages)

    prompt_ids, _ = prepare_model_inputs(
        processor_messages,
        state.tokenizer,
        state.processor,
        metadata={"images": images} if images else None,
        apply_chat_template=True,
        apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
    )

    osworld_sampling_params = dict(sampling_params)
    osworld_sampling_params["ignore_eos"] = True
    osworld_sampling_params["stop"] = ["</tool_call>"]

    payload = {
        "input_ids": prompt_ids,
        "sampling_params": osworld_sampling_params,
        "return_logprob": True,
    }

    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in images]

    output = await _vlm_post(url, payload, max_retries=60)
    response_text = output["text"]

    if "output_token_logprobs" in output["meta_info"]:
        response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        response_tokens = state.tokenizer.encode(response_text, add_special_tokens=False)
        log_probs = []

    return response_text, response_tokens, log_probs


def build_chat_messages(
    system_prompt: str,
    observation_history: list[dict],
    response_history: list[str],
) -> list[dict]:
    """Build chat messages from interaction history."""
    messages = [{"role": "system", "content": system_prompt}]

    for i, obs in enumerate(observation_history):
        # Add observation as user message
        content = []
        multimodal = obs.get("multi_modal_data") or {}
        include_image = obs.get("include_image", True)
        if include_image:
            for _, images in multimodal.items():
                for image in images:
                    content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": obs.get("obs_str", "")})
        messages.append({"role": "user", "content": content})

        # Add response if available
        if i < len(response_history):
            messages.append({"role": "assistant", "content": response_history[i]})

    return messages


def _summarize_observation(obs: dict) -> dict:
    """Summarize older observations to reduce token budget while preserving key signals."""
    summary_parts: list[str] = []
    last_action = obs.get("last_action")
    if last_action:
        summary_parts.append(f"[Last Action]\n{last_action}")
    action_result = obs.get("action_result")
    if action_result:
        summary_parts.append(f"[Last Action Result]\n{action_result}")
    terminal = obs.get("terminal")
    if terminal:
        summary_parts.append(f"[Terminal Output]\n{terminal}")
    screen_diff = obs.get("screen_diff")
    if screen_diff is not None:
        summary_parts.append(f"[Screen Diff]\n{screen_diff:.4f}")
    summary = "\n\n".join(summary_parts) if summary_parts else obs.get("obs_str", "")
    summarized = dict(obs)
    summarized["obs_str"] = summary
    return summarized


def _compress_observation_history(observations: list[dict], keep_last: int) -> list[dict]:
    """Keep full text for the first and last N turns; summarize older turns."""
    if keep_last <= 0 or len(observations) <= keep_last + 1:
        return observations
    cutoff = max(1, len(observations) - keep_last)
    compressed: list[dict] = []
    for idx, obs in enumerate(observations):
        if idx == 0 or idx >= cutoff:
            compressed.append(obs)
        else:
            compressed.append(_summarize_observation(obs))
    return compressed


def _decode_replay_images(images: list[Any]) -> list[Image.Image]:
    """Decode base64 images for replay samples into PIL Images."""
    decoded: list[Image.Image] = []
    for img in images:
        if isinstance(img, Image.Image):
            resized = resize_screenshot_for_vlm(img.convert("RGB"))
            if os.environ.get("OSWORLD_ASSERT_REPLAY_IMAGE_SIZE") == "1":
                pixels = resized.size[0] * resized.size[1]
                if pixels > QWEN3_MAX_PIXELS:
                    raise ValueError(f"Replay image exceeds max pixels after resize: {resized.size}")
            decoded.append(resized)
            continue
        if not isinstance(img, str):
            continue
        payload = img
        if img.startswith("data:image"):
            _, payload = img.split(",", 1)
        try:
            img_bytes = base64.b64decode(payload)
            resized = resize_screenshot_for_vlm(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            if os.environ.get("OSWORLD_ASSERT_REPLAY_IMAGE_SIZE") == "1":
                pixels = resized.size[0] * resized.size[1]
                if pixels > QWEN3_MAX_PIXELS:
                    raise ValueError(f"Replay image exceeds max pixels after resize: {resized.size}")
            decoded.append(resized)
        except Exception as exc:
            logger.warning(f"Failed to decode replay image: {exc}")
    return decoded


def _attach_replay_images(messages: list[dict], images: list[Image.Image]) -> list[dict]:
    """Attach decoded images to user messages containing <image> placeholders."""
    if not images:
        return messages
    img_iter = iter(images)
    normalized: list[dict] = []
    for msg in messages:
        if msg.get("role") != "user":
            normalized.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Assume images already embedded
            normalized.append(msg)
            continue
        if isinstance(content, str) and "<image>" in content:
            parts = content.split("<image>")
            content_list: list[dict] = []
            for idx, part in enumerate(parts):
                if part:
                    content_list.append({"type": "text", "text": part})
                if idx < len(parts) - 1:
                    try:
                        image = next(img_iter)
                    except StopIteration:
                        break
                    content_list.append({"type": "image", "image": image})
            if content_list:
                normalized.append({**msg, "content": content_list})
            else:
                normalized.append(msg)
            continue
        normalized.append(msg)
    return normalized


def _normalize_messages_for_processor(messages: list[dict]) -> list[dict]:
    """Normalize messages so processor.apply_chat_template can parse them."""
    normalized: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            content_list: list[dict] = []
            for item in content:
                if isinstance(item, dict):
                    content_list.append(item)
                elif isinstance(item, str):
                    content_list.append({"type": "text", "text": item})
            normalized.append({**msg, "content": content_list})
            continue
        if isinstance(content, str):
            normalized.append({**msg, "content": [{"type": "text", "text": content}]})
            continue
        normalized.append(msg)
    return normalized


def _normalize_replay_messages(
    conversations: list[dict],
    system_prompt: str | None,
    images: list[Any] | None = None,
) -> list[dict]:
    """Ensure replay conversations include OSWORLD_SYSTEM_PROMPT and attach images."""
    if not conversations:
        return []
    messages = [dict(msg) for msg in conversations]
    if system_prompt:
        if messages[0].get("role") == "system":
            messages[0] = {**messages[0], "content": system_prompt}
        else:
            messages = [{"role": "system", "content": system_prompt}] + messages
    if images:
        decoded_images = _decode_replay_images(images)
        messages = _attach_replay_images(messages, decoded_images)
    return messages


def _load_custom_config(path: str | None) -> dict:
    """Load a YAML config file if present."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_loss_mask_for_multimodal(
    text_token_ids: list[int],
    text_loss_mask: list[int],
    multimodal_token_ids: list[int],
) -> list[int]:
    """Expand text-only loss mask to multimodal token sequence length."""
    expanded = []
    text_idx = 0
    for token_id in multimodal_token_ids:
        if text_idx < len(text_token_ids) and token_id == text_token_ids[text_idx]:
            expanded.append(text_loss_mask[text_idx])
            text_idx += 1
        else:
            expanded.append(0)

    if text_idx != len(text_token_ids):
        missing = len(text_token_ids) - text_idx
        if missing <= 8:
            logger.warning(
                "Multimodal/text token alignment mismatch (missing=%d). "
                "Proceeding with best-effort loss mask.",
                missing,
            )
            return expanded
        raise ValueError(
            "Failed to align multimodal tokens with text tokens: "
            f"matched={text_idx} total_text={len(text_token_ids)}"
        )

    return expanded


def _extract_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        token_ids = tokenized["input_ids"]
    elif hasattr(tokenized, "input_ids"):
        token_ids = tokenized.input_ids
    else:
        token_ids = tokenized

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return token_ids


def _build_loss_mask_from_full_chat(
    messages: list[dict],
    apply_chat_template_fn,
    gen_token_length: int,
    chat_template_kwargs: dict,
) -> tuple[list[int], list[int]]:
    """Build loss mask using the same full-chat tokenization path as the processor."""
    full_ids = _extract_token_ids(apply_chat_template_fn(messages, tokenize=True, **chat_template_kwargs))
    loss_mask = [0] * len(full_ids)
    prev_ids: list[int] = []

    for idx, msg in enumerate(messages):
        current_ids = _extract_token_ids(
            apply_chat_template_fn(messages[: idx + 1], tokenize=True, **chat_template_kwargs)
        )
        span_len = len(current_ids) - len(prev_ids)
        if span_len < 0:
            raise ValueError("Chat template tokenization is not monotonic across prefixes.")

        if msg["role"] == "assistant":
            prefix_len = min(gen_token_length, span_len)
            span_loss = [0] * prefix_len + [1] * (span_len - prefix_len)
        else:
            span_loss = [0] * span_len

        if msg.get("step_loss_mask", 1) != 1:
            span_loss = [0] * span_len

        loss_mask[len(prev_ids) : len(prev_ids) + span_len] = span_loss
        prev_ids = current_ids

    return full_ids, loss_mask


def _build_multimodal_training_inputs(
    tokenizer,
    processor,
    messages: list[dict],
    tokenizer_type: str,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[list[int], int, list[int], dict | None]:
    """Build multimodal tokens, loss masks, and multimodal inputs for training."""
    def _expected_image_tokens(mm_inputs: dict) -> int | None:
        grid = mm_inputs.get("image_grid_thw") if isinstance(mm_inputs, dict) else None
        if grid is None:
            return None
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        merge_size = 1
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "spatial_merge_size"):
            merge_size = int(getattr(processor.image_processor, "spatial_merge_size") or 1)
        elif hasattr(processor, "config") and hasattr(processor.config, "vision_config"):
            merge_size = int(getattr(processor.config.vision_config, "spatial_merge_size", 1) or 1)
        total = 0
        for item in grid:
            if not item or len(item) < 3:
                continue
            total += int(item[0]) * int(item[1]) * int(item[2])
        if merge_size > 1:
            total = total // (merge_size**2)
        return total

    def _validate_multimodal_tokens(token_ids: list[int], mm_inputs: dict | None):
        if not mm_inputs:
            return
        has_images = any(k in mm_inputs for k in ("pixel_values", "image_grid_thw", "image_sizes"))
        if not has_images:
            return
        image_token_ids: set[int] = set()
        for attr in ("image_token_id", "image_token_ids"):
            if hasattr(tokenizer, attr):
                val = getattr(tokenizer, attr)
                if isinstance(val, (list, tuple, set)):
                    image_token_ids.update(int(v) for v in val)
                elif val is not None:
                    image_token_ids.add(int(val))
        for tok in ("<|image_pad|>", "<|video_pad|>"):
            try:
                tok_id = tokenizer.convert_tokens_to_ids(tok)
                if tok_id is not None and tok_id != tokenizer.unk_token_id:
                    image_token_ids.add(int(tok_id))
            except Exception:
                pass
        if image_token_ids:
            count = sum(1 for tid in token_ids if tid in image_token_ids)
            if count == 0:
                raise ValueError("Image features present but no image tokens in input_ids")
            expected = _expected_image_tokens(mm_inputs)
            if expected is not None and count != expected:
                raise ValueError(f"Image features and image tokens do not match: tokens: {count}, features {expected}")
    chat_template_kwargs = {
        key: value for key, value in (apply_chat_template_kwargs or {}).items() if key != "add_generation_prompt"
    }

    if processor is None:
        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=tokenizer_type)
        text_token_ids, text_loss_mask = _build_loss_mask_from_full_chat(
            messages,
            tokenizer.apply_chat_template,
            mask_generator.gen_token_length,
            chat_template_kwargs,
        )
        system_tokens_len = 0
        if messages and messages[0].get("role") == "system":
            system_tokens_len = len(
                _extract_token_ids(tokenizer.apply_chat_template([messages[0]], tokenize=True, **chat_template_kwargs))
            )
        response_length = max(0, len(text_token_ids) - system_tokens_len)
        return text_token_ids, response_length, text_loss_mask[-response_length:], None

    processor_messages = _normalize_messages_for_processor(messages)

    def _apply_chat_template(messages, **kwargs):
        tokenize = kwargs.pop("tokenize", False)
        kwargs.pop("add_special_tokens", None)
        normalized = _normalize_messages_for_processor(messages)
        prompt = processor.apply_chat_template(normalized, tokenize=False, **kwargs)
        if not tokenize:
            return prompt
        encoded = processor.apply_chat_template(normalized, tokenize=True, return_dict=True, **kwargs)
        return _extract_token_ids(encoded)

    def _encode_with_processor(prefix_messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], dict]:
        normalized = _normalize_messages_for_processor(prefix_messages)
        prompt_text = processor.apply_chat_template(normalized, tokenize=False, **chat_template_kwargs, tools=tools)
        multimodal_inputs = process_vision_info(normalized, processor)
        processor_output = processor(text=prompt_text, **multimodal_inputs)
        token_ids = _extract_token_ids(processor_output)
        extra = {k: v for k, v in processor_output.items() if k not in {"input_ids", "attention_mask"}}
        return token_ids, extra

    def _tokenize_with_processor(prefix_messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        token_ids, _ = _encode_with_processor(prefix_messages, tools=tools)
        return token_ids

    mask_generator = MultiTurnLossMaskGenerator(
        tokenizer,
        tokenizer_type=tokenizer_type,
        apply_chat_template_fn=_apply_chat_template,
    )
    multimodal_token_ids, loss_mask_full = mask_generator.get_loss_mask_with_tokenizer_fn(
        processor_messages,
        _tokenize_with_processor,
        tools=None,
    )

    full_token_ids, multimodal_inputs = _encode_with_processor(processor_messages)
    if full_token_ids != multimodal_token_ids:
        raise ValueError("Processor tokenization mismatch between loss mask and model inputs")

    system_tokens_len = 0
    if processor_messages and processor_messages[0].get("role") == "system":
        system_ids = _tokenize_with_processor([processor_messages[0]])
        if multimodal_token_ids[: len(system_ids)] != system_ids:
            raise ValueError("System prompt tokens do not match multimodal token prefix")
        system_tokens_len = len(system_ids)

    response_length = max(0, len(multimodal_token_ids) - system_tokens_len)
    loss_mask = loss_mask_full[system_tokens_len:]
    if len(loss_mask) != response_length:
        raise ValueError(
            "Loss mask length mismatch after multimodal expansion: "
            f"mask_len={len(loss_mask)} response_len={response_length}"
        )

    if isinstance(multimodal_inputs, dict) and not multimodal_inputs:
        multimodal_inputs = None
    _validate_multimodal_tokens(multimodal_token_ids, multimodal_inputs)
    return multimodal_token_ids, response_length, loss_mask, multimodal_inputs


def _extract_prompt_text(prompt: str | list) -> str:
    """Extract raw text from prompt (handles chat-templated or raw format)."""
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        # Chat template applied - extract text from first user message
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Multimodal content - find text portion
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "")
        return ""

    return str(prompt)


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Generate a complete multi-turn GUI interaction trajectory for GRPO."""
    # Timeout protection: OSWorld can hang on stuck VMs
    # Default 10 minutes per trajectory (reset ~90s + multi-turn steps + buffer)
    # Configurable via OSWORLD_TIMEOUT env var for easy adjustment
    default_timeout = int(os.environ.get("OSWORLD_TIMEOUT", "600"))
    timeout_seconds = getattr(args, "osworld_timeout", default_timeout)

    try:
        return await asyncio.wait_for(
            _generate_impl(args, sample, sampling_params),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        prompt_text = _extract_prompt_text(sample.prompt)
        logger.error(f"OSWorld rollout timed out after {timeout_seconds}s for sample {prompt_text[:100]}")
        # Avoid empty tokens which crash ring_flash_attn on empty cu_seqlens.
        state = OSWorldRolloutState(args)
        pad_token_id = state.tokenizer.pad_token_id or state.tokenizer.eos_token_id or 0
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0
        sample.tokens = [pad_token_id]  # Minimal non-empty sequence
        sample.response_length = 1
        sample.loss_mask = [0]  # Don't train on this
        sample.remove_sample = True
        sample.metadata = sample.metadata or {}
        sample.metadata["osworld"] = {"timeout": True, "task_reward": 0.0}
        return sample


async def _generate_impl(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """Internal implementation of generate with timeout protection."""
    state = OSWorldRolloutState(args)
    max_turns = state.max_turns
    train_truncate_turns = getattr(args, "train_truncate_turns", None)
    if train_truncate_turns is None:
        env_truncate = os.environ.get("OSWORLD_TRAIN_TRUNCATE_TURNS")
        if env_truncate:
            try:
                train_truncate_turns = int(env_truncate)
            except ValueError:
                logger.warning(
                    "OSWORLD_TRAIN_TRUNCATE_TURNS must be an int, got: %s",
                    env_truncate,
                )
                train_truncate_turns = None

    # Validate input
    assert sample.status in (
        Sample.Status.PENDING,
        Sample.Status.ABORTED,
    ), f"Sample status is {sample.status}"

    # Get task index from prompt (for logging only)
    prompt_text = _extract_prompt_text(sample.prompt)
    task_index = int(prompt_text) if prompt_text.isdigit() else 0

    logger.info(f"Starting OSWorld rollout for task {task_index}, max_turns={max_turns}")

    # Ensure metadata is a dict
    if isinstance(sample.metadata, str):
        try:
            sample.metadata = json.loads(sample.metadata)
        except json.JSONDecodeError:
            sample.metadata = {}
    elif not isinstance(sample.metadata, dict):
        sample.metadata = {}

    env: OSWorldEnvWrapper = build_env(sample, args, config_overrides=state.env_defaults)

    try:
        observation, reset_info = await asyncio.to_thread(env.reset)
        sample.metadata = sample.metadata or {}
        sample.metadata.update(on_reset(env, observation, sample, reset_info))

        system_prompt = env.system_prompt()
        observation_history: list[dict] = []
        response_history: list[str] = []

        done = False
        for _turn in range(max_turns):
            # Format current observation
            formatted_obs = observation
            observation_history.append(formatted_obs)

            obs_keep_last = int(os.environ.get("OSWORLD_OBS_KEEP_FULL_TURNS", "2"))
            compressed_obs = _compress_observation_history(observation_history, obs_keep_last)

            # Extract images from observation history for VLM call
            # SGLang needs image_data to match number of vision token blocks in prompt
            # build_chat_messages adds vision tokens for ALL observations in history
            all_images = []
            for obs in compressed_obs:
                if not obs.get("include_image", True):
                    continue
                multimodal = obs.get("multi_modal_data") or {}
                for _, img_list in multimodal.items():
                    all_images.extend(img_list)

            messages = build_chat_messages(system_prompt, compressed_obs, response_history)

            response_text, response_tokens, _log_probs = await generate_vlm_response(
                args, state, messages, all_images, sampling_params
            )

            response_history.append(response_text)

            observation, done, step_info = await asyncio.to_thread(env.step, response_text)
            if step_info and "step_signal" in step_info:
                step_signal = step_info["step_signal"]
                screen_changed = step_signal.get("screen_changed")
                observation["screen_changed"] = screen_changed
                observation["screen_diff"] = step_signal.get("screen_diff")
                observation["include_image"] = bool(screen_changed) if screen_changed is not None else True
                observation["last_action"] = step_info.get("raw_action") or step_signal.get("action_type")
            if done:
                break

        finalize_meta = await asyncio.to_thread(finalize_episode, env, observation, sample, response_history)
        sample.metadata.update(finalize_meta)

        kept_obs = observation_history
        kept_responses = response_history
        if train_truncate_turns and len(response_history) > train_truncate_turns:
            kept_obs = observation_history[-train_truncate_turns:]
            kept_responses = response_history[-train_truncate_turns:]

        kept_turns = len(kept_responses)
        dropped_turns = len(response_history) - kept_turns

        obs_keep_last = int(os.environ.get("OSWORLD_OBS_KEEP_FULL_TURNS", "2"))
        compressed_kept_obs = _compress_observation_history(kept_obs, obs_keep_last)
        messages = build_chat_messages(system_prompt, compressed_kept_obs, kept_responses)
        tokenizer_type = getattr(args, "loss_mask_type", "qwen")
        try:
            tokens, response_length, loss_mask, multimodal_inputs = _build_multimodal_training_inputs(
                state.tokenizer,
                state.processor,
                messages,
                tokenizer_type,
                apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
            )
        except ValueError as e:
            pad_token_id = state.tokenizer.pad_token_id or state.tokenizer.eos_token_id or 0
            sample.status = Sample.Status.ABORTED
            sample.reward = 0.0
            sample.tokens = [pad_token_id]
            sample.response_length = 1
            sample.loss_mask = [0]
            sample.remove_sample = True
            sample.metadata.setdefault("osworld", {})
            sample.metadata["osworld"]["multimodal_error"] = str(e)
            return sample

        if response_length == 0:
            pad_token_id = state.tokenizer.pad_token_id or state.tokenizer.eos_token_id or 0
            sample.status = Sample.Status.ABORTED
            sample.reward = 0.0
            sample.tokens = [pad_token_id]
            sample.response_length = 1
            sample.loss_mask = [0]
            sample.remove_sample = True
            sample.metadata.setdefault("osworld", {})
            sample.metadata["osworld"]["empty_response_tokens"] = True
            return sample

        sample.tokens = tokens
        sample.response = "\n".join(response_history)
        sample.response_length = response_length
        sample.loss_mask = loss_mask
        sample.multimodal_inputs = multimodal_inputs
        sample.multimodal_train_inputs = multimodal_inputs
        sample.rollout_log_probs = None

        # Compute task completion reward
        # For GSPO: custom reward fn will compute shaped reward; keep raw task reward in metadata.
        task_reward = env.compute_reward()
        sample.reward = None  # allow custom RM to compute shaped reward
        sample.metadata["osworld"]["task_reward"] = task_reward
        sample.metadata["raw_reward"] = task_reward
        if state.reward_config:
            sample.metadata["osworld"]["reward_config"] = state.reward_config
        sample.metadata["osworld"]["train_truncate_turns"] = (
            train_truncate_turns if train_truncate_turns is not None else 0
        )
        sample.metadata["osworld"]["train_kept_turns"] = kept_turns
        sample.metadata["osworld"]["train_dropped_turns"] = dropped_turns

        # Set status
        if done:
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.TRUNCATED

        logger.info(
            f"Finished OSWorld rollout for task {task_index}: "
            f"turns={len(response_history)}, reward={task_reward:.2f}"
        )

    finally:
        # Clean up environment
        env.close()

    return sample


def _extract_task_id(metadata: dict | None) -> str:
    """Extract task_id from sample metadata (handles task_config nesting)."""
    if not metadata:
        return ""
    # Direct task_id
    if "task_id" in metadata:
        return metadata["task_id"]
    # Nested in task_config (from --metadata-key task_config)
    if "id" in metadata:
        return metadata["id"]
    return ""


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
):
    """Batch-level rollout function with replay injection for GSPO.

    This function wraps the per-sample generate() and adds replay buffer
    injection when all online rollouts fail.

    Args:
        args: Training arguments
        rollout_id: ID of this rollout batch
        data_buffer: Buffer providing samples via get_samples()
        evaluation: Whether this is an eval rollout

    Returns:
        RolloutFnTrainOutput with samples organized as groups
    """
    from slime.rollout.base_types import RolloutFnTrainOutput

    # Get sampling params from args
    sampling_params = {
        "temperature": getattr(args, "rollout_temperature", 0.8),
        "top_p": getattr(args, "rollout_top_p", 0.95),
        "max_tokens": getattr(args, "rollout_max_response_len", 2048),
    }
    state = OSWorldRolloutState(args)

    batch_size = args.rollout_batch_size

    # Initialize replay buffer singleton (logs on first call)
    replay_buffer = get_replay_buffer()

    all_results: list[list[Sample]] = []

    sample_groups = data_buffer.get_samples(batch_size)
    if not sample_groups:
        return RolloutFnTrainOutput(samples=all_results)

    work_items: list[tuple[int, int, Sample]] = []
    coros = []
    for group_idx, group in enumerate(sample_groups):
        if not group:
            continue
        for sample_idx, sample in enumerate(group):
            sample.status = Sample.Status.PENDING
            sample.metadata = dict(sample.metadata) if sample.metadata else {}
            work_items.append((group_idx, sample_idx, sample))
            coros.append(generate(args, sample, sampling_params, evaluation=evaluation))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(asyncio.gather(*coros, return_exceptions=True))
    finally:
        loop.close()

    group_results: list[list[Sample | None]] = [[None for _ in group] for group in sample_groups]

    for (group_idx, sample_idx, sample), result in zip(work_items, results):
        if isinstance(result, Exception):
            logger.error(f"generate_rollout: Error generating sample {sample_idx}: {result}")
            sample.status = Sample.Status.ABORTED
            sample.reward = 0.0
            sample.tokens = [0]
            sample.response_length = 1
            sample.loss_mask = [0]
            sample.remove_sample = True
            group_results[group_idx][sample_idx] = sample
            continue
        group_results[group_idx][sample_idx] = result

    for group_idx, group in enumerate(group_results):
        if not group:
            all_results.append([])
            continue

        if any(sample is None for sample in group):
            logger.error("generate_rollout: Missing results for a sample group; dropping group.")
            all_results.append([])
            continue

        # At this point, the group is fully populated with Samples.
        group = [sample for sample in group if sample is not None]
        # Compute shaped rewards for all samples before replay injection check
        # This is required because custom rollout functions bypass the standard RM flow
        for sample in group:
            if sample.reward is None:
                sample.reward = compute_reward_from_metadata(sample, state.reward_config)

        base_sample = sample_groups[group_idx][0]
        task_id = _extract_task_id(base_sample.metadata)

        # Replay injection: inject successful trajectory if all online rollouts failed
        # Use RAW task rewards (0 or 1) for replay decision, not shaped rewards
        # Shaped rewards can be positive even when task fails due to partial scores
        replay_debug = os.environ.get("OSWORLD_REPLAY_DEBUG", "0") == "1"
        if replay_buffer is not None:
            # Extract raw task rewards from metadata for replay decision
            raw_task_rewards = []
            for s in group:
                osworld_meta = (s.metadata or {}).get("osworld", {})
                raw = osworld_meta.get("task_reward", 0.0)
                raw_task_rewards.append(raw)

            if replay_debug:
                has_task = replay_buffer.has_task(task_id)
                logger.info(
                    f"[REPLAY_DEBUG] task_id={task_id} raw_rewards={raw_task_rewards} "
                    f"threshold={replay_buffer.success_threshold} has_task={has_task}"
                )

            replay_sample = replay_buffer.maybe_inject(task_id, raw_task_rewards)
            if replay_sample:
                tokenizer_type = getattr(args, "loss_mask_type", "qwen")
                replay_messages = _normalize_replay_messages(
                    replay_sample.conversations,
                    OSWORLD_SYSTEM_PROMPT,
                    replay_sample.images,
                )
                replay_tokens, replay_response_length, replay_loss_mask, replay_multimodal_inputs = (
                    _build_multimodal_training_inputs(
                        state.tokenizer,
                        state.processor,
                        replay_messages,
                        tokenizer_type,
                        apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
                    )
                )

                if replay_tokens and replay_response_length > 0:
                    replay_metadata = {
                        **(base_sample.metadata or {}),
                        "task_id": task_id,
                        "is_replay": True,
                        "replay_reward": replay_sample.reward,
                    }
                    if state.reward_config:
                        osworld_meta = replay_metadata.get("osworld", {})
                        if not isinstance(osworld_meta, dict):
                            osworld_meta = {}
                        osworld_meta["reward_config"] = state.reward_config
                        replay_metadata["osworld"] = osworld_meta
                    injected = Sample(
                        prompt=base_sample.prompt,
                        index=base_sample.index,
                        status=Sample.Status.COMPLETED,
                        tokens=replay_tokens,
                        response_length=replay_response_length,
                        loss_mask=replay_loss_mask,
                        multimodal_inputs=replay_multimodal_inputs,
                        multimodal_train_inputs=replay_multimodal_inputs,
                        reward=replay_sample.reward,
                        metadata=replay_metadata,
                    )
                    group.append(injected)
                    logger.info(
                        f"Injected replay for task {task_id}: "
                        f"raw_task_rewards={raw_task_rewards}, replay_reward={replay_sample.reward}"
                    )

        # Drop aborted/invalid samples to avoid zero-length response issues in training.
        # Also drop samples missing multimodal inputs to prevent training crashes.
        filtered_results = []
        for sample in group:
            if getattr(sample, "remove_sample", False):
                continue
            if getattr(sample, "multimodal_train_inputs", None) is None:
                meta = sample.metadata if isinstance(sample.metadata, dict) else {}
                task_id = meta.get("task_id", "unknown")
                logger.warning(
                    "Dropping sample with missing multimodal_train_inputs "
                    f"(task_id={task_id})"
                )
                continue
            filtered_results.append(sample)
        all_results.append(filtered_results)

    return RolloutFnTrainOutput(samples=all_results)
