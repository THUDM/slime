from __future__ import annotations

import importlib
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_geo3k"
DEFAULT_ROLLOUT_CONFIG = {
    "max_turns": 5,
}


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _resolve_rollout_config(args: Any, env_module) -> dict[str, Any]:
    """Combine rollout defaults with optional overrides from args."""
    cfg = deepcopy(getattr(env_module, "DEFAULT_ROLLOUT_CONFIG", DEFAULT_ROLLOUT_CONFIG))
    if getattr(args, "max_turns", None) is not None:
        cfg["max_turns"] = args.max_turns
    return cfg


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = getattr(env_module, "build_env", None) or getattr(env_module, "create_env", None)
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)


def _format_observation(env_module, observation: dict) -> dict:
    """Convert an environment observation into a chat message."""
    formatter = getattr(env_module, "format_observation", None)
    if callable(formatter):
        return formatter(observation)

    observation = observation or {}
    content = []
    multimodal = observation.get("multi_modal_data") or {}
    for _, images in multimodal.items():
        for image in images:
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": observation.get("obs_str", "")})
    return {"role": "user", "content": content}


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """
    Encode a single observation turn that may include images/videos in the content list.
    """
    def _get_chat_template_preamble_len() -> int:
        """
        Compute (and cache) the token length of the chat template preamble (e.g., tool XML)
        so we can trim it for subsequent turns instead of re-appending it every time.
        """
        if not metadata or not metadata.get("tools"):
            return 0
        cache_key = "_chat_template_preamble_len"
        if cache_key in metadata:
            return metadata[cache_key] or 0

        # Compute delta between templates with tools vs without tools to isolate the tool preamble.
        prompt_with_tools = tokenizer.apply_chat_template(
            [message],
            tools=metadata.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
        prompt_without_tools = tokenizer.apply_chat_template(
            [message],
            tools=None,
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
        ids_with_tools = tokenizer.encode(prompt_with_tools or "", add_special_tokens=False)
        ids_without_tools = tokenizer.encode(prompt_without_tools or "", add_special_tokens=False)

        if not ids_with_tools[-len(ids_without_tools) :] == ids_without_tools:
            raise ValueError("Tool-prefixed chat template does not end with non-tool template; cannot trim preamble safely.")

        metadata[cache_key] = len(ids_with_tools) - len(ids_without_tools)
        return metadata[cache_key]

    preamble_len = _get_chat_template_preamble_len() if apply_chat_template else 0
    if apply_chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            [message],
            tools=metadata.get("tools") if metadata else None,
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if preamble_len:
        prompt_ids = prompt_ids[preamble_len:]

    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(existing: dict | None, new: dict | None) -> dict | None:
    """
    Concatenate per-image tensors to keep a single batched multimodal_train_inputs dict.
    """
    if not new:
        return existing
    if not existing:
        return new

    merged = dict(existing)
    for key, val in new.items():
        if key not in merged:
            merged[key] = val
            continue
        if isinstance(merged[key], torch.Tensor) and isinstance(val, torch.Tensor):
            merged[key] = torch.cat([merged[key], val], dim=0)
        elif isinstance(merged[key], list) and isinstance(val, list):
            merged[key] = merged[key] + val
        elif isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    rollout_config = _resolve_rollout_config(args, env_module)

    state = GenerateState(args)
    tokenizer = state.tokenizer
    processor = state.processor
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = sampling_params.copy()

    sample.metadata = sample.metadata or {}
    max_turns = rollout_config["max_turns"]
    env = _build_env(env_module, sample, args)
    try:
        observation, _reset_info = env.reset()

        #prepare generation inputs from sample.prompt and sample.multimodal_inputs
        if processor:
            processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
            prompt_ids = processor_output["input_ids"][0]
            sample.multimodal_train_inputs = {
                k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
            } or None
        else:
            prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
        image_data = []
        if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
            image_data = [
                encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]
            ]
        current_image_data = image_data

        # Initialize token/logprob/loss_mask tracking to be aligned with model inputs
        if not sample.tokens:
            sample.tokens = list(prompt_ids)
        response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
        sample.loss_mask = sample.loss_mask or []
        sample.rollout_log_probs = sample.rollout_log_probs or []
        sample.response_length = len(response_tokens)

        budget = None
        if args.rollout_max_context_len is not None:
            budget = args.rollout_max_context_len - len(sample.tokens)
        elif sampling_params.get("max_new_tokens") is not None:
            budget = sampling_params["max_new_tokens"] - len(sample.tokens)
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        for turn_idx in range(max_turns):
            if budget is not None:
                sampling_params["max_new_tokens"] = budget
            cur_sampling_params = sampling_params.copy()

            payload = {
                "input_ids": sample.tokens,
                "sampling_params": cur_sampling_params,
                "return_logprob": True,
            }

            if current_image_data:
                payload["image_data"] = current_image_data

            output = await post(url, payload)

            response_text = output["text"]
            if "output_token_logprobs" in output["meta_info"]:
                new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            else:
                new_response_tokens = tokenizer(response_text, add_special_tokens=False)["input_ids"]
                new_response_log_probs = [0.0] * len(new_response_tokens)

            # Append assistant response tokens/logprobs/masks
            sample.tokens.extend(new_response_tokens)
            response_tokens.extend(new_response_tokens)
            sample.loss_mask.extend([1] * len(new_response_tokens))
            sample.rollout_log_probs.extend(new_response_log_probs)
            sample.response_length = len(response_tokens)

            if budget is not None:
                budget -= len(new_response_tokens)
                if budget <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break

            finish_type = output["meta_info"]["finish_reason"]["type"]
            match finish_type:
                case "length":
                    sample.status = Sample.Status.TRUNCATED
                    break
                case "abort":
                    sample.status = Sample.Status.ABORTED
                    break

            #interact with environment to get feedback
            observation, done, step_info = env.step(response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            next_user_message = _format_observation(env_module, observation)

            # Encode the new observation turn and append its tokens.
            obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
                _encode_observation_for_generation(
                    tokenizer,
                    processor,
                    next_user_message,
                    sample.metadata,
                    getattr(args, "apply_chat_template", False),
                    args.apply_chat_template_kwargs,
                )
            )

            # Drop a leading BOS if present to avoid injecting it mid-stream.
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
                obs_prompt_ids = obs_prompt_ids[1:]

            sample.tokens.extend(obs_prompt_ids)
            response_tokens.extend(obs_prompt_ids)
            sample.loss_mask.extend([0] * len(obs_prompt_ids))  # user/obs + next assistant prefix => masked as zero
            sample.rollout_log_probs.extend([0.0] * len(obs_prompt_ids))  # keep logprob aligned with loss_mask
            sample.response_length = len(response_tokens)
                
            if obs_image_data:
                current_image_data = (current_image_data or []) + obs_image_data

            if obs_multimodal_inputs:
                if not sample.multimodal_inputs:
                    sample.multimodal_inputs = obs_multimodal_inputs
                elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
                    for key, val in obs_multimodal_inputs.items():
                        if val is None:
                            continue
                        if (
                            key in sample.multimodal_inputs
                            and isinstance(sample.multimodal_inputs[key], list)
                            and isinstance(val, list)
                        ):
                            sample.multimodal_inputs[key].extend(val)
                else:
                    sample.multimodal_inputs = obs_multimodal_inputs

            if obs_multimodal_train_inputs:
                # Concatenate per-image tensors (e.g., pixel_values, image_grid_thw) across turns.
                sample.multimodal_train_inputs = _merge_multimodal_train_inputs(
                    sample.multimodal_train_inputs, obs_multimodal_train_inputs
                )

            if budget is not None:
                budget -= len(obs_prompt_ids)
                if budget <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break
            
            if turn_idx + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                break

   
        # Decode only the response segment (excluding the initial prompt while including env's feedback tokens)
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)

        if sample.status is None:
            sample.status = Sample.Status.COMPLETED
        return sample
    finally:
        try:
            env.close()
        except Exception:
            pass
