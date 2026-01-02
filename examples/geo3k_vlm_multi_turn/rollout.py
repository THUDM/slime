from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_geo3k"


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


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env 
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
    Trim out the system/tool preamble added by the chat template so only the observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0
    dummy_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I am a user."},
    ]

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            dummy_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            dummy_messages + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
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

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs

def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """
    Merge per-turn multimodal_train_inputs with a single concat per key.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    processor = state.processor
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = sampling_params.copy()

    sample.metadata = sample.metadata or {}
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
        multimodal_train_inputs_buffer=[]
        if sample.multimodal_train_inputs:
            multimodal_train_inputs_buffer.append(sample.multimodal_train_inputs)

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
                new_response_tokens, new_response_log_probs = [], []

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
                    args.apply_chat_template,
                    args.apply_chat_template_kwargs,
                )
            )

            # Drop a leading BOS if present to avoid injecting it mid-stream.
            bos_id = tokenizer.bos_token_id
            if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
                obs_prompt_ids = obs_prompt_ids[1:]

            sample.tokens.extend(obs_prompt_ids)
            response_tokens.extend(obs_prompt_ids)
            sample.loss_mask.extend([0] * len(obs_prompt_ids))  # user/obs + next assistant prefix => masked as zero
            sample.rollout_log_probs.extend([float("-inf")] * len(obs_prompt_ids))  # keep logprob aligned with loss_mask
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
                # Defer concat until after the loop to avoid repeated allocations.
                multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

            if budget is not None:
                budget -= len(obs_prompt_ids)
                if budget <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break
            
            if turn_idx + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)

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
