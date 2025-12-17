from __future__ import annotations

import importlib
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_sokoban"
DEFAULT_ROLLOUT_CONFIG = {
    "max_turns": 20,
    "max_total_tokens": 8192,
    "stop_on_max_tokens": True,
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
    if getattr(args, "max_turns", None):
        cfg["max_turns"] = args.max_turns
    for key in ("max_total_tokens", "stop_on_max_tokens"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
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


def _merge_metadata(sample: Sample, updates: dict | None):
    if not updates:
        return
    sample.metadata = sample.metadata or {}
    for key, value in updates.items():
        if key in sample.metadata and isinstance(sample.metadata[key], dict) and isinstance(value, dict):
            sample.metadata[key] = {**sample.metadata[key], **value}
        else:
            sample.metadata[key] = value


def _handle_reset(env_module, env, observation: dict, sample: Sample, reset_info: dict | None):
    on_reset = getattr(env_module, "on_reset", None)
    if callable(on_reset):
        updates = on_reset(env=env, observation=observation, sample=sample, reset_info=reset_info)
        _merge_metadata(sample, updates)


def _finalize_episode(
    env_module,
    env,
    observation: dict,
    sample: Sample,
    responses: list[str],
) -> dict | None:
    finalize_fn = getattr(env_module, "finalize_episode", None)
    if callable(finalize_fn):
        result = finalize_fn(
            env=env,
            observation=observation,
            sample=sample,
            responses=responses,
        )
        updates = result or {}
        updates.setdefault("turns", len(responses))
        return updates
    return {}


def _encode_for_generation(tokenizer, processor, messages: list[dict], metadata: dict | None, apply_chat_template_kwargs: dict | None):
    """
    Encode the conversation for SGLang generation (with generation prompt) and return payload pieces.
    """
    from slime.utils.processing_utils import prepare_model_inputs

    prompt_ids, extra_info = prepare_model_inputs(
        messages,
        tokenizer,
        processor,
        metadata,
        apply_chat_template_kwargs,
    )

    image_data = [encode_image_for_rollout_engine(img) for img in extra_info.get("images", [])]
    return prompt_ids, image_data, extra_info.get("multimodal_inputs")


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
    stop_on_max_tokens = rollout_config["stop_on_max_tokens"]
    max_total_tokens = rollout_config["max_total_tokens"]
    max_context_len = getattr(args, "rollout_max_context_len", None)
    token_budget = min(max_total_tokens, max_context_len) if max_context_len is not None else max_total_tokens

    sample.metadata = sample.metadata or {}
    max_turns = getattr(args, "max_turns", None) or rollout_config["max_turns"]
    sample.rollout_response_length = sample.rollout_response_length or 0
    env = _build_env(env_module, sample, args)
    try:
        observation, reset_info = env.reset()
        _handle_reset(env_module, env, observation, sample, reset_info)

        # Use the preloaded prompt (contains system + first image) as the initial conversation state
        messages = deepcopy(sample.prompt)

        prompt_ids, image_data, multimodal_inputs = _encode_for_generation(
            tokenizer,
            processor,
            messages,
            sample.metadata,
            args.apply_chat_template_kwargs,
        )
        is_resume = bool(sample.tokens)

        if is_resume:
            max_new_tokens = sampling_params.get("max_new_tokens")
            if max_new_tokens is not None:
                if sample.rollout_response_length >= max_new_tokens:
                    sample.status = Sample.Status.TRUNCATED
                    return sample
                sampling_params["max_new_tokens"] = max_new_tokens - sample.rollout_response_length

        # Initialize token/logprob/loss tracking to be perfectly aligned with model inputs
        if not sample.tokens:
            sample.tokens = list(prompt_ids)
        response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
        sample.loss_mask = sample.loss_mask or []
        sample.rollout_log_probs = sample.rollout_log_probs or []
        sample.multimodal_inputs = multimodal_inputs if sample.multimodal_inputs is None else sample.multimodal_inputs
        sample.response_length = len(response_tokens)
        current_image_data = image_data

        generated_responses: list[str] = []
        status: Sample.Status | None = None

        for turn_idx in range(max_turns):
            remaining_budget = token_budget - len(sample.tokens)
            if stop_on_max_tokens and remaining_budget <= 0:
                status = Sample.Status.TRUNCATED
                break

            cur_sampling_params = sampling_params.copy()
            # Apply both remaining response budget and context budget
            max_new_tokens = cur_sampling_params.get("max_new_tokens")
            if remaining_budget > 0:
                effective_max_new = max_new_tokens if max_new_tokens is not None else remaining_budget
                cur_sampling_params["max_new_tokens"] = min(effective_max_new, remaining_budget)
            elif stop_on_max_tokens:
                status = Sample.Status.TRUNCATED
                break
            if cur_sampling_params.get("max_new_tokens", 0) <= 0:
                status = Sample.Status.TRUNCATED
                break

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
            def _fmt_tokens(seq: list[int], head: int = 10, tail: int = 30):
                if len(seq) <= head + tail:
                    return seq
                return seq[:head] + ["..."] + seq[-tail:]

            sample.tokens.extend(new_response_tokens)

            response_tokens.extend(new_response_tokens)
            sample.loss_mask.extend([1] * len(new_response_tokens))
            sample.rollout_log_probs.extend(new_response_log_probs)
            sample.rollout_response_length += len(new_response_tokens)
            if "max_new_tokens" in sampling_params and sampling_params["max_new_tokens"] is not None:
                sampling_params["max_new_tokens"] = max(0, sampling_params["max_new_tokens"] - len(new_response_tokens))
            sample.response_length = len(response_tokens)

            messages.append({"role": "assistant", "content": response_text})
            generated_responses.append(response_text)


            observation, done, step_info = env.step(response_text)
            step_record = {"turn": turn_idx, "info": step_info}
            sample.metadata.setdefault("trajectory", []).append(step_record)

            if done:
                status = Sample.Status.COMPLETED
                break

            if stop_on_max_tokens and len(sample.tokens) >= token_budget:
                status = Sample.Status.TRUNCATED
                break

            # Combine previous action text with the new observation image for the next user turn
            next_user_message = _format_observation(env_module, observation)
            messages.append(next_user_message)

            # Temporary solution: Re-encode the full conversation (including the new observation) so tokens/images stay aligned
            # TODO: make it more efficient by only re-encoding the new observation and the last assistant response but still keep tokens/images stay aligned.
            next_prompt_ids, next_image_data, next_multimodal_inputs = _encode_for_generation(
                tokenizer,
                processor,
                messages,
                sample.metadata,
                args.apply_chat_template_kwargs,
            )

            if sample.tokens != next_prompt_ids[: len(sample.tokens)]:
                raise RuntimeError(
                    "Token prefix mismatch after adding observation; generated response tokens do not align with re-encoded prompt."
                )

            delta_tokens = next_prompt_ids[len(sample.tokens) :]
            if stop_on_max_tokens and len(sample.tokens) + len(delta_tokens) > token_budget:
                status = Sample.Status.TRUNCATED
                break
            sample.tokens.extend(delta_tokens)
            response_tokens.extend(delta_tokens)
            sample.loss_mask.extend([0] * len(delta_tokens))  # user/obs + next assistant prefix => masked
            sample.rollout_log_probs.extend([0.0] * len(delta_tokens))  # keep logprob aligned with loss_mask
            sample.response_length = len(response_tokens)

            sample.multimodal_inputs = next_multimodal_inputs
            current_image_data = next_image_data

        
            if sampling_params.get("max_new_tokens", None) is not None and sampling_params["max_new_tokens"] <= 0:
                status = Sample.Status.TRUNCATED
                break
            
            if turn_idx + 1 >= max_turns:
                status = Sample.Status.TRUNCATED
                break
            
            finish_type = output["meta_info"]["finish_reason"]["type"]
            match finish_type:
                case "length":
                    status = Sample.Status.TRUNCATED
                    break
                case "abort":
                    status = Sample.Status.ABORTED
                    break

        # Decode only the response segment (everything after the initial prompt)
        metadata_updates = _finalize_episode(
            env_module,
            env,
            observation,
            sample=sample,
            responses=generated_responses,
        )
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        _merge_metadata(sample, metadata_updates)

        if status is None:
            status = Sample.Status.COMPLETED
        sample.status = status

        return sample
    finally:
        try:
            env.close()
        except Exception:
            pass
