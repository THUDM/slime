"""Tau2 (dual-control) rollout generation for slime.

Used via `--custom-generate-function-path rollout.generate`.
"""

from __future__ import annotations

import os
from typing import Any

from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import generate as sglang_generate
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

from actions import env_action_from_parsed_action, followup_messages_for_observation, parse_action
from env import compute_partial_score_from_reward_info, parse_reward_info
from prompting import build_tau2_agent_system_prompt

TOKENIZER = None
MASK_GENERATOR = None


def _get_tokenizer_and_mask_generator(args) -> tuple[Any, MultiTurnLossMaskGenerator]:
    global TOKENIZER, MASK_GENERATOR
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)
    return TOKENIZER, MASK_GENERATOR


def _get_tau2_user_sim_config() -> tuple[str, dict[str, Any]]:
    user_model = os.environ.get("TAU2_USER_MODEL", "openai/Qwen/Qwen3-4B-Instruct-2507")
    user_temperature = float(os.environ.get("TAU2_USER_TEMPERATURE", "0.7"))
    user_api_base = os.environ.get("TAU2_USER_API_BASE", "http://127.0.0.1:30001/v1")

    user_llm_args: dict[str, Any] = {"temperature": user_temperature}

    if user_api_base:
        user_llm_args["api_base"] = user_api_base
        user_llm_args["api_key"] = "dummy-key-for-local-server"
        user_llm_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    return user_model, user_llm_args


def _get_tau2_max_steps(args) -> int:
    return int(os.environ.get("TAU2_MAX_STEPS", "100"))


async def _generate_one_action(
    args,
    *,
    messages: list[dict[str, str]],
    sampling_params: dict[str, Any],
    max_repair_attempts: int = 1,
) -> tuple[str, Any | None, str | None]:
    stops = list(sampling_params.get("stop") or [])
    if "</tool_call>" not in stops:
        stops.append("</tool_call>")
    sampling_params = {**sampling_params, "stop": stops, "no_stop_trim": True}

    working_messages = list(messages)
    last_text = ""
    last_error: str | None = None

    for attempt in range(max_repair_attempts + 1):
        turn_sample = Sample(prompt=working_messages, metadata={})
        turn_sample = await sglang_generate(args, turn_sample, sampling_params.copy())
        if turn_sample.status == Sample.Status.ABORTED:
            return "", None, "sglang_aborted"

        text = (turn_sample.response or "").strip()
        last_text = text
        try:
            parsed = parse_action(text)
            return text, parsed, None
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_repair_attempts:
                break
            working_messages = working_messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "FORMAT ERROR. Re-output EXACTLY in the required <tool_call> format: "
                    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>. One action only.',
                },
            ]
            sampling_params = {**sampling_params, "temperature": 0.0}

    return last_text, None, last_error


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Multi-turn rollout for tau2-bench (dual-control)."""
    from tau2.gym.gym_agent import AgentGymEnv

    metadata = sample.metadata or {}
    task_index = metadata.get("task_index", 0)
    domain = metadata["domain"]
    task_split = metadata.get("split", "train")
    task_id = metadata["task_id"]

    user_llm, user_llm_args = _get_tau2_user_sim_config()
    max_steps = _get_tau2_max_steps(args)

    env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    observation, info = env.reset()
    tools = info.get("tools", [])
    tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]

    policy = info.get("policy", "")

    system_prompt = build_tau2_agent_system_prompt(domain=domain, policy=policy, tools_openai=tools_openai)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(
        followup_messages_for_observation(
            observation=observation,
            last_action_call="(reset)",
            last_action_was_tool=False,
            native_fc=True,
        )
    )

    assistant_turn_texts: list[str] = []
    tool_sequence: list[str] = []
    reward = 0.0
    reward_info: dict[str, Any] = {}
    info: dict[str, Any] = {}

    for _ in range(max_steps):
        assistant_text, parsed, err = await _generate_one_action(
            args,
            messages=messages,
            sampling_params=sampling_params,
            max_repair_attempts=1,
        )
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            assistant_turn_texts.append(assistant_text)

        if parsed is None:
            sample.remove_sample = True
            break

        if parsed.name not in ("respond", "done"):
            tool_sequence.append(parsed.name)

        env_action = env_action_from_parsed_action(parsed)
        observation, reward, terminated, _truncated, info = env.step(env_action)

        if terminated:
            reward_info = parse_reward_info(info)
            break

        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call=parsed.raw_action_call,
                last_action_was_tool=(parsed.name != "respond"),
            )
        )

    sample.prompt = messages
    sample.response = "\n\n".join(assistant_turn_texts).strip()
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED if terminated else Sample.Status.TRUNCATED

    reward_info = reward_info or parse_reward_info(info)
    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata.update(
        {
            "domain": domain,
            "split": task_split,
            "task_id": task_id,
            "task_index": task_index,
            "user_model": user_llm,
            "reward_info": reward_info,
            "partial_score": partial_score,
            "partial_components": partial_components,
            "tool_sequence": tool_sequence,
            "terminated": terminated,
        }
    )

    # Build tokens/loss-mask from the final multi-turn message list.
    _, mask_gen = _get_tokenizer_and_mask_generator(args)
    token_ids, full_loss_mask = mask_gen.get_loss_mask(messages)
    response_length = mask_gen.get_response_lengths([full_loss_mask])[0]

    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = full_loss_mask[-response_length:] if response_length > 0 else []

    return sample
