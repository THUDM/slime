"""Tau-bench integration for slime training."""

import logging
import os
from typing import Any

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": "gemini-2.5-flash-lite",  # Cheap Model for user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
    "user_model_provider": "gemini",
}
GEMINI_API_KEY = "NONE"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """Convert InteractionResult to a slime Sample."""
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    status = status_mapping.get(res.status)

    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        if res.loss_mask:
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """Run a single tau-bench interaction trajectory."""
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    env = get_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)

    result_sample = res_to_sample(interaction_result, task_index)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
