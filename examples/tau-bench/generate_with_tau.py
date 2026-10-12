"""
Tau-Bench Integration for slime Training

This module provides the main interface for training agents in tau-bench environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.
"""

import logging
import os
from typing import Any

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, agent_factory

from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

# Tau-bench configuration
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent_strategy": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": "gemini-2.5-flash-lite",  # Cheap Model for user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
    "user_model_provider": "gemini",
}
# Replace with your actual API key for user sim
GEMINI_API_KEY = "NONE"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult, task_index: int, args) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    Prompt ids are the initial chat template. Response segments are appended
    in order so sampled assistant tokens keep their logprobs and loss mask,
    and environment tokens stay masked.
    """
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=list(res.prompt_token_ids or []),
        reward=res.reward,
        status=Sample.Status(res.status.value),
        metadata=res.info or {},
    )
    for segment in res.segments:
        if not segment.token_ids:
            continue
        sample.append_response_tokens(
            args,
            tokens=segment.token_ids,
            log_probs=segment.log_probs if segment.trainable else None,
            trainable=segment.trainable,
            meta_info=segment.meta_info if segment.trainable else None,
        )
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.response = res.response

    logger.debug(
        f"res_to_sample: response_length={sample.response_length}, "
        f"loss_mask_len={len(sample.loss_mask)}, "
        f"tokens_len={len(sample.tokens)}"
    )
    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for slime training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # Initialize tau-bench environment
    env = get_env(
        env_name=tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
        task_index=task_index,
    )

    # Create trainable agent
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=tau_config,
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # Note: The sample.prompt field contains the task index for repeatability
    interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index, args)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
