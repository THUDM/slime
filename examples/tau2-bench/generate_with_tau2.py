"""
Tau2-Bench integration for slime Training.

Configure the domain/task split below, point slime at this file via
--custom-generate-function-path examples.tau2-bench.generate_with_tau2.generate
"""

import logging
import os
from typing import Any

from slime.utils.types import Sample

from .trainable_agent import Tau2TrainableAgent, res_to_sample

logger = logging.getLogger(__name__)

# Base configuration (edit here as needed).
TAU2_CONFIGS: dict[str, Any] = {
    "domain": "airline",  # tau2 domain: airline | retail | telecom | mock
    "task_split": "train",  # task split within the domain
    "max_steps": 100,  # safety cap on interaction steps
    # Explicit gemini provider prefix to avoid Vertex ADC path.
    # "user_llm": "gemini/gemini-2.5-flash-lite",
    "user_llm": "gpt-4.1",
    "user_llm_args": {},  # will inject api_key below
    "solo_mode": False,  # set True to disable user simulator
}

# Replace with your actual API key for user simulator (LiteLLM)
API_KEY = "NONE"
if API_KEY == "NONE":
    API_KEY = os.getenv("OPENAI_API_KEY")
# Also pass through args to force gemini path
TAU2_CONFIGS["user_llm_args"] = {"api_key": API_KEY}


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for tau2."

    agent = Tau2TrainableAgent(
        args=args,
        sampling_params=sampling_params,
        domain=TAU2_CONFIGS["domain"],
        task_split=TAU2_CONFIGS["task_split"],
        max_steps=TAU2_CONFIGS["max_steps"],
        user_llm=TAU2_CONFIGS["user_llm"],
        user_llm_args=TAU2_CONFIGS.get("user_llm_args") or {},
        solo_mode=TAU2_CONFIGS["solo_mode"],
    )

    task_id, task_index = agent._resolve_task_id(sample.prompt)  # noqa: SLF001 - simple helper
    logger.info("Starting tau2 rollout for task_id=%s (index=%s)", task_id, task_index)

    interaction_result = await agent.run_episode(task_id)
    result_sample = res_to_sample(interaction_result, task_index)

    logger.info("Finished tau2 rollout for task_id=%s", task_id)
    return result_sample
