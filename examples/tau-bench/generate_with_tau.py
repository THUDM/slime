import os
import logging
from typing import Any, Dict

import weave
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"], only tool-calling implemented for now
    "user_model": "gemini-2.0-flash-lite",  # Cheap Model for user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail, ["test"] for airline
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router", # Unused, required
    "model": "qwen2.5-3b", # Unused, reqired
    "user_model_provider": "gemini",
}
# Replace with your actual API key for user sim    
GEMINI_API_KEY = "NONE" 
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
tau_config = RunConfig(**TAU_CONFIGS)


@weave.op()
def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    status = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }.get(res.status)
    
    # Add debug logging
    logger.debug(f"res_to_sample: response_length="
                f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
                f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
                f"tokens_len={len(res.tokens) if res.tokens else 'None'}")
    
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
    
    # Ensure response_length is set correctly
    if hasattr(res, 'response_length'):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # Now loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback (not ideal)
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")
    
    return sample


@weave.op()
async def generate(args: Dict[str, Any], sample: Sample, sampling_params: dict):
    # Generate a full environment trajectory with Tau-Bench
    assert not args.partial_rollout, (
        "Partial rollout is not supported for this function at the moment."
    )
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction in task {task_index}")
    env = get_env(
        tau_config.env,
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
    # Samples are required to have prompt field. Instead of setting the actual sample, we set the index within the environment
    # for repeatability.
    res = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)
    res = res_to_sample(res, task_index)
    logger.info(f"Finished agent-environment interaction in task {task_index}")
    return res
