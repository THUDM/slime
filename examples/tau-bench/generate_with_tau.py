import os
from typing import Any, Dict

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

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
GEMINI_API_KEY = "KEY HERE" 
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult) -> Sample:
    status = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }.get(res.status)
    return Sample(
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )


async def generate(args: Dict[str, Any], sample: Sample, sampling_params: dict):
    # Generate a full environment trajectory with Tau-Bench
    assert not args.partial_rollout, f"Partial rollout is not supported for this function at the moment."
    env = get_env(
        tau_config.env,
        user_strategy=tau_config.user_strategy,
        user_model=tau_config.user_model,
        user_provider=tau_config.user_model_provider,
        task_split=tau_config.task_split,
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
    task_index = int(sample.prompt)
    print(f"Starting agent-environment interaction in task {sample.prompt}")
    res = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)
    print(f"Finished agent-environment interaction in task {sample.prompt}")
    return res_to_sample(res)
