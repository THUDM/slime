import random

from transformers import AutoTokenizer

from slime.utils.misc import load_function
from slime.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}


async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(args, key, value)

    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)

    # All samples returned here come from the same rollout execution (the agent
    # system expanded one prompt-sample into multiple training samples). Stamp
    # the shared rollout_id so the per-rollout loss reducer counts this rollout
    # once instead of len(samples) times, and so the by-rollout step splitter
    # keeps the siblings in the same training step.
    for s in samples:
        s.rollout_id = sample.index

    random.shuffle(samples)

    return samples
