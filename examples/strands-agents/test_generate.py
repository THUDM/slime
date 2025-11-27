"""
Simple test script for generate function with Strands Agent

# launch the server
nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8000 \
    --host 0.0.0.0 \
    --tool-call-parser qwen \
    --tp-size 8 \
    --mem-fraction-static 0.9 &

# test the generate function
python test_generate.py
"""

from argparse import Namespace
import asyncio
import json
import random

from generate_with_strands import generate, reward_func
from slime.utils.types import Sample


DEFAULT_DATA_PATH = "/root/data/dapo-math-17k.jsonl"


async def main():
    data_path = input(f"Enter path to .jsonl data file (default: {DEFAULT_DATA_PATH}): ").strip() or DEFAULT_DATA_PATH
    # Load one random sample
    with open(data_path) as f:
        samples = [json.loads(line) for line in f]

    sample = random.choice(samples)
    sample = Sample(prompt=sample["prompt"], label=sample["label"])

    # Simple args for Qwen3-8B server
    args = Namespace(
        model_name="Qwen/Qwen3-8B",
        hf_checkpoint="Qwen/Qwen3-8B",
        sglang_router_ip="localhost",
        sglang_router_port=8000,
        partial_rollout=False,
        sglang_server_concurrency=32,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=8,
        rollout_temperature=1.0,
        rollout_top_p=0.7,
        rollout_top_k=-1,
        rollout_max_response_len=20480,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        rollout_seed=42,
        n_samples_per_prompt=1,
    )
    sampling_params = {"max_new_tokens": 20480, "temperature": 1.0, "top_p": 1.0}

    print("Testing generate function with Qwen3-8B...")

    # Generate
    sample = await generate(args, sample, sampling_params)
    print(f"\nResponse:\n{sample.response}\n")
    print(f"Status: {sample.status}")
    print(f"Response Length: {sample.response_length}")
    print(f"Tool Calls: {getattr(sample, 'tool_call_count', 0)}")

    # Compute reward
    if not sample.status == Sample.Status.ABORTED:
        reward = await reward_func(args, sample)
        print(f"Ground Truth: {sample.label}\n")
        print(f"Score: {reward['score']}")
        print(f"Predicted: {reward['pred']}")
        print(f"Correct: {'✓' if reward['score'] > 0 else '✗'}")


if __name__ == "__main__":
    asyncio.run(main())
