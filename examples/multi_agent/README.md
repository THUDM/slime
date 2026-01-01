# Multi-Agent RL

This directory provides an example of running multi-agent reinforcement learning (RL) with slime.

## Environment Setup

The environment setup is identical to the standard RL setup used in slime.

## Running the Script

You can either define your own multi-agent system or use the provided default configuration.

```python
MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}
```

To start a run, execute:

```bash
cd slime/
bash examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh
```

## New Arguments

- Specify the agent rollout function with the `--custom-generate-function-path` argument.
- Set the `--rollout-max-context-len` argument according to your model’s context window.

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-context-len 16384
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)
```