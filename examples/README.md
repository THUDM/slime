# Examples

Concrete example workflows built on top of `slime`.

## Directory Structure

- **[common](./common)**: Shared helpers used by the multidomain and MOPD examples.
- **[MOPD](./MOPD)**: Multi-objective policy distillation examples.
- **[multidomain_v2](./multidomain_v2)**: Multidomain RL training and submit entrypoints.
- **[eval_multi_task](./eval_multi_task)**: Example for supporting evaluation multiple tasks with different configs.
- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs on a single-turn reasoning task using GRPO on the GEO3K dataset.
- **[geo3k_vlm_multi_turn](./geo3k_vlm_multi_turn)**: VLM multi-turn training on Geo3k dataset.
- **[multi_agent](./multi_agent)**: Example of running multi-agent RL with `slime`.
- **[on_policy_distillation](./on_policy_distillation)**: Example implementation for on-policy distillation, extending the reinforcement learning pipeline to support teacher–student distillation directly within on-policy training.
- **[retool](./retool)**: Demonstrates the retool functionality for tool-enabled language model generation.
- **[search-r1](./search-r1)**: A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling.
- **[strands_sglang](./strands_sglang)**: Integration example with the Strands-Agents scaffolding framework.
- **[tau-bench](./tau-bench)**: Training in an agentic multi-turn tool use environment (Tau-bench).
- **[train_infer_mismatch_helper](./train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
