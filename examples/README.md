# Examples

These examples provide concrete examples to leverage slime in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

- **[eval_multi_task](./eval_multi_task)**: Example for supporting evaluation multiple tasks with different configs.
- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs with FSDP on a single-turn reasoning task using GRPO on the GEO3K dataset.
- **[geo3k_vlm_multi_turn](./geo3k_vlm_multi_turn)**: VLM multi-turn training (FSDP backend) on Geo3k dataset.
- **[low_precision](./low_precision)**: Examples of FP8 training and inference for improved throughput and stability.
- **[multi_agent](./multi_agent)**: Example of running multi-agent RL with `slime`.
- **[on_policy_distillation](./on_policy_distillation)**: Example implementation for on-policy distillation, extending the reinforcement learning pipeline to support teacher–student distillation directly within on-policy training.
- **[reproducibility](./reproducibility)**: Guides on achieving bitwise experiment reproduction using deterministic modes.
- **[retool](./retool)**: Demonstrates the retool functionality for tool-enabled language model generation.
- **[search-r1](./search-r1)**: A minimal reproduction of Search-R1, featuring multi-turn conversation and tool-calling.
- **[strands-agents](./strands-agents)**: Integration example with the Strands-Agents scaffolding framework.
- **[tau-bench](./tau-bench)**: Training in an agentic multi-turn tool use environment (Tau-bench).
- **[train_infer_mismatch_helper](./train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
- **[true_on_policy](./true_on_policy)**: Ensures strictly equal log probabilities between inference (SGLang) and training engines.
- **[true_on_policy_vlm](./true_on_policy_vlm)**: "True On-Policy" training demonstration for VLM (Qwen3-VL).
