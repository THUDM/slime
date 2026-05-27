# Agentic RL Training Roadmap

slime is not limited to single-turn RL. Its main advantage for agentic training is the combination of high-performance training, SGLang rollout serving, and pluggable data-generation interfaces. This makes it suitable for multi-turn tool use, sandbox interaction, subagent branches, context compaction, and test-based rewards.

This page is a roadmap: use it to decide which docs and examples to read when plugging an agent workflow into slime.

## Where To Start

| Goal | Recommended entry point |
| :--- | :--- |
| Run a custom agent loop, tool calls, RAG, browser/terminal/sandbox interaction for each sample | [`--custom-generate-function-path`](customization.md#2-custom-generate-function---custom-generate-function-path), [writing a custom generation function](quick_start.md#writing-custom-generation-function) |
| Implement verifier rewards, test-based rewards, environment success checks, or an external reward service | [`--custom-rm-path`](customization.md#3-reward-model---custom-rm-path), [writing a custom reward function](quick_start.md#writing-custom-reward-function) |
| Return multiple training samples from one prompt, such as subagent, multi-agent, or context-compaction segments | [fan-out return from custom generate](customization.md#returning-multiple-training-samples-for-one-prompt), [`examples/multi_agent`](../_examples_synced/multi_agent/README.md) |
| Avoid blocking training on long-tail agent rollouts | [`examples/fully_async`](../_examples_synced/fully_async/README.md) |
| Study a full end-to-end agent example with sandboxing, real code edits, and test-based grading | [`examples/coding_agent_rl`](../_examples_synced/coding_agent_rl/README.md) |
| Improve SGLang serving throughput for multi-turn agents | [PD Disaggregation](../advanced/pd-disaggregation.md), [SGLang Config](../advanced/sglang-config.md) |
| Enable SGLang optimization flags, router policies, or multi-model serving | [How to Use SGLang](usage.md#how-to-use-sglang), [SGLang Config](../advanced/sglang-config.md), [Speculative Decoding](../advanced/speculative-decoding.md), [Low Precision Training](../advanced/low-precision.md) |

## Recommended Integration Pattern

Most agentic RL tasks should start with `--custom-generate-function-path`. This function converts one agent execution into slime-trainable `Sample` objects: fill `tokens`, `response_length`, `loss_mask`, and `status`, then either fill `reward` directly or let `--custom-rm-path` compute it.

If one prompt rollout corresponds to one training sample, return a single `Sample`. If one rollout splits into multiple trainable segments, such as subagent trajectories, main-agent continuations, or pre/post-compaction segments, return `list[Sample]` and set the same `rollout_id` on all sibling samples. slime then keeps those samples together for train-step splitting and loss aggregation instead of counting them as independent rollouts.

Reach for `--rollout-function-path` only when you need to replace the whole rollout orchestration. Common reasons include custom data-source scheduling, cross-rollout background queues, fully asynchronous generation, or workflows that cannot fit the default `sglang_rollout` prompt-by-sample structure.

## Agent Serving And Performance

Agentic rollouts tend to depend more heavily on serving configuration than ordinary single-turn generation: contexts are longer, requests are multi-turn, latency has a heavier tail, and the workflow may need actor, reference, reward, or tool-side models at the same time.

- Regular SGLang server arguments are passed as `--sglang-*`. For example, SGLang's `--context-length` becomes `--sglang-context-length`, and `--mem-fraction-static` becomes `--sglang-mem-fraction-static`.
- Router arguments are passed as `--router-*`. For multi-turn agents, consider `--router-policy consistent_hashing` so requests for the same `sample.session_id` go to the same worker and improve prefix-cache hit rate. See [Session-Affinity Routing for Multi-Turn Agents](../advanced/sglang-config.md#session-affinity-routing-for-multi-turn-agents).
- Use `--sglang-config` for more complex topologies: PD disaggregation, multi-model serving, heterogeneous server groups, and per-group SGLang overrides.
- For multi-turn or agentic RL, evaluate PD disaggregation. Prefill and decode have different workload shapes, and separating them makes it easier to scale each resource independently.
- For rollout-throughput optimization, also see [Speculative Decoding](../advanced/speculative-decoding.md) and [Low Precision Training](../advanced/low-precision.md).

## Reference Example

The full coding-agent example is [`examples/coding_agent_rl`](../_examples_synced/coding_agent_rl/README.md). It shows an end-to-end agent RL setup that is close to a real software-engineering workflow: each sample boots an isolated sandbox, the agent uses tools to edit code, the rollout captures a `git diff`, and a clean sandbox runs the tests to produce the reward.

This example also demonstrates agent fan-out training. Its middleware splits one trajectory into `subagent`, `wipe` (the chain frozen before compaction), and `final` segments. `generate()` returns `list[Sample]`, and all segments share the same `rollout_id`.

For smaller starting points, see [`examples/search-r1`](../_examples_synced/search-r1/README.md) for multi-turn tool use, [`examples/retool`](../_examples_synced/retool/README.md) for tool-augmented generation, and [`examples/multi_agent`](../_examples_synced/multi_agent/README.md) for the multi-agent pattern.
