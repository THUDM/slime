# Agentic RL

slime supports agentic RL by letting a custom generate function run an agent and
return trainable `Sample`s. The agent may speak in strings/messages, but the
training target stays token based.

## Basic Pattern

- Use `--custom-generate-function-path` to run the agent or environment for one
  dataset row.
- Fill `tokens`, `response_length`, `loss_mask`, `reward`, and `status`, or use
  `--custom-rm-path` to compute reward separately.
- Return `list[Sample]` when one rollout is split into subagent, compaction, or
  final segments. Give sibling samples the same `rollout_id`.
- Use `--rollout-function-path` only when the default rollout orchestration is
  not enough.

## Adapters

slime includes protocol adapters for existing agent runtimes:

- `slime.agent.adapters.anthropic`: Anthropic Messages API, used by Claude Code
  style agents.
- `slime.agent.adapters.openai`: OpenAI Chat Completions and Responses APIs,
  used by OpenAI SDK / OpenAI Agents SDK style clients.

Adapters follow a simple contract: message history in, sampled tokens out. They
render the chat template, call SGLang with `input_ids` and `return_logprob=True`,
and export the returned token ids/logprobs as training segments. They do not
re-tokenize response text to recover the training target.

For multi-turn agents, use a stable `session_id`. The adapters pass it as
`X-SMG-Routing-Key` so SGLang can route one session to the same worker and reuse
prefix cache.

## Example

[`examples/coding_agent_rl`](../_examples_synced/coding_agent_rl/README.md)
shows a full SWE-style loop: boot a sandbox, run Claude Code through the
Anthropic adapter, capture a `git diff`, run tests for reward, and export the
recorded token trajectory.

Serving tips: use `--router-policy consistent_hashing` for multi-turn agents,
consider PD disaggregation for long contexts, and use `--sglang-config` for
multi-model or heterogeneous serving.

See also: [custom generate](customization.md#2-custom-generate-function---custom-generate-function-path),
[custom reward](customization.md#3-reward-model---custom-rm-path), and
[SGLang Config](../advanced/sglang-config.md).
