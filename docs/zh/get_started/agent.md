# Agentic RL

slime 通过 custom generate function 接入 agentic RL。agent 侧可以使用字符串或消息协议，但训练目标必须保持 token based。

## 基本模式

- 用 `--custom-generate-function-path` 对一条数据运行 agent 或环境。
- 填好 `tokens`、`response_length`、`loss_mask`、`reward` 和 `status`；或者用 `--custom-rm-path` 单独计算 reward。
- 如果一次 rollout 会拆成 subagent、compact、final 等片段，返回 `list[Sample]`，并让 sibling samples 共享同一个 `rollout_id`。
- 只有默认 rollout 编排不够表达 workflow 时，才使用 `--rollout-function-path`。

## Adapters

slime 提供已有 agent runtime 可用的协议 adapter：

- `slime.agent.adapters.anthropic`：Anthropic Messages API，用于 Claude Code 风格 agent。
- `slime.agent.adapters.openai`：OpenAI Chat Completions 和 Responses API，用于 OpenAI SDK / OpenAI Agents SDK 风格 client。

adapter 的 contract 很简单：message history in，sampled tokens out。它会渲染 chat template，用 `input_ids` 和 `return_logprob=True` 调 SGLang，并把返回的 token ids/logprobs 导出为训练片段；不会从 response text 重新分词恢复训练目标。

多轮 agent 应使用稳定的 `session_id`。adapter 会把它作为 `X-SMG-Routing-Key` 传给 SGLang，让同一个 session 尽量落到同一个 worker，复用 prefix cache。

## 示例

[`examples/coding_agent_rl`](../_examples_synced/coding_agent_rl/README.md) 展示了完整 SWE 风格流程：启动 sandbox，通过 Anthropic adapter 跑 Claude Code，捕获 `git diff`，跑测试得到 reward，再导出记录下来的 token trajectory。

Serving 建议：多轮 agent 使用 `--router-policy consistent_hashing`；长上下文可以评估 PD 分离；多模型或异构 serving 用 `--sglang-config`。

相关文档：[custom generate](customization.md#2-自定义生成函数---custom-generate-function-path)、[custom reward](customization.md#3-奖励模型---custom-rm-path)、[SGLang Config](../advanced/sglang-config.md)。
