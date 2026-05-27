# Agentic RL 训练路线图

slime 的核心定位并不只是跑单轮 RL，而是把高性能训练、SGLang rollout serving、以及可插拔的数据生成接口组合起来，支持 agent 时代常见的多轮工具调用、sandbox 交互、subagent 分支、context compact 和 test-based reward。

这篇文档是一个导航页：当你要把 agent workflow 接进 slime 时，先用它判断该看哪些文档和例子。

## 从哪里开始

| 目标 | 推荐入口 |
| :--- | :--- |
| 给每条 sample 跑自定义 agent loop、tool call、RAG、browser/terminal/sandbox 交互 | [`--custom-generate-function-path`](customization.md#2-自定义生成函数---custom-generate-function-path)、[编写自定义生成函数](quick_start.md#编写自定义生成函数) |
| 做 verifier reward、test-based reward、环境成功判定或外部 reward 服务 | [`--custom-rm-path`](customization.md#3-奖励模型---custom-rm-path)、[编写自定义奖励函数](quick_start.md#编写自定义奖励函数) |
| 一个 prompt 会产生多个训练样本，例如 subagent、multi-agent、context compact | [custom generate 的 fan-out 返回](customization.md#一个-prompt-产生多个训练样本)、[`examples/multi_agent`](../_examples_synced/multi_agent/README.md) |
| agent rollout 有长尾耗时，希望训练不要被最慢样本卡住 | [`examples/fully_async`](../_examples_synced/fully_async/README.md) |
| agent 需要 sandbox、真实代码修改、测试验证和完整端到端样例 | [`examples/coding_agent_rl`](../_examples_synced/coding_agent_rl/README.md) |
| 多轮 agent 需要更高 SGLang serving 吞吐 | [PD 分离](../advanced/pd-disaggregation.md)、[SGLang Config](../advanced/sglang-config.md) |
| 想开启 SGLang 的优化 flag、router 策略或多模型 serving | [sglang 使用方法](usage.md#sglang-使用方法)、[SGLang Config](../advanced/sglang-config.md)、[投机采样](../advanced/speculative-decoding.md)、[低精度训练](../advanced/low-precision.md) |

## 推荐接入方式

大多数 agentic RL 任务应该先从 `--custom-generate-function-path` 开始。这个函数负责把一次 agent 运行转换成 slime 可训练的 `Sample`：填好 `tokens`、`response_length`、`loss_mask`、`status`，并在需要时填好 `reward` 或交给 `--custom-rm-path` 计算。

如果一次 prompt rollout 只对应一个训练样本，返回一个 `Sample` 即可。如果一次 rollout 会拆成多个训练片段，例如 subagent 轨迹、main-agent 轨迹、compact 前后的片段，则返回 `list[Sample]`，并给这些 sibling samples 设置相同的 `rollout_id`。这样 slime 会在训练 step 切分和 loss 聚合时把它们视作同一次 rollout，而不是重复计数。

只有当你需要替换整个 rollout 编排时，才优先考虑 `--rollout-function-path`。典型场景包括：自定义数据源调度、跨 rollout 的后台队列、完全异步生成，或者默认 `sglang_rollout` 的 prompt × sample 结构已经无法表达你的 workflow。

## Agent Serving 与性能配置

agentic rollout 往往比普通单轮 generation 更依赖 serving 配置：上下文更长、多轮请求更多、请求时长分布更重尾，并且可能同时需要 actor、reference、reward 或工具侧模型。

- 常规 SGLang server 参数通过 `--sglang-*` 传入。例如 `--context-length` 在 slime 中写作 `--sglang-context-length`，`--mem-fraction-static` 写作 `--sglang-mem-fraction-static`。
- router 参数通过 `--router-*` 传入。多轮 agent 可以考虑 `--router-policy consistent_hashing`，让同一个 `sample.session_id` 的多轮请求落到同一个 worker，提高 prefix cache 命中率。详见 [多轮 Agent 的会话亲和路由](../advanced/sglang-config.md#多轮-agent-的会话亲和路由)。
- 更复杂的拓扑使用 `--sglang-config`：它可以描述 PD 分离、多模型 serving、异构 server groups，以及每组不同的 SGLang overrides。
- 多轮或 agentic RL 通常建议评估 PD 分离。prefill 与 decode 的负载形态不同，拆开后更容易分别扩展资源。
- 对 rollout 吞吐敏感时，可以继续查看 [投机采样](../advanced/speculative-decoding.md) 和 [低精度训练](../advanced/low-precision.md)。

## 参考样例

完整的 coding-agent 样例见 [`examples/coding_agent_rl`](../_examples_synced/coding_agent_rl/README.md)。它展示了一个比较接近真实 agent RL 的端到端形态：每条 sample 启动独立 sandbox，agent 使用工具修改代码，生成 `git diff`，再在干净 sandbox 里跑测试得到 reward。

这个样例也演示了 agent fan-out 的训练方式：middleware 会把 trajectory 切成 `subagent`、`wipe`（compact 前被冻结的链）和 `final` 等片段，`generate()` 返回 `list[Sample]`，并让这些片段共享同一个 `rollout_id`。

如果你只需要更轻量的入门例子，可以先看 [`examples/search-r1`](../_examples_synced/search-r1/README.md) 的多轮工具调用、[`examples/retool`](../_examples_synced/retool/README.md) 的工具增强生成、以及 [`examples/multi_agent`](../_examples_synced/multi_agent/README.md) 的多 agent 模式。
