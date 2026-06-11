# 容灾

长时间 RL 任务的失败模式和短 SFT 任务很不一样：rollout engine 可能 hang，long-tail sample 可能拖住整个 round，serving state 也必须在权重更新后保持一致。slime 的容灾能力主要聚焦在 rollout 侧：让 rollout engine 可观测、可重启、可调试，同时不改变 training / rollout / Data Buffer 主路径。

开启容灾：

```bash
--use-fault-tolerance
```

## 当前覆盖范围

slime 当前提供 rollout-engine fault tolerance：

- 对 SGLang rollout server 做 health check；
- heartbeat timeout 后重启 rollout server；
- 重启后正确更新参数；
- 保存 debug rollout dump，用于不重新跑 rollout 的情况下 replay 训练侧问题；
- trace/profiling hook，用于检查 long-tail rollout 行为。

集群级抢占、trainer rank failure 和 full-job resume 仍应由集群调度器、Ray restart policy 和 slime checkpointing 共同处理。实际生产任务中，建议把 rollout fault tolerance、定期 checkpoint 和 debug dump 组合使用。

## Rollout Health Checks

rollout 过程中，slime 会定期向所有 SGLang server 发送 heartbeat 请求（`/health_generate`）。如果 heartbeat timeout，异常 SGLang server 会被停止。当前 rollout round 完成后，slime 会重启 server，并在其继续服务后续 rollout 请求前更新到正确参数。

主要参数：

- `--rollout-health-check-first-wait`：第一次 rollout 前等待多久再开始 heartbeat。大 MoE 模型首次运行可能需要 kernel compilation。默认 `300` 秒。
- `--rollout-health-check-interval`：heartbeat 间隔。默认 `10` 秒。
- `--rollout-health-check-timeout`：单次 heartbeat timeout。默认 `5` 秒。

示例：

```bash
--use-fault-tolerance \
--rollout-health-check-first-wait 600 \
--rollout-health-check-interval 10 \
--rollout-health-check-timeout 5
```

## Debug 与 Replay 路径

容灾只有在问题可复现时才真正有用。slime 提供 rollout-only 和 train-only 分离调试路径：

- `--debug-rollout-only`：只跑 rollout 并保存生成数据，不训练；
- `--save-debug-rollout-data /path/to/rollout_{rollout_id}.pt`：保存 rollout samples，后续可以检查或 replay；
- `--load-debug-rollout-data /path/to/rollout_{rollout_id}.pt`：加载已保存 rollout data，并跳过 SGLang 初始化；
- `--debug-train-only`：只跑训练侧逻辑，不跑 rollout。

这可以帮助定位问题属于 serving/rollout、data conversion、reward/verifier 逻辑，还是 Megatron training。

## 推荐生产模式

对于长时间任务：

1. 开启 `--use-fault-tolerance`。
2. 通过 `--save-interval` 定期保存 checkpoint。
3. 对新的 agentic 或 verifier-heavy workload 保存 rollout debug dump。
4. 使用 [Trace Viewer](../developer_guide/trace.md) 检查 long-tail samples 和 reward/model-call spans。
5. 使用 [Profiling](../developer_guide/profiling.md) 区分 rollout bottleneck 和 training bottleneck。
6. 对复杂 multi-model 或 PD topology，使用 [SGLang Config](sglang-config.md) 显式管理 SGLang 部署。

## 需要关注的信号

- 如果大 MoE 模型启动阶段 health check 失败，增大 `--rollout-health-check-first-wait`。
- 如果短暂负载高峰导致误判，增大 `--rollout-health-check-timeout`。
- 如果某个 server 在 weight sync 后反复重启，检查 SGLang log 和最近的 rollout debug dump。
- 如果失败发生在 trainer 而非 rollout，从 checkpoint 恢复，并用 debug replay 确认保存的 rollout batch 是否有效。

## 相关文档

- [Debugging](../developer_guide/debug.md)
- [Trace Viewer](../developer_guide/trace.md)
- [Profiling](../developer_guide/profiling.md)
- [CI](../developer_guide/ci.md)
