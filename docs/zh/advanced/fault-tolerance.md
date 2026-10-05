# 容灾

长时间 RL 任务的失败模式和短 SFT 任务很不一样：rollout engine 可能 hang，long-tail sample 可能拖住整个 round，serving state 也必须在权重更新后保持一致。slime 的容灾能力主要聚焦在 rollout 侧：让 rollout engine 可观测、可重启、可调试，同时不改变 training / rollout / Data Buffer 主路径。

开启容灾：

```bash
--use-fault-tolerance
```

## 当前覆盖范围

slime 提供 rollout-engine 容灾和 Megatron 手动重启恢复：

- 对 SGLang rollout server 做 health check；
- heartbeat timeout 后重启 rollout server；
- 重启后正确更新参数；
- 保存 debug rollout dump，用于不重新跑 rollout 的情况下 replay 训练侧问题；
- trace/profiling hook，用于检查 long-tail rollout 行为。
- 开启 Straw 传输或 debug rollout dump 时，Megatron 失败后保留 SGLang 集群，并重放训练数据。

集群级抢占和 Ray 集群丢失仍需集群调度器与 slime checkpointing 处理。保留的 serving 会话可以在同一个存活的 Ray 集群内跨训练 driver 失败继续使用。

## Megatron 手动重启

将 `--use-fault-tolerance` 与以下任意一种模式组合：

```bash
--rollout-data-transport straw --rollout-data-dir /shared/my-run/queue
```

或者：

```bash
--save-debug-rollout-data '/shared/my-run/rollout_{rollout_id}.pt'
```

Megatron OOM 后，等待失败的训练任务结束，调整训练配置，再向**同一个 Ray 集群**提交 `train.py`。可以修改 TP/CP/EP、microbatch 和 token 上限；colocate 模式下 trainer 需要保持在原 GPU placement 容量内，独立部署的 trainer 则可以单独重新分配训练 placement，不移动 rollout GPU。模型、rollout 配置、global batch size 和会话标识需要保持一致。Straw 用存储目录和 `--rollout-queue-run-id` 标识会话，debug 模式使用 dump 路径模板。独立的新训练任务应使用不同的会话标识。

保留的 rollout manager 会暂停 producer 接收新任务、释放失败的 trainer ranks，同时保留 SGLang 进程、router 和 rollout GPU placement。新 trainer 重新连接这些 engine，加载最近一次成功保存的模型、optimizer 和 RNG 状态，并重放该 checkpoint 之后的数据，包括已经训练成功但尚未保存模型更新的 batch。没有 checkpoint 时，从最初的模型开始重放。reward 后处理结果和 token 数据在 DP 分片之前保留，因此修改并行配置会重新分片，不会重新生成 samples 或再次运行 reward 后处理。

通过 `--save` 和 `--save-interval` 控制重放量及存储占用。恢复模式使用 `torch_dist` checkpoint，并自动开启可跨并行配置重新分片的 optimizer 保存格式。checkpoint 必须保存 optimizer 和 RNG 状态，因此不允许 `--no-save-optim` 和 `--no-save-rng`。Megatron 会在拓扑兼容时恢复 RNG，TP/PP 改变时则重新初始化 RNG，因此跨并行布局恢复不保证逐 bit 重现。debug dump 路径必须包含 `{rollout_id}`。恢复数据会保留到模型 checkpoint 提交成功或训练正常结束。serving 权重版本在 trainer 重启后继续递增。

两次尝试之间不要停止 Ray、重建 serving container 或执行 `pkill sglang` 等清理命令。`slime.utils.external_utils.command_utils.execute_train` 会在这些恢复模式下保留 Ray 和 SGLang；含无条件清理的 shell 启动脚本需要在重启时跳过清理。训练正常结束会释放保留的会话。这是手动重启流程，不会自动重试 trainer，也不需要修改 SGLang 本身。

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
- 如果 trainer 失败，修正配置后向保留的会话重新提交；如果 Ray 集群已丢失，则从持久 checkpoint 恢复，并用 debug replay 检查失败的 batch。

## 相关文档

- [Debugging](../developer_guide/debug.md)
- [Trace Viewer](../developer_guide/trace.md)
- [Profiling](../developer_guide/profiling.md)
- [CI](../developer_guide/ci.md)
