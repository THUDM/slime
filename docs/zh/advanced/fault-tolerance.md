# 容灾

长时间 RL 任务的失败模式和短 SFT 任务很不一样：rollout engine 可能 hang，long-tail sample 可能拖住整个 round，serving state 也必须在权重更新后保持一致。slime 的容灾能力主要聚焦在 rollout 侧：让 rollout engine 可观测、可重启、可调试，同时不改变 training / rollout / Data Buffer 主路径。

内部 serving 始终由独立的 detached owner 持有，并启用健康检查和异常 engine 恢复。以下开关保留用于兼容及原有 external serving 策略：

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

集群级抢占和 Ray 集群丢失仍需集群调度器与 slime checkpointing 处理。同一个存活的 Ray 集群内，训练 driver 或 rollout manager 退出后，serving 会话仍可继续使用；是否传入兼容开关不会改变内部 serving 的生命周期和健康检查策略。external cluster 路径保持原有行为。

## Megatron 手动重启

训练数据重放需要以下任意一种模式。这些模式始终启用恢复记录持久化，不依赖 `--use-fault-tolerance`。

```bash
--rollout-data-transport straw --rollout-data-dir /shared/my-run/queue
```

或者：

```bash
--save-debug-rollout-data '/shared/my-run/rollout_{rollout_id}.pt'
```

Megatron OOM 后，等待失败的训练任务结束，调整训练配置，再向**同一个 Ray 集群**提交 `train.py`。可以修改 TP/CP/EP、microbatch 和 token 上限；colocate 模式下 trainer 需要保持在原 GPU placement 容量内，独立部署的 trainer 则可以单独重新分配训练 placement，不移动 rollout GPU。模型、rollout 配置、global batch size 和会话标识需要保持一致。可以通过 `--rollout-session-id` 显式指定 serving 会话。未指定时，Straw 用存储目录和 `--rollout-queue-run-id` 标识会话，debug 模式使用 dump 路径模板，其他模式使用保存目录或模型与 rollout 配置。独立的新训练任务应使用不同的会话标识。

独立的 serving owner 持有 SGLang 进程、router、rollout GPU placement 和 Straw controller。manager 存活时会暂停 producer 接收新任务；manager 被杀后，重新提交会创建新 manager、接回原 owner，并读取原子写入的恢复记录。内置数据源恢复游标和 metadata；存活的 Straw controller 会隔离旧 reader，并接回已完成但尚未交给训练的预取结果。trainer ranks 登记在 serving owner 中，因此 manager 退出后仍能释放旧训练进程。新 trainer 重新连接这些 engine，加载最近一次成功保存的模型、optimizer 和 RNG 状态，并重放该 checkpoint 之后的数据，包括已经训练成功但尚未保存模型更新的 batch。没有 checkpoint 时，从最初的模型开始重放。reward 后处理结果和 token 数据在 DP 分片之前保留，因此修改并行配置会重新分片，不会重新生成 samples 或再次运行 reward 后处理。

通过 `--save` 和 `--save-interval` 控制重放量及存储占用。恢复模式使用 `torch_dist` checkpoint，并自动开启可跨并行配置重新分片的 optimizer 保存格式。checkpoint 必须保存 optimizer 和 RNG 状态，因此不允许 `--no-save-optim` 和 `--no-save-rng`。Megatron 会在拓扑兼容时恢复 RNG，TP/PP 改变时则重新初始化 RNG，因此跨并行布局恢复不保证逐 bit 重现。debug dump 路径必须包含 `{rollout_id}`。恢复数据会保留到模型 checkpoint 提交成功或训练正常结束。serving 权重版本在 trainer 重启后继续递增。

两次尝试之间不要停止 Ray、重建 serving container 或执行 `pkill sglang` 等清理命令。`slime.utils.external_utils.command_utils.execute_train` 每次启动都会保留运行中的 Ray head 和 SGLang；含无条件清理的 shell 启动脚本需要在重启时跳过清理。训练正常结束会释放保留的会话。这是手动重启流程，不会自动重试 trainer，也不需要修改 SGLang 本身。

没有 Straw 或 debug dump 时仍可保留 serving，但无法重放未保存的训练 batch。自定义数据源的构造函数和 rollout hook 签名保持不变；manager 重建支持内置数据源，自定义数据源需要提供兼容的 `state_dict` / `load_state_dict`，并保证自有 controller 的生命周期独立于 manager。serving owner 或整个 Ray 集群丢失时，需要从 checkpoint 冷启动。

## Rollout Health Checks

rollout 过程中，slime 定期向内部 SGLang server 发送 heartbeat 请求（`/health_generate`）。rollout 收尾时还会立即检查，不受检查间隔和首次等待限制。同步 rollout 在发 abort/drain 请求前先清理 router 中失效的 worker；serving owner 随后注销失效 engine 并移除 actor handle，避免 offload 等控制请求访问坏 server。HTTP 和 Ray 检查都有超时。重启仍在下一次权重更新前进行。

主要参数：

- `--rollout-health-check-first-wait`：resume 后后台检查的等待时间，供大 MoE 模型 kernel compilation 使用。收尾检查不受此等待限制。默认 `0` 秒。
- `--rollout-health-check-interval`：后台 heartbeat 间隔。默认 `600` 秒。
- `--rollout-health-check-timeout`：单次 heartbeat 或排队中的健康检查 RPC 超时。默认 `30` 秒。

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

1. 内部健康检查默认开启，按模型预热和响应时间调整 `--rollout-health-check-*`。
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
