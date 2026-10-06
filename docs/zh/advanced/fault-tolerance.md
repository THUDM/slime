# 容灾

slime 会检查 SGLang 推理引擎是否正常工作，移除失效的引擎，并在下次更新权重前重启它们。使用 Straw 传输或保存 rollout 调试数据时，还支持在 Megatron 训练失败后保留推理集群，调整训练配置，再继续训练。

这些能力默认对 slime 启动的推理集群生效，不需要传入 `--use-fault-tolerance`。该参数保留用于兼容旧命令；使用外部推理集群时，仍按原来的策略处理。

## Megatron 失败后如何继续训练

这个恢复流程要求 **Ray 集群仍在运行**。训练入口进程或 `RolloutManager` 退出，不会销毁 SGLang 引擎和路由器；重新提交训练任务后，slime 会找到原来的推理集群并继续使用。整个流程不需要修改 SGLang，也不会自动重试训练任务。

### 首次启动时保留恢复数据

以下两种方式任选其一。选择后，slime 会自动保存恢复所需的数据，不依赖 `--use-fault-tolerance`。

使用 Straw 保存 rollout 和训练数据：

```bash
--rollout-data-transport straw --rollout-data-dir /shared/my-run/queue
```

或者保存 rollout 调试数据，路径中必须包含 `{rollout_id}`：

```bash
--save-debug-rollout-data '/shared/my-run/rollout_{rollout_id}.pt'
```

通过 `--save` 和 `--save-interval` 定期保存模型 checkpoint。保存间隔越长，失败后需要重新训练的批次越多，也需要保留更多恢复数据。

恢复模式要求使用 `torch_dist` checkpoint，并保存优化器和随机数状态，因此不能设置 `--no-save-optim` 或 `--no-save-rng`。slime 会自动启用支持跨并行配置重新分片的优化器保存格式。

### 失败后重新提交

例如，Megatron 因 OOM 退出后：

1. 等待失败的训练任务结束。
2. 调整 TP/CP/EP、microbatch 大小或每张卡的 token 上限。
3. 向**同一个 Ray 集群**重新提交 `train.py`。

重新提交时，模型、rollout 配置、global batch size 和会话标识要保持一致。训推共置（colocate）时，训练进程需要放得进原先分配的 GPU 资源；训推分离时，可以重新分配训练侧资源，推理侧 GPU 不会移动。

两次提交之间不要停止 Ray、重建推理服务容器，或执行 `pkill sglang` 等清理命令。`slime.utils.external_utils.command_utils.execute_train` 会保留正在运行的 Ray head 和 SGLang；如果使用的 shell 脚本每次启动都会清理进程，重启时需要跳过这一步。训练正常结束后，slime 会释放保留的会话及其资源。

### 从哪一步恢复

新训练进程会加载最近一次成功保存的模型、优化器和随机数状态，并重新训练该 checkpoint 之后的批次。已经训练完成、但模型更新尚未保存的批次也会重放；如果还没有 checkpoint，则从最初的模型开始重放。

保留的数据包括生成结果、token 和 reward 后处理结果。修改训练并行配置后，slime 会重新分片这些数据，不会重新生成样本，也不会再次执行 reward 后处理。恢复数据会保留到对应的模型 checkpoint 提交成功，或训练正常结束。推理侧的权重版本号在重启后继续递增。

Megatron 会在并行布局兼容时恢复随机数状态；TP/PP 改变时会重新初始化。因此，改变并行布局后可以继续训练，但不保证结果逐比特一致。

## 如何找到原来的推理集群

slime 使用具名 Ray actor 管理推理集群。新任务根据会话名称找到该 actor，由它返回已有的路由器和引擎信息，不需要扫描路由器进程。

可以用 `--rollout-session-id` 显式指定会话标识。未指定时，按以下顺序选择标识来源，再计算哈希得到稳定的名称：

1. 使用 Straw 时，取存储目录的绝对路径和 `--rollout-queue-run-id`。
2. 否则，取 `--save-debug-rollout-data` 路径模板的绝对路径。
3. 未保存调试数据时，取 `--save` 目录的绝对路径。
4. 以上都没有时，取模型和 rollout 配置。

同一任务的两次提交需要使用相同标识；独立任务应使用不同标识，避免接入同一个推理集群。

代码中有三个组件负责恢复：

- `ServingCluster` 管理路由器、SGLang 引擎、GPU 资源、Straw 队列控制器和权重更新锁。它是具名的 detached Ray actor，不会随创建它的训练任务退出。
- `RolloutManager` 负责生成、读取数据、转换样本和划分训练数据。它可以继续使用，也可以在退出后重建；重建不会销毁 `ServingCluster` 持有的资源。
- `TrainingRecovery` 保存恢复记录，包括 checkpoint 边界、数据源读取进度和训练批次。读取进度与已接收批次在同一次写入中保存，转换后的批次则在 DP 分片前保存，避免重启时跳过数据或受旧并行配置限制。

如果 `RolloutManager` 还活着，它会暂停接收新的生成任务。如果它已经退出，新实例会接回 `ServingCluster`，恢复数据源进度，并由队列控制器阻止旧读取进程继续取任务。已经完成、但尚未交给训练的预取结果仍可使用。训练进程也登记在 `ServingCluster` 中，因此 manager 退出后仍能清理旧训练进程。

## 推理引擎的健康检查与重启

rollout 过程中，slime 定期请求 SGLang 的 `/health_generate` 接口，检查引擎是否正常响应。rollout 结束时还会立即检查一次，**不受后台检查间隔和首次等待时间限制**。

同步 rollout 在发送中止生成和等待请求结束的控制命令前，会先从路由器中移除失效的服务。返回训练前，`ServingCluster` 会再次检查引擎，注销失效服务并清除对应的 Ray actor 引用，避免后续显存卸载或权重更新请求访问已经停止的引擎。HTTP 请求和 Ray 调用都有超时限制。

缺失的引擎在下一次更新权重前重启，随后加载训练侧的权重。后台检查和 rollout 收尾检查使用同一套故障处理流程；更新权重或调整显存占用时会暂停后台检查。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rollout-health-check-first-wait` | `0` 秒 | 每次恢复后台检查后，先等待这段时间，给模型预热和算子编译留出时间。rollout 收尾检查不受影响。 |
| `--rollout-health-check-interval` | `600` 秒 | 后台健康检查的间隔。 |
| `--rollout-health-check-timeout` | `30` 秒 | 单次健康检查的超时，也限制等待 Ray 健康检查调用的时间。 |

例如，大 MoE 模型需要较长的预热时间时，可以设置：

```bash
--rollout-health-check-first-wait 600 \
--rollout-health-check-interval 10 \
--rollout-health-check-timeout 5
```

如果负载高峰导致健康检查误判，可以增大超时。如果引擎在权重更新后反复失败，应检查 SGLang 日志和最近保存的 rollout 数据。

## 分开调试推理和训练

保存 rollout 数据后，可以固定训练输入，单独排查训练问题：

- `--debug-rollout-only`：只初始化推理侧，不训练；可配合保存参数检查生成和打分结果。
- `--save-debug-rollout-data /path/to/rollout_{rollout_id}.pt`：保存每轮 rollout 的样本。
- `--load-debug-rollout-data /path/to/rollout_{rollout_id}.pt`：加载已保存的样本用于训练，跳过 SGLang 初始化。
- `--debug-train-only`：只初始化训练侧，不启动 SGLang。

对于耗时较长的生成请求，可以用[请求追踪](../developer_guide/trace.md)查看生成、打分和模型调用各自的耗时，再用[性能分析](../developer_guide/profiling.md)定位瓶颈。多模型或 PD 分离部署的配置见 [SGLang 配置](sglang-config.md)。

## 恢复范围与限制

未使用 Straw、也未保存 rollout 调试数据时，仍可保留推理集群，但无法重放未保存到 checkpoint 的训练批次。

`RolloutManager` 重建支持内置数据源。自定义数据源需要提供兼容的 `state_dict` / `load_state_dict`，并让自己的队列控制器独立于 manager 存活；数据源构造函数和 rollout hook 的签名不变。

如果 `ServingCluster` 或整个 Ray 集群已经丢失，需要重新启动推理服务并从 checkpoint 恢复。集群抢占和节点丢失仍需要调度系统与 checkpoint 配合处理。

更多调试方式见[调试指南](../developer_guide/debug.md)，恢复测试的覆盖范围见[持续集成](../developer_guide/ci.md)。
