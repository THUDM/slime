# 同步、partial 与 distributed fully async rollout

这是三个独立决定：**何时交给训练一个 batch**、**未完成的生成能否续跑**、**rollout 状态存在哪里**。{ref}`实验室 <lab>`用时间线展示组合效果，并导出真实参数。

| 选择 | 训练边界上的生成行为 | 未完成工作去哪里 |
|---|---|---|
| 同步基线 | 完成目标 batch，训练，更新权重 | 实验室基线不保留 partial |
| 同步 + partial rollout | 超采样，凑齐完整且被接受的组后停止剩余任务并训练 | 保存前缀，之后继续生成 |
| Fully async + object-store | 管理进程内的后台 worker 跨调用维持队列 | 返回的 aborted 组回到 data buffer |
| Fully async + straw | 符合条件的 Ray 节点各有生成进程，向全局 batch collector 供数 | 持久 tasks、partial/ready 组与打包张量 |

独立训推 GPU 使生成与训练可以重叠。Fully async 在权重发布前后仍需暂停接收新任务、协调进行中的工作，并没有消除所有同步点。straw 也可以搭配同步训练。

(partial)=
## Partial rollout 保存了什么？

A 组很快完成，B 组只生成了一段前缀。当**完整且被接受的组**已足够组成目标 batch 时，同步 sampler 可以中断剩余请求、训练、更新权重，然后从前缀继续 B。Partial rollout 复用已生成的工作，不是把未完成回复直接当作完整奖励样本训练。

```{mermaid}
sequenceDiagram
    participant S as SGLang
    participant Q as Buffer / straw
    participant T as Trainer
    S->>S: v0 下超采样多组
    S->>T: 目标数量的完整且被接受的组
    S->>Q: 未完成的 B：tokens + sampler metadata
    T->>S: 训练后发布 v1
    Q->>S: 从保存的前缀续跑 B
    S->>T: 完成的 B，可能跨 v0 / v1
```

实验室导出 `--partial-rollout`；同步配方还设置 `--over-sampling-batch-size 16`，目标 `--rollout-batch-size 8`。多采样一些组，才有机会先凑齐 batch，将慢组留下续跑。标准 sampler 按组判断完成，组内一个慢回答可能拖住该组；如果没有多余的进行中任务，可保留的 partial 可能很少。

假设前 $k$ 个 token 使用 $v_0$，后续使用 $v_1$，实际行为分布是：

$$
q(y\mid x)=\prod_{t=1}^{k}q_{v_0}(y_t\mid x,y_{<t})
\prod_{t=k+1}^{T}q_{v_1}(y_t\mid x,y_{<t}).
$$

每个 sampled token 的 log-prob 属于真正生成它的策略，R3 routes 与 SC sampler heads 也必须与这些 token 对齐。Token 级重要性比使用 $p_\theta(y_t\mid h_t)/q_{v(t)}(y_t\mid h_t)$，不能用一个当前 sampler 替代整条混合版本回复。见[重要性权重推导](policy-mismatch.md#tis)。

`--mask-offpolicy-in-partial-rollout` 在续跑时将已有前缀的 loss mask 置零。令 $t\le k$ 时 $m_t=0$、之后 $m_t=1$，token 目标变为 $\sum_t m_t\ell_t$，再按选择的规则归一化。它去掉旧 token 的 loss 贡献，但后续 token 仍条件于旧前缀，因此不会恢复完整的 on-policy 轨迹分布，也会减少每个生成 token 对应的学习信号。

自定义 agent hook 需要正确保留续跑 tokens、masks 和 metadata，并实现兼容的 abort 行为。保存工具调用的文本，不代表外部工具的副作用自动可以恢复。具体见[标准 sampler](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py)与 [hook 契约](../get_started/customization.md)。

(distributed)=
## 从本地持续队列到分布式 producers

两条异步路径都使用 `slime.rollout.fully_async_rollout.generate_rollout_fully_async`。默认 object-store 数据源选择管理进程内的后台 worker；选择 straw 后数据源提供 queue readers，转而使用 `DistributedRollout`。每个存活且具备 CPU 资源的 Ray 节点获得一个生成 actor。这些 CPU producers 向 SGLang 发请求，并不是额外的模型副本。

```{mermaid}
flowchart LR
    Q[straw prompt / partial tasks] --> W0[节点 0 producer]
    Q --> W1[节点 1 producer]
    Q --> WN[节点 N producer]
    W0 --> S[SGLang GPU engines]
    W1 --> S
    WN --> S
    S --> R[持久化完整组与张量]
    R --> B[全局 batch collector]
    B --> T[Megatron]
    T -->|权重与任务接收协调| S
```

快 worker 可以组成下一批，不必等待每个 worker 都完成。已完成结果的预取量有限，训练慢时通过 backpressure 控制生成。权重更新时暂停接收新任务、保存结果，再恢复。Aborted 组回到队列，不会当作完整训练数据。存储与续跑仍是独立概念：同步的 partial flag 控制 batch 边界上保存未完成任务；异步 worker 自己处理 aborted 组。

设 $C$ 为 `sglang_server_concurrency`，$E$ 为逻辑 rollout 引擎数，$G$ 为每 prompt 回复数，$W$ 为符合条件的 worker 节点数。分布式实现把

$$K=\left\lfloor\frac{CE}{G}\right\rfloor$$

个组槽位按商和余数分给各 worker。每个 worker 至少得到一组，因此要求 $CE\ge WG$。这是并发预算，不是吞吐；引擎延迟、工具、奖励计算与存储都会影响速度。引擎计数以 slime serving 配置为准，包括 PD 分组，不能用 GPU 数直接替代。见 [worker 选择](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py)和[分布式调度](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_distributed.py)。

## 样本过期与吞吐

当前 serving 版本为 $v$，组内最早生成 token 使用版本 $u$，则 staleness 为 $v-u$。刚完成的回复也可能因前缀很早生成而已经过期。straw 优先取完整组，再 partial 组，再新 prompt；各阶段内优先较老的 token 权重版本，同版本按 FIFO。排序不会自动丢弃旧样本。

若 producer 每秒稳定产出 $\lambda$ 个被接受的组，训练每秒消费 $\mu$ 个，吞吐上限为 $\min(\lambda,\mu)$。当 $\lambda>\mu$，无限队列约以 $\lambda-\mu$ 的速度增长，需要 backpressure。稳定队列按 Little 定律有 $L=\lambda W_q$，其中 $L$ 是平均排队组数、$W_q$ 是平均队内时间。只有其他环节吃得下，增加 producer 才有用。

同时观察 `staleness/mean`、`staleness/max`、有效组吞吐、reward、mask/clip 比例及训推 log-prob 差异。Off-policy 修正不保证任意延迟都无害。[系统耗时推导](rl-systems.md#async)与 [PipelineRL 论文](https://arxiv.org/abs/2509.19128)可作为流水线背景，实际行为以上述 slime 实现为准。

## straw 持久化与恢复

实验室要求填写共享 JuiceFS 目录、run ID 和部署声明，并导出：

```bash
--rollout-data-transport straw \
--rollout-data-dir /shared/juicefs/jobs/my-run/rollout_data \
--rollout-queue-run-id my-run \
--rollout-storage-profile juicefs \
--rollout-storage-declaration /shared/juicefs/deployment.json
```

各节点安装同版本 `straw-queue`，将经过验证的存储挂载到相同绝对路径。声明文件记录挂载和 durability 属性，不负责创建挂载。straw 保存 prompt tasks、partial/ready 组、训练 batch 与打包张量；租约使失联 worker 的未完成工作可再次被领取。R3 与 SC 张量记录可以在阶段间复用，不必复制字节。

straw 短配方每轮保存。恢复需要完成的**模型/优化器与 rollout 状态共同 checkpoint**，以及引用的数据池。它不恢复 GPU KV cache，也不保证后续随机生成完全相同。数据集、tokenizer、run ID、存储设置和 fully async worker 拓扑需与 checkpoint 兼容。整任务重启前停止上一任务，coordinator 不会自动 failover；保留 serving 的恢复见[故障容错](fault-tolerance.md)。完整设置、leases、GC、分支与回退语义见 [straw 指南](straw.md)及[文件系统契约](https://github.com/zhuzilin/straw/blob/main/docs/FILESYSTEM.md)。

连续异步路径不支持内联评估与 `--rollout-all-samples-process-path`，请独立评估。straw 不支持 `--buffer-filter-path`，采用自己的持久排序。Replay 见[调试](../developer_guide/debug.md)，改变并发前先做 [profiling](../developer_guide/profiling.md)。
