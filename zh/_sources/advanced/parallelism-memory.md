# 卡数、显存与并行方式

在{ref}`交互实验室 <lab>`的「布局」步骤中逐个点击图示。先确认各阶段能放下且学习正确，再优化每秒有效样本数。

## 从多少张卡起步？

实验室按每节点 8 GPU，参考 H100/H200 级别、高速互联服务器的仓库配方。这些是配方起步值，不是实测最小值，也不保证这一档所有硬件均能直接运行。短实验限制回复为 4,096 tokens，每轮 8 个 prompt、各采样 8 次；prompt 长度仍计入显存。MoE 和 Qwen 默认启用 CPU Adam 与重计算。

| 模型 | 训练卡数 | TP / PP / CP / EP | 每个推理引擎卡数 | 来源 |
|---|---:|---|---:|---|
| GLM-4-9B | 8 | 2 / 1 / 2 / 1 | 2 | [9B](https://github.com/THUDM/slime/blob/main/scripts/run-glm4-9B.sh) |
| GLM-4.7-Flash | 8 | 2 / 2 / 2 / 4 | 8 | [Flash](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B.sh) |
| GLM-4.7 | 64 | 8 / 4 / 2 / 16 | 32 | [355B](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-355B-A32B.sh) |
| GLM-5.3 | 256 | 4 / 8 / 8 / 32 | 64 | [744B](https://github.com/THUDM/slime/blob/main/scripts/run-glm5.2-744B-A40B.sh) |
| DeepSeek-R1 | 128 | 8 / 4 / 4 / 32 | 64 | [R1](https://github.com/THUDM/slime/blob/main/scripts/run-deepseek-r1.sh) |
| Qwen3.8-27B | 32 | 4 / 2 / 4 / 1 | 2 | [匹配的架构](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3.5-27B.sh) |

Flash 在实验室使用 EP4，使 EP × PP 能放进 8 卡。Qwen3.8 复用匹配架构配置，此次文档修改尚未对新 checkpoint 做 GPU 验证。第一次建议用 9B 或 Flash 理解循环，再对目标模型做短实验，测量峰值显存后分配长任务资源。

同卡训推分配一个池；训推分离需训练卡数加 rollout 卡数。PD 将 rollout 池分为完整 prefill/decode 引擎；外部推理资源不计入训练 Ray 集群。只增加卡数而不调整模型并行，可能只是增加副本，并不减少每卡模型显存。

## 显存账本

以一种混合精度 Adam 为例，$P$ 个参数的 BF16 权重占 $2P$ 字节、FP32 梯度 $4P$、FP32 master weights $4P$、两份 FP32 moments $8P$：

$$M_{\rm states}=(2+4+4+8)P=18P\ \text{bytes}.$$

实际 precision-aware 优化器的状态精度可能不同。激活、buffers、临时转换、分配器预留、reference policy 副本、KV cache 还要另算。30B 的 BF16 权重约 60 GB，两份 FP32 moments 约 240 GB，尚未分片。MoE 存储要看总参数量，不能只看激活参数量。

```{mermaid}
flowchart TB
    T[训练显存] --> W[权重 + 梯度]
    T --> O[优化器状态]
    T --> A[保存的激活]
    T --> B[Buffers + 临时分配]
    O --> C[CPU Adam：主机存储和计算]
    A --> R[重计算：少保存，再算一次]
    S[推理显存] --> SW[权重]
    S --> KV[KV cache + workspace]
```

理想的阶段 offload 使同卡显存需求接近 $\max(M_{\rm train},M_{\rm rollout})$，还需考虑常驻 buffers 与切换开销。分离布局则分别满足两个池的容量约束。

## TP：拆一个层

按矩阵列拆分：

$$W=[W_0\;W_1],\qquad Y=XW=[XW_0\;XW_1].$$

各卡计算一片；后续按输入行拆分的层计算 $Z=Y_0V_0+Y_1V_1$，需要归约各卡的部分输出。这解释了显存节省和频繁通信的来源，TP 应优先放在高速互联内。Sequence parallelism 还拆分部分激活操作，不能替代 CP。见 [Megatron 论文](https://arxiv.org/abs/2104.04473)和[并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。

## PP：拆模型层

```{mermaid}
flowchart LR
    A[前面的层] -->|激活| B[中间的层]
    B -->|激活| C[后面的层]
    C -->|梯度| B
    B -->|梯度| A
```

每个 stage 只保存自己的层。假设 $p$ 个等耗时阶段、$m$ 个 microbatch、每阶段耗时 $\tau$，简单 forward 流水线耗时 $(m+p-1)\tau$。利用率为 $m/(m+p-1)$，填充与排空的 bubble 比例为 $(p-1)/(m+p-1)$。实际训练还受 backward 调度、通信与分层不均衡影响。增加 microbatch 数能摊薄 bubble，但每个 microbatch 仍要放得下。见 [Megatron 论文](https://arxiv.org/abs/2104.04473)。

## EP：拆 MoE 专家

对选中的专家集合 $K(x)$：

$$y(x)=\sum_{e\in K(x)}g_e(x)E_e(x).$$

不同 GPU 保存不同专家。All-to-all dispatch 把 token 送到专家所在卡，combine 返回加权输出。权重均匀分片不代表 token 流量均匀，专家负载倾斜可能成为瓶颈。见 [GShard](https://arxiv.org/abs/2006.16668)。

Dense 层常用卡数关系为 $N=TP\cdot PP\cdot CP\cdot DP$。专家层有另一套含 expert TP、expert DP 的分组，不能在 dense 关系上再乘 EP。本实验室使用 expert TP=1，检查卡数能被 $TP\cdot PP\cdot CP$ 和 $EP\cdot PP$ 整除；模型维度与 Megatron 还会带来其他约束。

## CP：拆 token

CP 按 token 位置分配 queries；attention 仍需远端 keys/values，因此引入通信。它减少本地序列激活，不会自动拆模型权重。

| 单条 8-token 序列、CP2 | Rank 0 | Rank 1 |
|---|---|---|
| 默认 zigzag | 0、1、6、7 | 2、3、4、5 |
| slime allgather CP | 0、1、2、3 | 4、5、6、7 |

Zigzag 将每条补齐后的序列切为 $2c$ 块，rank $r$ 得到第 $r$ 与 $2c-r-1$ 块。前后配对平衡 causal attention：8 个位置连续切分时两卡分别处理 10、26 个 query-key 对，图中的 zigzag 各处理 18 对。这是配对数，不是 sparse kernel 真实耗时。

slime 的 `--allgather-cp` 打包序列，将补齐后的整体 token 流连续切分。DSA backend 收集所需远端 attention/index 数据，loss 处理再恢复样本归属。它不同于 zigzag 逐序列切分；不兼容的 backend 上误开 flag 会打乱 token 顺序。

```{mermaid}
flowchart LR
    A[Rank 0 本地 queries] --> C[对所需 keys 做 attention]
    B[Rank 1 keys / values] -->|backend 通信| C
    C --> D[Rank 0 本地输出]
```

Zigzag 描述布局，不等同于某一个固定 ring 算法；通信由 attention backend 决定。Allgather 可能产生较大聚合张量，所以 CP 不会将每类内存等比例缩小。

CP > 1 时，slime 当前只允许 `DeepseekV32ForCausalLM` 和 `GlmMoeDsaForCausalLM` 使用 allgather CP。实验室的 GLM-5.3 通过模型配置启用它，其他配方保持支持的布局。图示按钮不修改运行 flag。见[参数校验](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/arguments.py)、[打包](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/data.py)、[CP 索引](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/cp_utils.py)和 [Megatron 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。

## ZeRO 与 distributed optimizer

数据并行副本处理不同样本。ZeRO 依次分片优化器状态（stage 1）、梯度（stage 2）、参数（stage 3）。沿用每参数 18 字节示例，$D$ 个副本上的理想持久状态开销为：

$$M_0=18P,\quad M_1=6P+12P/D,\quad M_2=2P+16P/D,\quad M_3=18P/D.$$

推导即先将 master weights 和 moments 除以 $D$，再依次将梯度、BF16 权重除以 $D$；未计临时通信 buffers。$P$ 可以是本地模型并行分片，$D$ 是对应副本数；专家参数分组可能不同。见 [ZeRO](https://arxiv.org/abs/1910.02054)。

slime 默认启用 Megatron `use_distributed_optimizer`，分片优化器状态与更新计算，再聚合更新后的参数。这不等于完整 ZeRO-3 参数分片，也不表示使用 DeepSpeed runtime。见[后端默认参数](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/arguments.py)。

## CPU Adam 与重计算

Adam 保存 moments 并做偏差修正：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,$$
$$\hat m_t=m_t/(1-\beta_1^t),\quad\hat v_t=v_t/(1-\beta_2^t),\quad\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon).$$

CPU Adam 在主机保存优化器状态并执行更新。开关导出 `--optimizer-cpu-offload`、`--overlap-cpu-optimizer-d2h-h2d` 和 `--use-precision-aware-optimizer`。分片决定哪个 rank 负责状态；offload 决定该 rank 在哪里保存和更新状态。主机内存、CPU 算力和 PCIe 带宽接过一部分显存负担，激活与 KV cache 不受影响。背景见 [Adam](https://arxiv.org/abs/1412.6980)、[ZeRO-Offload](https://arxiv.org/abs/2101.06840)，slime 使用 Megatron 实现。

独立的重计算开关保存边界，在 backward 时重做部分 forward，以 FLOPs 换激活显存，见[激活 checkpointing](https://arxiv.org/abs/1604.06174)。训练 OOM 先判断状态还是激活占主导，再选 offload、更多模型并行或更小的 token microbatch。推理 OOM 则检查权重、KV 精度、并发、回复长度和引擎卡数。通过 [profiling](../developer_guide/profiling.md)逐项测量。
