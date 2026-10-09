# 看懂你正在搭建的 RL 系统

[打开实验室](../index.rst)。**布局**决定 GPU 的归属，**调度**决定工作顺序，**精度**决定数值计算，**推理拓扑**决定如何划分 serving 工作。它们是独立的配置维度。下面的成本模型用于理解取舍，不是性能预测。

## Placement

同步的一轮包含生成、奖励计算、训练和权重更新。同卡布局让 rollout 与训练交替驻留在相同 GPU 上；分离布局各自占一组 GPU，即使仍按同步顺序执行。External SGLang 进一步将引擎生命周期交给外部系统，它不自动意味着异步训练。

如果训练需要 $N_T$ 张 GPU、rollout 需要 $N_R$ 张，独立资源池需要 $N_T+N_R$ 张。同卡布局复用资源池，但每个阶段仍必须满足自己的并行度、显存与主机内存要求。向导假设每个训练节点 8 张 GPU，并检查 TP × PP × CP、expert-TP × EP × PP 的整除条件。整除是必要条件，不是显存证明。

设完整权重大小为 $S_W$、带宽为 $B_W$、控制和加载开销为 $t_0$，同步成本近似为：

$$t_W\approx S_W/B_W+t_0.$$

Delta 传输用变化字节 $S_\Delta$ 替代全量，但增加 snapshot、diff、编码和重建开销。只有节约的传输时间超过这些成本才有收益。slime 同卡使用 CUDA IPC；分离全量更新支持 NCCL 或磁盘。Delta 需要磁盘传输、共享目录、推理主机本地 checkpoint 和带补丁的 SGLang `/pull_weights` 接口。

相关系统研究：[HybridFlow](https://arxiv.org/abs/2409.19256)。实际接口见[外部引擎](external-rollout-engines.md)、[增量同步](delta-weight-sync.md)和[训练拓扑](megatron-config.md)。

## Async

令 $t_R$ 为 rollout 加 reward 时间，$t_T$ 为训练时间，$t_W$ 为权重更新时间，串行一轮约为：

$$t_{\rm sync}=t_R+t_T+t_W.$$

如果有独立资源、充分预热的队列，且生成与训练能够完美重叠，乐观的稳态下界为：

$$t_{\rm async}\gtrsim\max(t_R,t_T)+t_W,\qquad
\text{speedup}\lesssim\frac{t_R+t_T+t_W}{\max(t_R,t_T)+t_W}.$$

权重更新停顿、采样依赖、reward 延迟、队列预热和 GPU 争用都会削弱这个估计。它不代表实测加速比。回复长度差异越大，同步 barrier 的长尾等待往往越明显。

重叠的代价是 staleness。样本在权重版本 $v_s$ 生成、在 $v_t$ 被使用，定义年龄 $d=v_t-v_s$。固定 prefix，可拆分差异：

$$
\log p_{v_t}(a\mid h)-\log q_{v_s}(a\mid h)
=\underbrace{\log p_{v_t}-\log p_{v_s}}_{\text{参数过期}}
+\underbrace{\log p_{v_s}-\log q_{v_s}}_{\text{同版本引擎差异}}.
$$

确定性 kernel 可以减少第二项，不能消去第一项。长轨迹也可能混入多个权重版本，单个年龄统计只是一种诊断。TIS / SC 可以帮助稳定训练，但不意味着可以无限容忍旧样本。

slime 仍用 `train.py`，选择 `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`。它跨调用保留正在生成的样本；要有实际重叠收益，需要独立 GPU。监控 `staleness/mean`、`staleness/max`。生成配方不会在这个连续队列上直接做 evaluation，请使用独立评估路径或任务。见 [fully async 实现指南](../_examples_synced/fully_async/README.md)。相关研究：[PipelineRL](https://arxiv.org/abs/2509.19128)，作为概念参考，不表示 slime 实现了论文完整算法。

## Precision

设训练 logits 为 $z$，推理 logits 为 $z+\delta$。对 log-softmax 求导：

$$\frac{\partial\log p_a}{\partial z_j}=\mathbf1_{a=j}-p_j.$$

一阶展开得到：

$$\log q_a-\log p_a\approx\delta_a-\sum_jp_j\delta_j.$$

所有 token 相同的平移被消去，token 相关的扰动则改变概率。量化、kernel 差异和累加顺序都可能产生扰动；专家选择等不连续位置可能使一阶近似失效。

$P$ 个元素、每元素 $b$ bits 的原始存储约为 $Pb/8$ 字节，不含 scales、元数据、padding 和副本。位宽减半，只代表这个张量的原始存储减半，不代表整个 GPU 的显存占用减半。优化器状态、激活和 KV cache 是独立预算。

向导提供 BF16 训练搭配 BF16 / FP8 rollout、实验性 FP8 训练和 beta INT4 rollout。BF16 训练使用转换后的 `torch_dist`；FP8 rollout 使用独立的 block-quantized HF checkpoint，其 `quantization_config` 决定在线更新时的量化。Attention KV、hybrid 递归状态与权重的精度分别配置。FP8 训练采用 TE blockwise 配置，因 CPU Adam 不兼容而不启用 `fp8-param-gather`。

格式参考：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)。支持状态见[低精度指南](low-precision.md)。INT4 仍需模型和 kernel 验证，向导不承诺 GLM INT4 的性能。

### Hybrid cache

Hybrid 模型有两类不同的推理缓存。[Qwen3.8-27B 配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)有 48 层线性注意力和 16 层 full attention，递归状态使用 FP32；[GLM-5.3-Flash 配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)则组合 34 层 KDA 线性注意力与 11 层 DSA。它们的 attention 架构不同；SGLang 的 **Mamba cache** 是递归状态存储的统一名称，不代表这些模型使用完全相同的架构。递归更新机制可参考 [Gated DeltaNet 论文](https://arxiv.org/abs/2412.06464)。

| 缓存 | 保存什么 | 独立的 slime 配置 |
|---|---|---|
| Attention KV | attention 层的逐 token 缓存 | `--sglang-kv-cache-dtype`；FP8 需目标模型、后端与 GPU 支持 |
| Mamba / 线性注意力状态 | 递归状态与缓存快照 | `--sglang-mamba-ssm-dtype`，与 KV dtype 分开 |
| 卷积状态 | 线性注意力层的短程历史 | 使用独立的卷积 dtype；SSM dtype 参数不改变它 |

向导给递归状态提供模型默认、FP32、BF16 三个选项。默认不传 SSM 参数，Qwen3.8-27B 发布的配置采用 FP32。选 FP8 attention KV 不会把递归状态也变成 FP8。参数定义见 [SGLang 文档](https://docs.sglang.io/docs/advanced_features/server_arguments#mamba-cache)，[状态 dtype 与大小的实现](https://github.com/sgl-project/sglang/blob/v0.5.15.post1/python/sglang/srt/configs/mamba_utils.py)分别计算递归与卷积张量。

设缓存中有 $N$ 个 token 位置，以及 $K$ 个递归状态 slot（包含快照、调度缓冲）；每个 attention token 占 $c_A$ 字节，每个状态 slot 占 $c_R$ 字节，则：

$$B_{\rm cache}\approx N c_A + K c_R.$$

对普通 GQA 层集合 $\mathcal A$，逐层计算每个 token 的 K、V 元素；对线性注意力层集合 $\mathcal R$，计算每个 slot 的递归状态和卷积状态元素：

$$
c_A=\sum_{\ell\in\mathcal A}2H^{KV}_\ell d_\ell\frac{b_{KV}}8,
\qquad
c_R=\sum_{\ell\in\mathcal R}\left(P^{SSM}_\ell\frac{b_{SSM}}8+P^{conv}_\ell\frac{b_{conv}}8\right).
$$

估计每卡占用时，用分片后的实际张量维度。DSA / MLA 的缓存张量不同，应测量其真实字节数，不能套用 GQA 维度。单个递归状态 slot 的形状固定，但 **slot 数量**会随并发、prefix 快照和调度缓冲增加。

只把 BF16 attention KV 改成 FP8 时，$B'\approx Nc_A/2+Kc_R$，因此总缓存通常不会减半。同理，BF16 SSM 只改变递归状态项，卷积状态保持自身的 dtype。

可选的内存比例通过 `--sglang-mamba-full-memory-ratio` 设置 $r=B_R/B_A$。理想化地给两个池分配预算 $B$，联立 $B_A+B_R=B$ 与 $B_R=rB_A$，得到：

$$B_A=\frac{B}{1+r},\qquad B_R=\frac{rB}{1+r}.$$

这不是总 GPU 显存的百分比。调大它，会给递归状态 slot 更多空间、给 attention token 更少空间；应结合分配日志和实际负载选择。留空沿用推理栈默认值。外部引擎的参数写入 `serving-reference.sh`，slime 托管的引擎则使用 `--sglang-` 参数。

向导里可运行的 Flash 起点是 **GLM-4.7-Flash**。上面的 GLM-5.3-Flash hybrid 结构不能直接套用该起点或 GLM-5.3 的 DSA 配方。

## Deterministic

浮点加法不满足结合律。即使没有随机采样，batch shape 和归约树改变也可能产生不同结果。可复现性关心同一配置重复执行的输出；对齐则关心**不同实现**是否产生相同概率：

$$\text{repeatable}(p)\land\text{repeatable}(q)\not\Rightarrow p=q.$$

向导中的 deterministic 开关开启 Megatron deterministic mode、SGLang deterministic inference，以及文档中的 NCCL / TE / cuBLAS 环境变量。它是可复现配置，不是训推相等的证明。依赖栈必须支持当前模型和 kernel，请遵循[可复现性指南](reproducibility.md)，包括 FlashAttention 3 的准备步骤。

GLM-5 精确对齐还需要匹配 DSA attention、batch-invariant DeepGEMM forward、router 和 head 精度，以及 patched Megatron / DeepEP 路径。维护中的验证是**六层 GLM-5.2、EP8**，要求 `train_rollout_logprob_abs_diff < 1e-6`；另有逐层零差异 gate。这不证明完整 744B、任意拓扑、异步或 PD 下已经对齐。

原始技术工作：[SGLang deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)。扩展结论前先运行文档中的 gate。

## PD

Prefill 处理 prompt，decode 逐 token 生成。PD 是对 **serving 内部**的拆分，独立于训练和 serving 是否共享 GPU。研究动机见 [DistServe](https://arxiv.org/abs/2401.09670)，slime 的 SGLang / Mooncake 配置见 [PD 指南](pd-disaggregation.md)。

设 prefill / decode 引擎数为 $n_P,n_D$，在目标负载下的单引擎请求吞吐为 $\mu_P,\mu_D$；每请求 KV 传输量为 $S_{KV}$、网络带宽为 $B_{KV}$。根据流量守恒：

$$\lambda\le\min(n_P\mu_P,n_D\mu_D,B_{KV}/S_{KV}).$$

忽略传输时，令 $n_P\mu_P\approx n_D\mu_D$ 可以避免中间队列持续增长。因此 split 应根据 prompt / response 分布实测，而不是固定选 1:1。

传统 GQA / MHA 中，长度 $L$ 的 prompt 对应未分片 cache 近似为 $S_{KV}\approx2L N_{\rm layers}N_{\rm KVheads}d_{\rm head}b/8$，2 来自 K 和 V。MLA / DSA、压缩和分页布局必须使用实际表示，不能直接套这个公式。

向导生成含 prefill / decode 两组的 `sglang.yaml`。各组 GPU 数必须是每引擎 GPU 数的倍数，且总和等于 rollout 池。Mooncake 配方还需要可用的 RDMA 网卡与依赖。External 部署自行设置 PD；`--sglang-config` 与 `--rollout-external-engine-addrs` 互斥。

## HiCache

[官方 HiCache 指南](https://docs.sglang.io/docs/advanced_features/hicache_best_practices) 介绍了将 prefix cache 扩展到 GPU 之外的层级缓存。一个简单的每请求成本模型是：

$$\Delta t\approx h\,t_{\rm avoided\ prefill}-t_{\rm cache\ IO}-t_{\rm bookkeeping},$$

$h$ 为可复用 prefix 的命中率。命中率大于零也不证明加速，因为传输可能更贵。应测量 prefix 命中率、主机内存、cache I/O 时间和 TTFT。

向导使用 `--sglang-enable-hierarchical-cache --sglang-hicache-ratio 2`，只配置主机缓存，不创建 L3 存储服务。slime 管理的 PD 仅在 prefill 开启，与当前 engine builder 一致；不自动配置 decode offload 或分布式存储。KV 是特定权重下的激活，跨 actor 权重更新必须正确失效。因此要在有效权重窗口内测量可用命中。

## 先 Debug，再优化

先跑三轮短实验并设置独立评估。文本异常时先查模板、checkpoint 转换和权重更新。[Debug replay](../developer_guide/debug.md) 固定采样 batch；[Trace](../developer_guide/trace.md) 找等待阶段；[Profiling](../developer_guide/profiling.md) 再看具体 kernel 或引擎阶段。针对实测瓶颈优化，同时比较 reward 和 mismatch；吞吐本身不是 RL 的最终目标。
