# Tutorial：搭建并理解一个 RL 实验

{ref}`打开交互式 Quick Start <lab>`。选择模型，观察 GPU 布局变化，试不同组合，最后下载启动 shell。同一条教程适用于 GLM、Qwen3.8-27B 和 DeepSeek-R1，模型尺寸来自仓库已有配置。

## 你正在搭建的循环

```{mermaid}
flowchart LR
    A[Prompt / 环境] --> B[SGLang 生成回答]
    B --> C[Reward / 验证器]
    C --> D[Batch 或持续队列]
    D --> E[Megatron 更新策略]
    E -->|同步新权重| B
    E --> F[Checkpoint 与独立评估]
```

一个训练样本不只有问答，还携带 tokens、reward、masks，以及算法需要的概率或 replay 数据。SGLang 产生经验，Megatron 计算梯度；精度、kernel 或权重版本变化，都可能使两者分布不同。规模化 RL 的关键，就是在有效使用算力的同时处理这些差异。

实验室的「演示一轮」用于理解流程，不是 GPU 性能模拟，也没有真的启动训练。每个选择都能回退：使用上一步、步骤标题、撤销 / 重做或重置。页面在本地保存选择，刷新后可继续。分享链接包含架构选择，不包含机器路径、引擎地址和自定义 hook 路径；`experiment.json` 保存完整的本地配置。

## 1. 选择任务与模型

起步实验使用可验证数学，JSONL 字段为 `prompt` 和 `label`，应用 chat template，使用内置 `deepscaler` 奖励。每个 prompt 采样 8 次，采用 GRPO，先跑 3 轮。这是验证正确性的实验，不承诺产生能力提升。

自己的 Agent 或奖励任务选择「我的 Agent / 奖励」，再填写可导入的生成函数和奖励函数。Worker 环境必须能导入这些 hook；向导不会替任务发明奖励函数。详见 [Agent 工作流](agent.md)和[接口约定](customization.md)。

选择模型后，架构、TP / PP / CP / EP、转换方式和推理引擎大小会一起改变。这些是起点配置，不是显存计算器。默认训练节点数继承仓库配方：GLM-4.7-Flash 和 GLM-4-9B 为一台 8 卡节点；GLM-4.7 为 8 台；GLM-5.3 为 32 台；DeepSeek-R1 为 16 台。Qwen3.8-27B 采用四节点的 27B 混合注意力配置。

[Qwen3.8-27B 官方配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) 使用 `qwen3_5`，尺寸匹配 `scripts/models/qwen3.5-27B.sh`。文本 RL 路径复用该 model spec 和 HF 转换支持。这是架构核对，不是新 checkpoint 的 GPU 验证，也不是多模态 RL 配方。仍需要 FLA / Gated Delta Net 依赖，见[架构支持](../advanced/arch-support-beyond-megatron.md)。

[GLM-5.3 官方配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json)的 78 层、256 专家 DSA 架构及跨层 index-sharing 计划，与现有 `glm5.2-744B-A40B` 模型规格匹配。实验室复用这一架构源文件，但下载 GLM-5.3 checkpoint；已有六层 deterministic 回归不代表对新 checkpoint 的验证。

## 2. 先决定 GPU 归属，再决定调度

实验室提供起步卡数、总资源量和可点击的 TP/PP/EP/CP/ZeRO 图示。CPU Adam 与激活重计算开关会改变导出的 shell。见[内存预算与并行方式](../advanced/parallelism-memory.md)，包括 allgather/zigzag CP 对比。

| 选择 | 行为 | 你需要准备 |
|---|---|---|
| Colocated | 训推轮流使用同一组 GPU | 每个阶段的显存和 offload 内存 |
| Disaggregated | slime 管理独立训推资源池 | 足够同时容纳两组的 Ray 资源 |
| External SGLang | 连接由外部系统管理的引擎 | 引擎地址和兼容的权重传输 |
| 同步 | 完成一批 rollout，再训练并更新权重 | 顺序容易理解的基线 |
| Fully async | 生成跨更新持续运行 | 独立资源池和样本年龄监控 |

从同卡选择 fully async 时，向导会在同一个可撤销操作内切换到独立 GPU；选择同卡则恢复同步。其他不兼容项会明确显示，等待修正，不会悄悄导出另一个算法。

外部引擎不是任意文本 API：slime 需要 SGLang server 信息、权重更新接口和 sampler 元数据。生成的 `serving-reference.sh` 提供所需参数；部署相关的 host、port、多机 rank 和 RDMA 属于外部 launcher。训练 CLI 无法重配已经运行的引擎。

Partial rollout 保存未完成前缀以便之后续跑，也可以用于同步训练。straw 持久化数据和队列，搭配 fully async 后启用多 Ray 节点的分布式 producer。实验室独立提供这两个选项；时间线、混合版本概率与 checkpoint 边界见[rollout 调度](../advanced/rollout-scheduling.md)。

## 3. 选择数值行为

可以先用 BF16 / BF16 建立基线，大 MoE 也可以从维护中的 BF16 训练 + FP8 rollout 路径开始。后者保留 BF16 训练 checkpoint，推理使用独立量化的 HF checkpoint。Attention KV 精度又是另一个维度；hybrid 模型还需单独选择递归状态 dtype。Qwen3.8-27B 提供 attention KV、Mamba SSM 状态和可选的状态 / KV 内存比例，配图分别展示两个池。见 [hybrid 缓存账本](../advanced/rl-systems.md#hybrid-cache)，其中也说明 GLM-5.3-Flash 的不同架构。

FP8 训练仍为实验性，INT4 rollout 为 beta；低精度都需要匹配 GPU / kernel 栈。见[精度与内存推导](../advanced/rl-systems.md#precision)。

再决定如何修正差异：TIS 裁剪重要性权重，IcePop mask 区间外权重，R3 重放 MoE 路由，SC 中心化加权 score 并使用 REINFORCE。Deterministic 改善可复现性，但不会自动让不同引擎相等。「原理与推导」按钮链接[完整公式推导](../advanced/policy-mismatch.md)及原论文。

## 4. 根据负载配置 serving

普通 serving 是基线；测量表明 prefill / decode 需要不同资源时再尝试 PD。向导要求两组都是完整引擎，使用 Mooncake / RDMA，需要填写真实网卡名。`sglang.yaml` 已嵌入 `experiment.sh`，不必再手工复制单独配置文件。

重复 prefix 较多时，可在预算主机内存和 I/O 后启用 HiCache；PD 下此配方仅在 prefill 开启。并发数和 response token 上限根据真实显存与延迟调整，见[推理成本模型](../advanced/rl-systems.md#pd)。

开启 **EAGLE** 可加入投机采样：选择 checkpoint 内置 MTP 头或填写匹配的独立 EAGLE 头，再设置投机深度。配图会随 PD、HiCache、EAGLE 三个开关变化，展示候选生成与目标模型验证。见[配置、接受概率推导与指标](../advanced/speculative-decoding.md)。

## 5. 准备一次、转换，然后运行

先[安装运行环境](quick_start.md)。各主机保持相同 checkout、依赖、模型 / 数据路径和配置路径。多机需要共享文件系统或相同路径的本地副本；只有 head 容器能看到的路径不够。生成的 `.slime-lab.*` 配置目录也要通过相同仓库路径供 Ray worker 访问。

把 `experiment.sh` 下载到 slime 仓库根目录。核对路径与说明后：

```bash
# 在共享文件系统上执行一次：下载原始权重、数据，
# 并准备选中的 FP8 / INT4 rollout 权重。
bash experiment.sh prepare

# 将 BF16 权重转换为可重分片的 Megatron checkpoint。
bash experiment.sh convert
```

中小模型配方在一台 8 卡节点上转换。GLM-4.7、GLM-5.3 和 DeepSeek-R1 的转换使用 4 节点、每节点 8 GPU；在**四台节点上同时**运行 `convert`，填写相同的 `CONVERT_MASTER_ADDR` 和不同的 `CONVERT_NODE_RANK`（0–3）。`torch_dist` 可重分片，因此转换并行度不必与训练一致；转换本身仍需足够显存和主机内存。

GLM-5.3 与 DeepSeek-R1 发布的是 FP8 权重，`prepare` 会下载到独立 source 目录，再转 BF16，并准备所选推理格式。如果已经有 BF16 HF 和 `torch_dist`，可直接改路径并跳过准备。

根据页面显示的 GPU 总量启动 Ray head 并加入 worker：

```bash
# Head 节点
ray start --head --node-ip-address HEAD_IP --num-gpus 8 \
  --dashboard-host 0.0.0.0 --dashboard-port 8265

# 每个 worker 节点
ray start --address HEAD_IP:6379 --num-gpus 8
```

External 需要先部署并核对外部引擎，其 GPU 不计入训练 Ray 资源。磁盘同步需要两侧以相同路径挂载共享权重目录；delta 还需要 patched serving 接口与本地 checkpoint 目录。

在 head 的仓库根目录执行：

```bash
bash experiment.sh check
bash experiment.sh train
```

`check` 检查输入文件存在，不加载模型，也不认证容量。`train` 用明确的 runtime environment 提交作业；其他 Ray head 可设置 `RAY_DASHBOARD`。脚本不会隐式杀进程或启动集群。

默认是短实验；新实验请使用新的输出目录和 straw run ID，已有目录遵循后端恢复规则。恢复后续训练时，添加指向保存 checkpoint 的 `--load`，并遵循[恢复要求](../advanced/fault-tolerance.md)。保存间隔需要根据实验调整；默认短实验可能在到达保存间隔前结束。

## 6. 判断实验是否有效

先看生成文本，再看 reward、reference / trainer log-prob，然后检查训推差异和修正的 clipping / masking。Fully async 还要看 staleness。跑得快但奖励无意义，并不代表有效实验。

长时间训练前加入独立验证集。例如下载 AIME 后，同步训练可以添加：

```bash
--eval-interval 20 \
--eval-prompt-data aime /data/aime-2024/aime-2024.jsonl \
--n-samples-per-eval-prompt 8
```

Fully async 队列本身不实现 evaluation，应安排独立的受支持评估路径或任务。见[评估配置](usage.md)和 [fully async 指南](../_examples_synced/fully_async/README.md)。

输出异常时用 [Debug 与重放](../developer_guide/debug.md)；结果正确但运行缓慢时，用 [Trace](../developer_guide/trace.md)、[Profiling](../developer_guide/profiling.md)和[观测指标](../advanced/observability.md)决定下一步。首页展开式 Advanced 区域遵循相同的排查顺序。
