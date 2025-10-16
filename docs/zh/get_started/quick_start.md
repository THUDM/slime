# 快速使用

本文档从搭建环境开始，在一小时内带您快速上手 slime，涵盖环境配置，数据准备，训练启动和关键代码解析和魔改。

## 基础环境搭建

由于 slime 可能会包含针对 sglang/megatron 的临时补丁（patch）。为避免潜在的环境配置问题，强烈建议**用户使用我们提供的最新 Docker 镜像**，它已预置好所有依赖。

### 硬件支持说明

**slime** 支持多种 NVIDIA GPU 硬件平台：

- **B200 系列**：完全支持，运行步骤与 H 系列完全相同
- **H 系列 (H100/H200)**：官方支持，具有完整的 CI 测试保护，运行稳定可靠

**重要说明**：
- 最新的 Docker 镜像对 B 卡和 H 卡通用，无需额外配置
- Megatron 后端在 H 卡上具有 CI 保护，经过充分测试验证，推荐生产环境使用
- B 卡基本功能稳定，可作为开发和测试参考，但暂无 CI 保护
- 两种硬件平台使用完全相同的安装和启动流程

- 对于不方便使用 docker 的场景，请参考 [build_conda.sh](https://github.com/THUDM/slime/blob/main/build_conda.sh)；
- 对于 AMD 支持，请参考 [AMD 使用教程](../../en/platform_support/amd_tutorial.md)。

### 拉取并启动 Docker 容器

请执行以下命令，拉取最新镜像并启动一个交互式容器：

```shell
# 拉取最新镜像
docker pull slimerl/slime:latest

# 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

### 安装 slime

进入 Docker 容器后，请按照以下步骤克隆 slime 仓库并进行安装：

```bash
# 路径可根据实际情况调整
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

## 模型与数据集下载

可以从 Hugging Face、ModelScope 等平台下载所需的模型和数据集。以下是使用 `huggingface_hub` 下载示例资源的命令：

```bash

pip install -U huggingface_hub

# 下载模型权重 (GLM-Z1-9B)
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

# 下载训练数据集 (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

## 模型权重转换

### Hugging Face 格式 转换为 Megatron 格式

当使用 Megatron 作为训练后端时，需要先将 Hugging Face 格式的模型权重转换为 Megatron `torch_dist` 格式。

首先，加载目标模型的配置文件。`slime/scripts/models` 目录下包含了支持模型的配置文件。需要 `source` 对应模型的脚本，将配置参数加载到当前环境中。此处我们以 GLM4-9B 模型为例子，对于 Qwen3-4B，Qwen3-30B-A3B，是类似的。

```bash
cd /root/slime
source scripts/models/glm4-9B.sh
```

接下来，运行转换脚本。请注意以下参数：
- `--hf-checkpoint`: 指定已下载的 Hugging Face 模型权重路径。
- `--save`: 指定转换后 `torch_dist` 格式权重的保存路径。

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

对于更大的模型，可以使用 `torchrun` 来启动转换脚本，从而使用多张 GPU 甚至多机进行权重转换。

### Megatron 格式 转换为 Hugging Face 格式

可以通过这样的方式将训练过程中保存的 Megatron 格式的权重转换回 Huggingface 格式：

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist_ckpt/iter_xxx/ \
  --output-dir /root/GLM-Z1-9B-0414-iter_xxx \
  --origin-hf-dir /root/GLM-Z1-9B-0414
```

由于 Megatron 会对 embedding 做 padding，可能会出现转换出来的权重的 embedding 形状不匹配的问题。这时需要在转换时设置 `--vocab-size`。

## 训练脚本与参数概览

完成上述准备工作后，即可运行训练脚本。

```bash
cd /root/slime
bash scripts/run-glm4-9B.sh
```

我们还是以 run-glm4-9B.sh 脚本为例，简单分析主要参数的作用。

### MODEL_ARGS: 模型配置参数

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"
```

此部分通过 `source` 命令从 `scripts/models/glm4-9B.sh` 文件中加载模型配置。这些配置均为 Megatron 所需的超参数。由于 Megatron 无法直接从检查点（checkpoint）中读取模型配置，因此需要手动指定。我们在 `scripts/models/` 目录下提供了一些常用模型的配置示例。

> ⚠️ **注意**：
> 请务必检查模型配置文件中的参数（如 `--rotary-base`）是否与您当前使用的模型完全匹配。同一模型结构的不同版本可能使用不同的配置值。如果需要修改，您可以在 `source` 之后直接覆盖，例如：
> ```bash
> source "${SCRIPT_DIR}/models/glm4-9B.sh"
> MODEL_ARGS+=(--rotary-base 10000)
> ```

### CKPT_ARGS: 检查点与路径参数

```bash
CKPT_ARGS=(
   # 用于加载 tokenizer 等其他信息，实际上不会使用 hf 路径中的模型权重参数
   --hf-checkpoint /root/GLM-Z1-9B-0414
   # 参考模型 (Reference Model) 的 Megatron 格式检查点
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   # Actor 模型的加载路径。若为空或不存在有效的checkpoint，则从 --ref-load 加载
   --load /root/GLM-Z1-9B-0414_slime/
   # 训练过程中模型的保存路径
   --save /root/GLM-Z1-9B-0414_slime/
   # 模型保存间隔（步数）
   --save-interval 20
)
```

### ROLLOUT_ARGS: 数据生成（Rollout）参数

整个训练流程可视为一个 **“数据采样 → 权重更新”** 的闭环。

**阶段一：数据采样 (Rollout)**
- `--rollout-batch-size`：定义每轮采样的 **Prompt 数量**
- `--n-samples-per-prompt`：定义每个 Prompt 生成的 **回复数量** (用于 GRPO 类似算法)

> 两者相乘，决定了 **单轮采样产生的总样本数**。

**阶段二：模型训练 (Training)**
- `--global-batch-size`：定义 **执行一次参数更新（optimizer.step）** 所需的样本量
- `--num-steps-per-rollout`：定义使用当前采样数据，**总共执行多少次参数更新**  (我们默认为 1，使用 on-policy 训练)

> 两者相乘，决定了 **单轮训练消耗的总样本数**。

> ⚠️ 这里的 **参数更新** 指训练环节的 optimizer.step()，不同于训练引擎向推理引擎发起的权重同步(Weight Sync)。

在这个过程中，每轮的“产出”与“消耗”必须相等，遵循以下约束：
**`(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)`**

- 在 slime 中，如果设置了 `--num-steps-per-rollout` ，`--global-batch-size` 未设置则会被自动设置，设置了则会被用上述公式校验。

**训练流程次数控制**
-   `--num-rollout`: 控制整个 **“采样→训练”** 循环的**总执行轮次**。

```bash
ROLLOUT_ARGS=(
   # Prompt 数据集，JSONL 格式
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   # 若 Prompt 的 `input_key` 是 OpenAI message 格式，则应用 Chat Template
   --apply-chat-template
   # 是否在 Rollout 阶段打乱数据
   --rollout-shuffle

   # Reward Model 类型。slime 内置多种类型，也支持通过 --custom-rm-path 自定义
   --rm-type deepscaler

   # 这五个参数来控制 rollout 与 train 的关系
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 128

   # Rollout 采样参数
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   # 对 rollout 阶段收集的数据进行负载均衡。它确保了分配到每个训练进程（DP rank）的计算任务量大致相等，可能对训练速度有好处
   --balance-data
)
```

### EVAL_ARGS: 评估参数

评估过程会继承大部分 Rollout 参数，但您可以通过以下参数进行覆盖，以实现与训练不同的评估策略。

```bash
EVAL_ARGS=(
   # 评估间隔（Rollout 数）
   --eval-interval 5
   # 评估用的 Prompt 数据集
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   # 每个评估 Prompt 的采样数量
   --n-samples-per-eval-prompt 16
   # 评估时最大响应长度
   --eval-max-response-len 16384
   # 评估时采样参数
   --eval-top-p 0.7
)
```

### PERF_ARGS: 性能与并行参数

这部分主要包含 Megatron 的并行配置。`--use-dynamic-batch-size` 和 `--max-tokens-per-gpu` 是 slime 添加的特有优化。

-   `--max-tokens-per-gpu`: 每张 GPU 处理的最大 Token 数。启用动态批处理（`use_dynamic_batch_size`）后，系统会智能地将长短不一的样本打包，使每个 micro-batch 的总 Token 数接近此限制，从而提升训练效率。如果单个样本长度超过该值，它将独立形成一个 batch。在上下文并行（CP）模式下，`N` 张 CP 卡共享 `N * max_tokens_per_gpu` 的总长度。
-   `--use-dynamic-batch-size`: 启用动态批处理。此时会忽略 `--micro-batch-size`。


> 💡 **提示**：
>  slime 总是会通过 data packing 的方法训练模型，并且严格保证 per sample loss 或 per token loss 是正确的。因此，开启 dynamic batch size 不会对 loss 计算有影响，强烈推荐开启。

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1 # 启用动态批处理后此项被忽略
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)
```

### GRPO_ARGS: GRPO 算法参数

-   `--use-kl-loss`: 启用此选项将加载一个参考模型（Reference Model），并计算当前模型与参考模型之间的 KL 散度（KL Divergence）作为一项监控指标。KL 散度是否被计入最终的训练损失（Loss），取决于 `--kl-loss-coef` 参数。若该参数设置为 0，则 KL 散度仅作为观测指标显示，而不会参与损失计算。

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

- `--advantage-estimator`: 除去 [GRPO](https://arxiv.org/abs/2402.03300)，slime 还支持丰富的其他训练算法，例如 [GSPO](https://arxiv.org/abs/2507.18071)、[Reinforce++](https://arxiv.org/abs/2501.03262) 与 [Reinforce++ Baseline](https://arxiv.org/abs/2501.03262)、以及 [PPO](https://arxiv.org/abs/1707.06347)；
- `--calculate-per-token-loss`：slime 中默认的方案是 per sample loss，即 `mean(sum(sample_i) / len(sample_i))`，如果需要计算 per token loss，即 `sum(sum(sample_i)) / sum(len(sample_i))`，可以开启 `--calculate-per-token-loss`；
- `--use-tis`：如果需要开启 TIS (Truncated Importance Sampling)，可以开启这一设置。TIS 由此[博客](https://fengyao.notion.site/off-policy-rl)介绍。

### OPTIMIZER_ARGS: 优化器参数

```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

### SGLANG_ARGS: SGLang 服务参数

这部分参数用于配置 SGLang 推理服务。
- `--rollout-num-gpus-per-engine`: 基本等同于 SGLang 的 `tp_size`。
- 其他 SGLang 参数可以通过添加 `--sglang-` 前缀传递给 slime,  slime 会自动透传给 SGLang。例如，要设置 SGLang 的 `--log-level INFO` 参数，只需使用 `--sglang-log-level INFO` 即可。

> ⚠️ **注意**：
> slime 使用 `sgl-router` 调度多个 SGLang Server。在不开启 DP Attention 的情况下， `dp_size` 会通过 `rollout-num-gpus / rollout-num-gpus-per-engine` 计算得到。

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)
```

## 特性介绍

### Colocated Actor and Rollout

在默认的配置下，训练（Actor）和推理（Rollout）的资源是分开指定的，通过 ray 给训练部分分配 `actor_num_nodes * actor_num_gpus_per_node` 张 GPU，给推理分配 `rollout_num_gpus` 张 GPU，也即训推分离。

**标准（分离）配置**：
```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```
上述配置中，Actor 使用 4 张卡，Rollout 也使用 4 张卡，两者并行运行。

**训推一体化（Colocated）配置**：
要将训练和推理部署在同一组 GPU 上，请添加 `--colocate` 参数，开启后会忽略 `--rollout-num-gpus` 让训练和推理的卡数相等。


```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```
此时，训练和推理将共享全部 8 张 GPU。

> ⚠️ **注意**：
> 在训推一体化模式下，Megatron 初始化后才能被 offload 掉，会占据一定量的显存。您需要通过调整 `--sglang-mem-fraction-static` 参数来降低 SGLang 的显存占用比例，以避免显存不足。通常我们建议为 0.8。

> 此外，[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) 里面的一些优化只能在训推一体模式中使用，因为需要释放 GPU 显存。训推分离模式暂不支持。

### Dynamic Sampling

slime 支持更复杂的采样策略，例如 [DAPO](https://dapo-sia.github.io/) 中使用的动态采样。要启用此功能，需配置以下参数：

```bash
   --over-sampling-batch-size 64 \
   --dynamic-sampling-filter-path \
     slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

这里 `over_sampling_batch_size` 需要大于 `rollout_batch_size`，例如配置为：

```bash
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --over-sampling-batch-size 64 \
```

那么 sampling 会直接采样 64 条 prompt，每条 prompt 采样 8 次。因为 slime 内部进行的是异步采样，所以我们会先后获得每个 prompt 的 8 条回复。在收到回复时，会用 `dynamic_sampling_filter_path` 对应的函数进行筛选，如果通过，则留下这 8 条数据，否则则丢掉。

示例中的过滤函数 `check_reward_nonzero_std` 会检查一组样本的奖励（reward）标准差是否大于零，确保留下的每组样本其奖励分数都存在差异，从而避免数据过于单一，提升了数据的多样性。

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.reward for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```

如果过滤函数非常严格，导致大量 prompt 组被丢弃，系统会监控 ` remaining_batch_size` 中待处理的任务数量。一旦待处理的任务数因丢弃过多而降至目标数 (32) 以下，系统会自动触发新一轮的过采样，再次请求  `over_sampling_batch_size` (64) 个新的 prompt 重复上述流程。

### Partial Rollout

在动态采样过程中，大量请求可能会被提前中止（abort），造成计算资源浪费。通过启用 `--partial-rollout` 参数，可以将这些生成到一半的样本缓存起来，在下一个 Rollout 阶段继续生成，从而提升性能。

您还可以通过 `--buffer-filter-path` 自定义从缓存中提取数据的策略。默认策略是 `pop_first`，即按先进先出的顺序提取所需数量的样本。

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

即每次取出前 `num_samples` 个 prompt 对应的 `num_samples * n_samples_per_prompt` 条数据。

> 💡 **提示**：
> 每条 partial rollout sample 的 `sample.metadata` 中存储了第一次进行生成的 rollout id，可以用于数据过滤。



### bf16 训练 fp8 推理

slime 直接支持 bf16 训练，fp8 推理。对于 Qwen3-4B 模型，只需要下载如下模型：

```bash
hf download Qwen/Qwen3-4B-FP8 --local-dir /root/Qwen3-4B-FP8
```

并将 `--hf-checkpoint` 替换为：

```bash
   # 用于加载 tokenizer 等其他信息，实际上不会使用 hf 路径中的模型权重参数
   --hf-checkpoint /root/Qwen3-4B-FP8

   #  megatron checkpoint 还需要是最开始用 bf16 的 huggingface 转换的 dist 权重，不因为 FP8 rollout 而去做修改。
   --ref-load /root/Qwen3-4B_torch_dist
```

即可触发 fp8 推理。目前我们会将 bf16 权重直接 cast 为 fp8，后续会逐渐添加对精度影响更小的量化方案。

⚠️  训练的 megatron checkpoint 还需要是最开始用 bf16 的 huggingface 转换的。


## Multiturn 适配

slime 框架高度可扩展，支持复杂的 Agent 场景（如多轮交互与工具调用）。其核心机制是通过自定义函数，重写默认的数据生成 (Rollout) 与奖励计算 (Reward) 逻辑。

本部分以一个基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的实现为例，说明如何适配 slime 以支持多轮交互。

### 适配思路总结

适配 slime 以支持多轮交互主要包含三个步骤：

1.  **数据准备**：将多轮交互数据集适配为 slime 的 `Sample` 对象。将对话历史、真实标签等映射到 `prompt` 和 `label` 字段，并将工具定义、中间状态等额外信息存入 `metadata` 字段，供后续函数调用。

2.  **实现自定义生成函数**：编写函数模拟“模型生成动作 → 执行工具 → 拼接观察结果”的交互循环，并正确处理 Loss Masking。

3.  **实现自定义奖励函数**：编写函数评估完整的交互轨迹，返回最终的奖励分数。

### 数据准备与映射

为了向自定义函数传递复杂的上下文信息，您需要在**数据预处理阶段**就将所有相关的额外字段聚合起来。

**核心思想**：将数据集中除 `prompt` 和 `label` 之外的所有附加信息（如 `session_id`, `user_profile`, `tool_code` 等）合并，构造成一个**单一的、结构化的字段**（例如，一个名为 `metadata` 的列，其内容为 JSON 字符串）。

### 步骤一：在数据集中构造 `metadata` 字段

在训练开始前，您需要对原始数据集进行处理。例如，您的原始数据可能如下：

| question | final_answer | session_id | tool_code |
| :--- | :--- | :--- | :--- |
| "..." | "..." | "sess_123" | "code_A" |

您需要将其转换为：

| question | final_answer | metadata |
| :--- | :--- | :--- |
| "..." | "..." | `{"session_id": "sess_123", "tool_code": "code_A"}` |

### 步骤二：在训练脚本中指定映射

完成数据准备后，在训练脚本中，通过 `ROLLOUT_ARGS` 将这个预处理好的 `metadata` 列映射到 slime 的 `Sample.metadata` 字段。

```bash
ROLLOUT_ARGS=(
   # 1. 指定预处理后的数据集文件
   --prompt-data /root/nq_search/train_processed.json

   # 2. 将 "question" 列映射为输入 prompt
   --prompt-key question

   # 3. 将 "final_answer" 列映射为评估标签
   --label-key final_answer

   # 4. 将预先构造好的 "metadata" 列加载到 Sample.metadata
   #    slime 会自动将其解析为 Python 字典
   --metadata-key metadata
)
```

通过这种方式，您就可以在自定义的 `generate` 或 `reward` 函数中，通过 `sample.metadata['session_id']` 等方式，方便地访问到所有预先准备好的结构化信息。

### 编写自定义生成函数

首先，通过 `--custom-generate-function-path` 参数指定一个自定义的异步 Python 函数。

**函数签名**: `async def generate(args, sample: Sample, sampling_params) -> Sample:`

**核心实现要点**:

1.  **构建交互循环**: 创建循环以控制最大交互轮次（如 `for _ in range(max_turns):`）。
2.  **调用模型生成动作**: 每轮循环中，调用 SGLang 服务，让模型根据当前对话历史生成下一步动作（如 `<search>query</search>`）。
3.  **解析并执行动作**: 解析模型输出，识别动作与参数，并调用外部工具或 API（如 Google 搜索）。
4.  **构建观察结果**: 将工具返回的结果格式化后，追加到对话历史中，作为下一轮的输入。
5.  **处理 Loss Masking**: 这是 Agent 训练的关键。
    -  需要注意的是： `loss_mask` 应该和 `response` 一样长，其中需要算 loss 的 token 为 1，mask 掉的为 0
    -   **模型生成**的 token (如思考、动作指令) → `loss_mask` 设为 `1`，参与损失计算。
    -   **工具或环境返回**的 token (如 API 结果) → `loss_mask` 设为 `0`，不参与损失计算。
6.  **终止条件**: 当模型生成终止标签（如 `<answer>...`）或达到最大轮次时，结束循环。
7.  **封装返回**: 将完整的交互历史、token ID 和 `loss_masks` 填充到 `Sample` 对象中并返回。


**代码示例（概念）**:
```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    # ... 初始化 ...
    prompt, full_response, loss_masks = sample.prompt, "", []

    for _ in range(max_turns):
        # 1. 模型生成动作
        model_output = await call_sglang(prompt + full_response, ...)
        # ... tokenization and appending ...
        loss_masks += [1] * len(model_tokens) # loss_mask = 1
        full_response += model_output

        # 2. 解析并执行动作
        action, content = parse_action(model_output)
        if action == "search":
            # 3 & 4. 获取并追加观察结果
            tool_output = await google_search(content)
            # ... tokenization and appending ...
            loss_masks += [0] * len(tool_tokens) # loss_mask = 0
            full_response += tool_output

        elif action == "answer":
            break # 结束循环

    # 7. 填充并返回 Sample 对象
    sample.response = full_response
    sample.tokens = ...
    sample.loss_mask = loss_masks
    return sample
```

### 编写自定义奖励函数

类似地，通过 `--custom-rm-path` 指定自定义奖励函数。

**函数签名**: `async def reward_func(args, sample: Sample, **kwargs) -> float:`

该函数接收完整的 `Sample` 对象，根据最终交互结果计算得分。可以在此实现自定义计分逻辑，或调用外部的 Reward Model 服务。

### 在训练脚本中配置

最后，在训练脚本中，通过以下参数启用上述自定义函数：

```bash
CUSTOM_ARGS=(
   # 指定自定义生成函数的路径 (格式: path.to.your.file:function_name)
   --custom-generate-function-path your_module.multiturn_logic:generate

   # 指定自定义奖励函数的路径
   --custom-rm-path your_module.multiturn_logic:reward_func
)
```

## 大规模 MOE 模型的多机训练

为了启动多机任务，首先需要启动一个 ray 集群，即在 node 0 运行：

```bash
# Node0（HEAD）
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats

# 其他 Node
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
```

在 ray 集群启动后，可以在 node 0 提交任务，例如：

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        ... # e.g. no_proxy、接口变量等
     }
   }' \
   -- python3 train.py \
   --...（其他 Megatron/SGLang/slime 参数）
```

slime 针对大规模混合专家（MoE）模型的分布式训练进行了深度优化。我们提供了一些端到端的训练案例以供参考：

- [示例：64xH100 训练 GLM-4.5](models/glm4.5-355B-A32B.md)
- [示例：128xH100 训练 DeepSeek-R1](models/deepseek-r1.md)
