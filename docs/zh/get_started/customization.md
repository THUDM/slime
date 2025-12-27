# 自定义指南

slime 通过函数路径参数提供了广泛的自定义能力。这些参数允许你在训练和推理流程的各个阶段注入自定义逻辑，而无需修改核心代码库。

## 自定义接口概览

下表总结了所有可用的自定义接口及其用途。

| 接口参数 | 用途 |
| :--- | :--- |
| [`--rollout-function-path`](#1-rollout-函数---rollout-function-path) | 覆盖整个 rollout 生成逻辑。 |
| [`--custom-generate-function-path`](#2-自定义生成函数---custom-generate-function-path) | 仅覆盖生成步骤（例如用于 RAG 或工具调用）。 |
| [`--custom-rm-path`](#3-奖励模型---custom-rm-path) | 实现自定义奖励计算逻辑。 |
| [`--dynamic-sampling-filter-path`](#4-动态采样过滤器---dynamic-sampling-filter-path) | 在动态采样过程中过滤样本（例如 DAPO）。 |
| [`--buffer-filter-path`](#5-buffer-过滤器---buffer-filter-path) | 在训练前过滤 rollout buffer 中的样本。 |
| [`--rollout-sample-filter-path`](#6-rollout-样本过滤器---rollout-sample-filter-path) | 决定单个样本是否参与损失计算。 |
| [`--rollout-all-samples-process-path`](#7-rollout-全样本处理---rollout-all-samples-process-path) | 在 rollout 后处理所有样本（包括被过滤的样本）。 |
| [`--rollout-data-postprocess-path`](#8-rollout-数据后处理---rollout-data-postprocess-path) | 在计算 log probabilities 后对 rollout 数据进行后处理。 |
| [`--custom-loss-function-path`](#9-自定义损失函数---custom-loss-function-path) | 实现自定义训练损失计算。 |
| [`--custom-tis-function-path`](#10-自定义-tisrs-函数---custom-tis-function-path) | 实现用于离策略（off-policy）校正的自定义重要性采样。 |
| [`--custom-pg-loss-reducer-function-path`](#11-自定义-pg-loss-reducer---custom-pg-loss-reducer-function-path) | 自定义 pg_loss 的归约方式（如 Dr.GRPO）。 |
| [`--custom-reward-post-process-path`](#12-奖励后处理---custom-reward-post-process-path) | 在优势计算前对奖励进行自定义后处理。 |
| [`--custom-convert-samples-to-train-data-path`](#13-样本转训练数据---custom-convert-samples-to-train-data-path) | 覆盖样本到训练数据格式的转换逻辑。 |
| [`--custom-rollout-log-function-path`](#14-日志函数) | 训练 rollout 的自定义日志记录。 |
| [`--custom-eval-rollout-log-function-path`](#14-日志函数) | 评估 rollout 的自定义日志记录。 |
| [`--data-source-path`](#15-数据源---data-source-path) | 覆盖 rollout 提示词的数据源。 |
| [`--eval-function-path`](#16-评估函数---eval-function-path) | 专门为评估覆盖 rollout 函数。 |
| [`--custom-megatron-init-path`](#17-megatron-hook) | Megatron 设置后的自定义初始化。 |
| [`--custom-megatron-before-log-prob-hook-path`](#17-megatron-hook) | log probability 计算前的自定义逻辑。 |
| [`--custom-megatron-before-train-step-hook-path`](#17-megatron-hook) | 每个训练步骤前的自定义逻辑。 |
| [`--slime-router-middleware-paths`](#18-slime-router-中间件---slime-router-middleware-paths) | 向 slime router 添加自定义中间件。 |

## 详细接口参考

### 1. Rollout 函数 (`--rollout-function-path`)

**默认值**: `slime.rollout.sglang_rollout.generate_rollout`

**用途**: 覆盖整个 rollout 生成逻辑。

**函数签名**:
```python
async def generate_rollout(args, rollout_id, *, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

**使用场景**:
- 实现复杂的多轮对话
- 添加自定义采样策略
- 在生成过程中集成外部工具或 API

**示例**: 参见 [examples/multi_agent/rollout_with_multi_agents.py](../../../examples/multi_agent/rollout_with_multi_agents.py)

---

### 2. 自定义生成函数 (`--custom-generate-function-path`)

**默认值**: `None`（使用内置生成函数）

**用途**: 仅覆盖默认 rollout 函数中的生成步骤。

**函数签名**:
```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample
```

**使用场景**:
- 实现工具调用（tool-calling）或函数调用（function-calling）能力
- 添加检索增强生成（RAG）
- 多轮对话处理

**示例**: 参见 [examples/search-r1/generate_with_search.py](../../../examples/search-r1/generate_with_search.py)

---

### 3. 奖励模型 (`--custom-rm-path`)

**默认值**: `None`（基于 `--rm-type` 使用内置奖励模型）

**用途**: 实现自定义奖励计算逻辑。

**函数签名**（单样本模式）:
```python
async def custom_rm(args, sample: Sample) -> float
```

**函数签名**（批量模式，当启用 `--group-rm` 时）:
```python
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]
```

**使用场景**:
- 自定义基于规则的奖励
- 集成外部奖励模型服务
- 多维度奖励信号

**内置选项** (`--rm-type`):
- `math`: 数学答案验证
- `dapo`: DAPO 风格评分
- `deepscaler`: DeepScaler 基于规则的奖励
- `f1`: F1 分数计算
- `gpqa`: GPQA 奖励计算
- `ifbench`: IFBench 奖励计算
- `remote_rm`: 远程奖励模型服务（需要 `--rm-url`）

---

### 4. 动态采样过滤器 (`--dynamic-sampling-filter-path`)

**默认值**: `None`

**用途**: 在动态采样过程中过滤样本（例如 DAPO 风格的过滤）。

**函数签名**:
```python
def filter_function(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput
```

**返回类型**:
```python
@dataclass
class DynamicFilterOutput:
    keep: bool  # 是否保留该样本组
    reason: str | None  # 过滤原因（用于日志）
```

**使用场景**:
- 过滤所有响应具有相同奖励的样本
- 实现课程学习策略
- 基于质量的样本选择

**示例**: `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`

---

### 5. Buffer 过滤器 (`--buffer-filter-path`)

**默认值**: `None`

**用途**: 在训练前过滤 rollout buffer 中的样本。

**函数签名**:
```python
def buffer_filter(samples: list[list[Sample]]) -> list[list[Sample]]
```

**使用场景**:
- 在训练前移除低质量样本
- 实现基于优先级的样本选择
- 平衡样本分布

---

### 6. Rollout 样本过滤器 (`--rollout-sample-filter-path`)

**默认值**: `None`

**用途**: 决定单个样本是否参与损失计算。

**函数签名**:
```python
def filter_function(args, samples: list[Sample]) -> None
```

**注意**: 此函数应直接修改每个 `Sample` 对象的 `remove_sample` 属性。

**使用场景**:
- 基于响应质量过滤样本
- 实现选择性训练策略

---

### 7. Rollout 全样本处理 (`--rollout-all-samples-process-path`)

**默认值**: `None`

**用途**: 在 rollout 后处理所有样本（包括被过滤的样本）。

**函数签名**:
```python
def process_function(args, samples: list[list[Sample]]) -> None
```

**使用场景**:
- 记录和分析所有生成的样本
- 计算过滤和保留样本的统计数据

---

### 8. Rollout 数据后处理 (`--rollout-data-postprocess-path`)

**默认值**: `None`

**用途**: 在计算 log probabilities 后对 rollout 数据进行后处理。

**函数签名**:
```python
def postprocess_function(args, samples: list[list[Sample]]) -> None
```

**使用场景**:
- 基于计算值更新损失掩码
- 向样本添加额外元数据

---

### 9. 自定义损失函数 (`--custom-loss-function-path`)

**默认值**: `None`（需要 `--loss-type custom_loss`）

**用途**: 实现自定义训练损失计算。

**使用场景**:
- 新颖的 RL 目标函数
- 多目标优化
- 自定义正则化项

---

### 10. 自定义 TIS/RS 函数 (`--custom-tis-function-path`)

**默认值**: `None`

**用途**: 实现用于离策略（off-policy）校正的自定义重要性采样。

**使用场景**:
- 自定义重要性采样比率计算
- 高级离策略校正方法

**示例**: `examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp`

---

### 11. 自定义 pg_loss Reducer (`--custom-pg-loss-reducer-function-path`)

**默认值**: `None`

**用途**: 自定义 pg_loss 的归约方式，其他指标（pg_clipfrac、ppo_kl、entropy_loss 等）仍使用默认的 sum_of_sample_mean。

**函数签名**:
```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]
```

**使用场景**:
- Dr.GRPO：除以常数而非有效 token 数
- 自定义损失归一化策略

**示例**: `examples/DrGRPO/custom_reducer.py:get_pg_loss_reducer`

---

### 12. 奖励后处理 (`--custom-reward-post-process-path`)

**默认值**: `None`（使用默认的 GRPO 归一化）

**用途**: 在优势计算前对奖励进行自定义后处理。

**使用场景**:
- 自定义奖励归一化策略
- 奖励塑形（reward shaping）

---

### 13. 样本转训练数据 (`--custom-convert-samples-to-train-data-path`)

**默认值**: `None`（使用内置转换逻辑）

**用途**: 覆盖样本到训练数据格式的转换逻辑。

**函数签名**:
```python
def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
) -> dict
```

**返回类型**:
```python
dict: {
    "tokens": list[list[int]],           # 每个样本的 token ID
    "response_lengths": list[int],        # 响应长度
    "rewards": list[float],               # 归一化后的奖励
    "raw_reward": list[float],            # 原始奖励
    "truncated": list[int],               # 截断标志（0 或 1）
    "sample_indices": list[int],          # 样本索引
    "loss_masks": list[list[int]],        # 每个样本的损失掩码
    # 可选字段：
    "round_number": list[int],            # 轮次编号（用于 rollout buffer）
    "rollout_log_probs": list,            # log 概率（用于离策略校正）
    "rollout_routed_experts": list,       # 路由专家（用于 MoE）
    "metadata": list,                     # 训练元数据
    "multimodal_train_inputs": list,      # 多模态张量（用于 VLM）
    "teacher_log_probs": list,            # 教师 log 概率（用于蒸馏）
}
```

**使用场景**:
- 处理 `list[list[Sample]]` 输入
- 自定义训练数据格式需求
  
---

### 14. 日志函数

#### 训练 Rollout 日志 (`--custom-rollout-log-function-path`)

**函数签名**:
```python
def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool
```

**返回值**: `True` 跳过默认日志，`False` 继续默认日志。

#### 评估 Rollout 日志 (`--custom-eval-rollout-log-function-path`)

**函数签名**:
```python
def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool
```

**返回值**: `True` 跳过默认日志，`False` 继续默认日志。

---

### 15. 数据源 (`--data-source-path`)

**默认值**: `slime.rollout.data_source.RolloutDataSourceWithBuffer`

**用途**: 覆盖 rollout 提示词的数据源。

**基类**: `slime.rollout.data_source.DataSource`

**必需方法**:
```python
class CustomDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """返回 num_samples 个样本"""
        
    def add_samples(self, samples: list[list[Sample]]):
        """将样本添加回数据源"""
        
    def save(self, rollout_id):
        """保存状态用于检查点"""
        
    def load(self, rollout_id=None):
        """从检查点加载状态"""
```

---

### 16. 评估函数 (`--eval-function-path`)

**默认值**: 与 `--rollout-function-path` 相同

**用途**: 专门为评估覆盖 rollout 函数。

**使用场景**:
- 评估时使用不同的采样参数
- 评估专用逻辑

---

### 17. Megatron Hook

#### Megatron 初始化 (`--custom-megatron-init-path`)

**函数签名**:
```python
def custom_init(args) -> None
```

**用途**: Megatron 设置后的自定义初始化。

#### Log Prob 前 Hook (`--custom-megatron-before-log-prob-hook-path`)

**函数签名**:
```python
def custom_hook(args, model, store_prefix) -> None
```

**用途**: log probability 计算前的自定义逻辑。

#### 训练步骤前 Hook (`--custom-megatron-before-train-step-hook-path`)

**函数签名**:
```python
def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
```

**用途**: 每个训练步骤前的自定义逻辑。

---

### 18. slime Router 中间件 (`--slime-router-middleware-paths`)

**用途**: 向 slime router 添加自定义中间件用于请求处理。

**使用场景**:
- 请求/响应转换
- 自定义路由逻辑
- 缓存和优化


