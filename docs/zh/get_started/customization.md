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

## 通过 customization 接口实现 agentic workflow

agentic workflow——multi-turn tool use、sandbox interaction、environment feedback、verifier/test-based reward——是一类重要的训练数据生成 workflow。它们通过 slime 已有的 customization 接口接入，slime 本身并不需要变成一个单独的 agent framework。

绝大多数 agentic 场景下，**建议从 `--custom-generate-function-path` 加 `--custom-rm-path` 开始**，只有在默认 rollout 循环无法满足需求时再去覆盖整个 rollout function。

| 想做的事 | 应使用的接口 |
| :--- | :--- |
| 让每条 sample 跑自定义的 agent loop、tool call、RAG、sandbox 执行、browser/terminal 交互或多轮生成，同时复用 slime 默认 rollout loop | [`--custom-generate-function-path`](#2-自定义生成函数---custom-generate-function-path) |
| 实现 verifier reward、test-based reward、environment 成功判定、rule-based reward 或调用外部 reward 服务 | [`--custom-rm-path`](#3-奖励模型---custom-rm-path) |
| 替换整个 rollout 编排（只在 per-sample 自定义不够用时使用） | [`--rollout-function-path`](#1-rollout-函数---rollout-function-path) |
| 控制任务采样、缓冲、回填，或自定义 prompt / task 数据源 | [`--data-source-path`](#15-数据源---data-source-path) |
| 给 agentic 输出附加自定义 loss mask、metadata，或转换成训练数据 | [`--rollout-data-postprocess-path`](#8-rollout-数据后处理---rollout-data-postprocess-path)、[`--custom-convert-samples-to-train-data-path`](#13-样本转训练数据---custom-convert-samples-to-train-data-path) |
| 调试长耗时的 custom generation、verifier、tool call 或 sandbox 调用 | [`slime.utils.trace_utils`](../developer_guide/trace.md) 中的 trace 工具 |

这一模式的原生示例是 [`examples/search-r1`](../../../examples/search-r1/)：通过 `--custom-generate-function-path` 接入搜索增强的多轮生成，外层仍然走 slime 默认的 `sglang_rollout`。互补的示例可参考 [`examples/multi_agent`](../../../examples/multi_agent/README.md) 中基于 `--rollout-function-path` 的多 agent 模式，以及 [`examples/fully_async`](../../../examples/fully_async/README.md) 中适合 long-tail agentic 场景的 fully-async rollout。

## 详细接口参考

### 1. Rollout 函数 (`--rollout-function-path`)

**默认值**: `slime.rollout.sglang_rollout.generate_rollout`

**用途**: 覆盖整个 rollout 生成逻辑。

**函数签名**:
```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
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
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]
```

**使用场景**:
- 实现工具调用（tool-calling）或函数调用（function-calling）能力
- 添加检索增强生成（RAG）
- 多轮对话处理

#### 一个 prompt 产生多个训练样本

在 subagent、multi-agent、context compact 等 agentic 场景中，一次 prompt rollout 可能会自然拆成多个可训练片段。例如：主 agent 调用 subagent 后，subagent 的轨迹和主 agent 的后续轨迹都需要参与训练；或者发生 compact 后，compact 前后的上下文被切成多个 segment。

这种情况下不需要重写整个 rollout 函数，`custom_generate` 可以直接返回 `list[Sample]`。关键是：这些由同一次 rollout 拆出来的 sibling samples 必须设置相同的 `rollout_id`，这样 slime 会在训练切分和 loss 聚合时把它们视作同一次 rollout，而不是重复计数为多次独立 rollout。

```python
import copy

from slime.utils.types import Sample


async def custom_generate(args, sample: Sample, sampling_params: dict) -> list[Sample]:
    segments = await run_agent_and_split_segments(args, sample, sampling_params)
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index

    samples: list[Sample] = []
    for segment in segments:
        s = copy.copy(sample)
        s.tokens = segment.tokens
        s.response = segment.response
        s.response_length = segment.response_length
        s.loss_mask = segment.loss_mask
        s.reward = segment.reward
        s.status = Sample.Status.COMPLETED
        s.rollout_id = rollout_id
        samples.append(s)
    return samples
```

如果一个完整 trajectory 只有一个总奖励、但被拆成了 `K` 个训练片段，常见做法是在这些片段之间分配这个奖励（例如每个片段写入 `reward / K`），避免把同一次 rollout 的奖励重复放大。

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
def buffer_filter(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]
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
def process_function(args, samples: list[list[Sample]], data_source) -> None
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
        """保存状态用于 ckpt"""
        
    def load(self, rollout_id=None):
        """从 ckpt 加载状态"""
    
    def __len__(self) -> int:
    """
        返回当前数据源中可用样本的数量。该数量可能会随着样本的获取或添加而变化。
    """
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

### 18. MoE 路由重放

通过记录和重放专家路由决策来稳定 MoE RL 训练。

| 参数 | 说明 |
| --- | --- |
| `--use-routing-replay` | 训练中前向-反向路由一致性。([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) |
| `--use-rollout-routing-replay` | R3：在训练时重放 rollout 阶段的路由。slime 默认的 `sglang_router` 路径支持该功能。([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)) |

---

### 19. Disk 权重同步 Post-Write Hook（`--custom-update-weight-post-write-path`）

**签名**：
```python
def hook(args, version_dir: str, rollout_engines) -> None
```

**用途**：在 disk 权重同步（`--update-weight-transport disk`，full 或 delta 模式）的文件写完之后、
engine 读取之前，在每个训练 rank 上调用。用于在非 POSIX 共享文件系统上发布写入——例如 commit
一个对象存储挂载——否则其他 host 无法看到这些文件。hook 会在每个 rank 上被调用，需要自行去重
（例如每个容器只执行一次）。

读取侧的对应 hook 运行在推理引擎内部、engine 覆盖的每个 host 上，因此它是一个 sglang server
参数而不是 slime hook：传入 `--sglang-custom-pull-weights-pre-read-hook <import.path>`，签名为
`hook(source_dir: str, target_version: int)`——在 `/pull_weights` 读取已发布权重之前调用
（例如刷新挂载视图）。完整机制见 [Delta 权重同步](../advanced/delta-weight-sync.md)。

## 自定义函数路径的测试

slime 现在也提供了一组 CPU 契约测试，用于校验这些 customization 接口。测试会通过字符串形式的导入路径来动态加载组件，因此既能回归仓库内置 hook，也能验证用户通过和训练时完全相同的 CLI 参数传入的自定义实现。

这些测试统一放在 `tests/plugin_contracts/` 目录下，并按 hook 形态归并成少数几个文件：

- `tests/plugin_contracts/test_plugin_rollout_contracts.py`
  覆盖 `--rollout-function-path`
- `tests/plugin_contracts/test_plugin_generate_contracts.py`
  覆盖 `--custom-generate-function-path`
- `tests/plugin_contracts/test_plugin_path_loading_contracts.py`
  覆盖 `--eval-function-path`、`--custom-rm-path`、`--dynamic-sampling-filter-path`、`--buffer-filter-path`、`--data-source-path`、`--rollout-sample-filter-path`、`--rollout-all-samples-process-path`
- `tests/plugin_contracts/test_plugin_runtime_hook_contracts.py`
  覆盖 `--custom-rollout-log-function-path`、`--custom-eval-rollout-log-function-path`、`--custom-reward-post-process-path`、`--custom-convert-samples-to-train-data-path`、`--rollout-data-postprocess-path`

本地运行全部 customization 契约测试：

```bash
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

每个测试文件也支持直接通过 `python tests/plugin_contracts/<file>.py` 执行，这样可以和 `run-ci-changed` 保持兼容。

CI 中也提供了独立的 `run-ci-cpu-unittest` label，给 PR 打上该标签后会并行运行 CPU-only 的单元测试任务，包含上述契约测试以及其他轻量单测（无需 GPU）。

如果你要验证自己的自定义实现，可以直接设置环境变量，例如 `SLIME_CONTRACT_ROLLOUT_FUNCTION_PATH`、`SLIME_CONTRACT_CUSTOM_RM_PATH`，也可以在直接运行测试文件时传参，例如：

```bash
python tests/plugin_contracts/test_plugin_rollout_contracts.py \
  --rollout-function-path my_project.custom_rollout.generate_rollout
```

验证时只需将插件路径替换成你的模块路径，断言逻辑（函数签名、返回结构、副作用）保持不变即可。
