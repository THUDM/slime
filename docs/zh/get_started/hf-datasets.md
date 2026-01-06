# HuggingFace Datasets 集成

本文档介绍如何使用 HuggingFace Datasets 加载大规模数据集（100GB+）进行训练。

## 1. 快速开始

### 何时使用

| 场景 | 推荐方案 |
|------|---------|
| 数据集 > 10GB | **HF Datasets**（流式加载，内存 < 1GB） |
| 数据集 < 10GB | Legacy Dataset（默认，全量加载） |

### 基本用法（RL 训练）

```bash
python train.py \
  --use-hf-datasets \
  --hf-datasets-num-samples 17000 \
  --prompt-data zhuzilin/dapo-math-17k \
  --rollout-batch-size 32 \
  --num-rollout 100
```

### SFT 训练用法

```bash
python train.py \
  --use-hf-datasets \
  --hf-datasets-num-samples 335122 \
  --prompt-data nvidia/Nemotron-Agentic-v1 \
  --hf-dataset-split interactive_agent \
  --input-key messages \
  --tool-key tools \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --loss-type sft_loss \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --rollout-batch-size 128 \
  --num-rollout 60
```

**必需参数**：
- `--use-hf-datasets`：启用 HF Datasets 流式模式
- `--hf-datasets-num-samples`：数据集样本数（用于 epoch 追踪）

### 支持的数据格式

| 格式 | 示例 |
|------|------|
| HuggingFace Hub | `zhuzilin/dapo-math-17k` |
| HuggingFace Hub (指定 split) | `nvidia/Nemotron-Agentic-v1` + `--hf-dataset-split interactive_agent` |
| 本地 JSONL | `/path/to/data.jsonl` |
| 本地 Parquet | `/path/to/data.parquet` |

---

## 2. 参数详解

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-hf-datasets` | `False` | 启用 HF Datasets 流式模式 |
| `--hf-datasets-num-samples` | **必需** | 数据集样本数（用于 epoch 边界计算） |
| `--hf-dataset-split` | `train` | 数据集 split 名称（如 `train`、`interactive_agent`） |
| `--hf-dataset-shuffle-buffer` | `10000` | Shuffle buffer 大小 |
| `--hf-dataset-buffer-size` | `100` | 预取 buffer 大小 |
| `--hf-dataset-num-proc` | `8` | DataLoader worker 数量 |

### SFT 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rollout-function-path` | - | SFT 使用 `slime.rollout.sft_rollout.generate_rollout` |
| `--loss-type` | `policy_loss` | 损失类型，SFT 使用 `sft_loss` |
| `--calculate-per-token-loss` | `False` | 按 token 计算损失（推荐 SFT 开启） |
| `--disable-compute-advantages-and-returns` | `False` | 禁用 advantage 计算（SFT 必须开启） |
| `--input-key` | `input` | 输入字段名（多轮对话使用 `messages`） |
| `--tool-key` | - | 工具定义字段名（如 `tools`） |

### 数据格式

**JSONL 格式**（每行一个 JSON 对象）：

```json
{"input": "问题文本", "label": "答案文本", "metadata": {"sample_id": 0}}
```

**字段映射**：
- `--input-key input`：输入字段名
- `--label-key label`：标签字段名
- `--metadata-key metadata`：元数据字段名

**Chat Template 格式**（需配合 `--apply-chat-template`）：

```json
{"input": [{"role": "user", "content": "问题"}], "label": "答案"}
```

**多轮对话格式**（SFT 模式，配合 `--input-key messages`）：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    {"role": "user", "content": "请解释一下 Python"},
    {"role": "assistant", "content": "Python 是一种高级编程语言..."}
  ],
  "tools": [{"type": "function", "function": {"name": "search", "description": "搜索信息"}}]
}
```

**多轮对话 loss_mask 说明**：
- SFT 模式下，只对 assistant 的回复计算损失
- `loss_mask` 中 1 表示需要计算损失的 token，0 表示跳过
- 多轮对话中，中间的 user/system 轮次会被标记为 0

---

## 3. Checkpoint 支持

### 保存

训练时自动保存数据集状态：

```bash
python train.py --use-hf-datasets --save /path/to/ckpt --save-interval 50
```

### 恢复

从 checkpoint 继续训练：

```bash
python train.py --use-hf-datasets --load /path/to/ckpt
```

### 状态内容

Checkpoint 包含以下数据集状态：

```python
{
    "epoch_id": 2,                    # 当前 epoch
    "consumed_count": 15234,          # 当前 epoch 已消费样本数
    "global_consumed_count": 45234,   # 全局已消费样本数
    "hf_state_dict": {...}            # HF 原生迭代器状态
}
```

---

## 4. 故障排查

### 问题 1: ValueError: --hf-datasets-num-samples is required

**原因**：使用 `--use-hf-datasets` 时必须指定样本数

**解决**：添加 `--hf-datasets-num-samples <数量>`

### 问题 2: 训练卡住/数据加载慢

**原因**：DataLoader worker 不足

**解决**：增加 `--hf-dataset-num-proc 16`

### 问题 3: Dataset exhausted while skipping

**原因**：Checkpoint 损坏或 epoch 边界错误

**解决**：
```bash
rm -rf /path/to/ckpt/rollout/global_dataset_state_dict_*.pt
```

---

## 5. 开发者参考

### 架构概述

```
RolloutDataSource
    └── HFIterableDatasetAdapter
            └── PyTorch DataLoader
                    └── HuggingFace IterableDataset
```

**核心设计**：
- 使用 PyTorch DataLoader 进行多进程预取
- 使用 HF 原生 `state_dict()` / `load_state_dict()` 支持 checkpoint
- 使用 HF `set_epoch()` 实现可复现的 shuffle

### SFT Rollout 函数

SFT 模式使用专用的 rollout 函数 `slime.rollout.sft_rollout.generate_rollout`：

```python
# 核心逻辑
def generate_rollout(args, rollout_id, *, evaluation=False):
    # 1. 获取批量数据
    # 2. 应用 chat template，生成 loss_mask
    # 3. 返回 Sample 列表（无需生成，直接使用数据集内容）
    pass
```

**loss_mask 计算**：
- 对于多轮对话，只对 assistant 回复部分设置 mask=1
- `response_length` 计算为从第一个 mask=1 位置到序列末尾的长度

### 添加新 Backend

继承 `HFDatasetAdapterBase` 并实现 4 个方法：

```python
from slime.utils.hf_dataset import HFDatasetAdapterBase

class MyDatasetAdapter(HFDatasetAdapterBase):
    def get_next_batch(self, num_samples: int) -> list[Sample]:
        """返回下一批样本"""
        pass

    def shuffle(self, new_epoch_id: int):
        """基于 epoch_id 的 shuffle"""
        pass

    def get_checkpoint_state(self) -> dict:
        """获取 checkpoint 状态"""
        pass

    def load_checkpoint_state(self, state: dict):
        """恢复 checkpoint 状态"""
        pass
```

---

## 参考资料

- [HuggingFace Datasets 文档](https://huggingface.co/docs/datasets)
- 源码：`slime/utils/hf_dataset.py`
- SFT Rollout：`slime/rollout/sft_rollout.py`
- 测试：`tests/test_hf_datasets.py`
- SFT 测试：`tests/test_qwen3-0.6B_hf_datasets_sft.py`
