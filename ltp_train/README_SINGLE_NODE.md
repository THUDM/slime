# Slime 单机训练指南 (InternVL 多模态模型)

本文档介绍如何在**单机多卡**环境下使用 Slime 框架训练 InternVL 多模态模型。

相比多机训练，单机训练更简单、稳定，适合调试和中小规模实验。

## 目录

1. [快速开始](#1-快速开始)
2. [数据准备](#2-数据准备)
3. [Reward 函数定义](#3-reward-函数定义)
4. [训练脚本配置](#4-训练脚本配置)
5. [运行训练](#5-运行训练)
6. [监控与调试](#6-监控与调试)

---

## 1. 快速开始

### 1.1 一键训练（如果已准备好数据和 reward）

```bash
cd /mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime

# 直接运行训练脚本
bash scripts/train_my_task.sh
```

### 1.2 文件结构

```
slime/
├── data/
│   └── my_task_train.parquet          # 训练数据
├── slime/
│   └── reward/
│       └── my_task_reward.py          # Reward 函数
├── scripts/
│   └── train_my_task.sh               # 训练启动脚本
└── outputs/my_task/                   # 训练输出
```

---

## 2. 数据准备

### 2.1 数据格式

训练数据需要是 **parquet** 格式，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `problem` | string | 输入问题/指令 |
| `answer` | string | 标准答案 (用于计算 reward) |
| `images` | list[string] | 图片路径列表 |

### 2.2 创建数据

```python
import pandas as pd

# 示例数据
data = [
    {
        "problem": "Extract all text information from this image.",
        "answer": '{"company": "ABC Corp", "date": "2024-01-01"}',
        "images": ["/path/to/image1.jpg"]
    },
    {
        "problem": "What is the total amount on this invoice?",
        "answer": '{"total": "$150.00", "currency": "USD"}',
        "images": ["/path/to/image2.png"]
    }
]

df = pd.DataFrame(data)

# 保存完整数据
df.to_parquet("/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train.parquet", index=False)

# 保存调试数据（100条，用于快速验证）
df.head(100).to_parquet("/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train_debug.parquet", index=False)

print(f"数据已保存: {len(df)} 条")
```

### 2.3 验证数据

```bash
# 检查数据文件
python3 -c "
import pandas as pd
df = pd.read_parquet('data/my_task_train.parquet')
print(f'数据条数: {len(df)}')
print(f'字段: {list(df.columns)}')
print(f'\n第一条数据:')
print(df.iloc[0].to_dict())
"
```

---

## 3. Reward 函数定义

### 3.1 创建 Reward 文件

创建 `slime/reward/my_task_reward.py`：

```python
"""
My Task Reward Function
用于评估模型输出与标准答案的匹配程度
"""

import json
import re


def my_task_reward(predict_str: str, answer_str: str) -> float:
    """
    计算预测结果与标准答案的匹配程度

    Args:
        predict_str: 模型生成的答案 (字符串)
        answer_str: 标准答案 (字符串，通常是 JSON)

    Returns:
        reward: 0.0 ~ 1.0 之间的分数
    """
    try:
        # 解析标准答案
        ground_truth = json.loads(answer_str)

        # 从模型输出中提取 JSON
        # 模型输出可能包含 markdown 代码块
        json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*?\})'
        matches = re.findall(json_pattern, predict_str, re.DOTALL)

        if matches:
            json_str = matches[0][0] or matches[0][1] or matches[0][2]
            prediction = json.loads(json_str)
        else:
            prediction = json.loads(predict_str)

        # 计算匹配度
        correct_count = 0
        total_count = len(ground_truth)

        for key, value in ground_truth.items():
            if key in prediction and str(prediction[key]).lower() == str(value).lower():
                correct_count += 1

        return correct_count / total_count if total_count > 0 else 0.0

    except Exception:
        return 0.0
```

### 3.2 注册 Reward

在 `slime/reward/__init__.py` 中添加：

```python
from .my_task_reward import my_task_reward

__all__ = [
    # ... 其他 reward
    "my_task_reward",
]
```

### 3.3 测试 Reward 函数

```bash
python3 -c "
from slime.reward.my_task_reward import my_task_reward

# 测试用例
predict = '{\"company\": \"ABC Corp\", \"date\": \"2024-01-01\"}'
answer = '{\"company\": \"ABC Corp\", \"date\": \"2024-01-01\"}'
print(f'完全匹配: {my_task_reward(predict, answer)}')  # 应输出 1.0

predict = '{\"company\": \"ABC Corp\"}'
answer = '{\"company\": \"ABC Corp\", \"date\": \"2024-01-01\"}'
print(f'部分匹配: {my_task_reward(predict, answer)}')  # 应输出 0.5

predict = '错误的输出'
answer = '{\"company\": \"ABC Corp\"}'
print(f'无效输出: {my_task_reward(predict, answer)}')  # 应输出 0.0
"
```

---

## 4. 训练脚本配置

### 4.1 创建训练脚本

创建 `scripts/train_my_task.sh`：

```bash
#!/bin/bash
#
# My Task 单机训练脚本
#

set -e

# ========================================
# 配置参数
# ========================================

# 模型路径
MODEL_PATH="/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf"

# 数据路径（使用调试数据快速验证）
# DATA_PATH="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train_debug.parquet"
DATA_PATH="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train.parquet"

# 输出路径
OUTPUT_DIR="/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/my_task"

# GPU 数量
NUM_GPUS=8

# WandB 配置（可选）
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="slime-my-task"
export WANDB_NAME="my-task-test"
export WANDB_MODE=offline  # 使用 offline 模式

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ========================================
# 清理残留进程
# ========================================
echo "=========================================="
echo "  清理残留进程和临时文件"
echo "=========================================="
echo ""

echo_info "停止 Ray、SGLang、Redis 相关进程..."
pkill -9 -f ray 2>/dev/null || true
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f redis 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
sleep 2

echo_info "清理 Ray 临时文件..."
rm -rf /tmp/ray* 2>/dev/null || true
rm -rf /dev/shm/ray* 2>/dev/null || true
rm -rf /tmp/redis* 2>/dev/null || true
sleep 1

echo_info "残留进程清理完成 ✓"
echo ""

# ========================================
# 启动前检查
# ========================================
echo "=========================================="
echo "  训练启动检查"
echo "=========================================="
echo ""

# 1. 检查数据文件
echo_info "检查训练数据..."
if [ -f "$DATA_PATH" ]; then
    DATA_SIZE=$(ls -lh "$DATA_PATH" | awk '{print $5}')
    echo_info "数据文件存在: $DATA_PATH ($DATA_SIZE) ✓"
else
    echo_error "数据文件不存在: $DATA_PATH"
    exit 1
fi
echo ""

# 2. 检查模型路径
echo_info "检查模型路径..."
if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    echo_info "模型路径有效: $MODEL_PATH ✓"
else
    echo_error "模型路径无效或缺少 config.json"
    exit 1
fi
echo ""

# 3. 检查 GPU
echo_info "检查 GPU 可用性..."
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo_info "检测到 $GPU_COUNT 个 GPU"
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo_warn "可用 GPU ($GPU_COUNT) 少于配置 ($NUM_GPUS)，将使用 $GPU_COUNT 个 GPU"
    NUM_GPUS=$GPU_COUNT
fi
echo ""

# 4. 创建输出目录
echo_info "检查输出目录..."
mkdir -p "$OUTPUT_DIR"
echo_info "输出目录: $OUTPUT_DIR ✓"
echo ""

# ========================================
# 启动训练
# ========================================
echo "=========================================="
echo -e "  ${GREEN}所有检查通过，准备启动训练${NC}"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  - 模型: $MODEL_PATH"
echo "  - 数据: $DATA_PATH"
echo "  - 输出: $OUTPUT_DIR"
echo "  - GPU 数量: $NUM_GPUS"
echo ""

cd /mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime

# 运行训练
python examples/my_task/train_my_task.py

echo ""
echo_info "训练完成!"
echo_info "输出目录: $OUTPUT_DIR"
```

### 4.2 创建 Python 训练入口

创建 `examples/my_task/train_my_task.py`：

```python
#!/usr/bin/env python3
"""
My Task 训练入口
单机多卡 GRPO 训练
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime")

from slime.utils.external_utils import U


# ========================================
# 配置参数
# ========================================

MODEL_PATH = "/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf"
DATA_PATH = "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train.parquet"
OUTPUT_DIR = "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/my_task"
NUM_GPUS = 8

# ========================================
# 构建训练参数
# ========================================

train_args = f"""
--hf-checkpoint {MODEL_PATH}
--prompt-data {DATA_PATH}
--input-key problem
--label-key answer
--apply-chat-template
--multimodal-keys '{{"image":"images"}}'
--custom-rm-path slime.reward.my_task_reward.my_task_reward

--rollout-shuffle
--num-rollout 100
--rollout-batch-size 16
--n-samples-per-prompt 4
--rollout-max-response-len 2048
--rollout-temperature 1.0
--global-batch-size 64

--train-backend fsdp
--gradient-checkpointing
--update-weight-buffer-size 536870912

--advantage-estimator grpo
--kl-loss-coef 0.0
--kl-loss-type seq_monkey
--kl-coef 0.01
--entropy-coef 0.001
--eps-clip 0.2
--eps-clip-high 0.28

--optimizer adam
--lr 5.0e-7
--lr-decay-style cosine
--weight-decay 0.01

--rollout-num-gpus-per-engine 4
--sglang-mem-fraction-static 0.7
--sglang-decode-log-interval 500
--sglang-enable-metrics
--attn-implementation flash_attention_2
--sglang-cuda-graph-max-bs 32

--save {OUTPUT_DIR}
--save-interval 10

--actor-num-nodes 1
--actor-num-gpus-per-node {NUM_GPUS}
--colocate

--rollout-stop-token-ids 151645
"""

# WandB 配置
use_wandb = os.environ.get("WANDB_API_KEY") is not None
if use_wandb:
    train_args += """
--use-wandb
--wandb-project slime-my-task
--wandb-group my-task-test
--disable-wandb-random-suffix
"""

# ========================================
# 启动训练
# ========================================

if __name__ == "__main__":
    print("=" * 50)
    print("My Task 训练启动")
    print("=" * 50)
    print(f"模型: {MODEL_PATH}")
    print(f"数据: {DATA_PATH}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"GPU: {NUM_GPUS}")
    print("=" * 50)
    print()

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,  # FSDP 后端
    )
```

---

## 5. 运行训练

### 5.1 首次运行

```bash
cd /mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime

# 给脚本添加执行权限
chmod +x scripts/train_my_task.sh

# 运行训练
bash scripts/train_my_task.sh
```

### 5.2 快速调试

如果数据量大，先用调试数据验证：

```python
# 在 train_my_task.py 中临时修改 DATA_PATH
DATA_PATH = "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train_debug.parquet"
```

---

## 6. 监控与调试

### 6.1 查看实时日志

训练过程中会输出：
- `train_wait` - 等待数据
- `rollout/N` - 第 N 轮 rollout
- `train/step` - 训练步数
- `rollout/raw_reward` - 平均 reward

### 6.2 检查 WandB 日志

如果使用 offline 模式：

```bash
# 查看本地日志
ls -la /root/wandb/

# 训练完成后同步到云端
wandb online
wandb sync /root/wandb/offline-run-*/
```

### 6.3 常见问题

**Q: 显存不足 (OOM)**

A: 减小以下参数：
```python
--rollout-batch-size 8          # 原来是 16
--sglang-mem-fraction-static 0.6  # 原来是 0.7
--rollout-max-response-len 1024   # 原来是 2048
```

**Q: 数据加载慢**

A: 使用调试数据或检查图片路径是否正确

**Q: Reward 始终为 0**

A: 检查：
1. `answer` 字段是否为有效 JSON
2. `images` 路径是否正确
3. Reward 函数解析逻辑是否正确

---

## 附录：从单机扩展到多机

单机验证通过后，如果需要扩展到多机：

1. **复制配置文件**
   ```bash
   cp ltp_train/config.yaml ltp_train/config_multinode.yaml
   ```

2. **修改关键参数**
   ```yaml
   env:
     - name: ACTOR_NUM_NODES
       value: "2"  # 从 1 改为 2
   ```

3. **提交到 LTP**
   ```bash
   kubectl create -f ltp_train/config_multinode.yaml -n aigc-ceph
   ```

详细配置参考 `README.md`（多机版本）。

---

**有问题请联系**: zhengmingming@baidu.com
