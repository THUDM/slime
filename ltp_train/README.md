# Slime + LTP 平台训练指南 (InternVL 多模态模型)

本文档介绍如何在 LTP 平台上使用 Slime 框架训练 InternVL 多模态模型。

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [Reward 函数定义](#3-reward-函数定义)
4. [训练配置](#4-训练配置)
5. [提交训练](#5-提交训练)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 代码路径

确保你的 Slime 代码在共享存储上，所有节点都能访问：

```bash
/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime
```

### 1.2 模型路径

准备 HuggingFace 格式的模型：

```bash
/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf
```

模型需要包含：
- `config.json` - 模型配置
- `pytorch_model-*.bin` 或 `model-*.safetensors` - 权重文件
- `tokenizer.json`, `tokenizer_config.json` - 分词器
- `preprocessor_config.json` - 处理器配置

---

## 2. 数据准备

### 2.1 数据格式

训练数据需要是 **parquet** 或 **jsonl** 格式，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `problem` | string | 输入问题/指令 |
| `answer` | string | 标准答案 (用于计算 reward) |
| `images` | list[string] | 图片路径列表 (相对路径或绝对路径) |

### 2.2 数据示例

**Parquet 格式示例：**

```python
import pandas as pd

# 创建示例数据
data = [
    {
        "problem": "Extract all text information from this image.",
        "answer": "{\"company\": \"ABC Corp\", \"date\": \"2024-01-01\"}",
        "images": ["/path/to/image1.jpg"]
    },
    {
        "problem": "What is the total amount on this invoice?",
        "answer": "{\"total\": \"$150.00\", \"currency\": \"USD\"}",
        "images": ["/path/to/image2.png"]
    }
]

df = pd.DataFrame(data)
df.to_parquet("/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train.parquet", index=False)
```

**JSONL 格式示例：**

```json
{"problem": "Extract all text information from this image.", "answer": "{\"company\": \"ABC Corp\"}", "images": ["/path/to/image1.jpg"]}
{"problem": "What is the total amount?", "answer": "{\"total\": \"$150.00\"}", "images": ["/path/to/image2.png"]}
```

### 2.3 调试数据

建议先用小规模数据测试（如 100 条）：

```python
# 取前 100 条作为调试数据
debug_df = df.head(100)
debug_df.to_parquet("/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train_debug.parquet", index=False)
```

---

## 3. Reward 函数定义

### 3.1 Reward 函数位置

在 `slime/reward/` 目录下创建新的 reward 文件，例如 `my_task_reward.py`：

```python
"""
My Task Reward Function
"""

import json
import re
from typing import Any


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
        # 尝试解析标准答案
        ground_truth = json.loads(answer_str)

        # 尝试从模型输出中提取 JSON
        # 模型输出可能包含 markdown 代码块，需要提取
        json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*?\})'
        matches = re.findall(json_pattern, predict_str, re.DOTALL)

        if matches:
            # 取第一个匹配的 JSON
            json_str = matches[0][0] or matches[0][1] or matches[0][2]
            prediction = json.loads(json_str)
        else:
            # 尝试直接解析整个输出
            prediction = json.loads(predict_str)

        # 计算匹配度 (示例：计算键值对匹配比例)
        correct_count = 0
        total_count = len(ground_truth)

        for key, value in ground_truth.items():
            if key in prediction and prediction[key] == value:
                correct_count += 1

        return correct_count / total_count if total_count > 0 else 0.0

    except json.JSONDecodeError:
        # 如果解析失败，给 0 分
        return 0.0
    except Exception:
        return 0.0


# 批处理版本 (可选，用于优化性能)
def my_task_reward_batch(predictions: list[str], answers: list[str]) -> list[float]:
    """批量计算 reward"""
    return [my_task_reward(p, a) for p, a in zip(predictions, answers)]
```

### 3.2 注册 Reward 函数

在 `slime/reward/__init__.py` 中添加：

```python
from .my_task_reward import my_task_reward

__all__ = [
    # ... 其他 reward 函数
    "my_task_reward",
]
```

### 3.3 Reward 函数路径配置

在 `config.yaml` 中指定 reward 函数路径：

```yaml
trainargs:
  custom_rm_path: "slime.reward.my_task_reward.my_task_reward"
```

---

## 4. 训练配置

### 4.1 配置文件 (config.yaml)

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: internvl-my-task-grpo
  namespace: aigc-ceph
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: pytorch
              image: ccr-2vdh3abt-vpc.ccr.bj.volces.com/rd-zhengmingming/internvl3-rlhf:20250305-v5
              command: ["/bin/bash"]
              args:
                - "-c"
                - |
                  # 设置 Slime 路径
                  export SLIME_PATH=/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime

                  # 启动训练
                  bash /mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/ltp_train/run.sh
              resources:
                limits:
                  nvidia.com/gpu: "8"
              volumeMounts:
                - name: cephfs-volume
                  mountPath: /mnt/cfs_bj_mt
              env:
                - name: MODEL_PATH
                  value: "/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf"
                - name: TRAIN_DATA
                  value: "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train.parquet"
                - name: SAVE_PATH
                  value: "/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/my_task"
                - name: NUM_ROLLOUT
                  value: "100"
                - name: ACTOR_NUM_NODES
                  value: "2"
                - name: ACTOR_NUM_GPUS_PER_NODE
                  value: "8"
                - name: CUSTOM_RM_PATH
                  value: "slime.reward.my_task_reward.my_task_reward"
          volumes:
            - name: cephfs-volume
              cephfs:
                monitors:
                  - 10.139.1.76:6789
                  - 10.139.1.77:6789
                  - 10.139.1.78:6789
                user: admin
                path: /
                secretRef:
                  name: ceph-secret
```

### 4.2 run.sh 配置

```bash
#!/bin/bash
# Slime 训练启动脚本

set -e

# ------------------------------------------------------------------------
# 路径配置 (从环境变量读取或设置默认值)
# ------------------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"/mnt/cfs_bj_mt/experiments/zhengmingming/qfocr-annv9-30k-s4-qwen3-4b-v30-new-vocab-0303/iter_0004600_hf"}
TRAIN_DATA=${TRAIN_DATA:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/data/my_task_train.parquet"}
SAVE_PATH=${SAVE_PATH:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime/outputs/my_task"}
LOGS_PATH=${LOGS_PATH:-"${SAVE_PATH}/logs"}

mkdir -p ${SAVE_PATH}
mkdir -p ${LOGS_PATH}

# ------------------------------------------------------------------------
# 训练参数配置
# ------------------------------------------------------------------------
NUM_ROLLOUT=${NUM_ROLLOUT:-"100"}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-"16"}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-"4"}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-"2048"}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-"1.0"}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-"64"}
LR=${LR:-"5.0e-7"}
TRAIN_BACKEND=${TRAIN_BACKEND:-"fsdp"}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-"2"}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-"8"}
COLOCATE=${COLOCATE:-"true"}

# Reward 函数路径
CUSTOM_RM_PATH=${CUSTOM_RM_PATH:-"slime.reward.my_task_reward.my_task_reward"}

# 优化器参数
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-"grpo"}
KL_LOSS_COEF=${KL_LOSS_COEF:-"0.0"}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-"seq_monkey"}
KL_COEF=${KL_COEF:-"0.01"}
ENTROPY_COEF=${ENTROPY_COEF:-"0.001"}
EPS_CLIP=${EPS_CLIP:-"0.2"}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-"0.28"}
OPTIMIZER=${OPTIMIZER:-"adam"}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-"cosine"}
WEIGHT_DECAY=${WEIGHT_DECAY:-"0.01"}
ADAM_BETA1=${ADAM_BETA1:-"0.9"}
ADAM_BETA2=${ADAM_BETA2:-"0.999"}

# SGLang 配置
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-"4"}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-"0.7"}

# 保存配置
SAVE_INTERVAL=${SAVE_INTERVAL:-"10"}

# WandB 配置
USE_WANDB=${USE_WANDB:-"true"}
WANDB_PROJECT=${WANDB_PROJECT:-"internvl-my-task"}
WANDB_GROUP=${WANDB_GROUP:-"internvl3.5-4b-my-task"}
WANDB_MODE=${WANDB_MODE:-"offline"}

# ------------------------------------------------------------------------
# 启动 Ray 集群
# ------------------------------------------------------------------------
echo "=========================================="
echo "  Slime + LTP 平台训练启动"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  - 模型: ${MODEL_PATH}"
echo "  - 数据: ${TRAIN_DATA}"
echo "  - 输出: ${SAVE_PATH}"
echo "  - Reward: ${CUSTOM_RM_PATH}"
echo "  - 节点数: ${ACTOR_NUM_NODES}"
echo "  - 每节点 GPU: ${ACTOR_NUM_GPUS_PER_NODE}"
echo ""

# 清理残留进程
echo "清理残留进程..."
pkill -9 -f "sglang" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
pkill -9 -f "gcs_server" 2>/dev/null || true
pkill -9 -f "plasma" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

rm -rf /tmp/ray/* 2>/dev/null || true
rm -rf /dev/shm/ray_* 2>/dev/null || true
sleep 2

echo "启动 Ray Head..."
ray start --head --node-ip-address 127.0.0.1 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} --disable-usage-stats
sleep 5

# ------------------------------------------------------------------------
# 等待 Worker 节点加入
# ------------------------------------------------------------------------
EXPECTED_NODES=$((ACTOR_NUM_NODES))
echo "等待 ${EXPECTED_NODES} 个节点加入 Ray 集群..."

for i in {1..60}; do
    CURRENT_NODES=$(ray status --address 127.0.0.1:6379 2>/dev/null | grep -c "ray::" || echo "0")
    echo "第 $i 秒: 当前 ${CURRENT_NODES} 个节点"

    if [ "${CURRENT_NODES}" -ge "${EXPECTED_NODES}" ]; then
        echo "所有节点已加入!"
        break
    fi

    if [ $i -eq 60 ]; then
        echo "等待超时，但继续执行..."
    fi

    sleep 1
done

echo "当前 Ray 集群状态:"
ray status --address 127.0.0.1:6379 2>/dev/null || true

# ------------------------------------------------------------------------
# 构建训练参数
# ------------------------------------------------------------------------
CKPT_ARGS="--hf-checkpoint ${MODEL_PATH}"

ROLLOUT_ARGS=(
    "--prompt-data ${TRAIN_DATA}"
    "--input-key problem"
    "--label-key answer"
    "--apply-chat-template"
    "--rollout-shuffle"
    "--num-rollout ${NUM_ROLLOUT}"
    "--rollout-batch-size ${ROLLOUT_BATCH_SIZE}"
    "--n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}"
    "--rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}"
    "--rollout-temperature ${ROLLOUT_TEMPERATURE}"
    "--global-batch-size ${GLOBAL_BATCH_SIZE}"
    "--rollout-stop-token-ids 151645"
)

MULTIMODAL_ARGS='--multimodal-keys {"image":"images"}'
REWARD_ARGS="--custom-rm-path ${CUSTOM_RM_PATH}"

FSDP_ARGS=(
    "--train-backend ${TRAIN_BACKEND}"
    "--gradient-checkpointing"
    "--update-weight-buffer-size 536870912"
)

GRPO_ARGS=(
    "--advantage-estimator ${ADVANTAGE_ESTIMATOR}"
    "--kl-loss-coef ${KL_LOSS_COEF}"
    "--kl-loss-type ${KL_LOSS_TYPE}"
    "--kl-coef ${KL_COEF}"
    "--entropy-coef ${ENTROPY_COEF}"
    "--eps-clip ${EPS_CLIP}"
    "--eps-clip-high ${EPS_CLIP_HIGH}"
)

OPTIMIZER_ARGS=(
    "--optimizer ${OPTIMIZER}"
    "--lr ${LR}"
    "--lr-decay-style ${LR_DECAY_STYLE}"
    "--weight-decay ${WEIGHT_DECAY}"
    "--adam-beta1 ${ADAM_BETA1}"
    "--adam-beta2 ${ADAM_BETA2}"
)

SGLANG_ARGS=(
    "--rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    "--sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}"
    "--sglang-decode-log-interval 500"
    "--sglang-enable-metrics"
    "--attn-implementation flash_attention_2"
    "--sglang-cuda-graph-max-bs 32"
)

SAVE_ARGS=(
    "--save ${SAVE_PATH}"
    "--save-interval ${SAVE_INTERVAL}"
)

WANDB_ARGS=""
if [[ "${USE_WANDB}" == "true" ]]; then
    WANDB_ARGS="--use-wandb --wandb-project ${WANDB_PROJECT} --wandb-group ${WANDB_GROUP} --wandb-mode ${WANDB_MODE}"
fi

CLUSTER_ARGS=(
    "--actor-num-nodes ${ACTOR_NUM_NODES}"
    "--actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}"
)

if [[ "${COLOCATE}" == "true" ]]; then
    CLUSTER_ARGS+=("--colocate")
fi

# ------------------------------------------------------------------------
# 提交训练任务
# ------------------------------------------------------------------------
echo "提交 Slime 训练任务..."

SLIME_PATH=${SLIME_PATH:-"/mnt/cfs_bj_mt/workspace/zhengmingming/rl_from_zero/slime"}

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "'${SLIME_PATH}'",
    "MASTER_ADDR": "127.0.0.1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_DEBUG": "WARN",
    "NCCL_SOCKET_IFNAME": "eth,ens,ib,bond",
    "no_proxy": "127.0.0.1"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${SLIME_PATH}/train.py \
    ${CKPT_ARGS} \
    ${ROLLOUT_ARGS[@]} \
    ${MULTIMODAL_ARGS} \
    ${REWARD_ARGS} \
    ${FSDP_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${SAVE_ARGS[@]} \
    ${WANDB_ARGS} \
    ${CLUSTER_ARGS[@]} \
    2>&1 | tee -a ${LOGS_PATH}/train_${HOSTNAME}_$(date +%Y%m%d-%H%M%S).log

echo "训练任务已提交"
```

---

## 5. 提交训练

### 5.1 提交到 LTP 平台

```bash
kubectl create -f config.yaml -n aigc-ceph
```

### 5.2 查看日志

```bash
# 查看 Pod 状态
kubectl get pods -n aigc-ceph | grep internvl-my-task

# 查看 Master 节点日志
kubectl logs -f internvl-my-task-grpo-master-0 -n aigc-ceph

# 查看 Worker 节点日志
kubectl logs -f internvl-my-task-grpo-worker-0 -n aigc-ceph
```

### 5.3 监控训练

如果使用 WandB offline 模式，训练完成后同步：

```bash
wandb sync /root/wandb/offline-run-*/
```

---

## 6. 常见问题

### Q1: 数据加载太慢

**解决**: 创建小规模调试数据测试：

```python
debug_df = df.head(100)
debug_df.to_parquet("data/my_task_train_debug.parquet", index=False)
```

### Q2: Reward 函数如何调试

**解决**: 单独测试 reward 函数：

```python
from slime.reward.my_task_reward import my_task_reward

predict = '{"company": "ABC Corp"}'
answer = '{"company": "ABC Corp", "date": "2024-01-01"}'

reward = my_task_reward(predict, answer)
print(f"Reward: {reward}")  # 输出: 0.5
```

### Q3: 模型加载报错 `AutoModelForImageTextToText`

**解决**: 确保使用的是 InternVL 非 HF 官方格式（有 `llm_config` 字段的 config.json），并且 `get_model_cls()` 方法优先检测 `llm_config`。

### Q4: NCCL 网络错误

**解决**: 确保 `RUNTIME_ENV_JSON` 中包含：

```json
{
  "NCCL_IB_DISABLE": "1",
  "NCCL_SOCKET_IFNAME": "eth,ens,ib,bond"
}
```

### Q5: 如何调整生成参数

**解决**: 在 `ROLLOUT_ARGS` 中修改：

```bash
ROLLOUT_ARGS=(
    "--rollout-temperature 0.7"      # 降低温度使输出更确定
    "--rollout-max-response-len 1024" # 缩短最大长度
    "--rollout-top-p 0.9"            # 添加 top-p 采样
)
```

---

## 附录：完整文件结构

```
slime/
├── data/
│   ├── my_task_train.parquet          # 训练数据
│   └── my_task_train_debug.parquet    # 调试数据（可选）
├── slime/
│   └── reward/
│       ├── __init__.py
│       └── my_task_reward.py          # 你的 reward 函数
├── ltp_train/
│   ├── config.yaml                    # LTP 平台配置
│   ├── run.sh                         # 启动脚本
│   └── README.md                      # 本文档
└── outputs/
    └── my_task/                       # 训练输出
```

---

**有问题请联系**: zhengmingming@baidu.com
