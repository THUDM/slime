# Qwen3.5 MoE 多教师在线策略蒸馏 (MOPD) 训练指南

本文档以 Qwen3.5 MoE 系列模型（35B-A3B、397B-A17B 等）为例，详细说明如何使用 slime 进行 MOPD (Multi-Teacher On-Policy Distillation) 训练，包括前期准备、参数配置、训练启动、checkpoint 转换和数据集构造。

---

## 目录

1. [整体流程概览](#1-整体流程概览)
2. [前期准备](#2-前期准备)
   - 2.1 [HF 模型转 Megatron torch_dist 格式](#21-hf-模型转-megatron-torch_dist-格式)
   - 2.2 [准备 Teacher 模型](#22-准备-teacher-模型)
   - 2.3 [准备训练数据](#23-准备训练数据)
3. [SGLang 模式 vs Megatron 模式](#3-sglang-模式-vs-megatron-模式)
4. [启动 SGLang Teacher 服务](#4-启动-sglang-teacher-服务)
5. [训练脚本参数详解](#5-训练脚本参数详解)
   - 5.1 [模型参数](#51-模型参数)
   - 5.2 [Checkpoint 参数](#52-checkpoint-参数)
   - 5.3 [Rollout 参数](#53-rollout-参数)
   - 5.4 [MOPD 参数](#54-mopd-参数)
   - 5.5 [性能参数](#55-性能参数)
   - 5.6 [SGLang 参数](#56-sglang-参数)
6. [启动训练](#6-启动训练)
7. [训练后的 Checkpoint 转换](#7-训练后的-checkpoint-转换)
   - 7.1 [使用 Bridge 模式转换（推荐）](#71-使用-bridge-模式转换推荐)
   - 7.2 [使用手动映射转换](#72-使用手动映射转换)
   - 7.3 [VLM 模型补齐 Visual Encoder 权重](#73-vlm-模型补齐-visual-encoder-权重)
   - 7.4 [验证转换结果](#74-验证转换结果)
   - 7.5 [选择最佳 Checkpoint](#75-选择最佳-checkpoint)
8. [数据集构造说明](#8-数据集构造说明)
   - 8.1 [基本格式](#81-基本格式)
   - 8.2 [多领域路由（可选）](#82-多领域路由可选)
   - 8.3 [数据质量建议](#83-数据质量建议)
9. [常见问题](#9-常见问题)

---

## 1. 整体流程概览

```
┌─────────────────────────────────────────────────────────────────────┐
│  前期准备                                                           │
│  1. HF → torch_dist 转换（学生模型 + 教师模型[Megatron模式]）        │
│  2. 启动 SGLang 教师服务 [SGLang模式]                               │
│  3. 准备 JSONL 训练数据                                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  训练                                                               │
│  - 学生模型 rollout 生成响应                                         │
│  - 教师模型获取 log-probs (SGLang HTTP / Megatron 前向传播)          │
│  - 计算 MOPD 损失 + 反向传播                                        │
│  - 定期保存 Megatron checkpoint                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  推理部署                                                           │
│  1. Megatron torch_dist → HF safetensors 转换                       │
│  2. 补齐 VLM visual encoder 权重                                    │
│  3. SGLang / vLLM 推理                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 前期准备

### 2.1 HF 模型转 Megatron torch_dist 格式

训练前需要将 HuggingFace 格式的学生模型转换为 Megatron torch_dist 格式。Qwen3.5 MoE 系列模型（含 VLM）**必须使用 `--megatron-to-hf-mode bridge`** 模式进行转换，因为它们包含 GDN (Gated DeltaNet) 线性注意力层、attention_output_gate、MTP 等自定义架构特性，只有 `megatron.bridge` 才能正确处理这些特殊参数的映射。

```bash
cd /path/to/slime
source scripts/models/qwen3.5-35B-A3B.sh  # 或 qwen3.5-397B-A17B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node=8 \
    tools/convert_hf_to_torch_dist.py \
    --megatron-to-hf-mode bridge \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/Qwen3.5-35B-A3B \
    --save /path/to/Qwen3.5-35B-A3B_Torch_Dist_Bridge
```

**参数说明：**
- `source scripts/models/qwen3.5-35B-A3B.sh`：加载模型架构参数（层数、隐藏维度、MoE 配置等）
- `--megatron-to-hf-mode bridge`：**必须指定**。使用 `megatron.bridge` 进行权重映射，以正确处理 Qwen3.5 的自定义架构（GDN 线性注意力、attention_output_gate、visual encoder 等）
- `--hf-checkpoint`：原始 HuggingFace 模型目录（包含 config.json、safetensors 等）
- `--save`：输出的 torch_dist 检查点目录
- `--nproc-per-node=8`：建议使用 8 GPU 并行转换，速度更快

**注意事项：**
- **不要使用 `--megatron-to-hf-mode raw`（默认值）**，raw 模式使用 `mbridge`，不支持 Qwen3.5 VLM 的自定义架构
- VLM 模型的 visual encoder 权重会被加载到 Megatron 模型中，但**不会**被保存到 Megatron checkpoint（Megatron 只保存语言模型部分）
- 这些权重在后续转回 HF 时需要从原始模型补回（见[第 7 节](#7-训练后的-checkpoint-转换)）

### 2.2 准备 Teacher 模型

根据教师模式不同：

#### SGLang 模式（推荐）

只需准备 HF/safetensors 格式的教师模型，用于启动 SGLang 推理服务。**不需要**转换为 torch_dist 格式。

```bash
# 教师模型只需是 SGLang 可加载的格式
TEACHER_MODEL=/path/to/teacher_model_safetensors
```

#### Megatron 模式

需要将教师模型也转换为 torch_dist 格式：

```bash
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node=8 \
    tools/convert_hf_to_torch_dist.py \
    --megatron-to-hf-mode bridge \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/teacher_model \
    --save /path/to/teacher_model_torch_dist
```

> **注意：** Megatron 模式要求教师与学生**架构完全相同**。SGLang 模式则无此限制。

### 2.3 准备训练数据

训练数据使用 JSONL 格式，每行一个 JSON 对象。基本字段为 `messages`（对话格式）或 `prompt`（纯文本格式）。

详细格式见[第 8 节](#8-数据集构造说明)。

---

## 3. SGLang 模式 vs Megatron 模式

| 维度 | SGLang 模式 | Megatron 模式 |
|------|-----------|-------------|
| **教师运行位置** | 外部 SGLang 服务器 | 加载到训练进程 CPU 内存 |
| **教师架构要求** | 无限制（可与学生不同） | **必须与学生架构相同** |
| **CPU 内存开销** | 无额外开销 | 每个教师 ≈ 模型大小（397B ≈ 800GB/教师） |
| **支持蒸馏类型** | `token_level` + `top_k` | `token_level` + `top_k` + `full_vocab` |
| **top_k 尾部校正** | 精确计算（SGLang 返回归一化 log-probs） | 均匀分布估计（保守上界） |
| **故障处理** | 教师 503 时跳过（会触发 RuntimeError） | 无此问题 |
| **适用场景** | 教师架构不同、避免 CPU OOM | 需要全词表精确 KL、教师架构相同 |

**推荐选择：** 对于 MoE 大模型，**强烈推荐 SGLang 模式**，因为：
- Megatron 模式需额外 ~800GB CPU 内存/教师，多节点容易 OOM
- SGLang 的 top_k 尾部校正更精确
- 教师可以与学生架构不同（如用更大的教师蒸馏）

---

## 4. 启动 SGLang Teacher 服务

在训练之前，需要先启动教师模型的 SGLang 推理服务。

```bash
# 多 GPU 启动教师模型 (TP=8, EP=16, 共 16 GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m sglang.launch_server \
    --model-path /path/to/teacher_model/ \
    --host 0.0.0.0 \
    --port 13141 \
    --tp 8 --ep-size 16 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.7
```

**等待服务就绪：**

```bash
until curl -sf http://localhost:13141/health_generate > /dev/null; do
    echo "Waiting for teacher model server to start..."
    sleep 10
done
echo "Teacher server is ready!"
```

**关键参数说明：**
- `--tp 8`：张量并行数
- `--ep-size 16`：专家并行数（MoE 模型需要）
- `--mem-fraction-static 0.7`：KV cache 显存占比
- `--chunked-prefill-size 4096`：分块预填充大小

> **重要：** 教师模型和学生模型的**词表大小必须一致**，否则 top_k 的 token index 映射会出错。

---

## 5. 训练脚本参数详解

以 `run-qwen35-35B-A3B-mopd-topk-sglang.sh` 和 `run-qwen35-397B-A17B-mopd-topk-sglang.sh` 为例。

### 5.1 模型参数

模型架构参数通过 `source scripts/models/qwen3.5-35B-A3B.sh` 或 `source scripts/models/qwen3.5-397B-A17B.sh` 加载，核心参数：

**35B-A3B：**
```bash
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
   --num-attention-heads 16       # 注意力头数
   --num-query-groups 2           # GQA KV 组数
   --kv-channels 256              # Head 维度
   --num-layers 40                # 层数
   --hidden-size 2048             # 隐藏维度
   --num-experts 256              # MoE 专家数
   --moe-router-topk 8            # 每层激活专家数
   --attention-output-gate        # Qwen3.5 特有：注意力输出门控
   --moe-shared-expert-gate       # Qwen3.5 特有：共享专家门控
   # ... 其他参数见 scripts/models/qwen3.5-35B-A3B.sh
)
```

**397B-A17B：**
```bash
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
   --num-attention-heads 32
   --num-query-groups 2
   --kv-channels 256
   --num-layers 60
   --hidden-size 4096
   --num-experts 512
   --moe-router-topk 10
   --attention-output-gate
   --moe-shared-expert-gate
   # ... 其他参数见 scripts/models/qwen3.5-397B-A17B.sh
)
```

### 5.2 Checkpoint 参数

```bash
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}/           # 原始 HF 模型路径
   --load ${TORCH_DIST_CKPT}/            # torch_dist 初始/恢复检查点
   --save ${SAVE_DIR}/                    # 训练检查点保存路径
   --save-interval 32                     # 每 32 步保存一次
   --no-save-optim                        # 不保存优化器状态（节省磁盘）
   --no-ckpt-fully-parallel-save
)
```

- `--hf-checkpoint`：SGLang rollout 引擎从中加载权重
- `--load`：首次训练时指向 torch_dist 格式的初始权重，恢复训练时指向保存目录
- `--save`：训练检查点保存路径

### 5.3 Rollout 参数

```bash
ROLLOUT_ARGS=(
   --input-key messages                  # 数据中对话字段的 key
   --apply-chat-template                 # 应用聊天模板
   --rollout-shuffle                     # 打乱数据
   --rollout-batch-size 4                # 每次 rollout 的 batch size
   --n-samples-per-prompt 4              # 每个 prompt 采样次数
   --rollout-max-prompt-len 9216         # 最大 prompt 长度
   --rollout-max-response-len 2048       # 最大生成长度
   --rollout-temperature 0.8             # 采样温度

   --global-batch-size 16                # 全局 batch size
   --balance-data                        # 跨节点均衡数据
   --num-epoch 1                         # 训练轮数
)
```

### 5.4 MOPD 参数

```bash
MOPD_ARGS=(
   --advantage-estimator grpo            # 优势估计方法

   # -- MOPD 核心 --
   --use-mopd                            # 启用 MOPD
   --mopd-teacher-mode sglang            # 教师模式：sglang 或 megatron
   --mopd-distill-type top_k             # 蒸馏类型：token_level / top_k / full_vocab
   --mopd-topk-k 96                      # top_k 保留的 token 数

   # -- MOPD 超参 --
   --mopd-alpha 0.0                      # α=0 纯蒸馏（无需奖励模型）
   --mopd-eps-low 0.2                    # IS 权重截断下界
   --mopd-eps-high 5.0                   # IS 权重截断上界
   --mopd-sampling-logprobs-key rollout_log_probs
)
```

**参数详解：**

| 参数 | 说明 |
|------|------|
| `--mopd-alpha` | 蒸馏与 ORM 的混合系数。0 = 纯蒸馏（无需奖励模型），>0 = 蒸馏 + RL 组合 |
| `--mopd-distill-type top_k` | **推荐**。每位置只传教师 top-k 个 token 的 logits，内存省 ~97% |
| `--mopd-distill-type token_level` | 每位置只传 1 个标量（教师对采样 token 的 log-prob），最省内存但精度低 |
| `--mopd-distill-type full_vocab` | 精确 KL，但内存开销极大（397B 词表 248K × batch × seq），**仅 Megatron 模式支持** |
| `--mopd-topk-k` | top_k 保留的 token 数。k=128 适合大多数场景，V>200K 时推荐 256+ |
| `--mopd-eps-low/eps-high` | IS 权重截断范围。紧范围(如[0.5,2])低方差高偏差；松范围(如[0.1,10])高方差低偏差 |

**教师配置（环境变量）：**

```bash
# 教师列表：name=教师名称, domain=领域标识
export MOPD_TEACHERS_JSON='[{"name":"math-teacher","domain":"math"},{"name":"code-teacher","domain":"code"}]'

# SGLang 模式：domain -> URL 映射
export MOPD_TEACHER_URLS="{\"math\":\"https://$MATH_TEACHER_IP:$PORT/generate\",\"code\":\"https://$CODE_TEACHER_IP:$PORT/generate\"}"
```

### 5.5 性能参数

**35B-A3B（4 节点 × 8 GPU = 32 GPU）：**
```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 32       # 256 专家 / 32 EP = 8 专家/GPU
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 4
   --max-tokens-per-gpu 2048
   --train-memory-margin-bytes 536870912
)
```

**397B-A17B（32 节点 × 8 GPU = 256 GPU）：**
```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 128      # 512 专家 / 128 EP = 4 专家/GPU
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 4
   --max-tokens-per-gpu 2048
   --train-memory-margin-bytes 268435456
)
```

### 5.6 SGLang 参数

```bash
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16     # 每个 SGLang 引擎使用的 GPU 数
    --sglang-mem-fraction-static 0.10    # SGLang KV cache 显存占比
    --sglang-ep-size 16                  # SGLang 推理的 EP 大小
)
```

---

## 6. 启动训练

### SGLang 模式（推荐）

```bash
# 35B-A3B
bash examples/multi_teacher_on_policy_distillation/run-qwen35-35B-A3B-mopd-topk-sglang.sh

# 397B-A17B
bash examples/multi_teacher_on_policy_distillation/run-qwen35-397B-A17B-mopd-topk-sglang.sh
```

前提条件：
1. SGLang Teacher 服务已启动并就绪
2. 环境变量 `MOPD_TEACHERS_JSON` 和 `MOPD_TEACHER_URLS` 已设置
3. 学生模型的 torch_dist 检查点已转换（使用 `--megatron-to-hf-mode bridge`）

---

## 7. 训练后的 Checkpoint 转换

训练保存的是 Megatron torch_dist 格式，需要转回 HuggingFace safetensors 格式才能用于 SGLang/vLLM 推理。

Qwen3.5 MoE 系列模型包含自定义架构（GDN 线性注意力、attention_output_gate、VLM visual encoder 等），**必须使用 bridge 模式**进行反向转换，确保与正向转换的权重映射一致。

### 7.1 使用 Bridge 模式转换（推荐）

这是 Qwen3.5 MoE VLM 模型的**推荐转换方式**，使用 `megatron.bridge` 进行端到端转换，与正向转换（`--megatron-to-hf-mode bridge`）保持映射一致。

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf_bridge.py \
    --input-dir /path/to/save_dir/iter_0000009/ \
    --output-dir /path/to/output_hf \
    --origin-hf-dir /path/to/original/Qwen3.5-35B-A3B \
    --force
```

**关键参数：**
- `--input-dir`：训练产出的 torch_dist 检查点目录（包含 `common.pt` 和 `.metadata` 等文件）
- `--output-dir`：输出的 HF safetensors 目录
- `--origin-hf-dir`：**原始 HF 模型目录**（提供 config.json 用于推断模型架构 + 补齐缺失的 visual encoder 权重）
- `--force`：覆写已存在的输出目录

**为什么必须使用 bridge 模式？**

Qwen3.5 MoE VLM 模型有以下特殊性，手动映射脚本 (`convert_torch_dist_to_hf.py`) 无法正确处理：

| 特性 | 手动映射脚本 | Bridge 模式 |
|------|-------------|------------|
| `attention_output_gate` (QKV+G 融合) | 只做 Q/K/V 三分拆，**gate 权重丢失** | 正确的 Q/G/K/V 四分拆 |
| Visual encoder (`model.visual.*`) | 未处理，遇到即报错 | 自动映射 `vision_model.**` → `model.visual.**` |
| GDN 线性注意力层参数 | 部分覆盖 | 完整映射 |
| Expert 权重融合/拆分 | 依赖 `common.pt` 中的 args 推断 | Bridge 自动处理 fused/per-expert 格式 |

**重要：** 确保远端 slime 代码是最新的，`convert_torch_dist_to_hf_bridge.py` 必须包含 `import slime_plugins.megatron_bridge` 以注册自定义 `Qwen35VLMoeBridge`。否则会使用官方 bridge，导致 vision encoder 和 MTP 层参数不匹配。

### 7.2 使用手动映射转换（非 VLM 模型可选）

对于**非 VLM** 的 Qwen3.5 MoE 模型（如纯语言模型版本），可以使用手动映射脚本，支持多进程并行加速：

```bash
# 单进程版本
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/iter_0000009/ \
    --output-dir /path/to/output_hf \
    --origin-hf-dir /path/to/original/Qwen3.5-35B-A3B \
    --model-name qwen3_5_moe \
    --add-missing-from-origin-hf \
    --force

# 多进程并行版本（更快）
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf_parallel.py \
    --input-dir /path/to/iter_0000009/ \
    --output-dir /path/to/output_hf \
    --origin-hf-dir /path/to/original/Qwen3.5-35B-A3B \
    --model-name qwen3_5_moe \
    --add-missing-from-origin-hf \
    --force
```

> **警告：** VLM 模型（如 `Qwen3_5MoeForConditionalGeneration`）**不要**使用此脚本，因为缺少 `attention_output_gate` 和 visual encoder 的映射。

### 7.3 VLM 模型补齐 Visual Encoder 权重

Qwen3.5 MoE VLM 模型的 Megatron 检查点**只包含语言模型权重**，不包含 visual encoder（`model.visual.*`）。如果不补齐，推理时会出现 `probability tensor contains inf/nan or element < 0` 错误。

**方法 1：自动补齐**（Bridge 模式已内置）

使用 `--origin-hf-dir` 且包含完整 HF 模型时，bridge 模式会通过 `ReplicatedMapping("vision_model.**", "model.visual.**")` 映射，但训练 checkpoint 中不包含 visual encoder 权重。如果转换后缺少 visual encoder 权重，使用 `merge_missing_keys.py` 补齐：

```bash
python tools/merge_missing_keys.py \
    --origin-hf-dir /path/to/original/Qwen3.5-35B-A3B \
    --converted-dir /path/to/output_hf \
    --dry-run  # 先预览缺失的 key
```

去掉 `--dry-run` 执行实际补齐。

**方法 2：手动映射脚本的 `--add-missing-from-origin-hf`**

使用 `convert_torch_dist_to_hf.py` 时加 `--add-missing-from-origin-hf` 会自动从原始 HF 模型补充缺失的权重。

### 7.4 验证转换结果

```bash
# 检查 key 数量
python3 -c "
import json
with open('/path/to/output_hf/model.safetensors.index.json') as f:
    idx = json.load(f)
print(f'Total keys: {len(idx[\"weight_map\"])}')
# 对比原始模型：
import os
origin_keys = set()
from safetensors import safe_open
for f in sorted(os.listdir('/path/to/original/Qwen3.5-35B-A3B')):
    if f.endswith('.safetensors'):
        with safe_open(f'/path/to/original/Qwen3.5-35B-A3B/{f}', framework='pt', device='cpu') as sf:
            origin_keys.update(sf.keys())
print(f'Original model keys: {len(origin_keys)}')
missing = origin_keys - set(idx['weight_map'].keys())
if missing:
    print(f'WARNING: Missing keys ({len(missing)}): {sorted(missing)[:10]}...')
else:
    print('All keys present!')
"

# 检查是否有 NaN/Inf 权重
python3 -c "
from safetensors import safe_open
import json, glob, os
idx = json.load(open('/path/to/output_hf/model.safetensors.index.json'))
files = set(idx['weight_map'].values())
for f in sorted(files):
    path = f'/path/to/output_hf/{f}'
    with safe_open(path, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            t = sf.get_tensor(k)
            if t.isnan().any() or t.isinf().any():
                print(f'ERROR: {k} has NaN/Inf!')
print('Validation complete.')
"
```

### 7.5 选择最佳 Checkpoint

训练过程中每 `--save-interval` 步保存一次 checkpoint。选择最佳 checkpoint 的建议：

- **关注 `mopd_topk_kl` 指标**：应该持续下降，代表学生与教师的分布差距在缩小
- **关注 `entropy` 指标**：应保持相对稳定，突然暴跌说明模式坍塌
- **避免 loss=0 的 checkpoint**：如果出现教师服务不可用导致的 0 梯度步骤，该 checkpoint 的权重可能已退化
- 一般选择 KL 收敛到较低点且 entropy 仍然健康的 checkpoint

---

## 8. 数据集构造说明

### 8.1 基本格式

训练数据为 JSONL 格式，每行一个 JSON 对象。支持两种输入字段：

**对话格式（推荐）：**

```jsonl
{"messages": [{"role": "user", "content": "Explain the concept of gradient descent in machine learning."}, {"role": "assistant", "content": "Gradient descent is an optimization algorithm..."}]}
```

配合 `--input-key messages --apply-chat-template` 使用。

**纯文本格式：**

```jsonl
{"prompt": "Explain the concept of gradient descent in machine learning."}
```

配合 `--input-key prompt` 使用。

### 8.2 多领域路由（可选）

当有多个教师分别负责不同领域时，可以在 `metadata` 中指定每个样本应从哪个教师蒸馏。`mopd_domains` 的值必须与 `MOPD_TEACHERS_JSON` 中对应教师的 `domain` 字段匹配。

例如，当教师配置为：
```bash
export MOPD_TEACHERS_JSON='[{"name":"math-teacher","domain":"math"},{"name":"code-teacher","domain":"code"}]'
```

数据集可以这样指定领域路由：

```jsonl
{"messages": [...], "metadata": {"mopd_domains": ["math"]}}
{"messages": [...], "metadata": {"mopd_domains": ["code"]}}
{"messages": [...], "metadata": {"mopd_domains": ["math", "code"]}}
{"messages": [...]}
```

说明：
- `"mopd_domains": ["math"]` — 仅从 `math` 领域的教师（math-teacher）蒸馏
- `"mopd_domains": ["math", "code"]` — 同时从两个教师蒸馏
- 无 `mopd_domains` 字段 — 从**所有**教师蒸馏（默认行为）
- 也支持字符串简写：`"mopd_domains": "math"`

### 8.3 数据质量建议

1. **数据多样性**：覆盖目标领域的各种子任务，避免过度集中
2. **长度分布**：控制 prompt 长度分布，避免超长样本浪费 rollout 资源
3. **batch size 匹配**：`--rollout-batch-size` 应与 `--global-batch-size` 一致
4. **教师容量**：确保 SGLang 教师服务能处理并发请求量（与 rollout batch size 成正比）

---

## 9. 常见问题

### Q1: MOPD 和 OPD 能同时使用吗？

不能。`--use-mopd` 和 `--use-opd` 互斥。

### Q2: alpha=0 时需要奖励模型吗？

不需要。`--mopd-alpha 0.0` 是纯蒸馏模式，无需 `--rm-type`。系统会自动将 reward 设为 0。

### Q3: SGLang 教师服务 503 会怎样？

如果教师在 rollout 期间返回 503（服务不可用），该 batch 的教师数据会被跳过，代码会打印 warning 日志。在纯蒸馏模式（`--mopd-alpha 0.0`）下，跳过教师数据会导致该 batch 的 MOPD 损失为 0，总损失也为 0，梯度为 0，**训练实质上空转一步**。如果持续 503，模型会因为长期零梯度而退化。建议确保教师服务稳定运行。

### Q4: top_k 的 k 值怎么选？

- `k=128`：适合大多数场景，内存最低
- `k=1024`：更精确的 KL 近似，内存稍高
- 经验法则：`k/V > 0.05%` 即可捕获 >99% 的 KL 信号。V=248K 时 k≥128 即可

### Q5: 转换后推理报 `probability tensor contains inf/nan` 怎么办？

这通常是因为 VLM 模型的 visual encoder 权重缺失。使用 `merge_missing_keys.py` 补齐：

```bash
python tools/merge_missing_keys.py \
    --origin-hf-dir /path/to/original/model \
    --converted-dir /path/to/converted/model \
    --dry-run  # 先预览缺失的 key
```

### Q6: 训练中出现 loss=0 和 grad_norm=0 怎么办？

这通常意味着某个 rollout batch 的教师数据获取失败（如 SGLang 503），导致 MOPD 损失被跳过。在纯蒸馏模式（alpha=0）下，跳过 MOPD 损失后总损失为 0，不产生梯度，训练空转一步。如果频繁出现，模型会因长期零梯度而退化。检查教师服务日志，确保服务稳定。

### Q7: EP=128 时 DeepEP 报断言错误？

注释掉 `--moe-enable-deepep`。当前 EP=128 时 DeepEP 的 inter-node kernel 有已知问题，使用默认的 alltoall 即可。

### Q8: 如何恢复中断的训练？

只要 `--load` 和 `--save` 指向同一目录，训练会自动加载最新 checkpoint 继续训练。确保 `--save-interval` 设置合理以避免丢失过多进度。

### Q9: convert_torch_dist_to_hf_bridge.py 报 `TypeError: object of type '_io.BytesIO' has no len()` 怎么办？

这通常是因为 `slime_plugins.megatron_bridge` 模块未被注册，导致 `AutoBridge` 使用了官方的 bridge（而非自定义的 `Qwen35VLMoeBridge`），模型结构与 checkpoint 不匹配。确保脚本中包含以下导入：

```python
import slime_plugins.megatron_bridge  # noqa: F401  # register custom bridges before AutoBridge
```

如果看到日志中 `Using Bridge provider: Qwen35VLMoEModelProvider`（官方），说明自定义 bridge 未注册；正确应显示 `Qwen35VLMoeVLModelProvider`。

### Q10: convert_torch_dist_to_hf_bridge.py 报大量 vision_model / mtp 参数 "not in state dict"？

同 Q9，这是因为使用了错误的 bridge。官方 bridge 期望 `vision_model.decoder.layers.*`（Megatron-native 命名），而 VLM 训练使用的是 HF 命名的 `vision_model.blocks.*`。注册自定义 bridge 后，映射会变为 `vision_model.**` → `model.visual.**`，问题自动解决。
