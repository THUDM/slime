# P2P 分片权重同步

- [背景](#背景)
- [快速开始](#快速开始)
- [前置条件与自动回退](#前置条件与自动回退)
- [工作原理](#工作原理)
- [与默认 broadcast 对比](#与默认-broadcast-对比)
- [常见问题](#常见问题)

## 背景

slime 在非 colocate 模式下，默认通过 **all_gather + NCCL broadcast** 将 Megatron 训练权重同步到 SGLang rollout engine。对大模型而言，gather 全量参数再 broadcast 的开销显著。

P2P 分片同步让每个训练 TP rank 将自身 shard 经 `dist.send/recv` 直接发给对应推理 TP rank，跳过全量 gather/broadcast。

## 快速开始

请使用**官方 slime Docker 镜像**（`docker build` 时已在 `sglang.patch` / `sglang-top_p.patch` 之后自动应用 SGLang P2P patch，无需手动操作）。

非 colocate 训练，Qwen3-4B、Megatron TP=4、单 engine 占 4 GPU 的典型配置：

```bash
python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --tensor-model-parallel-size 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  --megatron-to-hf-mode bridge \
  --update-weight-mode full \
  --use-p2p-weight-update \
  --hf-checkpoint /path/to/Qwen3-4B \
  ...
```

关键开关：

| 参数 | 说明 |
|---|---|
| `--use-p2p-weight-update` | 启用 P2P 分片同步（不满足条件时自动回退 broadcast） |
| `--update-weight-mode full` | P2P 仅支持 full 模式 |
| `--megatron-to-hf-mode bridge` | **必须**；P2P 依赖 Megatron Bridge 的权重布局 |
| `--tensor-model-parallel-size` | 训练 TP，须与 SGLang TP 一致 |
| `--rollout-num-gpus-per-engine` | 单 engine GPU 数；SGLang TP = 该值 / `sglang_pp_size` |

## 前置条件与自动回退

开启 `--use-p2p-weight-update` 后，slime 在启动时检查以下条件。**任一不满足则自动使用 NCCL broadcast**，并在 rank 0 打印 `[P2P] ... using NCCL broadcast weight update instead.`：

| 条件 | 说明 |
|---|---|
| 非 colocate | colocate 走 `UpdateWeightFromTensor`，与 P2P 无关 |
| `--megatron-to-hf-mode bridge` | raw 模式不支持 P2P |
| 模型支持 shard 级转换 | 当前实现：**Qwen2 / Qwen3 稠密**；MoE 及其他架构暂回退 |
| Megatron TP == SGLang TP | 每个 rollout engine 的 TP 均须与训练 TP 对齐 |
| SGLang PP == 1 | `sglang_pp_size > 1` 时 P2P send/recv 无法正确配对，自动回退 |

TP 对齐关系：

```
Megatron TP  = tensor_model_parallel_size
SGLang TP    = rollout_num_gpus_per_engine / sglang_pp_size
```

## 工作原理

1. **词表参数**（embed / lm_head）：TP 组内小范围 all_gather，去 Megatron padding 后按 SGLang 分片边界切分。
2. **其余参数**：shard 级 Megatron→HF 转换（`convert_shard_to_hf`），不做 all_gather。
3. **每个 bucket**：`all_gather_object` 元数据 → rank-0 HTTP 通知 SGLang → NCCL barrier → 并行 `dist.send` → `ray.get` 等待 load 完成。

实现见 `slime/backends/megatron_utils/update_weight/update_weight_from_distributed_p2p.py`。

## 与默认 broadcast 对比

| | NCCL broadcast（默认） | P2P 分片 |
|---|---|---|
| 训练侧 | TP all_gather → 全量 HF 权重 | 各 rank 只转换/发送本地 shard |
| 通信 | broadcast 全量 chunk | `dist.send/recv` 按 shard |
| 适用模型 | bridge 支持的模型 | 当前仅 Qwen2/Qwen3 稠密 + bridge |
| TP 要求 | 无严格对齐要求 | Megatron TP == SGLang TP |

去掉 `--use-p2p-weight-update` 即恢复默认 broadcast 路径。

## 常见问题

**Q: 开了 P2P 但日志显示 NCCL broadcast？**  
A: 查看 rank 0 的 `[P2P]` 提示：常见原因是 `--megatron-to-hf-mode raw`、MoE/未支持模型、TP 不对齐、或 `sglang_pp_size > 1`。

**Q: 与 delta 同步的关系？**  
A: 互斥。P2P 属于 `--update-weight-mode full`；delta 同步见 [Delta 权重同步](delta-weight-sync.md)。
