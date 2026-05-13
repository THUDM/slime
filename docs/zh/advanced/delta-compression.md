# 增量权重同步（Delta Compression）

- [概述](#概述)
- [快速开始](#快速开始)
- [工作原理](#工作原理)
- [选择 wire 编码](#选择-wire-编码)
- [精度行为](#精度行为)
- [周期性 Full Sync](#周期性-full-sync)
- [为什么 colocate 模式不需要](#为什么-colocate-模式不需要)

## 概述

在**非 colocate**（non-colocated）模式下，slime 默认会在每一步训练时把所有参数完整地广播给 SGLang。在 355B 规模下，每步要走 NCCL ~170 GB，即使实际变化的权重比例很小。Delta-compression 会把上一次同步时的权重在 pinned CPU 内存里保留一份 snapshot，每步只广播稀疏编码后的 `(current − snapshot)`，SGLang 接收端以 `param += delta` 的方式累加。典型 RL 训练步的稠密度大约在 2–3%，wire 体积可缩减 ~30×。

## 快速开始

训练端开关与传输编码：

```bash
--update-weight-mode delta
--delta-compression sparse_indices
--delta-dtype fp32
--delta-full-interval 10000
```

SGLang 端唯一的旋钮（slime 通过 `--sglang-update-weight-delta-chunk-bytes` 自动转发）：

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

完整非 colocate 启动脚本见 [examples/delta_compression/run-glm4.7-355B-A32B-delta.sh](../../../examples/delta_compression/run-glm4.7-355B-A32B-delta.sh)。

## 工作原理

每次同步，训练端（仅 PP-source rank）：

1. **计算 delta**：在 GPU 上将当前权重和 pinned-CPU snapshot 同时提升到 `delta_dtype`（默认 fp32），相减，结果保留为 fp32。CPU 上的 snapshot 仍以模型自身的 dtype（通常 bf16）存放，不会因此变成 fp32。
2. **编码**：对每个参数，把非零位置和数值打包成两条扁平张量（`__packed_keys__`、`__packed_values__`），并维护一份 per-param manifest（`WeightDeltaSpec.params`），记录该参数在两条 buffer 里的 slice 范围。
3. **分桶广播**：多个参数共享一次 NCCL 广播，桶大小由 `--update-weight-buffer-size` 控制。
4. **异步刷新 snapshot**：把当前权重通过一条独立 CUDA stream 拷贝到 pinned CPU，与下一轮的广播、编码计算重叠。

SGLang 接收端：

1. **接收**：每个桶接收两条 packed 张量。
2. **懒解码**：以生成器逐参数 yield 解码后的稠密 delta 张量；下游 chunking 的 `chunk_byte_cap` 即可同时为 decode 阶段的峰值 HBM 设上限（`encoded_buffers + in-flight chunk`）。
3. **加性写入**：仍走模型的 `load_weights` 主路径，但通过一个 context manager 把 `torch.Tensor.copy_` / `fill_` 改写为 `add_`，并且只在写入目标落在某个参数的 storage 范围内时才生效。临时 buffer、dtype 转换、`post_load_weights`（FP8 scale 重计算、MoE bias 物化等）保持原始覆盖语义。

Wire protocol —— `WeightDeltaSpec`（encoding + 每参数 manifest）以及 `WeightDeltaParam`（name、dtype、shape、keys/values slice）—— 定义在 `sglang.srt.managers.io_struct` 中（由 slime 的 SGLang patch 注入）。

## 选择 wire 编码

`--delta-compression` 接受三个值：

| 值 | wire 排布 | 适用场景 |
|---|---|---|
| `sparse_indices` | int32 非零下标 + 值 | 极稀疏（density < ~3%） |
| `sparse_bitmask` | 每元素 1 bit 的 mask + 值 | 中等稀疏（density > ~3%） |
| `dense` | 每参数一条张量 | 调试 additive 路径 |

两种稀疏编码的等价点和值 dtype 无关。令 `n = numel`，`k = nnz`，`v = 值字节数`：

```
sparse_indices wire = k * (4 + v)
sparse_bitmask wire = ceil(n / 8) + k * v
```

二者相等时 `4k = n/8`，即 `k/n = 1/32 ≈ 3.125%`。低于该 density 选 indices，高于则选 bitmask。常见的小学习率 RL fine-tuning 阶段 `sparse_indices` 更省，训练早期大 LR 阶段几乎所有权重都在动时换 `sparse_bitmask`。

## 精度行为

`--delta-dtype` 控制的是**计算 dtype**，不仅仅是 wire dtype。减法在两个操作数都被提升到 `delta_dtype` 之后进行；接收端的 `param.data.add_(fp32_delta)` 让 PyTorch 内部以共同 dtype（fp32）做加法，然后再 cast 回 bf16 写入 param。这样可以保留那些在 bf16 减法下会直接舍入为零的小幅度 delta。

CPU snapshot 只占用 param dtype 的字节数（不会因此膨胀到 fp32 的存储）。

## 周期性 Full Sync

第一次同步永远是 full sync。之后每当 `committed_syncs % --delta-full-interval == 0` 再触发一次完整广播，同时把所有人的 snapshot 重新校准。保留周期性 full sync 的两个理由：

- 如果某次 delta 应用失败，full sync 是一个自愈点；
- 新加入的 rollout engine 在下一次 full sync 时被完整同步。

由于 sender 每步都会以自己的广播刷新 snapshot，实际使用中 interval 设得很大（例如 10000）是合理的。

## 为什么 colocate 模式不需要

Colocate 模式的权重同步走的是 CUDA IPC：SGLang 直接把 trainer 进程的参数 storage 映射到自己进程，wire 上只交换一个 IPC handle（~64 B），完全没有 NCCL 广播。Delta 编码的「wire 体积」优势归零，而 delta 计算 + 稀疏编码 + snapshot 维护反而是纯开销。所以 slime 在 argparse 阶段就拒绝 `--update-weight-mode delta --colocate` 的组合。
