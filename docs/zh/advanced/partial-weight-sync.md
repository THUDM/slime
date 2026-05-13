# 增量权重同步（Delta / Selective）

- [概述](#概述)
- [快速开始](#快速开始)
- [两种 partial 模式：delta 与 selective](#两种-partial-模式delta-与-selective)
- [工作原理](#工作原理)
- [选择 wire 编码](#选择-wire-编码)
- [精度行为](#精度行为)
- [周期性 Base Sync](#周期性-base-sync)
- [为什么 colocate 模式不需要](#为什么-colocate-模式不需要)

## 概述

在**非 colocate**（non-colocated）模式下，slime 默认会在每一步训练时把所有参数完整地广播给 SGLang。完整广播的体积随模型规模线性增长，即使两步之间实际变化的权重比例很小，broadcast 仍然主导整个权重同步阶段。Partial-update 模式会把上一次同步时的权重在 pinned CPU 内存里保留一份 snapshot，每步只广播变化位置的数据，SGLang 接收端只更新这些位置。在 RL fine-tuning 阶段、学习率不大的常见设置里，每步 diff 都很稀疏（只有百分之几的权重发生变化），wire 体积也按比例减少。

**参考资料 / 先验工作。** `delta` 模式的加性更新思路参考了 [Cursor Composer 2](https://cursor.com/resources/Composer2.pdf) 和 [Fireworks AI — Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think)。`selective` 模式的灵感来自 [arXiv:2509.19128](https://arxiv.org/abs/2509.19128)。

## 快速开始

训练端开关与传输编码：

```bash
--update-weight-mode delta                # 'delta' / 'selective' / 'full'（默认）
--update-weight-partial-encoding sparse_indices
--update-weight-delta-dtype fp32          # 仅 delta 模式生效
--update-weight-base-sync-interval 30     # 可以设成非常大的数（例如 10000）以彻底关闭周期性
                                          # base sync —— partial 模式的 apply 是 lossless 的
                                          # （详见下面 "周期性 Base Sync" 一节）。
```

SGLang 端唯一的旋钮（slime 通过 `--sglang-update-weight-partial-chunk-bytes` 自动转发）：

```bash
--sglang-update-weight-partial-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

完整非 colocate 启动脚本见 [examples/partial_weight_sync/run-glm4.7-355B-A32B-partial.sh](../../../examples/partial_weight_sync/run-glm4.7-355B-A32B-partial.sh)。

## 两种 partial 模式：delta 与 selective

两种模式共用 sender 流水线（snapshot、mask 计算、稀疏编码、桶式广播）和 wire 格式，区别只在 values 的语义以及 receiver 的 apply 方式：

| | `--update-weight-mode delta` | `--update-weight-mode selective` |
|---|---|---|
| wire 上的 values | `(current − snapshot)`，cast 到 `--update-weight-delta-dtype`（默认 fp32） | 变化位置的新权重，dtype 同 snapshot |
| 接收端"未变化"信号 | 隐式（delta 在未变化位置为 0） | 解码后的稠密张量在未变化位置填 NaN |
| 接收端 apply | `param += delta`（in-place add，自动提升到 fp32 计算后再 cast 回 param dtype） | `param[~isnan(src)] = src[~isnan(src)]`（selective overwrite） |
| values 部分 wire 字节 | 4×nnz @ fp32 | 2×nnz @ bf16（½× delta） |
| 是否 lossless | 当 `delta-dtype` 高于 param dtype 时 lossless | 永远 lossless（无算术） |

当你想要更小的 wire、不需要 fp32 算术余量时选 `selective`；当你需要 fp32 减法去保住 sub-bf16 级别的小 delta 时选 `delta`。

## 工作原理

每次同步，训练端（仅 PP-source rank）：

1. **计算 payload**：delta 模式下，将当前权重与 pinned-CPU snapshot 同时提升到 delta_dtype 然后相减；selective 模式下，先在 bf16 上取 `current != snapshot` 的 mask，再生成新权重值并在 unchanged 位置填 NaN。
2. **编码**：将 active 位置稀疏编码为两条扁平张量（`__packed_keys__`、`__packed_values__`）和一份 per-param manifest（`PartialWeightSpec.params`）。
3. **分桶广播**：多个参数共享一次 NCCL 广播，桶大小由 `--update-weight-buffer-size` 控制。
4. **异步刷新 snapshot**：把当前权重通过独立 CUDA stream 拷贝到 pinned CPU，与下一轮的广播、编码计算重叠。

SGLang 接收端：

1. **接收**：每个桶接收两条 packed 张量。
2. **懒解码**：以生成器逐参数 yield 解码后的稠密张量；unchanged 位置按模式填入 sentinel（delta 模式填 0，selective 模式填 NaN）。下游 chunking 的 `chunk_byte_cap` 同时为 decode 阶段的峰值 HBM 设上限（`encoded_buffers + in-flight chunk`）。
3. **加性写入**：仍走模型 `load_weights` 主路径，但通过一个 context manager 重写 `Tensor.copy_` / `fill_`：
   - `delta` 模式下 `_additive_load_context` 把落入 param storage 的 copy_ 重写为 `add_`（PyTorch 自动提升到 fp32 完成加法、再 cast 回 store，保留 fp32 精度）。
   - `selective` 模式下 `_selective_load_context` 把落入 param storage 的 copy_ 重写为 mask-overwrite（`param[~isnan(src)] = src[~isnan(src)]`），unchanged 位置保持不动。

非 param 的写入（scratch buffer、dtype 转换、`post_load_weights` 中的 FP8 scale 重计算 / MoE bias 物化等）在两种 context 下都保持原始覆盖语义。

Wire protocol —— `PartialWeightSpec`（encoding + per-param manifest）和 `PartialWeightParam`（name、dtype、shape、keys/values slice）—— 定义在 `sglang.srt.managers.io_struct`（由 slime 的 SGLang patch 注入）。

## 选择 wire 编码

`--update-weight-partial-encoding` 接受三个值：

| 值 | wire 排布 | 适用场景 |
|---|---|---|
| `sparse_indices` | int32 active 下标 + 值 | 低变化率（< ~3%） |
| `sparse_bitmask` | 每元素 1 bit 的 mask + 值 | 中等变化率（> ~3%） |
| `dense` | 每参数一条张量 | 调试 apply 路径 |

两种稀疏编码的等价点和值 dtype 无关。令 `n = numel`，`k = nnz`，`v = 值字节数`：

```
sparse_indices wire = k * (4 + v)
sparse_bitmask wire = ceil(n / 8) + k * v
```

二者相等时 `4k = n/8`，即 `k/n = 1/32 ≈ 3.125%`。低于该 density 选 indices，高于则选 bitmask。常见的小学习率 RL fine-tuning 阶段 `sparse_indices` 更省，训练早期大 LR 阶段几乎所有权重都在动时换 `sparse_bitmask`。

## 精度行为

`delta` 模式下 `--update-weight-delta-dtype` 控制的是**计算 dtype**，不仅仅是 wire dtype。减法在两个操作数都被提升到 `delta_dtype` 之后进行；接收端的 `param.data.add_(fp32_delta)` 让 PyTorch 内部以共同 dtype（fp32）做加法，然后再 cast 回 bf16 写入 param。这样可以保留那些在 bf16 减法下会直接舍入为零的小幅度 delta。

`selective` 模式下没有算术，接收端直接把 trainer 的精确 bf16 值写回 param，因此精度天然 bit-perfect，与 `--update-weight-delta-dtype` 无关（该 flag 在 selective 模式下被静默忽略）。

CPU snapshot 在两种模式下都只占用 param dtype 的字节数（不会因此膨胀到 fp32 的存储）。

## 周期性 Base Sync

每次任务的第一次同步永远是 *base sync*（一次完整广播，重建 snapshot）。之后每当 `committed_syncs % --update-weight-base-sync-interval == 0` 再触发一次 base sync。

在 `--update-weight-delta-dtype fp32`（delta 模式）或 selective 模式下，partial apply 都是**无损（lossless）**的：每个 bf16 值都可以精确表示为 fp32，`current_fp32 − snapshot_fp32` 得到两个 bf16 值的精确差，接收端的 `bf16_param.add_(fp32_delta)` 在自动提升到 fp32 完成加法、再 cast 回 bf16 之后，会逐比特地复现 trainer 的 bf16 状态；selective 模式则因为直接覆盖而天然无损。因为不会有误差累积，无论中间累积了多少次 partial 同步，接收端的状态都不会偏离对应的 base sync 结果，从正确性角度并不需要周期性 base sync。把 `--update-weight-base-sync-interval` 设得很大（例如 `10000`）就等于关闭周期性 base sync，在实践中是安全的。

保留少量 base sync 的运营性理由主要是恢复点——例如一个中途加入的 rollout engine 需要先拿到完整状态才能应用后续 partial 更新。如果你为了进一步压缩 wire 体积而把 `--update-weight-delta-dtype` 设为 `bf16`（不高于 param dtype 的精度，仅对 delta 模式有意义），apply 就不再 lossless，这时 interval 才需要给一个合理的有限值。

## 为什么 colocate 模式不需要

Colocate 模式的权重同步走的是 CUDA IPC：SGLang 直接把 trainer 进程的参数 storage 映射到自己进程，wire 上只交换一个 IPC handle（~64 B），完全没有 NCCL 广播。Partial 编码的「wire 体积」优势归零，而 partial 更新的额外开销（snapshot 维护、减法/取 mask、稀疏编码）反而是纯开销。所以 slime 在 argparse 阶段就拒绝 `--update-weight-mode delta --colocate` 和 `--update-weight-mode selective --colocate` 的组合。
