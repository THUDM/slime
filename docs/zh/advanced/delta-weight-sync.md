# Delta 权重同步

- [背景](#背景)
- [快速开始](#快速开始)
- [工作原理](#工作原理)
- [编码选择](#编码选择)
- [为何不支持 colocated](#为何不支持-colocated)

## 背景

slime 默认的权重同步会在每一步广播全部参数，开销随模型规模线性增长，即使每步真正变化的权重只有几个百分点。Delta 同步在内存中保留上一次同步后的参数快照（pinned CPU），只发送字节发生变化的位置。

最主要的应用场景是 **训练 / 推理跨数据中心解耦** —— 训练器和推理引擎运行在不同数据中心，通过共享文件系统通信（带宽通常在百 MB/s 级别）。在这种环境下，全量广播不可行，而 ~3% 密度的稀疏 delta（355B 模型约 5 GB）是可行的。同一套 delta 机制在数据中心内部跑 NCCL，作为验证基线，确认 wire 编码和 apply 逻辑正确。

参考资料：选择性覆写借鉴自 [arXiv:2509.19128](https://arxiv.org/abs/2509.19128)，跨数据中心的动机来自 [Fireworks AI — Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think)。

## 快速开始

磁盘传输（跨数据中心训推解耦，主要场景）：

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd                 # ≤ 300 MB/s 共享 FS 推荐
--update-weight-delta-dir /shared/fs/delta-updates
```

NCCL 传输（数据中心内部验证基线）：

```bash
--update-weight-mode delta
--update-weight-transport nccl
--update-weight-encoding indices                     # 计算最少，无压缩
```

接收端调优（两种传输都适用）：

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))  # 每次 load_weights 字节上限
--sglang-update-weight-delta-read-workers 4                           # 并行 I/O 线程数（仅磁盘传输）
```

完整启动脚本见 [examples/delta_weight_sync/run-glm4.7-355B-A32B-delta.sh](../../../examples/delta_weight_sync/run-glm4.7-355B-A32B-delta.sh)。

## 工作原理

两种传输共用同一条发送管线、同一种 wire 布局以及同一套接收端解码器；只有每个 bucket 的承载层不同。

**发送端（每次同步，仅 PP 源 rank）：**

1. **求差**：通过逐字节比较 `current.view(int_dtype) != snapshot.view(int_dtype)` 检测变化。无算术、无损、与 dtype 无关。
2. **编码**：将变化的 (位置, 值) 对打包成 `__positions__` 字节块 + `__values__` 张量 + per-param 解码 manifest。编码方式（`indices` / `deltas` / `deltas_zstd`）只影响位置如何打包，值始终按参数本身的 dtype 原样发送。
3. **打包并发送**：每个 chunk 编码后累积至 `--update-weight-buffer-size` 字节再 flush：
   - NCCL：广播 `(__positions__, __values__)`，Ray RPC 同时携带 `DeltaSpec`（编码 + per-param manifest）。
   - 磁盘：每个 flush 写一个 safetensors 文件到 `weight_v{N:06d}/` 目录，后台线程负责 I/O 和可选的 zstd 压缩，不阻塞关键路径。
4. **更新快照**：刚发送的值在 side stream 上 D2H 拷贝，与下一个 chunk 的编码重叠。

**同步结束（仅磁盘）：** 写 `DONE` 标记，rank 0 对每个引擎触发一次 HTTP push，所有引擎确认后清理目录。

**接收端：** 两种传输最终都进入同一个 `_apply_delta_payload(encoding, params, positions, values)` 帮助函数。它把每个参数的切片解码成全形状张量，未变化位置填 NaN，然后通过 `model.load_weights(...)` 应用；过程中 `_delta_apply_context` 替换 `Tensor.copy_` / `Tensor.fill_`，对参数存储执行 NaN 掩码覆写。辅助写入（scratch buffer、fp8 scale、MoE bias 等通过 `post_load_weights` 写入的派生张量）保留正常语义。

选择性覆写没有任何算术运算 —— 接收端在变化位置直接写入训练端的精确字节 —— 因此天然无损，也不存在数值漂移问题，无需周期性 base 同步。

## 编码选择

`--update-weight-encoding` 决定位置如何打包。三种编码共用同一种 wire 布局（`__positions__` uint8 块 + `__values__` 张量 + per-param manifest），解码端根据 metadata 分派。

| 取值 | 位置编码 | 推荐场景 |
|---|---|---|
| `indices` | int32 绝对位置（4 字节 / nnz） | NCCL 或高速集群内 FS（≥ ~600 MB/s） |
| `deltas` | uint16 增量（异常时 uint32 兜底，2% 密度下约 2 字节 / nnz） | 中等带宽 FS（~300-500 MB/s） |
| `deltas_zstd` | `deltas` 文件再用 zstd L1 压缩 | 跨数据中心 / 跨区共享 FS（≤ ~300 MB/s） |

**为何 gap 编码更省**：`mask.nonzero()` 返回的位置已经升序排列。密度 `p` 时连续非零位置的期望间隔为 `1/p`，且 `P(gap > 65535) ≈ exp(-p · 65535)`，p = 2% 时这个概率实际上为零，所以 uint16 完全够用，uint32 仅作 per-param 兜底。位置开销比 `indices` 减半，且无损。

**`deltas_zstd` 的额外收益**：在 gap 字节流上做 zstd L1 还能再减少 ~35-40%，代价是每文件约 250ms 压缩 + 150ms 解压。当共享 FS 带宽 ≤ 300 MB/s 时，带宽节省超过额外计算开销。

## 为何不支持 colocated

Colocated 同步通过 CUDA IPC：进程间传递的只是一个内存句柄（~64 B）。Delta 编码的"wire 节省"在此为零，而其簿记开销（快照 + 求差 + 稀疏编码）反而是纯损失。slime 在参数校验阶段拒绝 `--update-weight-mode delta --colocate`。
