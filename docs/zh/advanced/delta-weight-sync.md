# Delta 权重同步

Delta 权重同步只发送两次同步之间发生变化的字节，而不是每次都写一份完整 checkpoint，以此让非 colocate 的 rollout engine 保持最新。它面向大模型、跨集群或跨数据中心的训推解耦场景——这种场景下每次都写整份 actor 权重是主要开销。

它**只支持 disk transport**，并且通过**原生**的 `update_weights_from_disk` 端点 reload，因此推理引擎不需要任何 delta 相关的支持。

## 配置

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/delta-updates
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt
--update-weight-delta-encoding xor          # 或: overwrite
--update-weight-delta-checksum xxh3-128     # 或: blake3, adler32
```

| 参数 | 作用 |
|---|---|
| `--update-weight-disk-dir` | 训练端发布 delta、rollout host 读取 delta 的共享文件系统目录。 |
| `--update-weight-local-checkpoint-dir` | host 本地（如 NVMe）的完整 HF checkpoint，delta 原地 apply 到这里。每个 host 在 engine 启动时由 `--hf-checkpoint` 物化。 |
| `--update-weight-delta-encoding` | 磁盘上的 delta 编码：`xor`（默认）或 `overwrite`。 |
| `--update-weight-delta-checksum` | 逐 tensor 完整性 checksum：`xxh3-128`（默认）、`blake3` 或 `adler32`。 |

delta 始终用 zstd（level 1）压缩；profiling 显示对这类数据它在 wire 大小和解压速度上都优于 lz4 / gzip / snappy / brotli，所以不做成可配置项。

## 工作原理

1. **Seed。** 第一次同步时，训练端为每个参数捕获一份 CPU snapshot——从 `--hf-checkpoint` seed，而这正是每个 rollout host 物化本地 checkpoint 的来源。此次不发布任何东西；这份 snapshot 就是下一次同步 diff 的基准。
2. **Publish。** 之后每次同步，训练端把每个 gather 出的 HF tensor 与 snapshot 做 diff，编码、压缩，写到 `--update-weight-disk-dir` 下的新版本目录 `weight_v{N:06d}/`。该目录是一份 canonical HF checkpoint——`model-NNNNN.safetensors` 文件装着压缩后的 diff tensor，外加 `model.safetensors.index.json`（tensor 名 → 文件）承载 apply 元数据——所以这个产物是可移植的，不绑定训练端的并行 layout。随后 snapshot 推进到新值，供下次 diff。
3. **Apply。** 每个 rollout host 把新版本的 delta 原地 apply 进它的本地 checkpoint。apply 在 tensor 之间并行，并逐 tensor 校验（见“完整性”）。
4. **Reload。** engine 通过原生 `update_weights_from_disk` 路径 reload 打过补丁的本地 checkpoint——它从不接触 delta 格式。

由于 snapshot 是从 `--hf-checkpoint`（engine 真正的 base）seed，而不是从当前 GPU 权重 seed，即使 Megatron→HF 往返不是逐字节相等（例如 embedding / LM head 中被裁掉的 vocab padding 行），该方案对任意模型也都正确。

## 编码

两种编码都是字节级、与 dtype 无关的，所以量化 checkpoint 也走同一条路径。engine 从每个版本的 index 元数据读取所用编码。

- **`xor`**（默认）：写 `new ^ old`。wire 最小、apply 最快（顺序访问、对 cache 友好；未变化的字节是 0，被压缩器压到极小）。它是一个对合（involution），所以必须**恰好对正确的 base apply 一次**——apply 两次会还原。
- **`overwrite`**：写变化的位置及其新的绝对值。wire 更大、apply 是对 cache 不友好的分散写，但**幂等**：重复 apply（或把部分 apply 的 delta 补完）无论执行多少次都收敛到同一状态。当“可重复 apply”比 wire 大小更重要时用它。

## 完整性

训练端把每个 tensor 新状态的逐 tensor checksum 存进版本里。apply 之后每个 host 重新计算 checksum，**任何不匹配都会 raise**，所以损坏的 delta 或错误的 base 会直接报错失败，而不会把坏权重提供出去。apply 还拒绝乱序执行：一个版本只会在它声明的 base 版本之上 apply。

`--update-weight-delta-checksum` 选择算法。checksum 不是 apply 的瓶颈（apply 受解压 + XOR 限制），所以这是一个 digest 属性的选择，而非速度选择：`xxh3-128`（默认）是最宽的快速非加密 digest；`blake3` 是加密 digest，用于不可信存储；`adler32` 用于与期望它的系统互操作。

## 共享文件系统可见性 hook

在 POSIX 共享文件系统（NFS、Lustre……）上不需要额外步骤。对于需要显式 commit/refresh 才能让写入跨 host 可见的对象存储卷，可以提供两个可选 hook（通过 import 路径加载——slime 里不存在任何厂商特定代码）：

- `--custom-delta-pre-push-path`：在一个版本的文件写完之后、通知 engine 读取之前调用（例如 commit volume）。签名：`hook(args, version_dir, rollout_engines)`。
- `--custom-delta-pre-read-path`：在每个 rollout host 读取 delta 目录之前调用（例如 refresh volume）。签名：`hook(delta_dir, target_version)`。
