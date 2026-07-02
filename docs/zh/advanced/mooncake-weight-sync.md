# Mooncake TransferEngine 权重同步

Mooncake TransferEngine transport 是面向 slime-managed SGLang engine 的实验性
全量权重同步路径。它不建立 NCCL 权重更新组，也不把 checkpoint 写到共享文件系统；
slime 会先让每个 SGLang engine 分配本地 Mooncake receiver buffer，再通过 Mooncake
TransferEngine 把每个权重 bucket 写入 receiver buffer，最后通知 SGLang 从本地 buffer
加载权重。

这条路径只适用于 slime 负责 SGLang engine 生命周期的场景。如果 engine 通过
`--rollout-external-engine-addrs` 由外部系统预启动，请继续使用 NCCL 或 disk transport。

## 快速开始

```bash
--update-weight-mode full
--update-weight-transport mooncake
--mooncake-metadata-server P2PHANDSHAKE
--mooncake-protocol tcp
--mooncake-buffer-size $((8 * 1024 * 1024 * 1024))
--mooncake-buffer-count 1
```

当前只支持 `P2PHANDSHAKE` metadata 模式。SGLang receiver 初始化时会返回每个
engine 的 target name，因此这条路径不依赖外部 Mooncake metadata service。

如果使用 RDMA，把 protocol 切到 `rdma`，并传入 Mooncake runtime 使用的设备名：

```bash
--mooncake-protocol rdma
--mooncake-device-name mlx5_0
```

## 当前限制

Mooncake transport 当前刻意收敛在一组窄契约内：

- 只支持 `--update-weight-mode full`；delta mode 会被拒绝。
- 只支持 slime-managed SGLang engine；external rollout engine 会被拒绝。
- 不支持 `--colocate`；colocated 权重同步使用 CUDA IPC tensor handle。
- 当前只支持每个 rollout engine 1 张 GPU。
- Megatron pipeline model parallel size 必须为 1。
- SGLang DP size 和 EP size 必须为 1。
- 不支持 SGLang DP attention。
- `--mooncake-metadata-server` 必须保持 `P2PHANDSHAKE`。
- `--mooncake-rpc-port-base` 是预留参数，参数校验会拒绝设置它。
- `--mooncake-buffer-count` 是为未来 double buffering 预留的参数，当前必须是
  `1`；发送端每次更新都使用 slot 0。

如果使用 `--sglang-config`，这些限制只会检查接收训练权重的 server group
（`update_weights: true`）。冻结的 reference/reward model 和 placeholder group
不需要 Mooncake receiver。

## Buffer 大小

`--mooncake-buffer-size` 必须能容纳 slime 发送的最大权重 bucket。如果不设置，
slime 会使用 `--update-weight-buffer-size`。

如果某个 bucket 大于 receiver buffer，slime 会在写入 Mooncake 前让本次同步失败。
可以增大 `--mooncake-buffer-size`，或减小 `--update-weight-buffer-size`，保证每个
bucket 都能放进 receiver buffer。

`--mooncake-buffer-count` 只是为了让 receiver API 保持未来 double buffering 的形状。
当前实现不会因为设置多个 buffer 而提升并发，且参数值不是 `1` 时会被拒绝。

## 与其他传输方式的关系

`--update-weight-mode` 决定发送什么，`--update-weight-transport` 决定如何送到
SGLang：

| mode | transport | 行为 |
|---|---|---|
| `full` | `nccl` | 通过 trainer-engine NCCL group 广播所有 HF 权重 chunk |
| `full` | `disk` | 写完整 HF checkpoint，然后调用 `update_weights_from_disk` |
| `full` | `mooncake` | 通过 Mooncake TransferEngine 把每个 bucket 写入 SGLang receiver buffer |
| `delta` | `nccl` | 通过 NCCL 广播稀疏变化位置和值 |
| `delta` | `disk` | 写稀疏 safetensors，然后调用 `update_weights_from_disk(load_format="delta")` |

Mooncake transport 只用于上表中的 `full` + slime-managed SGLang 场景。External
engine、异构 GPU 集群或共享文件系统跨集群部署，见
[External Rollout Engines](external-rollout-engines.md) 和
[Delta 权重同步](delta-weight-sync.md)。
