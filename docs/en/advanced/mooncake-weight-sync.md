# Mooncake TransferEngine Weight Sync

Mooncake TransferEngine transport is an experimental full-weight sync path for
slime-managed SGLang engines. Instead of building an NCCL update group or
writing a checkpoint to a shared filesystem, slime asks each SGLang engine to
allocate a local Mooncake receiver buffer, writes each weight bucket into that
buffer with Mooncake TransferEngine, and then asks SGLang to load the bucket from
its local buffer.

Use this only when slime owns the SGLang engine lifecycle. If your engines are
pre-launched by another system through `--rollout-external-engine-addrs`, use
NCCL or disk transport instead.

## Quick Start

```bash
--update-weight-mode full
--update-weight-transport mooncake
--mooncake-metadata-server P2PHANDSHAKE
--mooncake-protocol tcp
--mooncake-buffer-size $((8 * 1024 * 1024 * 1024))
--mooncake-buffer-count 1
```

`P2PHANDSHAKE` is the only supported metadata mode today. The SGLang receiver
returns a per-engine target name during initialization, so slime does not require
an external Mooncake metadata service for this path.

For RDMA deployments, switch the protocol and pass the device name used by the
Mooncake runtime:

```bash
--mooncake-protocol rdma
--mooncake-device-name mlx5_0
```

## Current Limits

Mooncake transport intentionally starts with a narrow contract:

- It supports only `--update-weight-mode full`; delta mode is rejected.
- It supports only slime-managed SGLang engines; external rollout engines are
  rejected.
- It does not support `--colocate`; colocated sync uses CUDA IPC tensor handles.
- It currently supports one GPU per rollout engine.
- Megatron pipeline model parallel size must be 1.
- SGLang DP size and EP size must be 1.
- SGLang DP attention is not supported.
- `--mooncake-metadata-server` must stay `P2PHANDSHAKE`.
- `--mooncake-rpc-port-base` is reserved and rejected by argument validation.
- `--mooncake-buffer-count` is reserved for future double buffering and must be
  `1`; the sender uses slot 0 for every update.

For `--sglang-config`, these limits are checked only for server groups that
receive training weights (`update_weights: true`). Frozen reference/reward
models and placeholder groups do not need Mooncake receivers.

## Buffer Sizing

`--mooncake-buffer-size` must be large enough for the largest weight bucket sent
by slime. If it is unset, slime uses `--update-weight-buffer-size`.

If a bucket is larger than the receiver buffer, slime fails the sync before
writing into Mooncake. Increase either `--mooncake-buffer-size` or reduce
`--update-weight-buffer-size` so every bucket fits the receiver buffer.

`--mooncake-buffer-count` is present only to keep the receiver API shape ready
for future double buffering. It does not increase concurrency in the current
implementation, and values other than `1` are rejected.

## Relationship With Other Transports

`--update-weight-mode` decides what gets sent, and `--update-weight-transport`
decides how it reaches SGLang:

| mode | transport | behavior |
|---|---|---|
| `full` | `nccl` | broadcast every HF weight chunk over a trainer-engine NCCL group |
| `full` | `disk` | write a complete HF checkpoint, then call `update_weights_from_disk` |
| `full` | `mooncake` | write each bucket into SGLang receiver buffers through Mooncake TransferEngine |
| `delta` | `nccl` | broadcast sparse changed positions and values over NCCL |
| `delta` | `disk` | write sparse safetensors, then call `update_weights_from_disk(load_format="delta")` |

Use Mooncake transport only for the `full` + slime-managed SGLang case above.
For external engines, heterogeneous GPU fleets, or cross-cluster deployments
with a shared filesystem, see [External Rollout Engines](external-rollout-engines.md)
and [Delta Weight Sync](delta-weight-sync.md).
