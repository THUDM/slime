# Delta-Compression Weight Sync

- [Overview](#overview)
- [Quick Start](#quick-start)
- [How it Works](#how-it-works)
- [Choosing the Wire Encoding](#choosing-the-wire-encoding)
- [Precision Behaviour](#precision-behaviour)
- [Periodic Full Sync](#periodic-full-sync)
- [Why Not Colocated](#why-not-colocated)

## Overview

For **non-colocated** runs, slime's default weight sync broadcasts every parameter on every training step. The full broadcast scales linearly with model size and dominates the sync phase even when only a small fraction of weights actually change between steps. Delta-compression keeps a pinned-CPU snapshot of the last sync's weights, broadcasts only `(current − snapshot)` sparse-encoded, and the SGLang receiver applies it additively (`param += delta`). During typical RL fine-tuning at conservative learning rates the per-step delta is sparse — a few percent of weights — so the wire shrinks proportionally.

## Quick Start

Enable the mode and pick a wire encoding on the trainer side:

```bash
--update-weight-mode delta
--delta-compression sparse_indices
--delta-dtype fp32
--delta-full-interval 30   # safe to set very large (e.g. 10000) to disable
                           # periodic full syncs — with fp32 delta the apply
                           # is lossless (see "Periodic Full Sync" below).
```

And one knob on the SGLang side (auto-mirrored by slime as `--sglang-update-weight-delta-chunk-bytes`):

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

See [examples/delta_compression/run-glm4.7-355B-A32B-delta.sh](../../../examples/delta_compression/run-glm4.7-355B-A32B-delta.sh) for a complete non-colocated launcher.

## How it Works

Per sync, on the trainer (PP-source rank only):

1. **Compute delta**: lift the current weights and pinned-CPU snapshot of the last sync's weights to `delta_dtype` (fp32 by default), subtract, keep the fp32 result. The CPU snapshot stays at the model's dtype (typically bf16) — the cast happens only on GPU during compute.
2. **Encode**: for each parameter, extract nonzero positions and values into two flat packed tensors (`__packed_keys__`, `__packed_values__`) plus a per-param manifest (`WeightDeltaSpec.params`) describing slice offsets back into those buffers.
3. **Bucket and broadcast**: pack multiple parameters per NCCL broadcast (`--update-weight-buffer-size` controls the bucket cap).
4. **Snapshot new prev**: D2H copy of the just-sent weights onto a side stream so it overlaps with downstream broadcast/encode work.

On the SGLang receiver:

1. **Broadcast**: receive the two packed tensors per bucket.
2. **Decode lazily**: yield one decoded dense delta tensor per parameter; the consumer's `chunk_byte_cap` bounds peak HBM during decode (`encoded_buffers + in-flight chunk`).
3. **Additive apply**: route the deltas through the model's normal `load_weights` path, but with `torch.Tensor.copy_` / `fill_` rewired to `add_` whenever the destination falls inside a parameter's storage range. Auxiliary writes (scratch buffers, dtype temporaries, `post_load_weights` for fp8-scale recompute / MoE bias materialization) keep their normal overwriting semantics.

The wire protocol — `WeightDeltaSpec` (encoding + per-param manifest), and per-param `WeightDeltaParam` (name, dtype, shape, key/value slice ranges) — is defined in `sglang.srt.managers.io_struct` (added by the slime SGLang patch).

## Choosing the Wire Encoding

`--delta-compression` accepts three values:

| value | wire layout | when to pick |
|---|---|---|
| `sparse_indices` | int32 nonzero offsets + values | very sparse deltas (density < ~3%) |
| `sparse_bitmask` | 1 bit per element + values | moderately sparse deltas (density > ~3%) |
| `dense` | identity, one tensor per param | debugging the additive apply path |

The break-even density between the two sparse encodings is independent of the value dtype. With `n = numel`, `k = nnz`, `v = value bytes`:

```
sparse_indices wire = k * (4 + v)
sparse_bitmask wire = ceil(n / 8) + k * v
```

Equal when `4k = n/8`, i.e. `k/n = 1/32 ≈ 3.125%`. Below that, indices is smaller; above, bitmask is smaller. For typical RL fine-tuning at moderate learning rates, `sparse_indices` wins; for early-training high-LR phases where most weights move every step, switch to `sparse_bitmask`.

## Precision Behaviour

`--delta-dtype` is the *math* dtype, not just the wire dtype. The subtraction is performed at `delta_dtype` on both operands (after promoting from the param dtype), and the receiver's `param.data.add_(fp32_delta)` lets PyTorch do the addition at the common dtype (fp32) before casting the result back into the bf16 param. This recovers small-magnitude deltas that would otherwise round to zero through a bf16 subtraction.

The CPU snapshot still occupies only the param dtype's bytes (no fp32 inflation of pinned memory).

## Periodic Full Sync

The first sync of every job is always full. After that, slime sends deltas until `committed_syncs % --delta-full-interval == 0`, at which point a full sync runs again.

With `--delta-dtype fp32` (any precision higher than the param dtype), the delta apply is **lossless**: every bf16 value is exactly representable in fp32, the subtraction `current_fp32 − snapshot_fp32` produces the exact difference between the two stored bf16 values, and the receiver's in-place `bf16_param.add_(fp32_delta)` reconstructs the trainer's bf16 state bit-for-bit when the fp32 result is rounded back to bf16. Because no error accumulates across deltas, receiver state never drifts from a full-sync reference no matter how many deltas elapse — periodic full sync is not needed for correctness. Setting `--delta-full-interval` to a very large integer (e.g. `10000`) effectively disables it and is fine in practice.

The only operational reasons to keep an occasional full sync are recovery points — e.g. a rollout engine that joins mid-training and needs a complete state before it can apply deltas. If you set `--delta-dtype bf16` (or anything not higher than the param dtype) to save wire bytes, the apply is no longer lossless and a finite interval starts to matter.

## Why Not Colocated

Colocated weight sync uses CUDA IPC: the engine maps the trainer's parameter storage directly into its own process. There is no NCCL broadcast, and "wire size" is one IPC handle per param (~64 B). Delta encoding's `bytes saved on the wire` benefit is zero, while the delta compute + sparse encode + baseline snapshot are pure overhead. Slime rejects `--update-weight-mode delta --colocate` at argparse time.
