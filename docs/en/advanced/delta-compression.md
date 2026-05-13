# Partial Weight Sync (Delta / Selective)

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Modes: delta vs selective](#modes-delta-vs-selective)
- [How it Works](#how-it-works)
- [Choosing the Wire Encoding](#choosing-the-wire-encoding)
- [Precision Behaviour](#precision-behaviour)
- [Periodic Base Sync](#periodic-base-sync)
- [Why Not Colocated](#why-not-colocated)

## Overview

For **non-colocated** runs, slime's default weight sync broadcasts every parameter on every training step. The full broadcast scales linearly with model size and dominates the sync phase even when only a small fraction of weights actually change between steps. Partial-update modes keep a pinned-CPU snapshot of the last sync's weights and broadcast only the changed-position payload; the SGLang receiver applies it without re-touching unchanged params. During typical RL fine-tuning at conservative learning rates the per-step diff is sparse — a few percent of weights — so the wire shrinks proportionally.

## Quick Start

Enable a partial mode on the trainer side:

```bash
--update-weight-mode delta                # 'delta' / 'selective' / 'full' (default)
--update-weight-partial-encoding sparse_indices
--update-weight-delta-dtype fp32          # delta mode only
--update-weight-base-sync-interval 30     # safe to set very large (e.g. 10000) to disable
                                          # periodic base syncs — partial-mode apply is
                                          # lossless (see "Periodic Base Sync" below).
```

And one knob on the SGLang side (auto-mirrored by slime as `--sglang-update-weight-partial-chunk-bytes`):

```bash
--sglang-update-weight-partial-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

See [examples/delta_compression/run-glm4.7-355B-A32B-delta.sh](../../../examples/delta_compression/run-glm4.7-355B-A32B-delta.sh) for a complete non-colocated launcher.

## Modes: delta vs selective

Both modes share the same sender pipeline (snapshot, mask determination, sparse encoding, bucketed broadcast) and the same wire format. They differ only in what the values mean and how the receiver applies them:

| | `--update-weight-mode delta` | `--update-weight-mode selective` |
|---|---|---|
| Values on wire | `(current − snapshot)` cast to `--update-weight-delta-dtype` (default fp32) | new param values at changed positions, in the snapshot's dtype |
| "Unchanged" signal at receiver | implicit (zero delta at unchanged positions) | NaN sentinel in the decoded dense tensor |
| Receiver apply | `param += delta` (in-place add, auto-promotes for fp32 math, casts back to param dtype) | `param[~isnan(src)] = src[~isnan(src)]` (selective overwrite) |
| Wire bytes (values portion) | 4×nnz @ fp32 | 2×nnz @ bf16 (½× delta) |
| Lossless? | yes when `delta-dtype` > param dtype | yes by construction (no arithmetic) |

Pick `selective` when you want the smaller wire and don't need fp32 arithmetic margin; pick `delta` when you'd rather keep the arithmetic path for compatibility or want to amplify sub-bf16 deltas via the fp32 subtraction.

## How it Works

Per sync, on the trainer (PP-source rank only):

1. **Compute the payload**: for delta, lift current weights and pinned-CPU snapshot to delta_dtype and subtract; for selective, take the bf16 mask `current != snapshot` and emit new values with NaN at unchanged positions.
2. **Encode**: sparse-encode active positions into two flat packed tensors (`__packed_keys__`, `__packed_values__`) plus a per-param manifest (`PartialWeightSpec.params`).
3. **Bucket and broadcast**: pack multiple parameters per NCCL broadcast (`--update-weight-buffer-size` controls the bucket cap).
4. **Snapshot new prev**: D2H copy of the just-sent weights onto a side stream so it overlaps with downstream broadcast/encode work.

On the SGLang receiver:

1. **Broadcast**: receive the two packed tensors per bucket.
2. **Decode lazily**: yield one decoded dense tensor per parameter; unchanged positions are filled with the mode's sentinel (0 for delta, NaN for selective). The consumer's `chunk_byte_cap` bounds peak HBM during decode (`encoded_buffers + in-flight chunk`).
3. **Apply**: route the decoded tensors through the model's normal `load_weights` path, but with `Tensor.copy_` / `fill_` rewired by a context manager:
   - For `delta`: `_additive_load_context` redirects writes that target param storage to `add_` (PyTorch auto-promotes for fp32 math and casts back on store, so deltas keep fp32 precision).
   - For `selective`: `_selective_load_context` redirects writes that target param storage to a masked overwrite (`param[~isnan(src)] = src[~isnan(src)]`), leaving NaN positions untouched.

Auxiliary writes (scratch buffers, dtype temporaries, `post_load_weights` for fp8-scale recompute or MoE bias materialization) keep their normal overwriting semantics in both contexts.

The wire protocol — `PartialWeightSpec` (encoding + per-param manifest), and per-param `PartialWeightParam` (name, dtype, shape, key/value slice ranges) — is defined in `sglang.srt.managers.io_struct` (added by the slime SGLang patch).

## Choosing the Wire Encoding

`--update-weight-partial-encoding` accepts three values:

| value | wire layout | when to pick |
|---|---|---|
| `sparse_indices` | int32 active offsets + values | low change density (< ~3%) |
| `sparse_bitmask` | 1 bit per element + values | moderate change density (> ~3%) |
| `dense` | identity, one tensor per param | debugging the apply path |

The break-even density between the two sparse encodings is independent of the value dtype. With `n = numel`, `k = nnz`, `v = value bytes`:

```
sparse_indices wire = k * (4 + v)
sparse_bitmask wire = ceil(n / 8) + k * v
```

Equal when `4k = n/8`, i.e. `k/n = 1/32 ≈ 3.125%`. Below that, indices is smaller; above, bitmask is smaller. For typical RL fine-tuning at moderate learning rates, `sparse_indices` wins; for early-training high-LR phases where most weights move every step, switch to `sparse_bitmask`.

## Precision Behaviour

For `delta` mode, `--update-weight-delta-dtype` is the *math* dtype, not just the wire dtype. The subtraction is performed at `delta_dtype` on both operands (after promoting from the param dtype), and the receiver's `param.data.add_(fp32_delta)` lets PyTorch do the addition at the common dtype (fp32) before casting the result back into the bf16 param. This recovers small-magnitude deltas that would otherwise round to zero through a bf16 subtraction.

For `selective` mode there is no arithmetic — the receiver overwrites changed positions with the trainer's exact bf16 values — so precision is bit-perfect regardless of `--update-weight-delta-dtype` (the flag is silently ignored).

The CPU snapshot occupies only the param dtype's bytes in both modes (no fp32 inflation of pinned memory).

## Periodic Base Sync

The first sync of every job is always a *base sync* (a full broadcast that re-establishes the snapshot). After that, slime sends partial syncs until `committed_syncs % --update-weight-base-sync-interval == 0`, at which point a base sync runs again.

With `--update-weight-delta-dtype fp32` (delta mode) or in selective mode, the partial apply is **lossless**: every bf16 value is exactly representable in fp32, the subtraction `current_fp32 − snapshot_fp32` produces the exact difference between the two stored bf16 values, and the receiver's in-place `bf16_param.add_(fp32_delta)` reconstructs the trainer's bf16 state bit-for-bit when the fp32 result is rounded back to bf16. Selective is lossless by construction (direct overwrite). Because no error accumulates across partial syncs, receiver state never drifts from a base-sync reference no matter how many partial syncs elapse — periodic base sync is not needed for correctness. Setting `--update-weight-base-sync-interval` to a very large integer (e.g. `10000`) effectively disables it and is fine in practice.

The only operational reason to keep an occasional base sync is recovery — e.g. a rollout engine that joins mid-training and needs a complete state before it can apply partial updates. If you set `--update-weight-delta-dtype bf16` (delta only, not higher than the param dtype) to save wire bytes, the delta apply is no longer lossless and a finite interval starts to matter.

## Why Not Colocated

Colocated weight sync uses CUDA IPC: the engine maps the trainer's parameter storage directly into its own process. There is no NCCL broadcast, and "wire size" is one IPC handle per param (~64 B). Partial encoding's `bytes saved on the wire` benefit is zero, while the partial-update bookkeeping (snapshot + subtract/mask + sparse encode) is pure overhead. Slime rejects `--update-weight-mode delta --colocate` and `--update-weight-mode selective --colocate` at argparse time.
