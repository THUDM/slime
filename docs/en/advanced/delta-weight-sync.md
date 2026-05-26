# Delta Weight Sync

- [Why](#why)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Encoding Choice](#encoding-choice)
- [Why Not Colocated](#why-not-colocated)

## Why

Slime's default sync broadcasts every parameter every step. The cost scales linearly with model size and dominates the sync phase, even though only a few percent of weights change between consecutive RL steps. Delta sync keeps a pinned-CPU snapshot of the last broadcast and ships only the positions whose bytes differ.

The motivating use case is **training/inference disaggregation** — running the trainer and the rollout engines in *different datacenters* over a shared filesystem with bandwidth on the order of 100s of MB/s, where a full broadcast is infeasible but a sparse delta (~3% density, ~5 GB for a 355B model) is. The same delta machinery also runs over NCCL inside a single datacenter, where it serves as the validation baseline that proves the wire encoding and apply logic are correct.

Prior art: selective overwrite is inspired by [arXiv:2509.19128](https://arxiv.org/abs/2509.19128); the cross-DC disaggregation motivation is from [Fireworks AI — Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think).

## Quick Start

Disk transport (training/inference disaggregation — the main use case):

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd                 # best for ≤ 300 MB/s shared FS
--update-weight-delta-dir /shared/fs/delta-updates
```

NCCL transport (intra-datacenter validation baseline):

```bash
--update-weight-mode delta
--update-weight-transport nccl
--update-weight-encoding indices                     # lowest compute, no compression
```

Receiver-side tuning (applies to both transports):

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))  # byte cap per load_weights call
--sglang-update-weight-delta-read-workers 4                           # parallel I/O threads (disk only)
```

See [examples/delta_weight_sync/run-glm4.7-355B-A32B-delta.sh](../../../examples/delta_weight_sync/run-glm4.7-355B-A32B-delta.sh) for a complete launcher.

## How It Works

Both transports share one sender pipeline, one wire layout, and one receiver-side decoder; only the per-flush carrier differs.

**Sender (per sync, PP-source rank only):**

1. **Diff** the current weights against the pinned-CPU snapshot via bytewise compare (`current.view(int_dtype) != snapshot.view(int_dtype)`) — lossless, dtype-agnostic, no arithmetic.
2. **Encode** changed (position, value) pairs into a packed `__positions__` byte blob + `__values__` tensor + per-param decoding manifest. The encoding (`indices`, `deltas`, `deltas_zstd`) governs only how positions are packed; values are sent verbatim in the param's dtype.
3. **Bucket** per-chunk encodes up to `--update-weight-buffer-size` bytes, then flush:
   - NCCL: broadcast `(__positions__, __values__)` to the rollout engines with a `DeltaSpec` (encoding + per-param manifest) carried in the Ray RPC.
   - Disk: write one safetensors file per flush under `weight_v{N:06d}/`. Async background thread does the I/O + optional zstd compression off the critical path.
4. **Snapshot the just-sent values** via a D2H copy on a side stream so it overlaps with the next chunk's encode.

**End-of-sync (disk only):** write a `DONE` marker, then rank 0 fires one HTTP push per engine and removes the directory after every engine acknowledges.

**Receiver:**

For both transports, the receiver ends up calling the same `_apply_delta_payload(encoding, params, positions, values)` helper. It decodes each param's slice into a full-shape tensor with NaN at unchanged positions, then routes it through `model.load_weights(...)` under a `_delta_apply_context` that patches `Tensor.copy_` / `Tensor.fill_` to perform NaN-masked overwrite. Auxiliary writes (scratch buffers, fp8 scales, MoE biases via `post_load_weights`) keep their normal semantics.

Selective overwrite has no arithmetic — the receiver writes the trainer's exact bytes at changed positions — so it's lossless by construction and there's no notion of drift to fight with periodic base re-syncs.

## Encoding Choice

`--update-weight-encoding` picks how positions are packed. All three share the same on-wire layout (`__positions__` uint8 blob + `__values__` tensor + per-param manifest); decoder dispatches on the metadata.

| value | positions | when to pick |
|---|---|---|
| `indices` | int32 absolute positions (4 bytes / nnz) | NCCL or fast intra-cluster FS (≥ ~600 MB/s) |
| `deltas` | uint16 gap-deltas with uint32 fallback (~2 bytes / nnz at 2% density) | medium FS bandwidth (~300-500 MB/s) |
| `deltas_zstd` | `deltas` wrapped in zstd L1 on disk | cross-DC / cross-region shared FS (≤ ~300 MB/s) |

**Why gap-encoded positions are smaller**: positions come out of `mask.nonzero()` already sorted ascending. At density `p`, the expected gap between consecutive nonzero positions is `1/p`, and `P(gap > 65535) ≈ exp(-p · 65535)`. At p = 2% that's effectively zero, so uint16 fits with a uint32 per-param fallback for pathological inputs. Half the position bytes of `indices`, lossless.

**Break-even with `indices`** at our density (~2%): `deltas` halves the positions blob (which dominates the wire); `zstd` shaves another ~35-40% on top by compressing the gap byte stream, at the cost of ~250ms/file compress + ~150ms/file decompress. The crossover with `indices` is where compress/decompress compute exceeds the bandwidth savings — empirically around 500 MB/s for `deltas` and 300 MB/s for `deltas_zstd`.

## Why Not Colocated

Colocated weight sync uses CUDA IPC: only a memory handle (~64 B) crosses processes. Delta encoding's "bytes saved on the wire" benefit is zero, while the bookkeeping (snapshot + diff + sparse encode) is pure overhead. Slime rejects `--update-weight-mode delta --colocate` at argparse time.
