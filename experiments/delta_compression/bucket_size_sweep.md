# Delta Bucket Size Sweep — GLM4.7-355B-A32B

## Setup
- Model: GLM4.7-355B-A32B (MoE, 355B total, 32B active)
- Config: `glm47_355b_a32b_noncolocate_delta_compression_indices`
- Transport: `sparse_indices`
- Infrastructure: 16 nodes x 8 H100 GPUs
- Parallelism: TP=8, PP=4, EP=16
- Total weights: ~170 GB (split across PP ranks)
- Delta density: ~2-3% (sparse encoded ~5 GB total)

## Results

| Bucket Limit | # Flushes (total) | Delta Sync Time | Commit |
|---|---|---|---|
| 512 MB sparse-est (default) | ~598 | 60.5s | `1ca456eb` |
| 10 GB dense | ~18 | 52.5s | `946bbb51` |
| 20 GB dense | ~10 | 52.8s | `f2dacc52` |

## Per-Flush Overhead Analysis (10 GB bucket)

With 18 flushes in 52.5s = **~2.9s per flush** on average.

Each flush involves:
1. **Materialize**: sparse encode all tensors in bucket (~0.04s)
2. **Lock acquire**: acquire rollout engine lock (0.003-2.9s, high variance from PP rank contention)
3. **NCCL broadcast**: send sparse-encoded data to all rollout engines (~0.5-1.4s)
4. **Lock release**: release rollout engine lock
5. **Receiver apply**: decode + batched load_weights (~0.2-1.4s, overlapped with sender)

The dominant cost is **lock contention between PP ranks**. With PP=4, four ranks compete for the same rollout engine lock, serializing their broadcasts.

## Theoretical Minimum with Current Architecture

Even with 1 flush per PP section (non-expert + expert = 2 per rank, 8 total across 4 PP ranks):
- Lock acquire overhead: ~8 lock cycles at ~1-2s contention each = ~8-16s
- NCCL broadcast: ~8 broadcasts of ~600 MB each at ~0.7s = ~5.6s
- But serialized through lock, so total = sum not parallel

Estimated minimum: **~20-30s** with this per-flush-locking architecture.

## Root Cause

The current architecture serializes weight updates through a global rollout engine lock:
```
PP rank 0: [lock → broadcast chunk 1 → unlock] → [lock → broadcast chunk 2 → unlock] → ...
PP rank 1: [lock → broadcast chunk 1 → unlock] → ...
PP rank 2: [lock → broadcast chunk 1 → unlock] → ...
PP rank 3: [lock → broadcast chunk 1 → unlock] → ...
```

All four PP ranks compete for the same lock. Even with fewer flushes, the sequential lock→broadcast→unlock pattern remains.

## What Would Be Needed to Beat 50s Baseline

To truly beat the non-delta baseline (50s), we need to eliminate the per-flush lock overhead entirely:

### Option A: Single lock for entire delta sync
- Acquire lock once at the start of update_weights
- All PP ranks broadcast without re-acquiring
- Release lock once at the end
- Requires: changing the lock granularity

### Option B: Pipelined broadcasts without lock
- Remove the per-flush lock entirely for delta mode
- Each PP rank broadcasts its chunks without waiting for lock
- Receiver handles out-of-order application
- Requires: ensuring receiver can accept concurrent broadcasts

### Option C: Accumulate all deltas, single broadcast
- All PP ranks gather+compute deltas into memory
- One PP rank (rank 0) collects all deltas and does a single broadcast
- Eliminates PP rank serialization entirely
- Requires: cross-PP-rank delta collection

### Option D: Non-blocking delta transfer (future)
- Compute deltas asynchronously while rollout is generating
- Transfer deltas while rollout is running (before lock acquire)
- Only hold lock briefly for the final apply signal
- Most applicable for the disaggregated trainer/rollout future

## Inline D2H Commit + Bucket Architecture Results (2026-04-15)

### Context
Inline D2H baseline commit: copy current weight to CPU pinned per-tensor
during delta compute (non_blocking). Prevents gathered buffer OOM by not
storing GPU refs in baseline_updates.

| Config | Bucket Type | Bucket Limit | Flushes/PP | Delta Sync | OOM? | Commit |
|---|---|---|---|---|---|---|
| Eager sparse + inline D2H | sparse | 1 GB | 16 | 90.9s | No | `15d53c00` |
| Eager sparse + inline D2H | sparse | 5 GB | 3 | 70.4s | No | `507c7742` |
| Dense + inline D2H + materialize-at-flush | dense | 10 GB | ~18 | TBD | TBD | `8bf1a5ef` |

### Key Finding: Eager materialization is counterproductive
With eager materialization, the bucket accumulates sparse-encoded data.
A 5 GB sparse bucket broadcasts ~2 GB per flush at 3-10s each.
Per-flush profiling (5 GB sparse, 70.4s total):
- materialize=0.003-0.021s (consolidation only, already encoded)
- lock=0.003-6.5s (PP rank contention)
- broadcast=0.3-9.7s (**broadcasting GB of sparse data is slow**)

Without eager materialization, the bucket holds dense deltas. At flush:
10 GB dense → ~500 MB sparse → broadcast 0.5s. Much faster per-flush.
