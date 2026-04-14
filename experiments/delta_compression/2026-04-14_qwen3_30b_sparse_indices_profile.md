# Qwen3-30B-A3B Sparse Indices Profiling Report

**Config**: `qwen3_dapo_noncolocate_delta_compression_indices_profile`
**Commit**: `d202772e` (delta-compression-feature branch)
**Transport**: `sparse_indices`
**Model**: Qwen3-30B-A3B (MoE, 30B total, 3B active)
**Infrastructure**: 4 nodes x 8 H200 GPUs (32 total: 16 actor + 16 rollout)
**Modal App**: `ap-Nqtb064nxzGDFGSfW2lzng`
**Date**: 2026-04-14
**Steps**: 3 (num_rollout=3), step 0 = full sync, steps 1-2 = delta sync

## Step 0: Full Sync (Baseline Establishment)

- **update_weights_total**: 17.8s (PP rank 0: 17.810s, PP rank 1: 17.812s)
- full_chunks=225, full_tensors=9433, full_bytes=30,532,120,576 (30.5 GB)
- baseline_h2d=0.000s (no delta baseline yet)
- delta_compute=0.000s
- baseline_commit=10.958s (GPU->CPU pinned memory, dominant cost)
- tp_gather: 0.546-0.574s
- First expert bucket ep_gather: 0.496s, subsequent: 0.007-0.012s
- Per-chunk broadcast: 0.040-0.507s (first chunk higher due to NCCL warmup)

## Step 1: Delta Sync (sparse_indices)

- **update_weights_total**: 7.2s (PP rank 0: 7.239s, PP rank 1: 7.238s)
- **Improvement vs full sync**: 2.5x faster (17.8s -> 7.2s)

### Sender-Side Breakdown

| Phase | Time |
|-------|------|
| baseline_h2d (CPU->GPU) | 0.696s |
| delta_compute (subtract + cast) | 0.960s |
| baseline_commit (GPU->CPU) | 0.591s |
| **Sender subtotal** | **~2.2s** |

### Sparsity Statistics

- total_elements: 15,266,060,288 (~15.3B elements)
- total_nonzeros: 88,834,467 (~89M nonzeros)
- **density: 0.5819%** (99.4% sparse)
- delta_chunks: 225, delta_sent_tensors: 9,433

### Chunk Efficiency (per flush)

**Non-expert chunk** (PP rank 0, first flush):
- original_tensors=217, zero_nnz_tensors=67
- dense_bytes=1,541,091,328 (1.54 GB)
- encoded_bytes=30,736,818 (30.7 MB)
- **Compression: 50x**
- materialize=0.039s, broadcast=0.257s

**Expert chunk** (bulk of data):
- original_tensors=9,216, zero_nnz_tensors=225-293
- dense_bytes=28,991,029,248 (29 GB)
- encoded_bytes=468-502 MB
- **Compression: 58-62x**
- materialize=0.631-0.647s, lock=0.004-1.131s, broadcast=1.116-1.308s

### Receiver-Side Breakdown (SGLang)

**Non-expert batch** (217 tensors):
- recv=0.000s, decode=0.006-0.012s, load_weights=0.020-0.022s
- **total: 0.028-0.033s**

**Expert batch** (9,216 tensors):
- recv=0.000s, decode=0.182-0.183s, load_weights=0.774-0.784s
- **total: 0.958-0.972s**
- apply_mode=per_tensor (current: one load_weights call per tensor)

## Step 2: Delta Sync (second delta)

- **update_weights_total**: 7.26s (from perf metrics: `perf/update_weights_time: 7.262`)
- Consistent with step 1, confirming steady-state behavior

## Key Observations

1. **Delta compression works extremely well for bandwidth**: 30.5 GB -> ~500 MB (60x compression) at 0.58% density.

2. **Sender bottleneck at 30B scale**: ~2.2s for compute/H2D/commit phases. At 355B with ~10x more parameters, this would scale to ~22s.

3. **Receiver is fast at 30B**: ~1s total. Per-tensor load_weights for 9216 expert tensors takes 0.78s. At 355B with ~10x more expert tensors, this could become the dominant cost.

4. **NCCL broadcast is fast**: 1.1-1.3s for 500MB of sparse data. At 355B the data volume would be ~5GB (still sparse), so maybe ~10-13s.

5. **Zero-nnz tensors are common**: 67/217 non-expert tensors (31%) and 225-293/9216 expert tensors (2.5-3.2%) have zero deltas. Skipping these saves receiver work.

6. **Sender `lock` time varies**: 0.004s to 1.131s depending on rollout engine contention.

## Extrapolation to 355B

The 355B model has:
- ~170 GB full weights (vs 30.5 GB)
- ~5.6x ratio
- TP=8 PP=4 EP=8-16 (vs TP=4 PP=2 EP=8)
- More expert layers, more MoE parameters

Expected delta sync at 355B (extrapolated):
- Sender compute: ~12-15s (baseline H2D + delta compute + commit scale with parameter count)
- NCCL broadcast: ~6-10s (data volume scales, but still compressed)
- Receiver apply: ~5-10s (per-tensor overhead scales with tensor count)
- **Expected total: ~25-40s** (vs current measured ~120s)

The ~120s measured at 355B suggests there are additional scaling factors beyond linear, possibly:
- PP=4 serialization (each PP stage processes sequentially)
- Expert parallelism overhead at EP=16
- More chunks = more lock contention

## Optimization Strategy (Directional)

Based on this data, the two most impactful optimizations are:

1. **Skip zero-delta tensors on sender**: 2.5-31% of tensors are all-zero. Eliminating them from sparse encoding, NCCL broadcast, and receiver apply saves proportional work.

2. **Batch receiver load_weights**: Currently 9,216 individual `load_weights()` calls for expert tensors. Batching into mini-batches (e.g., 32 MiB cap) would reduce Python/framework overhead per call.
