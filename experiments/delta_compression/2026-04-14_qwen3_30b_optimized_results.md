# Qwen3-30B-A3B Optimized Delta Compression Results

**Config**: `qwen3_dapo_noncolocate_delta_compression_indices_profile`
**Commit**: `1ca456eb` (delta-compression-feature branch, optimized)
**Transport**: `sparse_indices`
**Model**: Qwen3-30B-A3B (MoE, 30B total, 3B active)
**Infrastructure**: 4 nodes x 8 H200 GPUs (32 total: 16 actor + 16 rollout)
**Modal App**: `ap-EPnwYqik7a2ymYs6BsZ7V6`
**Date**: 2026-04-14

## Optimizations Applied

1. **Skip zero-delta tensors** (sender): Zero-delta tensors excluded from sparse metadata and packed tensors. Still tracked in baseline_updates for correctness.
2. **Batched load_weights** (receiver): Per-tensor `load_weights()` calls replaced with mini-batches (32 MiB cap). Reduces Python/framework overhead per call.

## Results Comparison

| Metric | Baseline (pre-opt) | Optimized | Change |
|--------|-------------------|-----------|--------|
| **update_weights_total** | **7.24s** | **5.97s** | **-17.5%** |
| Receiver total (expert) | 0.972s | 0.330s | **-66%** |
| Receiver load_weights | 0.784s | 0.177s | **-77%** |
| Receiver decode | 0.183s | 0.143s | -22% |
| Receiver load_calls | 9,216 | 742 | -92% |
| Sender materialize | 0.647s | 0.450s | -30% |
| Sender broadcast | 1.116s | 0.441s | -60% |
| Density | 0.58% | 0.75% | (different step) |

## Step 1: Delta Sync (Optimized)

- **update_weights_total**: 5.967s (PP rank 0), 5.968s (PP rank 1)
- delta_chunks=225, delta_sent_tensors=9,433

### Sender-Side Breakdown

| Phase | Time |
|-------|------|
| baseline_h2d (CPU->GPU) | 0.671s |
| delta_compute (subtract + cast) | 0.424s |
| sparsity_scan (count_nonzero) | 0.725s |
| baseline_commit (GPU->CPU) | 0.590s |
| **Sender subtotal** | **~2.4s** |

### Zero-Tensor Skip Stats

- Non-expert chunk: skipped_zero=73 out of 217 tensors (33.6%)
- Expert buckets: skipped_zero=22-31 out of 1640 per bucket (~1.5%)
- Fewer tensors in metadata = less NCCL data + less receiver work

### Receiver-Side (SGLang, batched mode)

**Expert batch** (dominant):
- tensor_count=7,417 (down from 9,216 due to zero-skip)
- load_calls=742 (down from 9,216 per-tensor calls)
- recv=0.000s, decode=0.143s, load_weights=0.177s, **total=0.330s**

### Sparsity Statistics

- total_elements: 15,266,060,288
- total_nonzeros: 114,527,445
- density: 0.75%

## Key Takeaways

1. **Batched load_weights is the biggest win**: Reducing from 9,216 individual calls to 742 batched calls cut receiver load_weights time by 77% (0.784s -> 0.177s).

2. **Zero-skip helps proportionally**: 33.6% of non-expert tensors skipped. Saves NCCL bandwidth and receiver decode/apply work.

3. **Overall 17.5% improvement** on Qwen3-30B (7.24s -> 5.97s). Receiver improvement is 66% but sender still dominates total time.

4. **355B extrapolation**: At 355B where current delta step is ~120s, the receiver side is a larger fraction. 66% receiver speedup + reduced NCCL from zero-skip should translate to meaningful improvement.
