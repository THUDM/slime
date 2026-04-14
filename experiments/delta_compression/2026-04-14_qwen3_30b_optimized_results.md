# Qwen3-30B-A3B Optimized Delta Compression Results

**Config**: `qwen3_dapo_noncolocate_delta_compression_indices_profile`
**Commit**: `1ca456eb` (delta-compression-feature branch, with optimizations)
**Transport**: `sparse_indices`
**Model**: Qwen3-30B-A3B (MoE, 30B total, 3B active)
**Infrastructure**: 4 nodes x 8 H200 GPUs (32 total: 16 actor + 16 rollout)
**Modal App**: `ap-EPnwYqik7a2ymYs6BsZ7V6`
**Date**: 2026-04-14

## Optimizations Applied

1. **Sender: skip zero-delta tensors** — Tensors with all-zero deltas are excluded from sparse metadata, packed values, and receiver work. Still tracked in baseline_updates for correctness.

2. **Receiver: batched load_weights** — Decoded tensors accumulated into 32 MiB mini-batches, then applied with a single `load_weights()` call per batch. Reduces Python/framework call overhead.

## Results Comparison

| Metric | Pre-Optimization | Optimized | Improvement |
|--------|-----------------|-----------|-------------|
| **Delta update_weights_total** | 7.2s | **6.4s** | 11% faster |
| Full sync update_weights_total | 17.8s | 17.0s | ~same |
| Receiver non-expert total | 0.029s | 0.012s | 2.4x |
| Receiver expert total | 0.972s | **0.376s** | **2.6x** |
| Non-expert tensors processed | 217 | 149 (skipped 75) | 34% fewer |
| Expert tensors processed | 9216 | 8945 (skipped 271) | 3% fewer |
| Non-expert load_calls | 217 | 41 | 5.3x fewer |
| Expert load_calls | 9216 | 895 | 10.3x fewer |

## Detailed Profiling (Step 1 - Delta Sync)

### Sender Side (PP rank 0)
- update_weights_total: **6.418s**
- Non-expert chunk: materialize=0.037s, broadcast=0.241s, encoded_bytes=30.5MB (from 1.54GB dense)
- Expert chunk: materialize=0.652s, broadcast=0.547s, encoded_bytes=513MB (from 29GB dense)
- skipped_zero: 75/218 non-expert (34.4%), 271/9216 expert (2.9%)
- metadata_bytes: 32KB non-expert, 2.07MB expert

### Receiver Side (all DP ranks consistent)
- Non-expert batch: 149 tensors, 41 load_calls, decode=0.005s, load_weights=0.006s, total=**0.012s**
- Expert batch: 8945 tensors, 895 load_calls, decode=0.157s, load_weights=0.209s, total=**0.376s**
- apply_mode: **batched** (confirmed)

### Sparsity Statistics
- Density: ~0.58% (consistent with pre-optimization)
- Per-tensor: layernorm weights often zero (density=0), attention/expert weights 1-3% density, lm_head 0.04% density

## Conclusions

1. **Batched load_weights is the bigger win**: Reducing from 9216 to 895 calls saved 0.6s on receiver (0.972s → 0.376s). This scales well to 355B.

2. **Zero-skip helps moderately**: 34% of non-expert tensors skipped. Expert tensors are rarely all-zero (only 3%), so the savings are smaller there.

3. **Overall improvement**: 6.4s vs 7.2s = 11% faster. At 355B scale (10x more parameters), the savings should be proportionally larger since the overhead being eliminated grows with tensor count.

4. **355B extrapolation**: If the current 355B ~120s is dominated by sender phases that scale with parameter count, and receiver apply takes ~10s at 355B scale, the batching optimization alone could save ~6s on the receiver (bringing 120s → ~114s). Combined with sender improvements and the zero-skip, a more significant reduction is expected. The 355B experiment needs to run to get actual numbers.
