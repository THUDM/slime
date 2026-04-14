# Ask Codex Input

## Question

# Second Codex Reasonability Review: Delta Compression Optimization Plan

## Current Candidate Plan

### Goal
Optimize delta compression weight updates for GLM4.7-355B large MoE model to beat the 50s baseline (currently ~120s/step).

### Proposed Acceptance Criteria

- AC-1: Comprehensive phase-level profiling infrastructure on both sender (Megatron/slime) and receiver (SGLang) sides, covering: TP/EP gather, HF conversion, baseline H2D transfer, delta compute, sparsity scan, materialization, NCCL broadcast, SGLang receive/decode, and load_weights application. Each phase timed independently.
  - Positive: profiling output shows timing for each phase
  - Negative: missing any phase or showing aggregate-only timing
  
- AC-2: Steady-state delta compression steps (step 2+) achieve avg update_weights_time < 50s on GLM4.7-355B-A32B non-colocate config.
  - Positive: 3-step run shows step 2-3 averaging under 50s
  - Negative: step 2-3 avg >= 50s fails
  
- AC-3: First step may be slower than baseline (CPU pinning + baseline building acceptable).
  - Positive: step 1 completes successfully regardless of time
  
- AC-4: Correctness preserved - optimized delta compression produces numerically equivalent results to non-delta baseline within floating-point tolerance (bf16 precision).
  - Positive: loss values within tolerance of baseline
  - Negative: loss diverges significantly from baseline
  
- AC-5: Peak GPU memory usage does not exceed non-delta baseline by more than 10%. No new OOM conditions.
  - Positive: max GPU memory within bounds
  - Negative: OOM or >10% excess memory
  
- AC-6: Each experiment recorded with: config name, commit hash, transport method, per-phase timings, sparsity stats, total update_weights_time per step, and outcome summary.
  - Positive: experiment log is complete and parseable
  - Negative: missing fields or uncommitted results

### Proposed Milestones

**Milestone 1: Profiling Infrastructure**
- Phase A: Add sender-side phase-level timing (delta_weight_update.py, update_weight_from_distributed.py)
- Phase B: Add receiver-side phase-level timing (sglang patch)
- Phase C: Add sparsity statistics collection (per-tensor nnz, density, size)
- Run profiling experiment on current delta_compression_indices config

**Milestone 2: Bottleneck Analysis**
- Analyze profiling data to identify top bottlenecks
- Compare sender vs receiver time breakdown
- Determine sparsity distribution at 355B scale
- Produce analysis document with findings

**Milestone 3: Targeted Optimization**
- Based on profiling findings, implement optimizations. Candidate directions include:
  a) Merge estimation + materialization into single pass (eliminate double count_nonzero/nonzero)
  b) Skip all-zero tensors before send
  c) Adaptive transport selection per bucket based on measured density
  d) Batch reconstruction on SGLang side instead of per-tensor
  e) Direct scatter-add kernel instead of reconstruct-then-apply
  f) Reduce CPU↔GPU synchronization barriers
  g) Optimize bitmask pack/unpack operations
- Each optimization tested independently before combining

**Milestone 4: Validation**
- Run optimized config against 50s baseline target
- Verify correctness via loss comparison
- Document final performance and approach

### Path Boundaries

**Upper Bound**: All identified optimizations implemented and tested, with comprehensive profiling infrastructure remaining in codebase as optional (flag-gated). Adaptive transport selection working. Both slime and sglang patch updated.

**Lower Bound**: The single most impactful optimization identified through profiling is implemented, achieving <50s steady-state. Basic profiling added temporarily for analysis then cleaned up. At minimum one transport mode optimized.

### Key Architectural Decisions
- Changes limited to: delta_weight_update.py, update_weight_from_distributed.py, and sglang_delta_compression.patch
- Config files remain unchanged (already runnable)
- First-principles approach: understand why it's slow, then fix the root cause

## Review Request

Please review this candidate plan for reasonability. Output in this exact format:

AGREE: Points you accept as reasonable
DISAGREE: Points you consider unreasonable and why
REQUIRED_CHANGES: Must-fix items before convergence
OPTIONAL_IMPROVEMENTS: Non-blocking improvements
UNRESOLVED: Opposite opinions needing user decisions

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-04-14_07-19-37
