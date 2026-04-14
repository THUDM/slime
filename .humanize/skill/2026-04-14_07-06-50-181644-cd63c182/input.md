# Ask Codex Input

## Question

# Codex First-Pass Analysis: Delta Compression Optimization for Slime RL Framework

## Repository Context
This is the "slime" RL post-training framework at /home/ec2-user/nan_wonderland/slime. It trains large language models (MoE architectures) using reinforcement learning with non-colocated weight updates between training nodes (Megatron) and inference engines (SGLang).

## Core Problem
The delta compression feature sends weight differences (deltas) instead of full weights during RL training updates. For small MoE models (Qwen3-30B-A3B), delta compression achieves ~7s per step (comparable to baseline). But for large MoE (GLM4.7-355B-A32B), it degrades to ~120s per step vs 50s baseline. The goal is to make delta compression faster than the 50s baseline for large MoE.

## Key Architecture
- Training side (slime/backends/megatron_utils/update_weight/delta_weight_update.py): DeltaCompressionTracker computes deltas, then materializes them via one of three transport modes
- Three transport modes: dense (sends full delta tensor), sparse_indices (stores int32 indices + values for non-zeros), sparse_bitmask (packs 1-bit mask + values for non-zeros)
- SGLang side (docker/patch/latest/sglang_delta_compression.patch): Receives encoded deltas via NCCL broadcast, reconstructs per-tensor, applies additively using monkey-patched copy_/fill_ operations
- Full weights for GLM4.7-355B are ~170GB; deltas are sparse elementwise but almost all tensors have some non-zero values

## Key Properties of the Delta
- Each RL update produces sparse weight diffs (elementwise sparse)
- Almost all tensors have at least some non-zero values (rare for entire tensor to be zero)
- This was observed on small MoE; pattern may differ for large MoE
- The sparsity is the key property that should be exploited

## Current Implementation Details
- sparse_indices: For each tensor, finds non-zeros with torch.nonzero(), packs all indices (int32) and values into concatenated tensors, sends 2 tensors + metadata via NCCL
- sparse_bitmask: For each tensor, creates boolean mask, packs 8 bools into 1 byte, concatenates masks and values, sends 2 tensors + metadata
- SGLang reconstruction: Per-tensor sequential iteration - allocates scratch buffer, zeros it, scatters/unpacks values, then applies via additive load_weights
- Baseline storage: Full weight baseline stored in CPU pinned memory, delta computed as current - baseline

## Draft Requirements
The user wants:
1. Profile comprehensively to understand where time is spent (both training side and SGLang side)
2. Understand the three transport modes deeply from first principles
3. Optimize to beat 50s baseline for large MoE
4. First step can be slower (needs to pin CPU memory), but subsequent steps must be faster
5. Start from commit a6334ac6
6. May need to update both local slime code and sglang patch
7. Record all experiments and findings
8. Approach from first principles, not workarounds

## Questions for Analysis

Please provide your analysis in the following format:

CORE_RISKS: What are the highest-risk assumptions and potential failure modes?

MISSING_REQUIREMENTS: What requirements or edge cases are likely omitted?

TECHNICAL_GAPS: What feasibility or architecture gaps exist?

ALTERNATIVE_DIRECTIONS: What viable alternatives exist with tradeoffs?

QUESTIONS_FOR_USER: What questions need explicit human decisions?

CANDIDATE_CRITERIA: What should the acceptance criteria look like?

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-04-14_07-06-50
