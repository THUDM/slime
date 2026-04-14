# Optimize Delta Compression Weight Updates for Large MoE (GLM4.7-355B)

## Goal Description

Optimize the delta compression weight update path in slime so that, for the GLM4.7-355B-A32B non-colocate configuration, steady-state delta steps significantly improve over the current ~120s/step toward the non-delta baseline of ~50s/step.

The delta compression feature sends only weight differences (deltas) between RL training steps instead of full weights. These deltas are elementwise sparse. The core insight is that sparsity should be exploited to reduce both wire transfer size and encode/decode overhead. Currently, the implementation works well at small MoE scale (~7s, comparable to baseline) but degrades at large MoE scale due to overhead that grows faster than the bandwidth savings.

The approach must be first-principles: profile comprehensively to identify actual bottlenecks, then apply targeted optimizations that address root causes rather than workarounds. Start with exact methods; lossy techniques (thresholding, quantization) may be explored as a follow-up only if exact methods prove insufficient.

Starting point: commit `a6334ac6` on the `delta-compression-feature` branch.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Phase-level profiling on both sender (slime/Megatron) and receiver (SGLang) sides, with chunk-level efficiency metrics.
  - Positive Tests (expected to PASS):
    - Profiling output shows independent timing for each phase: TP/EP gather, HF conversion, baseline H2D transfer, delta compute, sparsity scan, materialization, NCCL broadcast, SGLang receive/decode, load_weights application
    - Chunk-level metrics present: chunk count, tensors per chunk, zero-nnz tensor count, encoded bytes vs dense bytes, lock acquire/broadcast/apply time
    - Sparsity statistics per tensor: nnz count, density ratio, tensor size
  - Negative Tests (expected to FAIL):
    - Profiling that only shows aggregate update_weights_time without per-phase breakdown
    - Missing either sender-side or receiver-side timings
  - AC-1.1: Profiling code is temporary (added for analysis, removed after optimization is complete)
    - Positive: Final optimized commit has no profiling instrumentation in main code paths
    - Negative: Profiling print/logging statements left in production code

- AC-2: Steady-state delta steps show significant improvement toward the 50s baseline on GLM4.7-355B-A32B non-colocate config.
  - AC-2.1: The measurement window is steps 2-3 in a 3-step experiment (num_rollout=3). Step 1 is the initial full-sync that establishes the CPU-pinned baseline and is excluded from the target.
    - Positive Tests (expected to PASS):
      - A 3-step run where steps 2-3 average significantly better than the current ~120s/step
      - Meaningful improvement demonstrated (directional goal toward 50s)
    - Negative Tests (expected to FAIL):
      - Steps 2-3 averaging >= 100s (no meaningful improvement from current state)
      - Performance regression from 120s baseline
  - AC-2.2: The baseline reference is `glm47_355b_a32b_noncolocate.py` (no delta compression) running on the same Modal setup, which averages ~50s/step.

- AC-3: First update_weights call (full-sync, baseline establishment) may exceed the target without failing.
  - Positive Tests (expected to PASS):
    - Step 1 completes successfully regardless of duration
    - CPU-pinned baseline memory is allocated and populated
  - Negative Tests (expected to FAIL):
    - Step 1 OOMs or crashes during baseline establishment

- AC-4: Correctness preserved after optimization.
  - AC-4.1: After a delta update, weights on inference engines match expected values. Verified using `check_weights(compare)` which checks rollout engine weights against train-side source weights. Run after at least one delta step.
    - Positive: `check_weights(compare)` passes after delta step
    - Negative: Weight mismatch detected by check_weights
  - AC-4.2: After a forced full-sync round, `check_weights(compare)` passes and weights exactly match the train-side source weights. This verifies no error accumulation from delta steps.
    - Positive: Full-sync followed by check_weights passes
    - Negative: Accumulated drift detected after full-sync
  - AC-4.3: Loss values during training remain within reasonable range compared to non-delta baseline (secondary evidence, not primary gate).

- AC-5: Memory usage within reasonable bounds.
  - AC-5.1: No new OOM conditions introduced on either train or rollout side.
  - AC-5.2: Memory measurement protocol: `torch.cuda.reset_peak_memory_stats()` before measured update window, `torch.cuda.max_memory_allocated()` after. Compare per-process on train side and max-over-rollout-engines against the same procedure on non-delta baseline. Reasonable overhead acceptable (user confirmed: "within reason, but GPU-resident baseline cache will definitely OOM").
    - Positive: Experiment completes without OOM; memory overhead is sensible
    - Negative: OOM on any process, or gratuitous memory waste

- AC-6: Each experiment recorded with complete metadata.
  - Positive Tests (expected to PASS):
    - Log file committed to repo containing: config name, commit hash, transport method, per-phase timings, sparsity stats, total update_weights_time per step, outcome summary
    - Git commit message documents what was tried and the result
  - Negative Tests (expected to FAIL):
    - Experiment results not committed or missing key fields
    - No way to reconstruct what was tried from the git history

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

Multiple targeted optimizations implemented across both sender (slime) and receiver (SGLang patch), informed by comprehensive profiling data. All optimizations address root causes identified through profiling. The sparse_indices transport mode is fully optimized for large MoE. Both slime code and sglang patch updated. Experiment logs committed with full metadata. Steady-state delta steps approach or beat the 50s baseline.

### Lower Bound (Minimum Acceptable Scope)

At least one coordinated sender+receiver optimization that produces meaningful speedup from the current ~120s/step. Profiling data collected and analyzed to identify the primary bottleneck. The optimization addresses a real root cause (not a workaround). At minimum the sparse_indices transport is improved. Experiment results recorded and committed.

### Allowed Choices

- Can use: CPU-pinned memory for caching (following existing ref_model pattern in slime), CUDA streams for overlap, fused operations, batched reconstruction, adaptive per-chunk transport selection, kernel-level optimizations for scatter/pack/unpack
- Can use: Temporary profiling code during analysis (must be removed in final version)
- Cannot use: GPU-resident baseline cache (will OOM at 355B scale)
- Cannot use: Lossy compression techniques in initial optimization (may explore as follow-up only if exact methods insufficient)
- Should not: Change launcher configs (already runnable and stable), introduce unnecessary abstraction layers, add permanent instrumentation/logging to main code paths

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The optimization should follow this strategy:

1. **Profile to find the real bottleneck.** The current ~120s/step at 355B scale (vs ~7s at 30B scale) suggests overhead that scales super-linearly. Likely candidates:
   - Sender: Multiple full passes over tensor data (delta compute -> count_nonzero for bucket sizing -> nonzero for materialization). At 355B with ~170GB weights, each extra pass costs significant time.
   - Sender: CPU<->GPU baseline transfer with `torch.cuda.synchronize()` barriers.
   - Receiver: Per-tensor sequential reconstruction (zero scratch -> scatter values -> load_weights per tensor). The non-delta path batches `load_weights`, but the sparse delta path calls it once per tensor.
   - Wire: May actually be smaller with sparse transport, but gains eaten by encode/decode overhead.

2. **Optimize the dominant bottleneck first.** Possible directions (to be confirmed by profiling):
   - Merge estimation and materialization into a single pass: currently `estimate_delta_transport_byte_size()` does `count_nonzero()` for every tensor, then materialization does `nonzero()` again. These could be combined.
   - Skip all-zero tensors: currently every tensor goes through the pipeline even if its delta is entirely zero.
   - Batch reconstruction on SGLang side: instead of per-tensor `load_weights()`, reconstruct multiple tensors and apply in batch.
   - Direct scatter-add: instead of zeroing a scratch buffer, scattering values, then applying additively, scatter-add directly into the model parameters.
   - Reduce synchronization barriers: the CPU<->GPU copies use `torch.cuda.synchronize()`. Non-blocking copies with proper stream management could help.

3. **Validate correctness and measure improvement.**

### Relevant References

- `slime/backends/megatron_utils/update_weight/delta_weight_update.py` - Core delta compression: DeltaCompressionTracker, transport materialization, sparsity estimation
- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` - Distributed weight update orchestration, bucket management, delta integration
- `slime/backends/megatron_utils/update_weight/common.py` - PendingHFUpdateBucket, HFUpdate dataclasses
- `docker/patch/latest/sglang_delta_compression.patch` - SGLang-side receive, decode, and apply logic for all three transport modes
- `slime/backends/sglang_utils/sglang_engine.py` - Engine-side update_weights_from_distributed API, check_weights
- `slime/utils/arguments.py` - CLI arguments for delta compression (lines 144-179)
- `/home/ec2-user/multinode-training-guide/slime/configs/glm47_355b_a32b_noncolocate_delta_compression*.py` - Experiment configs
- `/home/ec2-user/multinode-training-guide/slime/modal_train.py` - Modal training launcher
- `working_delta_compression_indices.log` - Existing performance log for sparse_indices at 355B scale (~120s/step)

## Dependencies and Sequence

### Milestones

1. **Profiling & Instrumentation**: Add temporary profiling to identify actual bottlenecks
   - Phase A: Add sender-side phase-level timing to delta_weight_update.py and update_weight_from_distributed.py (baseline H2D, delta compute, sparsity scan, materialization, NCCL broadcast time)
   - Phase B: Add receiver-side phase-level timing to sglang_delta_compression.patch (receive, decode per-tensor, load_weights per-tensor, total apply)
   - Phase C: Add sparsity statistics collection (per-tensor nnz, density ratio, tensor element count)
   - Phase D: Run profiling experiment on current `glm47_355b_a32b_noncolocate_delta_compression_indices` config at 355B scale
   - Phase E: Collect and commit profiling results

2. **Bottleneck Analysis**: Analyze profiling data to determine optimization strategy
   - Step 1: Identify the top time-consuming phases on both sender and receiver
   - Step 2: Determine actual sparsity distribution at 355B scale (may differ from 30B observations)
   - Step 3: Calculate theoretical wire savings vs actual overhead
   - Step 4: Decide optimization strategy: if sparsity is high enough for sparse transport to win, optimize the encode/decode overhead; if not, pivot strategy
   - Step 5: Document analysis findings

3. **Targeted Optimization**: Implement changes based on profiling findings
   - Implement the optimization(s) identified in Milestone 2
   - May require coordinated changes in both slime code and sglang patch
   - Focus on sparse_indices transport first (most data available)
   - Test each change independently where possible

4. **Validation**: Verify improvement and correctness
   - Run optimized config, measure update_weights_time for steps 2-3
   - Run check_weights to verify correctness
   - Commit experiment log with full results
   - If target not met with exact methods, document findings and propose lossy follow-up

Milestone 1 must complete before Milestone 2. Milestone 2 must complete before Milestone 3. Milestone 3 must complete before Milestone 4.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add sender-side phase-level timing to delta_weight_update.py and update_weight_from_distributed.py | AC-1 | coding | - |
| task2 | Add receiver-side phase-level timing to sglang_delta_compression.patch | AC-1 | coding | - |
| task3 | Add sparsity statistics collection (per-tensor nnz, density, size) | AC-1 | coding | - |
| task4 | Run profiling experiment on delta_compression_indices config at 355B | AC-1, AC-6 | coding | task1, task2, task3 |
| task5 | Analyze profiling data: identify top bottlenecks, sparsity distribution, theoretical vs actual savings | AC-1 | analyze | task4 |
| task6 | Decide optimization strategy based on analysis findings | AC-2 | analyze | task5 |
| task7 | Implement sender-side optimization(s) in slime code | AC-2, AC-4, AC-5 | coding | task6 |
| task8 | Implement receiver-side optimization(s) in sglang patch | AC-2, AC-4, AC-5 | coding | task6 |
| task9 | Run optimized experiment, measure improvement | AC-2, AC-3, AC-6 | coding | task7, task8 |
| task10 | Verify correctness via check_weights after delta and full-sync steps | AC-4 | coding | task9 |
| task11 | Remove temporary profiling code from final version | AC-1.1 | coding | task10 |
| task12 | Commit all experiment logs and document findings | AC-6 | coding | task10 |

## Claude-Codex Deliberation

### Agreements

- Profiling before optimization is the correct approach. Aggregate update_weights_time alone is insufficient; phase-level breakdown is needed on both sender and receiver sides.
- The candidate bottlenecks are plausible: sender-side multiple passes over tensor data (count_nonzero + nonzero), CPU<->GPU synchronize barriers, and receiver-side per-tensor sequential reconstruction with per-tensor load_weights calls.
- Allowing step 1 to be slower is reasonable since it establishes the CPU-pinned baseline.
- The first-principles framing is correct: understand why it's slow, then fix root causes.
- Recording experiment metadata (config, commit, transport, timings, outcome) is essential for tracking progress.
- The bucket mechanism is central: the existing fast non-delta path is bucket-centric while the sparse delta path is tensor-centric, and this architectural mismatch is likely part of the problem.

### Resolved Disagreements

- **AC-4 correctness check (Round 1)**: Codex argued loss-based correctness is too weak. Claude agreed and switched to direct `check_weights(compare)` as primary gate, loss as secondary. Resolution: use check_weights mechanism that already exists in the codebase.
- **AC-2 measurement window (Round 1)**: Codex flagged ambiguity in "step 2+". Claude agreed and clarified: step 1 is the initial full-sync (baseline establishment), steps 2-3 are the measured delta steps.
- **File boundary (Round 1)**: Codex argued three-file restriction was too tight. Claude agreed and relaxed to allow temporary config/flag changes and small helper modifications for profiling.
- **Lower bound scope (Round 1)**: Codex argued single optimization too narrow. Claude agreed and expanded to coordinated sender+receiver changes.
- **AC-5 measurement (Round 2)**: Codex requested concrete protocol. Claude agreed: reset_peak_memory_stats before, max_memory_allocated after.
- **AC-4.2 wording (Round 2)**: Codex clarified "match non-delta baseline" should mean "match train-side source weights via check_weights" not a separate control run. Claude agreed.

### Convergence Status

- Final Status: `converged`
- Rounds: 2
- Round 1: 5 required changes, all addressed
- Round 2: 3 wording refinements, all addressed. No DISAGREE items remaining.

## Pending User Decisions

- DEC-1: Exactness of compression
  - Claude Position: Start with exact methods
  - Codex Position: Should define correctness tolerance explicitly
  - Tradeoff Summary: Exact methods may not achieve <50s; lossy methods trade precision for speed
  - Decision Status: `Start exact, explore lossy as follow-up if needed`

- DEC-2: Performance target interpretation
  - Claude Position: <50s as hard target
  - Codex Position: Need precise measurement contract
  - Tradeoff Summary: Hard target risks declaring good work as failure; directional goal acknowledges uncertainty
  - Decision Status: `Directional goal - significant improvement from 120s toward 50s`

- DEC-3: Profiling code disposition
  - Claude Position: Keep as flag-gated tooling
  - Codex Position: Either option reasonable, depends on user preference
  - Tradeoff Summary: Flag-gated is useful for future work; temporary keeps codebase minimal
  - Decision Status: `Temporary - remove after analysis`

- DEC-4: Transport optimization focus
  - Claude Position: Profile all three for informed choice
  - Codex Position: Either approach viable; indices has most data
  - Tradeoff Summary: Profiling all is thorough but costs 3x experiment time; focusing is faster
  - Decision Status: `Focus on sparse_indices first, pivot to others only if needed`

- DEC-5: Memory budget
  - Claude Position: Reasonable overhead acceptable
  - Codex Position: Need explicit limits
  - Tradeoff Summary: GPU memory is tight at 355B; CPU-pinned memory is the safe approach
  - Decision Status: `Within reason. Use CPU-pinned memory (like existing ref_model pattern). GPU-resident baseline will OOM.`

- DEC-6: Experiment logging
  - Claude Position: Commit experiment logs
  - Codex Position: Need concrete format/location
  - Tradeoff Summary: Logs + commit messages provides best traceability
  - Decision Status: `Both - commit log files AND use descriptive commit messages`

- DEC-7: Validation runs
  - Claude Position: One confirmation rerun for variance
  - Codex Position: At least one confirmation rerun recommended
  - Tradeoff Summary: Confirmation reduces false positives but costs ~1h per rerun
  - Decision Status: `One run sufficient`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead
- Value clarity over cleverness; simple control flow and explicit behavior
- No unnecessary abstraction or helper functions/layers that obscure the real flow
- Changes should be minimal but structurally sound
- Code should feel native to the existing slime codebase
- Instrumentation/logging should be cheap, minimal, and justified (and temporary per DEC-3)
- First-principles solutions over incremental workarounds

### Experiment Execution
- Run experiments via: `EXPERIMENT_CONFIG=glm47_355b_a32b_noncolocate_delta_compression_indices modal run -d /home/ec2-user/multinode-training-guide/slime/modal_train.py::train`
- Stop stuck apps via: `modal app stop ap-X`
- Each experiment runs 3 steps (num_rollout=3); step 1 is full-sync, steps 2-3 are delta
- Experiments take ~1h each at 355B scale
- May need to update both local slime code and the sglang patch in `docker/patch/latest/sglang_delta_compression.patch`

### Key Technical Context
- Full weights for GLM4.7-355B are ~170GB
- Weight deltas are elementwise sparse (almost all tensors have some non-zeros, but most elements per tensor are zero)
- Three transport modes exist: dense (full delta), sparse_indices (int32 indices + values), sparse_bitmask (packed bitmask + values)
- Baseline is stored on CPU-pinned memory; delta computed as `current_weight - baseline` then cast to bf16
- SGLang receives encoded tensors via NCCL broadcast, reconstructs per-tensor, applies additively via monkey-patched copy_/fill_

--- Original Design Draft Start ---

now we have a slime launcher at `/home/ec2-user/multinode-training-guide/slime`, the ultimate goal is to optimize the non colocate weight update time.  right now if you run `/home/ec2-user/multinode-training-guide/slime/configs/glm47_355b_a32b_noncolocate.py` the avg update_weights_time is around 50s per step. and your job is to optimize `/home/ec2-user/multinode-training-guide/slime/configs/glm47_355b_a32b_noncolocate_delta_compression*.py` any variant of delta compression to make it faster than baseline. it is fine to have first step to be slower since the first step you need to pin cpu memory. but you need to optimize to make following steps to better than baseline(which is 50s). and you need to approach this from first principle way rather than work around. you can see for all delta compression 3 configs we all use local slime, which is `/home/ec2-user/nan_wonderland/slime` and we will be patching sglang patch `/root/slime/docker/patch/latest/sglang_delta_compression.patch` to docker. you can check the docker image in `/home/ec2-user/multinode-training-guide/slime/modal_train.py` which is `slimerl/slime:nightly-dev-20260329a`. so basically we patch sglang in `/sgl-workspace/sglang/` in the docker. so right now current local slime has a working(runnable version) and the performance of it in small moe(qwen3-30b-a3b) is comparable to baseline(both around 7 second). but now i want to optimize for big moe, thats why you are given this task. to run any experiment you can run `EXPERIMENT_CONFIG=glm47_355b_a32b_noncolocate_delta_compression* modal run -d /home/ec2-user/multinode-training-guide/slime/modal_train.py::train`. feel free to come up with your own fix or etc. for current we have a log for delta_compression_indices. it is in `/home/ec2-user/nan_wonderland/slime/working_delta_compression_indices.log` you can take a reference, just to let you know current performance is 120s per step on avg. right now i hardcoded to make each experiment to run at most 3 steps(`num_rollout=3`) since you just need 3 steps to tell how good the result is. your end goal is to optimize the delta compression method we have. the tldr is that each rl update is sparse and we want to just send the diff(delta). in this case we should utilize the property of delta(sparse). feel free to add some print/debug in your process. and you need to get yourself familiar with using modal `https://modal.com/llms.txt`. and you should do `modal app stop ap-X` to stop app if it does not stop when you want to stop. you might need to update both local slime and sglang patch to make your stuff works. make sure you record what you try and how it goes. each experiment takes maybe ~1h to finish(since we are doing very large moe so). we will start from commit `a6334ac6` in slime repo. and you need to commit/record information so i can know you progress, what you have tried. make sure you understand the stuff from first principle. this is not an engineering problem. this is more about how to use the property cleverly. ideally you dont need to change launcher since the config is relative stable since they are already runnable.

some notes
1. i want to be clear there. the weight diff in each step is sparse elementwise, which means entire tensor is sparse. but almost all tensors in weights has at least some values. so basically they are all sparse but very rare will have some tensor being all 0. i got this conclusion when i did small moe(qwen3-30b-a3b) so conclusion might change when we have big moe. you should do comprehensive profiling to know what is the thing you should optimize and propose solution.
2. you should start with profiling to understand things clearly and then propose solution. basically you should start with adding some profiling code and sure there are enough information for you to know what to improve
3. you need to really understand current delta_compression three transport mode. i optimize them a lot. they were way worse before in small moe. and now they got comparable results but in large moe it gets worse. i was expecting to be better since you can compress so the numebr of weight update call got decreased. i think you need to profile both sides. maybe profiling some exciting methods....? maybe you can profile indices/bitmask at the same time? you are not required to propose something completely new. optimizing existing methods or introduce new thing for them are also fine. but you need to view things from first principle.

My general coding taste / style:

  - I value clarity over cleverness.
  - I want code to be easy to read, easy to trace, and easy to reason about.
  - I prefer simple control flow and explicit behavior.
  - I do not like unnecessary abstraction.
  - I do not like helper functions, layers, or files that exist only to "organize" code while making the real flow harder to see.
  - I dislike duplication, but I also dislike abstraction that hides important differences in behavior.
  - I want names to be precise and reflect real responsibility.
  - I care a lot about clean boundaries and coherent design.
  - If something feels like a workaround, patch, or hack, I usually want to step back and rethink the design.
  - I prefer first-principles solutions over incremental ugliness.
  - I want code to feel native to the existing codebase, not like a separate style was pasted in.
  - I want changes to be minimal but structurally sound.
  - I care about maintainability more than short-term cleverness.
  - I prefer correctness first, then optimization.
  - I want optimizations to be real and semantically honest, not accidental or misleading.
  - I do not want instrumentation, logging, or debugging code to pollute the main implementation.
  - If logging exists, it should be cheap, minimal, and justified.
  - I prefer practical solutions that are easy to operate and easy to debug.
  - In general, I want code that is clean, direct, unsurprising, and robust.
--- Original Design Draft End ---
