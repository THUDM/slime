Analysis below is against commit `a6334ac6`.

**CORE_RISKS**
- The biggest risk is assuming wire size is the main bottleneck. The current delta path still copies the baseline CPU->GPU before diffing and GPU->CPU after commit, with hard `torch.cuda.synchronize()` barriers in [delta_weight_update.py](/home/ec2-user/nan_wonderland/slime/slime/backends/megatron_utils/update_weight/delta_weight_update.py:104) and [delta_weight_update.py](/home/ec2-user/nan_wonderland/slime/slime/backends/megatron_utils/update_weight/delta_weight_update.py:135). At 355B scale, that host-device traffic can erase any network savings.
- The sparse receive path is sparse on the wire, but dense in compute. It zeroes a full scratch buffer and reconstructs each tensor sequentially before applying it in [sglang_delta_compression.patch](/home/ec2-user/nan_wonderland/slime/docker/patch/latest/sglang_delta_compression.patch:154) and [sglang_delta_compression.patch](/home/ec2-user/nan_wonderland/slime/docker/patch/latest/sglang_delta_compression.patch:186).
- `sparse_indices` is only wire-efficient if density is very low: below 33% for fp16/bf16, below 50% for fp32. `sparse_bitmask` is wire-efficient up to about 93.75% density for fp16/bf16 and 96.875% for fp32, but still pays near-dense decode cost. If 355B deltas are denser than expected, static transport choice will lose.
- Sender work scales badly: delta compute, `count_nonzero()` for bucket sizing, then `nonzero()` or bitmask packing is multiple full passes over the same data in [delta_weight_update.py](/home/ec2-user/nan_wonderland/slime/slime/backends/megatron_utils/update_weight/delta_weight_update.py:144), [delta_weight_update.py](/home/ec2-user/nan_wonderland/slime/slime/backends/megatron_utils/update_weight/delta_weight_update.py:219), and [delta_weight_update.py](/home/ec2-user/nan_wonderland/slime/slime/backends/megatron_utils/update_weight/delta_weight_update.py:268).
- The sparse SGLang path applies `self.model.load_weights([(name, decoded_weight)])` one tensor at a time, while the dense delta path batches once. That makes repeated loader/name-resolution overhead a likely major cost [sglang_delta_compression.patch](/home/ec2-user/nan_wonderland/slime/docker/patch/latest/sglang_delta_compression.patch:167).
- Monkey-patching `torch.Tensor.copy_` and `fill_` globally is a correctness risk and can hide expensive internal behavior [sglang_delta_compression.patch](/home/ec2-user/nan_wonderland/slime/docker/patch/latest/sglang_delta_compression.patch:303).
- All-zero tensors are not pruned before send. The code appends every delta tensor unconditionally, so unchanged tensors still incur metadata, decode, and apply overhead [delta_weight_update.py](/home/ec2-user/nan_wonderland/slime/slime/backends/megatron_utils/update_weight/delta_weight_update.py:148).

**MISSING_REQUIREMENTS**
- Define “subsequent steps” precisely: step 2 onward, or all delta steps except periodic full-sync steps.
- Fix the benchmark contract: exact GLM4.7-355B config, engine count, TP/PP/EP, interconnect, buffer size, transport, dtype.
- State whether the method must be exact or whether lossy compression is allowed.
- Set memory limits for pinned CPU baseline, receiver scratch, and any new GPU caches.
- Define correctness tolerance across repeated delta steps and after a forced full sync.
- Define experiment logging format and where it should live in-repo.
- Call out edge cases: mixed dtypes, frozen-parameter training, huge tensors, quantized models.

**TECHNICAL_GAPS**
- There is no real phase-level profiling. You need timings for: TP/EP gather, HF conversion, baseline H2D, delta compute, sparsity scan, materialization, NCCL broadcast, SGLang receive, sparse decode, and `load_weights`.
- Current stats are misleading: `delta_sent_bytes` logs dense delta bytes, not actual transported bytes.
- Bucket sizing itself is expensive because `estimate_delta_transport_byte_size()` does a full `count_nonzero()` scan before materialization.
- `sparse_indices` adds an int32->int64 conversion on the receiver before `index_copy_`, which is another full pass.
- `sparse_bitmask` allocates and unpacks a boolean mask per tensor.
- The distributed delta path is tensor-centric, while the existing fast additive path elsewhere is bucket-centric (`flattened_bucket_delta`). That architectural mismatch is likely central.

**ALTERNATIVE_DIRECTIONS**
- Best first candidate: add a distributed flattened-bucket delta path, so send/apply stays bucketed and `load_weights` stays batched.
- Make transport adaptive per bucket: dense for high-density, bitmask for moderately sparse, indices only for very sparse.
- If sparse transport stays, decode directly into parameter views or a fused scatter-add kernel instead of reconstructing dense scratch tensors.
- Remove the extra sender scan by merging size estimation with materialization, or sampling density first.
- Revisit delta source entirely: if optimizer updates can be exposed directly, that may remove the CPU baseline round-trip.
- If only a subset of params trains, skip known-unchanged params before diffing.

**QUESTIONS_FOR_USER**
- Must the optimization remain exact, or is approximate compression acceptable?
- What exact run defines the 50s baseline?
- What hardware/interconnect is used on training and rollout sides?
- Can we trade memory for speed?
- Do periodic full-sync steps count against the target, or only steady-state delta steps?
- Do you want experiment logs committed into the repo, and where?
- Can the wire format change freely, or must it stay compatible with an existing SGLang deployment?

**CANDIDATE_CRITERIA**
- Steady-state GLM4.7-355B delta steps beat the 50s baseline after the initial baseline-building step.
- The speedup holds across a full delta interval, not just a single step.
- Every run includes per-phase timings plus real transported-byte stats.
- Correctness matches the agreed reference within a stated tolerance after repeated delta steps and a forced resync.
- Extra pinned CPU and GPU memory stay within agreed limits, with no new OOMs.
- Each experiment records config, commit, density stats, transport bytes, timings, and result summary.
