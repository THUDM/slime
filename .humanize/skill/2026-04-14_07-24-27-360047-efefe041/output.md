AGREE:
- AC-4 is materially improved. Making direct weight verification primary and loss secondary fixes the biggest correctness gap.
- AC-2/AC-3 now cleanly separate the initial baseline-establishing full sync from the steady-state delta target.
- AC-1/AC-6 now ask for the right chunk-level efficiency data, including encoded vs dense bytes and lock/broadcast/apply timing.
- The relaxed file boundary is reasonable and matches how this codebase is actually wired.
- AC-5 now uses a defensible GPU-memory metric on both sides.
- The lower bound is now realistic: coordinated sender+receiver optimization is the correct minimum viable scope here.

DISAGREE:
- None of the Round 1 required changes remain unreasonable. The revisions are mostly aligned.

REQUIRED_CHANGES:
- Tighten AC-4’s validation procedure. `check_weights(compare)` exists end-to-end in [train.py](/home/ec2-user/nan_wonderland/slime/train.py:27), [slime/ray/rollout.py](/home/ec2-user/nan_wonderland/slime/slime/ray/rollout.py:565), and [slime/backends/sglang_utils/sglang_engine.py](/home/ec2-user/nan_wonderland/slime/slime/backends/sglang_utils/sglang_engine.py:370), but your plan should say exactly when it runs during validation: at minimum after one delta step and after one forced full-sync step.
- Reword AC-4.2. As written, “match non-delta baseline” suggests a separate baseline run, while the existing mechanism checks rollout weights against the train-side source weights, not against another run. Better wording: after a forced full-sync round, `check_weights(compare)` passes and weights exactly match the source/train weights.
- Tighten AC-5’s measurement protocol. Define the reset window and aggregation rule: use `torch.cuda.reset_peak_memory_stats()` immediately before the measured update window, sample `torch.cuda.max_memory_allocated()` immediately after, and compare max-per-process on train plus max-over-rollout-engines against the same baseline procedure. Without that, the 10% threshold is not reproducible.

OPTIONAL_IMPROVEMENTS:
- Make AC-6 specify a concrete log sink/schema, even if it is just structured log lines with fixed field names.
- State explicitly that profiling is flag-gated and off by default outside analysis runs.
- Add a fallback note in Milestone 2 that low effective sparsity can justify pivoting away from sparse transport without reopening scope.

UNRESOLVED:
- Whether AC-4.2 should mean “compare against train/source weights after forced full sync” or “compare against a separate non-delta control run.” The first matches the current code path; the second is heavier and should be an explicit decision.
