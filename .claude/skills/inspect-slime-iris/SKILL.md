---
name: inspect-slime-iris
description: Diagnose the scheduling and runtime health of a Slime GPU job submitted to Iris.
---

# Inspect a Slime Iris job

Read `.agents/ops/iris.md`, then resolve the exact Iris job ID. Inspect without
mutating the job first.

1. Record the logical state, per-task state, pending reason, attempt count, worker,
   and task-image digest.
2. If pending, distinguish quota/capacity, constraints, image pull, and controller or
   worker failures before suggesting changes.
3. If running, verify GPU discovery, Ray startup, SGLang engines, rollout output,
   advancing training steps, and checkpoints when due.
4. If failed, preserve the first causal error and the task attempt that emitted it;
   later shutdown errors are usually consequences.
5. Report evidence, likely cause, and the smallest next diagnostic. Do not cancel,
   retry, or resubmit without authority.

Treat RUNNING with no advancing work as unhealthy, not as successful admission.
