---
name: cleanup-slime-iris
description: Safely cancel or clean up a Slime Iris job after resolving its exact identity and preserving diagnostics.
---

# Clean up a Slime Iris job

Read `.agents/ops/iris.md`. Cleanup changes external state and requires explicit
authority for the exact job.

1. Resolve the full Iris job ID and show its current state. Never target a name prefix,
   user namespace, or all jobs.
2. Preserve the revision, task image, terminal or pending reason, attempt history, and
   relevant logs before cancellation.
3. If the job is already terminal, do not issue a cancellation. Report whether any
   durable output or checkpoints remain.
4. For an authorized active job, cancel the exact ID with Iris, then verify it reaches
   a terminal state.
5. Report what was stopped, what remains durable, and whether resubmission needs a new
   job name.

Do not delete object-store artifacts, checkpoints, or registry images as part of job
cleanup unless they are separately and explicitly in scope.
