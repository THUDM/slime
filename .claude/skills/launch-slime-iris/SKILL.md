---
name: launch-slime-iris
description: Validate and submit a single-node Slime GPU command through the repository's optional Iris launcher.
---

# Launch Slime on Iris

Read `infra/iris/launcher.py` and `.agents/ops/iris.md` before forming a command.

1. Confirm the checkout is clean and committed. Record the revision and immutable
   task-image digest.
2. Verify that the command is single-node and does not bootstrap workers from an SSH
   hostfile. Do not bypass the launcher's `--nodes 1` guard.
3. Select the cluster, resources, priority, timeout, and retry budgets explicitly.
   Forward secrets by name with `--secret-env`; never place secret values in argv.
4. Run the exact command with `--dry-run`. Review the redacted request and command.
5. Remove `--dry-run`, submit, and record the returned Iris job ID.
6. Verify GPU visibility, Ray initialization, rollout progress, and advancing
   training steps. A RUNNING scheduler state alone is not evidence of health.

Do not cancel a job without authority. Preserve its ID and diagnostic evidence before
an authorized stop or cleanup.
