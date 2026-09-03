---
name: debug
description: Debug code bugs and operational faults while recording distilled hypotheses, evidence, and outcomes in Marin's shared Echo work log.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: Debug

Invoke `consult-echo` before forming a new diagnosis when prior decisions,
incidents, or exact errors could help. Use `task-logbook` to append distilled
milestones to Echo during any multi-step investigation. Do not create a
repository debug-log file unless the user explicitly asks for one.

For infrastructure or operational faults, first read any operations runbook the
repo provides and follow its matching section. Its guardrails take precedence.

Work one hypothesis at a time:

1. Record the initial symptom and evidence in the Echo work log.
2. State one falsifiable hypothesis and the smallest check that distinguishes it.
3. Run the check, then append the result and its evidence URL or command output.
4. Repeat until the cause is established or the investigation is blocked.
5. Add a regression test for a code fix when one can catch the failure.

At resolution, invoke `write-ops-log` for an infrastructure incident or a
durable multi-step diagnosis. The final Echo wiki entry records the reusable
cause, recovery, and evidence; the work log remains the chronological record.
