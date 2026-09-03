---
name: write-ops-log
description: Publish a durable debugging or infrastructure incident record to Echo. Use when a multi-step investigation reaches a conclusion, a failure needs a postmortem, or a recovery pattern should be visible across Marin projects.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: Write Ops Log

Invoke `consult-echo` and search the exact error, subsystem, and incident before
writing. Edit the existing entry when the same incident is continuing. Create
one entry for a distinct incident; do not turn every retry into a new record.

Draft an Open Knowledge Format document in a temporary file:

```markdown
---
type: wiki-note
title: <Failure and operational consequence>
use_when: when <future symptom should retrieve this incident>
tags:
  - incident
  - debugging
  - <repo-or-subsystem>
  - severity-<low|medium|high|critical>
  - <resolved|unresolved>
---

## Impact

<Affected work, duration, and user-visible consequence.>

## Cause and resolution

<Diagnostic discriminator, established cause, recovery, and verification.>

## Evidence and prevention

<Canonical issue, PR, run, dashboard, and follow-up links.>
```

Include the shortest chronology needed to make the diagnosis credible. Keep
raw logs and command transcripts in their source systems.

Publish with
`uv run .agents/skills/consult-echo/scripts/echo.py wiki add --file <path>` or
revise with
`uv run .agents/skills/consult-echo/scripts/echo.py wiki edit <id> --file <path>`.
Return the canonical Echo URL and link it from the associated issue or PR. Do
not add a repository debug-log file.
