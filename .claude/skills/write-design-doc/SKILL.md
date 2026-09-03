---
name: write-design-doc
description: Publish or revise a concise cross-project design proposal in Echo. Use for an explicit design task or a design-level change whose context, decision, interfaces, tradeoffs, and rollout need review without adding a repository design document.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: Write Design Doc

Invoke `consult-echo` and inspect the current checkout before drafting. Edit the
closest existing design when the same decision is continuing; add a new entry
only for a distinct design.

Write a compact Open Knowledge Format document in a temporary file:

```markdown
---
type: wiki-note
title: <Concrete decision or proposal>
use_when: when <future task should load this design>
tags:
  - design
  - <repo-or-subsystem>
---

## Context

<Constraint, observed problem, and evidence links.>

## Decision

<Interfaces, invariants, ownership, and important alternatives.>

## Validation and rollout

<How the design will be tested, introduced, observed, and reversed if needed.>
```

Keep it near one page. Include code only when it resolves an interface or data
model ambiguity. State material risks and unresolved choices plainly.

Publish the design:

```bash
uv run .agents/skills/consult-echo/scripts/echo.py wiki add --file /tmp/<design>.md
uv run .agents/skills/consult-echo/scripts/echo.py wiki edit <id> --file /tmp/<design>.md
```

Return the canonical Echo URL and link it from the coordinating issue or PR.
Do not add a parallel file under `.agents/projects/` or `docs/` unless the user
explicitly requests a repository-owned specification.
