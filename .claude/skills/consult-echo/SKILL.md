---
name: consult-echo
description: Search and cite Marin's shared Echo wiki, repository index, issues, pull requests, and optional Discord history. Use when prior cross-project decisions, incidents, workflows, exact errors, or reusable knowledge could inform a task, and before adding or revising an Echo wiki entry.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: Consult Echo

Search Echo before rediscovering prior work:

```bash
uv run .agents/skills/consult-echo/scripts/echo.py search \
  "how do I diagnose a stalled TPU collective" --limit 10
uv run .agents/skills/consult-echo/scripts/echo.py get <domain:id>
```

Wiki, Marin repository files, pull requests, and issues are searched by default.
Add `--domain discord` only when discussion history is relevant. Echo's file
index follows `marin-community/marin` main; use `rg` for the current checkout,
including fork-only and uncommitted files.

Open the complete result and cite its canonical URL when it affects the work.
Run another search before writing so a near-duplicate can be edited instead.

Choose the durable home narrowly:

- Append distilled milestones for an active task to Echo's work log with
  `task-logbook`.
- Publish a specific incident or durable debugging outcome with `write-ops-log`.
- Publish or revise a cross-project design with `write-design-doc`.
- Update repository docs when guidance belongs to that product or subsystem.
- Edit an existing Echo wiki synthesis when several projects need the same
  decision, guardrail, or diagnostic pattern.

Keep raw logs, diffs, test output, and run artifacts in their source systems.
Echo entries summarize the decision or result and link that evidence.
