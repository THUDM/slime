---
name: task-logbook
description: Maintain a cross-project task or research log in Echo. Use for multi-step investigations, experiments, long-running implementation work, or any task whose milestones must survive the current agent session without adding a repository logbook file.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: Task Logbook

Use Echo's append-only work log instead of `.agents/logbooks/`, `.agents/ops/`,
or another repository progress file. Pick one stable project slug such as
`harbor:issue-123` or `vllm:grugmoe-refresh` and reuse it for the whole thread.

Append one entry after each milestone that changes the next action:

```bash
uv run .agents/skills/consult-echo/scripts/echo.py work-log add \
  --project "<repo>:<issue-or-task>" \
  --title "<one-line result>" \
  --body "<evidence, interpretation, and next action>"
```

Record decisions, falsified hypotheses, measured results, launches, failures,
and handoff state. Link the issue, PR, run, dashboard, or immutable artifact that
supports the entry. Omit routine commands and narrative that will not affect a
future reader.

The work log is chronological and append-only. Correct a mistaken entry with a
new entry. Keep coordinating issues concise and link the Echo conversation page
when the detailed history matters.

At completion, publish reusable conclusions through `write-ops-log`,
`write-design-doc`, repository docs, or an existing Echo synthesis. Do not copy
the whole work log into that durable summary.
