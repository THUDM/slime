---
name: file-issue
description: File a GitHub issue for a bug or improvement found this session.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: File GitHub Issue

Create a GitHub issue in the current repository from bugs, regressions, or
improvements identified in the current conversation. `gh` defaults to the repo of
the current checkout; pass `--repo <owner>/<name>` only to target a different one.

## Background

Read first: `AGENTS.md`.

Before drafting, read:

- `.agents/skills/writing-style/SKILL.md`
- `.agents/skills/writing-style/issues.md`
- `.agents/skills/writing-style/ai-writing-donts.md`

## Issue Kinds and Body Structure

Pick the kind, then use the matching body structure below. There are no GitHub
issue templates — these structures live here.

| Kind | When to use | Labels |
|---|---|---|
| **bug** | A bug or regression was found | `bug`, `agent-generated` |
| **task** | An improvement, refactor, or feature request | `agent-generated` + priority if known |
| **experiment** | An experiment needs tracking | `experiment`, `agent-generated` |

### Bug body

```markdown
<what is broken and its impact -- concrete symptoms or error messages>

Reproduce:
1. <step>
2. <step>

Expected: <what should happen instead>

<optional: concise evidence or confirmed root cause>
```

### Task body

```markdown
<what needs to be done and why -- enough context for anyone on the team>

Done when:
<specific, testable completion criteria>
```

### Experiment body

```markdown
## TL;DR

<One-paragraph current summary. Leave blank only when the work is just being kicked off.>

## Description

<Context someone outside the thread can understand.>

## Hypothesis or Goal

<What are you trying to learn, fix, or achieve?>

## Status

<Current state; update as evidence lands.>

## Links

* Logbook:
* Report:
* Important updates:

## Decision Log

## Conclusion
```

## Workflow

### 1. Gather Context from Conversation

Extract from the conversation:

- **What is broken or missing** -- concrete symptoms, error messages, failing test output.
- **Where it happens** -- file paths, line numbers, module names.
- **How to reproduce** -- steps, commands, or minimal config that triggers it.
- **Root cause** (if known).
- **Severity** -- blocks work, causes data loss, or cosmetic?

If it's ambiguous what to file, ask the user before proceeding.

### 2. Classify the Issue

Pick the kind (bug, task, or experiment). If unsure, ask the user.

### 3. Duplicate Check

Search for existing issues first:

```bash
gh issue list --state open --search "<keyword>"
```

If a match exists, tell the user and offer to comment on it instead.

### 4. Draft the Issue

**Title**: At most 80 characters, optionally prefixed with a scope tag. State a
factual symptom for a bug (e.g. `[training] Gradient accumulation drops the last
microbatch`) and an imperative outcome for a task (e.g. `[training] Handle
partial accumulation steps`). Do not add `bug:`, `task:`, or another type
prefix.

**Body**: Use the section structure for the chosen kind (see above).

**Rules for the body:**

- No filler ("I noticed...", "During our conversation...").
- No markdown images or tables.
- Reference code with `file:line` links, not inline dumps.
- Keep every fact needed to understand and act on the issue. Remove history,
  repetition, and implementation narration that does not define the problem or
  completion criteria; experiment issues may retain more tracking context.
- Do not repeat the title in a `Description` section.
- Do not inventory files, functions, or proposed implementation steps that are
  not required to define the problem or completion criteria.
- Include error messages or stack traces in code blocks, trimmed to the
  relevant frames.
- For task issues: include concrete `Done when` criteria.
- For bug issues: include numbered reproduction steps.

### 5. Compress and Inspect the Payload

Apply the writing-style final compression pass to the exact title and body that
will be sent to GitHub. For a bug or task, verify the title is at most 80
characters. Every remaining sentence must add a symptom, impact, reproduction
step, observation, expected behavior, or completion criterion.

This review is required even when the user explicitly asked to file the issue.
It is an author self-check, not a request for approval.

### 6. Confirm or File Directly

If the user explicitly asked to file an issue, skip the preview — file it and
share the link. If the agent surfaced the issue (not explicitly requested),
show the drafted title and body and wait for approval or edits.

### 7. File the Issue

Write the body to a uniquely named temp file, then pass it with `--body-file`.
Do not inline the body with shell substitution (`--body "$(cat <<'EOF' ...)"`)
— multiline text can be corrupted by pasted output or escaping mistakes. Do not
reuse a fixed path like `/tmp/issue-body.md`; concurrent agent runs can
overwrite each other's drafts on shared hosts.

```bash
body_file="$(mktemp "${TMPDIR:-/tmp}/issue-body.XXXXXX.md")"
trap 'rm -f "$body_file"' EXIT

cat > "$body_file" <<'EOF'
<body>
EOF

issue_url="$(gh issue create \
  --title "<title>" \
  --label "agent-generated" \
  --body-file "$body_file")"
```

Add kind-appropriate labels (`bug`, `experiment`). If a relevant label does not
exist, skip it rather than creating new labels. For task issues, add a priority
label (`p1`, `p2`, `p3`) if the user specifies one or severity is clear.

Before creating the issue, re-open the body file and verify it contains no
unrelated shell output (pre-commit logs, pytest session headers, prompt
transcripts). If it does, clean the draft before posting.

After creating the issue, fetch its published text with
`gh issue view "$issue_url" --json title,body` and correct any text added or
altered by the publishing tool.

### 8. Report Back

Print the issue URL.

## Writing Style

Terse: every sentence conveys new information; no preamble or editorializing; no
restating code a link covers; annotate code links, don't narrate them.

## Rules

0. Never credit yourself in the issue.
1. Always add the `agent-generated` label.
2. Confirm with the user before filing only when the agent surfaced the issue
   (not when the user explicitly asked to file).
3. If the conversation does not contain a clear bug or actionable improvement,
   say so and ask the user what they want to file.
4. Use the smallest matching body structure. Omit optional context and headings
   that add no information.
