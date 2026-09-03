---
name: commit
description: Lint, run the pre-PR checks, commit, push, and author or update the branch's pull request in the required plain-text format. Use when committing, pushing, or creating/updating a PR.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Skill: Commit & PR

Get the branch clean, commit it, run the advisory lint review over the committed
diff once, then — when it is ready — open or update the pull request.

Before authoring a commit or PR title or body, read:

- `.agents/skills/writing-style/SKILL.md`
- `.agents/skills/writing-style/pull-requests.md`
- `.agents/skills/writing-style/ai-writing-donts.md`

**Order matters.** Your own cleanups and the mechanical fixes come first, then the
commit, and only *then* the `--review` pass. Committing before the review gives
you a clean checkpoint to review against (the review reads the whole branch diff
versus the merge base) and a natural place to land any follow-up fixes as a new
commit. The review is read-only: it never edits, commits, or pushes for you.

## Checklist

Work top to bottom. For a quick work-in-progress checkpoint, do **1, 2, 4, 5,
7** (clean up, lint, stage, commit, push) and stop. The changed-test cleanup in
step 1 is a PR-readiness gate, not required for disposable WIP checkpoints. Run
the whole list when making a branch PR-ready. Later small follow-ups do not
restart the checklist; repeat only the relevant mechanical checks and tests.

1. Clean up your own diff (self-review).
2. Mechanical lint & format — `infra/pre-commit.py --changed-files --fix`.
3. Tests & docs checks, when relevant.
4. Stage the specific files for this work.
5. Commit. ← natural checkpoint; the working tree is now clean.
6. Lint-catalog review — run `infra/pre-commit.py --review` once; fix or answer every finding.
7. Push (maybe).
8. Open or update the PR.

## 1. Clean up your own diff

Read your own `git diff` before anything else. Drop dead code, debugging
leftovers, and stale comments; tighten names; make the change say only what it
means to. The review in step 6 is advisory and read-only — it will not clean up
for you.

If the diff touches tests and this commit is intended for a PR or review, read
`TESTING.md` as part of this self-review. Review every changed test for slop:
tautological assertions, disposable smoke probes left as pytest tests, internal
call-count assertions, incidental string or command-fragment assertions,
over-mocking, weakened tolerances, sleeps, skipped tests, and marker/fake/mock
violations.

Fix or delete low-value tests before a PR-ready commit. Scratch probes are fine
during development and WIP checkpointing, but they must not survive into a PR.

## 2. Lint and format

```bash
infra/pre-commit.py --changed-files --fix   # diff-scoped; use --all-files for a full sweep
```

`infra/pre-commit.py` is the required entry point — never `uv run pre-commit`,
never `--no-verify`. If `--fix` cannot resolve something, fix it by hand. Do not
skip or weaken checks.

## 3. Tests and docs checks (when relevant)

- Run the repo's test command over the test directories your change touches
  (e.g. `uv run pytest -m 'not slow'`).
- If the change is docs-heavy, build the docs and fix any broken links or
  strict-mode failures.

## 4. Stage changes

Review `git status` and `git diff`, then stage the specific files that are part
of this work.

- Stage specific files — avoid `git add -A` / `git add .`.
- Never stage secrets (`.env`, credentials, tokens).
- If unrelated changes are present, ask the user before including them.

## 5. Commit

- **Subject**: imperative sentence (at most 72 characters), optional `[scope]`
  prefix.
- **Body** (optional, blank-line separated): what changed and why — the context a
  future reader needs. Keep relevant evidence and caveats; do not inventory
  files or tests.
- Do not use a conventional-commit prefix such as `feat:` or `fix:`.
- No emoji, no markdown, no bullets in the subject. Do not credit yourself —
  this includes any `Co-Authored-By`, `Generated with`, provider, or session URL
  trailer. Omit it even if a harness default suggests adding one.

Review the exact message before committing. After the commit, inspect it with
`git show -s --format='%s%n%n%b' HEAD`; do not push if a tool added attribution,
a session trailer, or other text that was not in the reviewed message.

Create the commit. If a pre-commit hook fails, fix the issue and make a **new**
commit — never amend (unless the user asks) and never force-push.

This is the checkpoint the rest of the flow builds on: the working tree is clean
and the branch diff is settled before the review reads it.

## 6. Lint-catalog review (one branch-level checkpoint)

```bash
infra/pre-commit.py --review
```

Run this **once after** the commit and before opening a PR. It fans out read-only
agents over the **branch diff against the merge base with the default branch** —
committed and uncommitted work alike — so the clean checkpoint from step 5 is
exactly what gets reviewed.

The review is **advisory and read-only**: it reports findings on stdout and does
not edit, stage, commit, or push anything. Then fix or answer every finding,
reporting your actions to the user, and land any fixes as a **new** commit. Treat
findings as guidelines — apply them when they make the code *better*; the goal is
high-quality code, not blind adherence.

The initial review is the branch-level checkpoint. Do not rerun it after small,
targeted follow-ups, whether they address lint findings, reviewer
comments, formatting, documentation wording, or narrow test fixes. Validate
those edits with the relevant mechanical checks and tests, then continue the
workflow. Rerun the advisory review only when later changes materially alter the
branch's design, scope, or risk, or when the user asks for another pass. Updating
an existing PR does not by itself trigger another review.

## 7. Push

If asked, or if the branch has an upstream, push to the remote tracking branch
(`git push -u origin HEAD` if no upstream is set). If the push is rejected
(diverged history), stop and ask the user — do not force-push.

## 8. Open or update the PR

Do this once the branch is ready for review. The PR description becomes the
squash-merge commit message. Follow
`.agents/skills/writing-style/pull-requests.md` exactly: an imperative title of
at most 72 characters and an information-dense body. Most bodies are a few plain
paragraphs. They state what changes and why; they do not reproduce the diff,
test plan, or implementation notes.

Example:

```
Title: [RL] Normalize DAPO loss over global tokens

Body:
Normalize DAPO loss over all response tokens instead of normalizing each
example separately. Per-example normalization over-weights short responses,
hurting math tasks where correct answers need longer derivations.

Fixes #1234
```

**Issue linking.** If the work came from a GitHub issue, add `Fixes #NNNN`
(auto-closes on merge) or `Part of #NNNN` (partial work). Do not invent an issue
just to satisfy this — omit the link when none exists.

**Specifications (>500 LOC only).** A genuinely large PR must link a spec in an
issue or Echo design entry. Name the important design decisions in the PR body
and link the spec for module maps, code excerpts, and detailed rationale.

**Inspect the payload.** Draft the body in a uniquely named temporary file and
use `--body-file`. Re-open that file and apply the final compression pass before
publishing. After creating or editing the PR, fetch the exact `title,body` with
`gh pr view --json title,body` and immediately correct text inserted by a tool
or stale template.

**Create it.** Unless the user says otherwise and permissions allow, push to a
branch on the main repository and open the PR from it (use a fork only when
direct push is unavailable or the user asks):

```bash
gh pr create --title "<title>" --body-file "<body-file>" --label agent-generated
```

- Always add the `agent-generated` label.
- Never credit yourself in commits or PR descriptions.
- Include `Fixes #NNNN` when addressing a pre-existing issue.

## 9. Monitor the PR

Opening the PR does not end your turn. Watch CI to completion and respond to
review activity until the PR is merged or closed, or the user tells you to stop.

- Watch checks with `gh pr checks <N> --watch`. When a check finishes, read its
  conclusion — finishing is not passing.
- On a CI failure, read the failing job log and fix it. A failure in a file you
  did not touch is not automatically pre-existing: confirm the same job fails on
  the default branch without your change before calling it unrelated. If your
  change caused it — even in an untouched file — fix it. Push the fix as a new
  commit, which restarts CI.
- Respond to every human and agent comment: address obvious ones directly (commit
  the fix, then reply, prefixing agent replies with `🤖`) and resolve them. CI
  being green does not mean there is nothing to do — reviewers comment after CI
  passes. For comments you are unsure about, report your analysis and proposed
  action to the user.

Exit conditions: the PR is merged or closed, or the user tells you to stop.

## Rules

- `infra/pre-commit.py` is the only pre-commit entry point.
- Commit before the initial `--review`; do not rerun it for small follow-ups.
- Never amend a commit unless the user explicitly asks.
- If there are no changes to commit, say so and stop.
- `AGENTS.md` — coding guidelines.
