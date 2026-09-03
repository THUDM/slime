---
name: write-tests
description: Write or revise tests with an emphasis on behavior, regression coverage, pytest style, and avoiding "slop tests." Use when adding tests, fixing failing tests, reviewing test quality, or deciding what test would catch a bug.
---

<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Write Tests

Use this skill when a change needs tests or when existing tests look too coupled
to implementation details.

Read `TESTING.md` before writing or reviewing non-trivial tests. It is the shared
behavior-focused testing policy, also used by `commit`. Read `AGENTS.md` for the
repo's coding standards. If the repo has package- or module-specific testing
docs, read the nearest one for local commands, markers, fakes, mocks, and
numerical tolerances before choosing a test style.

## Workflow

1. Find existing tests for the touched behavior before creating a new file.
2. Check for repo-specific testing rules for commands, markers, fakes, mocks, and
   numerical tolerances.
3. Name the behavior that should fail if the code is wrong.
4. Write the smallest test that observes that behavior through a public API,
   structured output, persisted state, or real side effect.
5. Prefer a regression test before the fix when fixing a bug. Ensure the test
   fails before implementing the bug fix.
6. Keep test setup realistic but small. Use fixtures and parameterization to
   remove duplication.
7. Run the narrow test first, then the relevant package test command. Before a
   PR, run the repo lint entry point required by `AGENTS.md`.

## Default Commands

- Run the repo's test command over the test directories your change touches
  (e.g. `uv run pytest -m 'not slow'`).
- For package-specific commands, use the relevant package testing doc.

For PR preparation, run `infra/pre-commit.py --changed-files --fix` or
`infra/pre-commit.py --all-files --fix` as appropriate. Do not replace it with
`uv run pre-commit`.
