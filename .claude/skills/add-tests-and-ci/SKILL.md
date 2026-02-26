---
name: add-tests-and-ci
description: Guide for adding or updating slime tests and CI wiring. Use when tasks require new test cases, CI registration, test matrix updates, or workflow template changes.
---

# Add Tests and CI

Add reliable tests and integrate them with slime CI flow.

## When to Use

Use this skill when:

- User asks to add tests for new behavior
- User asks to fix or update existing tests in `tests/`
- User asks to update CI workflow behavior
- User asks how to run targeted checks before PR

## Step-by-Step Guide

### Step 1: Pick the Right Test Pattern

- Follow existing naming: `tests/test_<feature>.py`
- Start from nearest existing test file for your model/path
- Keep test scope small and behavior-focused

### Step 2: Keep CI Compatibility

When creating CI-discoverable tests, ensure top-level constants and conventions match repository patterns (including `NUM_GPUS = <N>` where expected).

### Step 3: Run Local Validation

```bash
pytest tests/test_*.py -k "<new_feature_keyword>" -vv
pytest -m "not skipduringci"
pre-commit run --all-files --show-diff-on-failure --color=always
```

### Step 4: Update Workflow Template Correctly

For CI workflow changes:

1. Edit `.github/workflows/pr-test.yml.j2`
2. Regenerate workflows:

```bash
python .github/workflows/generate_github_workflows.py
```

3. Commit both template and generated workflow files

### Step 5: Provide Verifiable PR Notes

Include:

- Which tests were added/changed
- Exact commands executed
- GPU assumptions for each test path
- Why this coverage protects against regression

## Common Mistakes

- Editing generated workflow file only
- Adding tests without following existing constants/conventions
- Making tests too large or non-deterministic
- Skipping local validation and relying only on remote CI

## Reference Locations

- Pytest config: `pyproject.toml`
- Tests: `tests/`
- CI template: `.github/workflows/pr-test.yml.j2`
- CI guide: `docs/en/developer_guide/ci.md`
