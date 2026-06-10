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

- CI executes registered test files with `python tests/<file>.py`, not only pytest discovery. New CPU pytest files should include:

```python
import pytest

NUM_GPUS = 0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

- `run-ci-changed` extracts a top-level `NUM_GPUS = <N>` constant from added/modified `tests/test_*.py` and `tests/plugin_contracts/test_*.py`; if missing, it defaults to 8 GPUs. Set `NUM_GPUS = 0` for CPU-only tests.
- For GPU/e2e tests, follow the nearby file pattern (`prepare()`, `execute()`, `NUM_GPUS`, and any model/dataset constants).

### Step 3: Register Tests in GitHub CI

Whenever adding, moving, or renaming a test file, update the GitHub workflow template before finishing:

1. Add the test to the appropriate matrix in `.github/workflows/pr-test.yml.j2`.
   - CPU-only pytest/unit tests usually belong in `cpu-unittest` with `num_gpus: 0`.
   - GPU/e2e tests should be placed beside the nearest similar model/path test with the matching `num_gpus` and environment fields.
2. Regenerate workflows:

```bash
python .github/workflows/generate_github_workflows.py
```

3. Include both `.github/workflows/pr-test.yml.j2` and the generated `.github/workflows/pr-test.yml` in the change set.

Only skip fixed matrix registration when the test is intentionally helper-only or manually invoked; state that reason in the final response.

### Step 4: Run Local Validation

- Run the exact existing test files you changed, if any.
- For new registered tests, run the same shape CI will use, for example `python tests/test_new_file.py`.
- Run repository-wide checks only when they are already part of the task or workflow.
- Avoid documenting placeholder test commands that may not exist in the current tree.

### Step 5: Keep Workflow Template as Source of Truth

For CI workflow changes unrelated to a new, moved, or renamed test:

1. Edit `.github/workflows/pr-test.yml.j2`
2. Regenerate workflows:

```bash
python .github/workflows/generate_github_workflows.py
```

3. Include both the template and generated workflow file in the change set (`.j2` and `.yml`). If the user asked for a commit, commit both.

### Step 6: Provide Verifiable PR Notes

Include:

- Which tests were added/changed
- Where each new/renamed test was registered in `.github/workflows/pr-test.yml.j2`
- Exact commands executed
- GPU assumptions for each test path
- Why this coverage protects against regression

## Common Mistakes

- Editing generated workflow file only
- Relying on `run-ci-changed` discovery for a new test that should run in the regular PR matrix
- Forgetting `NUM_GPUS = 0` on a CPU-only changed test, causing `run-ci-changed` to default to 8 GPUs
- Adding a CPU pytest file that passes under `pytest tests/foo.py` but fails under CI's `python tests/foo.py`
- Adding tests without following existing constants/conventions
- Making tests too large or non-deterministic
- Skipping local validation and relying only on remote CI

## Reference Locations

- Pytest config: `pyproject.toml`
- Tests: `tests/`
- CI template: `.github/workflows/pr-test.yml.j2`
- CI guide: `docs/en/developer_guide/ci.md`
