# Slime testing

Read the [shared Marin testing policy](.agents/marin-style/TESTING-core.md)
before writing or reviewing tests. Tests should validate observable behavior and
avoid pinning private helpers, incidental logs, or implementation details.

## Test execution

Slime CI executes registered tests as Python programs. A new CPU-only pytest file
must include:

```python
import pytest

NUM_GPUS = 0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

Register new, moved, or renamed tests in `.github/workflows/pr-test.yml.j2`, then
regenerate `.github/workflows/pr-test.yml`:

```bash
python .github/workflows/generate_github_workflows.py
```

Run the exact changed test programs locally. GPU and distributed tests require
the topology declared by their `NUM_GPUS` value and should only run on an
appropriate idle allocation.

Agent and sandbox changes usually belong in `tests/test_agent/`. Reuse its fakes
at remote I/O boundaries while asserting through public behavior.
