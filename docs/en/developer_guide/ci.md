# CI (Continuous Integration)

slime uses GitHub Actions for CI. Tests are triggered by **PR labels** — adding a specific label to a PR will run the corresponding test suite.

## How It Works

The workflow is defined in `.github/workflows/pr-test.yml` (auto-generated from `pr-test.yml.j2`). Each CI job:

1. Runs on a self-hosted GPU runner inside a Docker container (`slimerl/slime:latest`).
2. Installs slime with `pip install -e . --no-deps`.
3. Acquires the required GPUs via `tests/ci/gpu_lock_exec.py --count <num_gpus>`.
4. Executes the test file: `python tests/<test_file>.py`.

Each test file follows a standard pattern: a `prepare()` function downloads models/datasets, and an `execute()` function builds CLI arguments and calls `U.execute_train(...)`.

## CI Labels

Add a label to your PR to trigger the corresponding test suite:

| Label | Job | Description |
|---|---|---|
| `run-ci-short` | `e2e-test-short` | Lightweight smoke tests with Qwen2.5-0.5B (4 GPUs). Fast feedback loop. |
| `run-ci-fsdp` | `e2e-test-fsdp` | FSDP backend tests (true on-policy, VL, megatron-fsdp alignment). |
| `run-ci-megatron` | `e2e-test-megatron` | Core Megatron training tests covering dense, MoE, PPO, MTP, OPD, etc. |
| `run-ci-precision` | `e2e-test-precision` | Numerical precision validation (parallel check, megatron-fsdp alignment). |
| `run-ci-ckpt` | `e2e-test-ckpt` | Checkpoint save/load correctness (sync and async-save). |
| `run-ci-image` | `e2e-test-image` | Full test suite run on `slimerl/slime-test:latest` image (for image validation). |
| `run-ci-changed` | `e2e-test-changed` | **Dynamically** detects new/modified test files in the PR and runs only those. |

All labels also run when triggered via `workflow_dispatch` (manual run from the Actions tab).

## Key Labels Explained

### `run-ci-changed` — Run Only New or Modified Tests

This is the most useful label for development. When you add a new test file or modify an existing one, just add `run-ci-changed` to your PR and CI will:

1. **Detect** which `tests/test_*.py` files are added or modified relative to `origin/main` (via `git diff --diff-filter=AM`).
2. **Extract** the `NUM_GPUS` value from each detected test file automatically.
3. **Build** a dynamic GitHub Actions matrix and run each test in parallel.

This means you don't need to manually register your new test in the workflow — just make sure your test file has a top-level `NUM_GPUS = <N>` constant and `run-ci-changed` will pick it up.

**Example**: If your PR adds `tests/test_qwen3_8B_opd_sglang.py` with `NUM_GPUS = 8`, adding the `run-ci-changed` label will automatically run that test on 8 GPUs.

### `run-ci-image` — Full Suite on Test Image

This runs **all** registered tests on the `slimerl/slime-test:latest` Docker image. Use this label to:

- Validate a newly built Docker image before release.
- Run the entire test suite for a comprehensive pre-merge check.

Since this includes every test, it consumes significant GPU time — use it sparingly and prefer more targeted labels for routine development.

### `run-ci-megatron` — Core Megatron Tests

This is the primary label for validating Megatron-backend changes. It covers:

- Dense models: GLM4-9B, Qwen3-4B (PPO)
- MoE models: Qwen3-30B-A3B (with/without DeepEP + FP8), Moonlight-16B-A3B
- Specialized: MiMo-7B MTP, Qwen2.5-0.5B debug rollout-then-train, OPD with sglang teacher

All tests use 8 GPUs. If you are modifying Megatron training logic, loss computation, or checkpoint conversion, this is the label to use.

## Writing a New Test

1. Create `tests/test_<your_test_name>.py` following the standard pattern:

```python
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4  # This constant is used by run-ci-changed

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    # Download datasets as needed ...

def execute():
    # Build argument strings and call U.execute_train(...)
    ...

if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
```

2. **For quick validation**: Just push your test file and add `run-ci-changed` to the PR. It will be auto-detected.

3. **To register in a permanent label group**: Edit `.github/workflows/pr-test.yml.j2`, add an entry to the desired job's `tests` list, then regenerate:

```bash
cd .github/workflows && python generate_github_workflows.py
```

Remember to commit both the `.j2` and the generated `.yml` file.

## Workflow Generation

The workflow file `pr-test.yml` is auto-generated from the Jinja2 template `pr-test.yml.j2`. **Do not edit `pr-test.yml` directly.** To make changes:

1. Edit `.github/workflows/pr-test.yml.j2`.
2. Run `python .github/workflows/generate_github_workflows.py`.
3. Commit both files.
