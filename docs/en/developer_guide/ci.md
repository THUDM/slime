# CI (Continuous Integration)

slime CI has two layers:

1. **Always-on CPU correctness tests** that run on every PR, every push to `main`, and manual `workflow_dispatch`.
2. **Label-gated GPU end-to-end tests** that validate real Megatron + SGLang training and rollout paths on self-hosted GPU runners.

This split is intentional. Most invariants should be checked quickly without waiting for the GPU fleet, while full training/rollout behavior is still covered by GPU e2e jobs.

## How It Works

The workflow is defined in `.github/workflows/pr-test.yml`, which is auto-generated from `.github/workflows/pr-test.yml.j2`.

### CPU Jobs

CPU jobs run on GitHub-hosted `ubuntu-latest` runners:

- `cpu-unittest` installs CPU PyTorch and lightweight dependencies, then runs registered unit and contract tests with `python tests/<test_file>.py`.
- `agent-adapter-test` does the same for agent adapter tests, with extra dependencies such as `openai`, `openai-agents`, and `anthropic`.

CPU jobs do not use Docker, do not acquire GPUs, and do not call `tests/ci/gpu_lock_exec.py`.

### GPU E2E Jobs

GPU jobs run on self-hosted GPU runners. Each job:

1. Starts a Docker container, usually `slimerl/slime:latest`; image validation uses `slimerl/slime-test:latest`.
2. Installs the latest `straw-queue` wheel from PyPI and slime with `pip install -e . --no-deps`.
3. Acquires the requested GPUs with `tests/ci/gpu_lock_exec.py --count <num_gpus>`.
4. Executes the registered test file with `python tests/<test_file>.py`.

GPU tests usually follow the e2e pattern: `prepare()` downloads models/datasets, and `execute()` builds CLI arguments and calls `U.execute_train(...)`.

The straw e2e tests set `--rollout-data-transport straw` explicitly, so local execution and
`run-ci-changed` use the same backend as the fixed matrix:

- R3: `test_qwen3_30B_A3B_r3.py` and `test_moonlight_16B_A3B_r3.py`.
- SC: `test_qwen2.5_0.5B_score_centering.py`, covering both top-k and top-p.
- Fully async, fanout, PPO, MTP, PD/Mooncake, distributed SGLang configuration,
  mixed-offload fault recovery, debug replay and release-train.
- Checkpoint save/load tests share the straw pool across phases and verify
  queue and training-state recovery. `test_straw_checkpoint_fork.py` covers
  step selection, repeated rollback, automatic branch selection and debug replay.

R3, SC and fully async also enable online GC. Ordinary straw tests use isolated
temporary directories that are cleaned up after execution. These single-host
GPU tests exercise the local filesystem profile; multi-host JuiceFS durability
requires separate validation. The other e2e cases retain their Ray object-store
or NIXL transport coverage. GPU containers install straw in both fixed and
changed-test jobs.

`test_qwen2.5_0.5B_pipeline_rl.py` runs three real GRPO steps with fully async
rollout on 4 GPUs. Its probes verify that the same HTTP requests keep generating
across weight updates, and that policy weights actually change. The fixed
matrix covers NCCL and full disk weight sync with `--flush-cache-interval 0`,
plus periodic refresh with NCCL and interval `2`. `test_pipeline_rl.py` checks
the cache schedule and SGLang control payloads on CPU. These are functional
checks; they do not measure PipelineRL learning quality or throughput gains.

`test_qwen2.5_0.5B_training_recovery.py` uses 4 GPUs and two Ray jobs on the
same cluster. It triggers a real CUDA allocation OOM with TP=1, verifies serving
still responds after that job exits, then restarts with TP=2 and a new DP size.
It checks serving process/router/placement identity, retained batch contents,
training scheduler progress, nonzero gradients and the final checkpoint. The
fixed matrix covers debug dumps, replay of completed but uncheckpointed Straw
batches with online GC, and Straw recovery from a model/optimizer checkpoint
with Megatron YAML overrides. All three scenarios also kill the rollout manager
and verify its new process reconnects to the original serving owner. Straw replay
and debug replay run without health checking; checkpoint recovery enables it.
`test_training_recovery.py` checks
driver fencing, replay retention/resharding, launcher behavior and checkpoint
boundaries on CPU.

### Changed-Test Job

`run-ci-changed` dynamically detects added or modified files under `tests/test_*.py` and `tests/plugin_contracts/test_*.py` relative to `origin/main`.

For each changed test file, it extracts a top-level `NUM_GPUS = <N>` constant and builds a matrix. If `NUM_GPUS` is missing, CI defaults to `8`, so CPU-only tests should declare:

```python
NUM_GPUS = 0
```

The changed-test job itself runs through the self-hosted Docker path. When `NUM_GPUS = 0`, it runs the test without acquiring GPUs.

## CI Jobs and Triggers

The CPU matrix marks straw-dependent tests with `straw: true`. Those jobs install
the latest `straw-queue` wheel from PyPI before running the test. Queue
tests do not require a Rust toolchain or access to the straw source repository.

`test_optional_straw.py` runs without installing straw in its CPU job. It checks
that the default rollout and training data path works without the package,
startup prints `pip install straw-queue`, and selecting straw fails with an
actionable installation error.

`test_straw_fully_async_recovery.py` is an automatic CPU integration test: it
SIGKILLs a job containing two local Ray nodes, then verifies recovery from the
same filesystem queue in a fresh process, with and without online GC. Its tiny
R3/SC payloads are bounded and it uses deterministic inference/reward fixtures.
To run it locally after installing the compatible straw wheel:

```bash
PYTHONPATH=. python tests/test_straw_fully_async_recovery.py
```

| Trigger | Job | Type | Description |
|---|---|---|---|
| Automatic | `cpu-unittest` | CPU | Always-on unit and contract tests for argument validation, schedules, rewards, samples, rollout validation, checkpoint utilities, and plugin contracts. |
| Automatic | `agent-adapter-test` | CPU | Always-on agent adapter tests with optional provider SDK dependencies. |
| `run-ci-sglang-config` | `e2e-test-sglang-config` | GPU | SGLang config tests for advanced rollout engine deployment and mixed/offload scenarios. |
| `run-ci-megatron` | `e2e-test-megatron` | GPU | Core Megatron training tests covering dense, MoE, PPO, MTP, OPD, fully-async rollout, PD/Mooncake, and debug replay paths. |
| `run-ci-precision` | `e2e-test-precision` | GPU | Numerical precision validation and parallel consistency checks. |
| `run-ci-ckpt` | `e2e-test-ckpt` | GPU | Checkpoint save/load correctness, including CPU/GPU optimizer states and async save. |
| `run-ci-image` | `e2e-test-image` | GPU | Runs the `run-ci-megatron` matrix on `slimerl/slime-test:latest`. |
| `run-ci-changed` | `e2e-test-changed` | Mixed | Runs only changed tests, using each file's `NUM_GPUS` value. |

`workflow_dispatch` can be used from the Actions page for manual validation. It runs the registered jobs according to the workflow conditions.

## CPU Unit Tests

The CPU suite is the first line of defense for correctness. It is designed to catch silent RL infrastructure bugs before a change reaches expensive GPU runs.

The registered CPU suite currently covers:

- Megatron argument and HF config validation;
- DP/CP scheduling utilities and CP loss invariance;
- metric reporting and distributed metric aggregation;
- reward-model grading utilities for math, GPQA, F1, DeepScaler, and DAPO-style math;
- `Sample` behavior, rollout validation, and agent trajectory merging;
- HF checkpoint saver behavior;
- customization hook contracts for rollout functions, generate functions, runtime hooks, and path loading.

Agent adapter tests are kept in a separate CPU job because they need extra SDK dependencies.

Useful local commands:

```bash
python tests/test_agent/test_trajectory_manager_branching.py
python -m pytest tests/test_megatron_argument_validation.py tests/plugin_contracts/test_plugin_generate_contracts.py
```

## GPU E2E Tests

GPU e2e tests validate the integrated training/rollout behavior that CPU tests cannot cover:

- `run-ci-sglang-config`: advanced SGLang deployment paths, including config-based engine layouts.
- `run-ci-megatron`: main Megatron backend coverage for dense/MoE recipes, fully-async rollout, OPD, PPO-style paths, PD/Mooncake, and debug rollout-then-train replay.
- `run-ci-precision`: numerical consistency across parallel settings.
- `run-ci-ckpt`: checkpoint save/load combinations and async save.
- `run-ci-image`: the same matrix as `run-ci-megatron`, but on the release/test image.

Use targeted labels for routine PRs. Use `run-ci-image` sparingly because it consumes significantly more GPU time.

## Writing a New Test

### CPU Tests

For CPU-only tests:

1. Add the test under `tests/test_*.py`, `tests/utils/test_*.py`, or `tests/plugin_contracts/test_*.py`, following nearby patterns.
2. Add a top-level `NUM_GPUS = 0` if the file may be run by `run-ci-changed`.
3. Make the file executable directly:

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

4. If the test should run permanently, register it in the `cpu-unittest` or `agent-adapter-test` job in `.github/workflows/pr-test.yml.j2`, then regenerate the workflow.

### GPU E2E Tests

For GPU e2e tests:

1. Create `tests/test_<your_test_name>.py` following the existing `prepare()` / `execute()` pattern.
2. Declare the required GPU count with `NUM_GPUS = <N>`.
3. Download required models/datasets in `prepare()`.
4. Build arguments and call `U.execute_train(...)` in `execute()`.
5. Register the test in the appropriate GPU job in `.github/workflows/pr-test.yml.j2`, then regenerate the workflow.

Example skeleton:

```python
import os
import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")

def execute():
    # Build argument strings and call U.execute_train(...)
    ...

if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
```

## Workflow Generation

The workflow file `pr-test.yml` is auto-generated from the Jinja2 template `pr-test.yml.j2`. Do not edit `pr-test.yml` directly.

To change the permanent CI matrix:

1. Edit `.github/workflows/pr-test.yml.j2`.
2. Run:

```bash
python .github/workflows/generate_github_workflows.py
```

3. Commit both `.github/workflows/pr-test.yml.j2` and the generated `.github/workflows/pr-test.yml`.

## Choosing Checks for a PR

- Pure argument parsing, reward, schedule, sample, trajectory, or hook-contract changes: rely on CPU tests first.
- SGLang topology or rollout engine deployment changes: use `run-ci-sglang-config`.
- Megatron training, loss, checkpoint conversion, or model recipe changes: use `run-ci-megatron`; add `run-ci-precision` or `run-ci-ckpt` when relevant.
- Docker image or dependency changes: use `run-ci-image`.
- New or modified tests: use `run-ci-changed` for quick targeted validation.
