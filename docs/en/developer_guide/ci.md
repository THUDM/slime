# CI (Continuous Integration)

slime runs CPU tests when a PR is opened or updated, when code is pushed to `main`, and when the workflow is triggered manually. GPU end-to-end tests are triggered by PR labels and run real Megatron training and SGLang rollout on self-hosted machines. For routine changes, start with CPU tests and select GPU tests that cover the affected behavior.

## Triggering Tests

| Trigger | CI job | Environment | Coverage |
|---|---|---|---|
| Automatic | `cpu-unittest` | CPU | Argument validation, batch scheduling, metrics, rewards, samples, checkpoint utilities, and extension interfaces. |
| Automatic | `agent-test` | CPU | Agent adapters, with the required model-provider SDKs installed. |
| `run-ci-sglang-config` | `e2e-test-sglang-config` | CPU/GPU | SGLang deployment configuration, including multiple models, engine layouts, and fault recovery with memory offload. |
| `run-ci-megatron` | `e2e-test-megatron` | GPU | Megatron training, including dense models, MoE, PPO, MTP, OPD, fully async rollout, PD/Mooncake, debug replay, and Codex/sunabako agent training. |
| `run-ci-precision` | `e2e-test-precision` | CPU/GPU | Numerical precision and consistency across parallel configurations. |
| `run-ci-ckpt` | `e2e-test-ckpt` | GPU | Checkpoint saving and loading, including CPU/GPU optimizer states and async saves. |
| `run-ci-image` | `e2e-test-image` | GPU | The same tests as `run-ci-megatron`, using the `slimerl/slime-test:latest` image. |
| `run-ci-changed` | `e2e-test-changed` | CPU/GPU | Only added or modified tests, with GPU counts taken from each file's `NUM_GPUS`. |

You can also run the workflow manually through `workflow_dispatch` on the GitHub Actions page. Manual runs execute the registered CPU and GPU jobs. Pushes to `main` automatically run only the CPU jobs.

The workflow is defined in `.github/workflows/pr-test.yml`, generated from `.github/workflows/pr-test.yml.j2`. To change test lists or trigger conditions, edit the template and regenerate the workflow.

## How Tests Run

### CPU Tests

CPU jobs run on GitHub-hosted `ubuntu-latest` runners. They install CPU PyTorch and test dependencies, then execute `python tests/<test_file>.py`. They do not use Docker or acquire GPUs.

`cpu-unittest` checks:

- Megatron arguments and Hugging Face model configuration;
- DP/CP batch scheduling and loss invariance under CP partitioning;
- metric reporting and distributed aggregation;
- reward calculation for math, GPQA, F1, DeepScaler, and DAPO;
- `Sample` behavior, rollout validation, and agent trajectory merging;
- Hugging Face checkpoint saving and interface contracts for custom rollout, generation functions, and runtime hooks.

Agent adapter tests run in a separate `agent-test` job because they also require SDKs such as `openai`, `openai-agents`, and `anthropic`.

CPU test entries marked with `straw: true` install the latest `straw-queue` wheel from PyPI. They do not need a Rust toolchain or the straw source repository. `test_optional_straw.py` deliberately runs without straw to check that default data transport still works and that explicitly selecting straw produces a clear installation hint.

Useful local commands:

```bash
python tests/test_agent/test_trajectory_manager_branching.py
python -m pytest tests/test_megatron_argument_validation.py tests/plugin_contracts/test_plugin_generate_contracts.py
```

### GPU End-to-End Tests

GPU jobs run on self-hosted machines. Each test follows these steps:

1. Start a Docker container, usually `slimerl/slime:latest`; image validation uses `slimerl/slime-test:latest`.
2. Install the latest `straw-queue` wheel from PyPI and the current slime checkout with `pip install -e . --no-deps`.
3. Acquire the required GPUs with `tests/ci/gpu_lock_exec.py --count <num_gpus>`.
4. Execute `python tests/<test_file>.py`.

Test files usually use `prepare()` to download models and datasets, then `execute()` to build training arguments and call `U.execute_train(...)`.

### Running Only Changed Tests

With the `run-ci-changed` label, CI finds added or modified `tests/test_*.py` and `tests/plugin_contracts/test_*.py` files relative to `origin/main` and creates a test job for each file.

The GPU count comes from the file's top-level `NUM_GPUS = <N>`. Without this declaration, CI requests 8 GPUs. CPU-only tests should therefore declare:

```python
NUM_GPUS = 0
```

These jobs still run in Docker on self-hosted machines, but do not acquire GPUs when `NUM_GPUS = 0`.

## Data Transport and Recovery Tests

### Coding-agent training

`tests/test_agent_sunabako_codex_e2e.py` uses Codex **0.162.1** and a single small
MiMo task, `format-code-task-000003` (implementing smol-evm's missing `SHL` opcode).
Training, SGLang, and both sandbox roles run on one host with eight H100 80GB GPUs.
It covers reading an existing repository, editing source, executing commands,
multi-turn Responses API translation, sampled-token/logprob capture, grading in
a fresh sandbox, and one real optimizer step. The unchanged repository must
fail because `SHL` is missing; the agent must earn reward 1 by passing the official
bitwise tests, including six new left-shift cases for ordinary values, zero shifts,
overflow and shifts at least 256 bits wide. Each training rank must report finite nonzero gradients and changed
model parameters. Reward normalization is disabled for this one-sample smoke
test so a successful rollout produces a learning signal. Sampling uses temperature 1,
`top_p=0.95`, disabled top-k and score centering. Both nucleus IDs/offsets and the
original sampler probabilities are checked against the actual training tensors.
R3 is disabled because Qwen3.8-27B is dense; CPU tests cover replay metadata
through agent forks.

This is one task, one sample and one update, with a 180-second agent budget and
a 600-second training-job deadline. Its Megatron CI matrix entry has a 15-minute limit
including setup. It records and validates the complete agent episode, then
trains on its first and last real segments, unchanged, to bound repeated-context
training cost. `agent-full.pt` retains every segment; the full example continues
to train on the entire trajectory. No solution or hidden tests are supplied to
the agent.
Model, dataset and image downloads, plus TileLang/Triton
compilation, are cached. Populate the shared cache before the first timed CI run:
cold downloads and compilation can exceed the 15-minute job limit.
It does not run a SWE benchmark or the full example dataset.
The `run-ci-megatron` label includes this GPU test; automatic `agent-test` CPU checks
continue to cover edge cases. `run-ci-changed` also discovers this top-level test.

Measured on 2026-10-10 with eight H100 80GB GPUs, cached assets/compilation and
a local sandbox cluster in RSS test mode: **457 seconds total**, including
**68 seconds for the agent and independent grading**. The grader passed 10/10
tests and all eight training ranks changed parameters. All 2,271 sampled tokens
across 13 turns passed the token/replay audit; the bounded update trained the
first and last segments (318 tokens). `sc_correction` was -0.00255 and mean
absolute train/rollout logprob difference was 0.00989.

The shared `tests/ci/setup_agent_e2e.sh` upgrades sunabako to the latest PyPI wheel
with `pip install --upgrade --no-deps --only-binary=sunabako sunabako` inside the test
container, then installs the example requirements. This applies to
`run-ci-megatron`, `run-ci-image`, and `run-ci-changed`; publishing a new sunabako
wheel is enough for CI to use it without rebuilding the slime image. It imports the OCI
image with skopeo/umoci and uses the native runtime in the existing privileged
CI container, without starting a Docker daemon. The explicitly enabled RSS test
mode is only a bounded functional check, **not aggregate hard RAM enforcement**.
Production sunabako still requires a writable delegated cgroup and fails closed.

For a preconfigured cluster, install the same requirements and run:

```bash
HF_CHECKPOINT=/path/to/Qwen3.8-27B \
SUNABAKO_CLUSTER=/path/to/cluster.json \
SUNABAKO_IMAGES=/path/to/images.json \
ADAPTER_PUBLIC_HOST=<training-node-ip> \
python tests/test_agent_sunabako_codex_e2e.py
```

The image map must include the selected task on every sandbox node. Set
`SUNABAKO_ALLOW_TEST_MEMORY=1` explicitly only when testing without hard cgroups.
Optional `SLIME_AGENT_TEST_DATA` reuses the downloaded MiMo parquet/mapping;
`SLIME_AGENT_CODEX_NATIVE_TARBALL` reuses the official platform archive.
`SLIME_AGENT_TEST_RUN_DIR` must name a new directory and retains the CLI logs,
grader output, rollout/train tensors, optimizer evidence and `result.json`.
The test owns its Ray head and does not stop unrelated clusters. GitHub Actions
uploads the evidence on success and failure.

### straw

straw end-to-end tests explicitly set `--rollout-data-transport straw`, so local runs, `run-ci-changed`, and the fixed test list use the same transport. Coverage includes:

- R3: `test_qwen3_30B_A3B_r3.py` and `test_moonlight_16B_A3B_r3.py`.
- SC: `test_qwen2.5_0.5B_score_centering.py`, checking both top-k and top-p data.
- Fully async rollout, fanout, PPO, MTP, PD/Mooncake, distributed SGLang, fault recovery with mixed memory offload, debug replay, and continued rollout after releasing training resources.
- Checkpoint saving and loading: phases share a straw storage pool to check queue and training-state recovery. `test_straw_checkpoint_fork.py` also checks step selection, repeated rollback, automatic branch selection, and debug replay.

R3, SC, and fully async tests also enable online GC. Ordinary straw tests use isolated temporary directories and clean them up afterward. Single-host GPU tests use the local filesystem profile; multi-host JuiceFS durability needs separate validation. Other end-to-end tests continue to cover Ray object-store and NIXL transport.

`test_straw_fully_async_recovery.py` is an automatic CPU integration test. It uses SIGKILL to terminate a job with two local Ray nodes, then starts a new process to recover from the same filesystem queue. It tests both enabled and disabled online GC, using small R3/SC payloads and deterministic inference and reward fixtures. After installing a compatible straw wheel, run it locally with:

```bash
PYTHONPATH=. python tests/test_straw_fully_async_recovery.py
```

### PipelineRL

`test_qwen2.5_0.5B_pipeline_rl.py` runs fully async rollout and three actual GRPO training steps on 4 GPUs. It checks whether the same HTTP request continues generating across weight updates and whether training changes the policy weights.

The fixed test list covers three configurations: `--flush-cache-interval 0` with NCCL or full disk weight synchronization, and `--flush-cache-interval 2` with NCCL for periodic cache flushing. The CPU test `test_pipeline_rl.py` checks the flush schedule and SGLang control requests. These tests check functionality, not learning quality or throughput gains.

### Manual Megatron Restart

`test_qwen2.5_0.5B_training_recovery.py` uses 4 GPUs and two successive training jobs on the same Ray cluster. The first uses TP=1 and deliberately triggers a real CUDA OOM. After verifying that serving still responds when the job exits, it resubmits training with TP=2, which also changes the DP size.

The test checks that healthy SGLang processes, routers, and GPU placements are reused; replayed batch contents match; and training scheduler progress, finite nonzero gradients, and the final checkpoint are correct. The fixed test list includes:

| Data storage | RolloutManager state | Checks |
|---|---|---|
| straw with online GC | Remains alive | Reconnect trainers and replay completed training batches that were not checkpointed. |
| straw with online GC | Killed after failure | Reconnect a new manager to the original serving cluster and replay the same batches. |
| straw with a model/optimizer checkpoint and Megatron YAML configuration | Killed during training | Restore from the checkpoint and check configuration and recovery state. |
| Rollout debug files | Killed after failure | Restore data from debug files and reconnect a new manager to the original serving cluster. |
| straw with disk-delta weight synchronization | Killed after failure | Publish restored weights as a new full baseline, then continue delta updates. |
| straw with PD/Mooncake serving | Killed after failure | Wedge the prefill actor, replace it within the reset timeout, and retain the healthy decode actor. |

`test_qwen3_30B_A3B_training_recovery.py` uses 8 GPUs for the same OOM/checkpoint/manager-loss workflow with a MoE model, R3, and stateless Adam. It omits optimizer tensors while checking scheduler progress, compares persisted routing bytes across the TP/DP change, and completes training after recovery. The dense cases cover ordinary Adam with optimizer checkpoints.

Internal serving health checks are enabled with or without the compatibility flag `--use-fault-tolerance`. CPU coverage includes configuration and checkpoint boundaries in `test_training_recovery.py`, lost disk-update replies in `test_disk_delta_recovery.py`, and real Ray manager SIGKILL or conversion-reply loss in `test_rollout_manager_recovery.py`.

### Removing Failed Engines at Rollout Completion

`test_qwen2.5_0.5B_rollout_health.py` stops a real SGLang HTTP server just before rollout completes, leaving its router registration intact. Two four-GPU cases either retain or kill the corresponding Ray actor. They check bounded rollout completion, deregistration before training, engine recovery at weight update, and a final training checkpoint.

Both cases omit `--use-fault-tolerance` and set the background interval and initial wait to 600 seconds. This verifies that rollout-completion checks run immediately without waiting for background checks.

## Adding Tests

### Adding CPU Tests

Follow nearby examples and place tests under `tests/test_*.py`, `tests/utils/test_*.py`, or `tests/plugin_contracts/test_*.py`. Declare a top-level `NUM_GPUS = 0` if the file will be run by `run-ci-changed`.

CI executes test files directly, so pytest files need an entry point:

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

To run a test automatically, register it in the `cpu-unittest` or `agent-test` list in `.github/workflows/pr-test.yml.j2`, then regenerate the workflow.

### Adding GPU End-to-End Tests

1. Create `tests/test_<your_test_name>.py` using the existing `prepare()` / `execute()` structure.
2. Declare the required GPU count with a top-level `NUM_GPUS = <N>`.
3. Download models and datasets in `prepare()`.
4. Build arguments and call `U.execute_train(...)` in `execute()`.
5. Register the test in the appropriate GPU job in `.github/workflows/pr-test.yml.j2`, then regenerate the workflow.

Example:

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

## Generating the Workflow

Do not edit the generated `.github/workflows/pr-test.yml` directly. After editing `.github/workflows/pr-test.yml.j2`, run:

```bash
python .github/workflows/generate_github_workflows.py
```

Commit both the template and the generated workflow file.

## Choosing Checks for a PR

- Argument parsing, rewards, batch scheduling, samples, trajectories, or extension interfaces: start with the corresponding CPU tests.
- SGLang deployment or engine layouts: add `run-ci-sglang-config`.
- Megatron training, loss, checkpoint conversion, or model training configurations: add `run-ci-megatron`. For numerical or checkpoint changes, also add `run-ci-precision` or `run-ci-ckpt` as needed.
- Docker images or dependencies: add `run-ci-image`. It runs the full Megatron test list and uses more GPU time.
- Added or modified tests: add `run-ci-changed` to validate the changed test files directly.
