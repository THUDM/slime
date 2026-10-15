# Coding-agent RL with Qwen3.8-27B

This example trains Qwen3.8-27B on the full code split of
[Xiaomi MiMo's RL dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss).
Codex and Claude Code use the served model to edit repositories in Sunabako
sandboxes. A fresh sandbox applies each resulting patch and runs the official
MiMo tests to obtain its reward.

[`run_qwen38_27b_fully_async.sh`](run_qwen38_27b_fully_async.sh) is the training
entry point. Its default allocation is three training nodes, three inference
nodes and two sandbox nodes. Each GPU node has eight GPUs; the 128k configuration
has been exercised on H100 80GB GPUs. Training and rollout proceed asynchronously.

| Setting | Default |
| --- | --- |
| Training batch | 8 prompts × 8 trajectories = 64 |
| Optimizer updates | 200 |
| Concurrent agent trajectories | 168 |
| Training parallelism | TP=4, PP=3, CP=2; pipeline layers 22/22/20 |
| Inference | 12 engines, TP=2 |
| Mamba / attention cache memory budget ratio | 0.5 |
| Context / per-turn output | 131,072 / 8,192 tokens |
| Compaction | At 70% context, at most twice per trajectory for either CLI |
| Agent / grader / whole-rollout timeout | 3,600 / 600 / 7,200 seconds |

## Environment

Use the same Slime, Megatron-LM and SGLang environment on all six GPU nodes.
Megatron must provide native GDN with tensor/context parallelism. SGLang must
support hybrid-attention HiCache, MTP speculative decoding and score-centering
metadata for top-p sampling. Mount the checkout, model, data and run directories
at the same absolute paths on these nodes.

Install the example dependencies on every GPU node:

```bash
python -m pip install -r examples/coding_agent_rl/requirements-sunabako.txt
```

The example requires `sunabako>=0.1.2`. Provision the two sandbox
nodes using [Sunabako's setup instructions](https://github.com/zhuzilin/sunabako),
including its patched PRoot binary, `skopeo` and `umoci`. These nodes do not join
Ray. Configure SSH access from every GPU node and allow sandbox connections to
the GPU nodes' adapter HTTP ports.

Set a node memory admission cap below available RAM, `memory_overcommit=1`, and
per-sandbox memory/PID limits. The launcher requests 8 GiB and 1,024 processes
per sandbox; image-map entries can override these with `memory_mb` and
`pids_max`. Use delegated cgroups for memory enforcement. The optional
`SUNABAKO_ALLOW_TEST_MEMORY=1` RSS mode is for environments without writable
cgroups and also requires Sunabako's explicit test setting.

Create a cluster file with the allocated sandbox nodes:

```json
{
  "nodes": [
    {
      "name": "sandbox-0",
      "host": "<sandbox-node-0>",
      "port": 22,
      "state_dir": "/var/lib/sunabako",
      "binary": "/usr/local/bin/sunabako"
    },
    {
      "name": "sandbox-1",
      "host": "<sandbox-node-1>",
      "port": 22,
      "state_dir": "/var/lib/sunabako",
      "binary": "/usr/local/bin/sunabako"
    }
  ]
}
```

Configure Sunabako's `image_cache_dir` and state directory on local storage with
reflink support. Bound disk use with `image_cache_max_bytes` and
`image_cache_min_free_bytes`; bound preparation work with
`image_cache_import_concurrency` and `rootfs_clone_concurrency`. Compressed task
images stay on shared storage, and Sunabako extracts and caches them locally.
The image set needs substantially more space than the dataset files.

Enable SSH connection reuse on the GPU nodes and size the sandbox SSH daemon's
`MaxSessions` for concurrent RPCs. Use the actual host names from the cluster
file in `~/.ssh/config`:

```text
Host sandbox-node-0 sandbox-node-1
    ControlMaster auto
    ControlPersist 600
    ControlPath ~/.ssh/control/%C
```

Create `~/.ssh/control` with mode 700 before establishing the connections.

## Prepare data and images

Run these commands from the Slime checkout after installing the example
dependencies. The download host needs `hf` and `skopeo`; `DATA_ROOT` must be
mounted at the same absolute path on the GPU and sandbox nodes.

First download the complete code split and its official image-name mapping.
This pinned dataset revision contains 2,698 code tasks:

```bash
export DATA_ROOT=/shared/mimo
export MIMO_REVISION=639865fd3374018d6cb29b9fb82dd531406fcf5f
hf download XiaomiMiMo/MiMo-V2.6-RL-oss code.parquet image-mapping.jsonl \
  --repo-type dataset --revision "$MIMO_REVISION" --local-dir "$DATA_ROOT/data"
```

Convert every task for both CLIs. Omitting `--ids` selects the full code split;
the following command produces 5,396 rows in `train-mixed.jsonl`:

```bash
python examples/coding_agent_rl/prepare_mimo.py \
  --data "$DATA_ROOT/data" --output "$DATA_ROOT/train-mixed.jsonl" \
  --agents codex claude_code
```

Each task has one row per CLI, with `metadata.agent` selecting `codex` or
`claude_code`. Every prompt group uses one CLI. Both receive equal input weights;
filtering groups with uniform rewards can change the mixture in accepted
batches. For a smaller subset, pass instance IDs through `--ids`.

Download the task images as compressed OCI layouts. The helper uses
`image-mapping.jsonl` through the converted rows to resolve the official Docker
Hub tags, and downloads each unique image once even though it has two rows:

```bash
python examples/coding_agent_rl/prepare_sunabako_images.py \
  --data "$DATA_ROOT/train-mixed.jsonl" --root "$DATA_ROOT/oci" \
  --output "$DATA_ROOT/images.json" --format oci --workers 8
```

Wait for this command to finish successfully before launching training. It
writes `images.json` after every requested image is ready. Re-run the same
command after an interrupted download: completed images and completed OCI blobs
are reused. `--workers` controls download concurrency. OCI mode keeps compressed
layers on shared storage; Sunabako extracts them into each node's bounded local
cache when a sandbox first needs the image.

The resulting layout is:

| Path under `DATA_ROOT` | Contents |
| --- | --- |
| `data/code.parquet`, `data/image-mapping.jsonl` | Original tasks and official image mapping |
| `train-mixed.jsonl` | All tasks, with one row per CLI |
| `oci/<image-tag>/` | Compressed OCI blobs, manifest and completion metadata |
| `images.json` | Sunabako map from image names to those absolute bundle paths |

The launcher defaults to `$DATA_ROOT/train-mixed.jsonl` and
`$DATA_ROOT/images.json`. `PROMPT_DATA` and `SUNABAKO_IMAGES` can override these
paths.

The agent receives the public problem statement and runs as `agent`. The official
test patch and command are reserved for grading in a fresh sandbox, which runs
as root to use the image-provided toolchains.

Provide native CLI archives through `SLIME_AGENT_CC_NATIVE_TARBALL` and
`SLIME_AGENT_CODEX_NATIVE_TARBALL`. The harness installs them offline inside each
sandbox. The compaction hooks have been exercised with Claude Code 2.1.295 and
Codex 0.162.1. Codex's archive is the official `@openai/codex@0.162.1-linux-x64`
platform package. Both CLIs call the local adapter and the served Qwen model;
they do not require a paid model-provider API account.

## Start training

Start a dedicated Ray cluster on the six GPU nodes. Set `HEAD_IP`, `RAY_PORT`
and `DASHBOARD_PORT` to the head's address and available ports. On the head:

```bash
ray start --head --node-ip-address "$HEAD_IP" --port "$RAY_PORT" \
  --num-gpus 8 --num-cpus 64 --disable-usage-stats \
  --dashboard-host 0.0.0.0 --dashboard-port "$DASHBOARD_PORT"
```

On each of the other five GPU nodes, set `NODE_IP` to that node's address:

```bash
ray start --address "$HEAD_IP:$RAY_PORT" --node-ip-address "$NODE_IP" \
  --num-gpus 8 --num-cpus 64 --disable-usage-stats
```

Use your allocation's addresses and ports, and bypass HTTP proxies for inter-node
traffic. Launch from a persistent shell on the head:

```bash
export HF_CHECKPOINT=/shared/models/Qwen3.8-27B
export MEGATRON_DIR=/shared/Megatron-LM
export DATA_ROOT=/shared/mimo
export PROMPT_DATA="$DATA_ROOT/train-mixed.jsonl"
export SUNABAKO_IMAGES="$DATA_ROOT/images.json"
export SUNABAKO_CLUSTER=/shared/config/sandbox-cluster.json
export SLIME_AGENT_CC_NATIVE_TARBALL=/shared/toolchain/claude-code-linux-x64.tgz
export SLIME_AGENT_CODEX_NATIVE_TARBALL=/shared/toolchain/codex-linux-x64.tgz
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
export RAY_DASHBOARD_ADDRESS="http://$HEAD_IP:$DASHBOARD_PORT"
export SOCKET_IFNAME="<inter-node-interface>"
export RUN_ROOT=/shared/runs/coding-agent-001
export LOG_ROOT=/shared/logs/coding-agent-001
export WANDB_PROJECT=my-agent-rl WANDB_NAME=coding-agent-001
bash examples/coding_agent_rl/run_qwen38_27b_fully_async.sh --start-rollout-id 0
```

Supply W&B credentials through the environment; omit `WANDB_PROJECT` to disable
tracking. Runtime environment files are private and removed when the submission
client exits. Additional trainer arguments can be appended to the command.

The launcher uses native GDN, activation recomputation, CPU optimizer offload
and stateless Adam. Stateless Adam resets moment history on every step. Sampling
uses temperature 1, `top_p=0.95` and medium reasoning effort; training enables
score centering, exact top-p replay and token-wise importance sampling.
MTP speculative decoding and HiCache are enabled.

`SERVER_CONCURRENCY=14` across twelve engines provides 168 agent slots.
These slots include agents doing sandbox work or waiting for model calls.
`GROUP_PREFETCH_FACTOR=2` keeps up to 42 prompt groups available to fill slots
as individual trajectories finish. A group enters
training only when all eight trajectories finish and its rewards are nonuniform.

`FLUSH_CACHE_INTERVAL=3` publishes weights after every optimizer update while
flushing caches every third update. Partial rollout preserves the live CLI,
sandbox and sampled token prefix across interrupted model calls.

Each CLI may compact twice, then continue; a third compaction attempt stops the
agent and grades its existing patch. There is no turn-count cap. The agent budget
includes inference waiting and tool execution. The grader timeout covers its
command. The whole-rollout timeout includes sandbox preparation, agent execution,
patch extraction, fresh-sandbox grading and trajectory export.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `NUM_ROLLOUT` | `200` | Optimizer updates |
| `ROLLOUT_BATCH_SIZE` / `SAMPLES_PER_PROMPT` | `8` / `8` | Training groups and trajectories per group |
| `SERVER_CONCURRENCY` | `14` | Agent slots per inference engine |
| `GROUP_PREFETCH_FACTOR` | `2` | Prompt-group prefetch multiplier |
| `MAX_CONTEXT_LEN` / `MAX_RESPONSE_LEN` | `131072` / `8192` | Context and per-turn output limits |
| `SLIME_AGENT_CC_MAX_COMPACTIONS` / `SLIME_AGENT_CODEX_MAX_COMPACTIONS` | `2` / `2` | Per-trajectory compaction limits |
| `SWE_AGENT_TIME_BUDGET_SEC` | `3600` | Agent wall-clock budget |
| `SWE_EVAL_TIMEOUT_SEC` | `600` | Grader command timeout |
| `SWE_ROLLOUT_GUARD_SEC` | `7200` | Whole-rollout timeout |
| `SWE_BOOT_CONCURRENCY` | `0` | Optional boot semaphore; zero disables it |
| `SUNABAKO_MEMORY_MB` / `SUNABAKO_PIDS_MAX` | `8192` / `1024` | Per-sandbox resource requests |
| `SUNABAKO_CREATE_TIMEOUT_SEC` | `600` | SDK creation deadline; increase for cold imports |
| `ADAPTER_PUBLIC_HOST` / `ADAPTER_PORT` | `auto` / `0` | Advertised node address and dynamically assigned port |
| `SAVE_INTERVAL` | `0` | Checkpoint interval; zero disables saving |

## Outputs and restart

`$LOG_ROOT/run.log` contains trainer, rollout and inference logs. W&B files,
sandbox artifacts and timing traces also live under `LOG_ROOT`. The shared
`$RUN_ROOT/rollout_data` holds Straw payloads, queue state and recovery metadata;
Slime's `straw-queue` dependency is required on all GPU nodes. Checkpoints are
written under `RUN_ROOT` only when `SAVE_INTERVAL` is positive.

For a fresh experiment, stop the driver and dispose its retained rollout manager
and serving cluster, including workers, engines and owned sandboxes. After all
writers exit, delete that experiment's Straw directory if it is no longer needed.
Use new `RUN_ROOT`, `LOG_ROOT` and `WANDB_NAME` values and load the original model.
A new submission alone can reattach to retained serving state when its identity
and storage paths are reused.

To inspect a stopped run's timing traces:

```bash
cat "$LOG_ROOT"/traces/rollouts-*.jsonl > "$LOG_ROOT/coding-traces.jsonl"
python tools/trace_timeline_viewer.py "$LOG_ROOT/coding-traces.jsonl" --no-serve
```

Traces cover preparation, model requests, agent execution and grading. Nested
spans overlap; do not sum them as independent durations. `SWE_TRACE_DIR` selects
the output directory, and an empty value disables trace files.

## How trajectories become training samples

The adapters retain the exact SGLang input/output token IDs and rollout
log-probabilities. Model outputs receive `loss_mask=1`; tool observations and
other context receive `loss_mask=0`. Training never reconstructs sampled tokens
by re-tokenizing decoded text.

A changed or compacted history can produce multiple `Sample` segments. They
retain their prompt-group and trajectory identities and share one outcome
reward. GRPO normalizes over distinct trajectories, then distributes advantages
to their segments. Shared generated prefixes contribute loss only once, and all
segments of a trajectory stay in the same optimizer step.

`generate.py` coordinates the harness, adapter and grading. `swe.py` prepares
workspaces and evaluates patches. `sandbox.py` selects the provider, and
`sunabako_sandbox.py` implements the shared sandbox interface. `check_rollout.py`
can validate real trajectories against an already-running SGLang server without
starting trainers; run it with `--help` for its inputs. The
[coding-agent CI test](https://thudm.github.io/slime/developer_guide/ci.html#coding-agent-training)
covers real Codex and Claude Code training, token provenance and optimizer updates.
