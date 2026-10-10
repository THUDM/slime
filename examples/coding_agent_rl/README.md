# Coding-agent RL with Qwen3.8-27B, MiMo SWE and sunabako

The real Codex training smoke test is documented in the
[CI guide](../../docs/en/developer_guide/ci.md#coding-agent-training).
It uses one small MiMo task, one sample and one optimizer update on eight GPUs.

This example trains **Qwen3.8-27B** on the **code subset of
[Xiaomi MiMo's public RL dataset](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)**.
Claude Code uses the served model to read and edit a repository in a sunabako
sandbox. A second fresh sandbox applies the resulting `git diff` and runs the
official MiMo tests to produce the reward. Model tokens, loss masks and rollout
log-probabilities are returned to slime for GRPO training.

The main launcher is `run_qwen38_27b_sunabako_2plus2.sh`: two 8-GPU nodes run
training and model serving; two other nodes run the agent and grader sandboxes.
**sunabako is the default sandbox provider.** Install its independent package
into slime's environment; dataset preparation and training code live here.

## Codex compatibility

`SWE_AGENT=codex` selects `CodexHarness` and `ResponsesAdapter`. The harness is
tested with **Codex CLI 0.162.1** (npm latest checked on 2026-10-10).
Current Codex supports the [Responses protocol](https://learn.chatgpt.com/docs/config-file/config-reference)
for custom providers. The adapter implements stateless `/v1/responses` requests
with full history and `store=false`, JSON/SSE replies, function tools, custom
text tools and namespaced tools. It preserves parallel call/result correlation
and the model's original tokens and logprobs. The separate `OpenAIAdapter`
continues to serve legacy Chat Completions clients.

Set `SLIME_AGENT_CODEX_NATIVE_TARBALL` to the official platform npm archive
`@openai/codex@0.162.1-linux-x64`. The harness installs its native binary and
resources offline; Node is unnecessary for this path. The existing
`SLIME_AGENT_NODE_TARBALL` + `SLIME_AGENT_CODEX_TARBALL` npm installation path is
also supported. `SLIME_AGENT_CODEX_EXTRA_ARGS` and
`SLIME_AGENT_CODEX_EXTRA_ENVS` (a JSON object) pass additional CLI settings.
The CI test downloads and integrity-checks its pinned platform archive.

Codex runs noninteractively with JSON event logs inside the outer sandbox;
sunabako owns isolation. Its provider URL points to the local slime adapter,
and the bearer value is a rollout session ID. No OpenAI account or paid API
is involved: the served Qwen model generates the training tokens. Web search,
WebSockets and subagents are disabled by default. Stored responses,
`previous_response_id`, hosted tools, multimodal output and the server-side
`/v1/responses/compact` endpoint are not implemented.

## Install and configure the sandbox nodes

Start with the standard slime training environment. On both training nodes:

```bash
python -m pip install -r examples/coding_agent_rl/requirements-sunabako.txt
```

To test a local sunabako build, install its wheel and the data tools:

```bash
python -m pip install /path/to/sunabako-0.1.0-*.whl 'pyarrow>=18' 'huggingface-hub>=0.34'
```

Provision both sandbox nodes with the sunabako CLI, its patched PRoot dependency,
`skopeo`, `umoci`, and node configuration. Set a cgroup memory-pool limit on each
node. Create `$DATA_ROOT/cluster.json` with **only the two sandbox nodes**:

```json
{
  "nodes": [
    {
      "name": "sandbox-0",
      "host": "<first-sandbox-node-ip>",
      "port": 22,
      "state_dir": "/var/lib/sunabako",
      "binary": "/usr/local/bin/sunabako"
    },
    {
      "name": "sandbox-1",
      "host": "<second-sandbox-node-ip>",
      "port": 22,
      "state_dir": "/var/lib/sunabako",
      "binary": "/usr/local/bin/sunabako"
    }
  ]
}
```

SSH uses the allocation's existing credentials. The sandbox nodes must be able
to reach the training/serving nodes' adapter HTTP port. For this example,
sunabako uses the outer container's network and does not publish ports.

sunabako defaults to cgroup enforcement. `SUNABAKO_ALLOW_TEST_MEMORY=1` is an
explicit functional-test setting for machines without delegated writable
cgroups, and also requires the node's test-only flag. RSS monitoring cannot
enforce a hard aggregate memory limit. Production runs use delegated cgroups
and `SUNABAKO_ALLOW_TEST_MEMORY=0`.

## Prepare the Xiaomi MiMo code dataset

Run from the slime checkout. Use a shared directory available at the same path
on both training nodes:

```bash
export DATA_ROOT=/shared/slime-mimo
hf download XiaomiMiMo/MiMo-V2.6-RL-oss code.parquet image-mapping.jsonl \
  --repo-type dataset --local-dir "$DATA_ROOT/data"

python examples/coding_agent_rl/prepare_mimo.py \
  --data "$DATA_ROOT/data" --output "$DATA_ROOT/train.jsonl" \
  --ids format-code-task-002661 format-code-task-002829
```

These two tasks provide a small first run. The Qwen3.8-27B rollout check on
`format-code-task-002661` earned reward 1 with **8/8 official tests passing**;
the original image failed three tests. The other task completed normally with
reward 0. This verifies one successful model-produced fix, not full-dataset
accuracy or a completed positive-reward training update.

Omit `--ids` to convert the full code dataset (2,698 tasks in the checked
snapshot):

```bash
python examples/coding_agent_rl/prepare_mimo.py \
  --data "$DATA_ROOT/data" --output "$DATA_ROOT/train-full.jsonl"
```

The converter maps dataset image names through the official
`image-mapping.jsonl`. It puts only the public problem statement into the agent
prompt; the official test patch and test command stay in the grading payload.
The fresh grader resets official test paths before applying the test patch.
MiMo graders run as root because some images initialize their toolchain from
root's login profile; the coding CLI runs as `agent`.

On **each sandbox node**, import the selected tasks' images. Supply that node
with the prepared JSONL and run this example's image helper:

```bash
python examples/coding_agent_rl/prepare_sunabako_images.py \
  --data "$DATA_ROOT/train.jsonl" \
  --root /var/lib/sunabako-images/mimo \
  --output "$DATA_ROOT/images.json"
```

Use the same absolute image root on both nodes. If `DATA_ROOT` is not shared
with sandbox nodes, copy the JSONL to each node and copy the resulting
`images.json` back to the training nodes. The helper imports OCI images using
sunabako, reuses matching bundles, preserves their environment, and uses the
dataset's repository working directory rather than the image's entrypoint
working directory. No Docker daemon is needed. The resulting mapping has this
shape:

```json
{
  "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-002661": {
    "rootfs": "/var/lib/sunabako-images/mimo/format-code-task-002661/rootfs",
    "workdir": "/testbed",
    "env": {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
  }
}
```

For the full dataset, pass `train-full.jsonl` to the image helper as well.
Importing all task images can consume substantial disk space. Some images keep
Python under `/root/.pyenv`; adjust their mapping's `env.PATH` so the `agent`
user can invoke the image's toolchain. This does not change the official tests.

## Run Qwen3.8-27B on two training nodes and two sandbox nodes

Supply host-side Node 22 and Claude Code npm tarballs; the harness uploads them
into each sandbox at boot. Launch from a long-lived shell on the first training
node, with the slime checkout and data available at the same paths on both
training nodes:

```bash
export DATA_ROOT=/shared/slime-mimo
export HF_CHECKPOINT=/path/to/Qwen3.8-27B
export TRAIN_HEAD=<first-training-node-ip>
export TRAIN_WORKER=<second-training-node-ip>
export SSH_PORT=22
export GLOO_SOCKET_IFNAME=<training-network-interface>
export SUNABAKO_CLUSTER="$DATA_ROOT/cluster.json"
export SUNABAKO_IMAGES="$DATA_ROOT/images.json"
export SUNABAKO_ALLOW_TEST_MEMORY=0
export SLIME_AGENT_NODE_TARBALL=/path/to/node-v22.x-linux-x64.tar.xz
export SLIME_AGENT_CC_TARBALL=/path/to/claude-code.tgz
export RUN_ROOT="$DATA_ROOT/run-001"
bash examples/coding_agent_rl/run_qwen38_27b_sunabako_2plus2.sh
```

The launcher defaults to one rollout containing two prompts with two samples
each, followed by one GRPO update. For longer runs, set `NUM_ROLLOUT`. Set
`PROMPT_DATA="$DATA_ROOT/train-full.jsonl"` to use the full dataset after
provisioning its images. `ROLLOUT_BATCH_SIZE` and `SAMPLES_PER_PROMPT` control
sampling; `GLOBAL_BATCH_SIZE` defaults to their product.

The model context is 64k (`MAX_CONTEXT_LEN`), with an 8192-token per-turn output
limit (`MAX_RESPONSE_LEN`). Both limits are passed to the Claude CLI, with
compaction at 70% of the context window. The six code tools are enabled
explicitly: `Bash`, `Read`, `Edit`, `Write`, `Glob`, and `Grep`.
The default agent budget is 1200 seconds. Sampling uses temperature 1.0,
`top_p=0.95`, no top-k truncation (slime's default `top_k=-1`), and
`reasoning_effort=medium`; override template options
with `CHAT_TEMPLATE_KWARGS`. Each model turn retains the exact nucleus IDs and
offsets, which the trainer replays when normalizing token probabilities. Add
`--use-score-centering` to the launcher to also retain every nucleus token's
original sampler probability and train with the SC REINFORCE objective. The GPU
e2e enables this combination. The adapter forwards
`--apply-chat-template-kwargs` and per-sample template options on every turn.
Without an explicit effort, this checkpoint's template selects `xhigh`.
Decode CUDA graphs are enabled for up to four requests; set
`SGLANG_DISABLE_CUDA_GRAPH=1` when diagnosing graph compatibility.

Training uses TP=2, PP=8 and DP=1. The launcher uses the shared Qwen dense-model
architecture arguments in `scripts/models/qwen3.5-27B.sh` with the Qwen3.8-27B
checkpoint. It propagates configuration to Ray workers and calls
`examples.coding_agent_rl.generate.generate` for every sample.

The adapter also supports `--use-rollout-routing-replay` (R3) for MoE models.
Replay snapshots stay in separate per-turn segments, preserving full prompt
routes as well as response routes; the segments retain their original rollout
and prompt-group IDs. Missing requested metadata fails the rollout. Qwen3.8-27B
is dense, so this example does not exercise MoE routing on GPU.

With top-p SC, `sc_correction` uses the same centered scalar convention as the
non-top-p path: it subtracts a baseline whose score gradient is zero. Matching
trainer and sampler distributions therefore give zero correction, up to rounding,
while retaining the original SC gradient, including TIS/MIS. The scalar is not
the mean absolute difference between sampled-token logprobs.

`ADAPTER_PUBLIC_HOST=auto` advertises the rollout actor's node IP. Set an explicit
address if that IP is not reachable from the sandbox nodes. Use a new `RUN_ROOT`
for each experiment. `SUNABAKO_START_RAY=0` reuses the experiment's Ray cluster;
set a new `RUN_LOG` when retrying a run whose serving engines are retained.
Existing logs are never overwritten. CLI trajectories, sandbox commands and
grader output are saved under `$RUN_ROOT/sandboxes` before sandbox cleanup.

## Check real rollouts before training

Start an SGLang server with Qwen3.8-27B, the `qwen3_coder` tool parser and the
`qwen3` reasoning parser. Set the same sandbox and CLI environment as above,
then run:

```bash
export ADAPTER_PUBLIC_HOST=<host-ip-reachable-from-sandbox-nodes>
export SUNABAKO_ARTIFACTS="$DATA_ROOT/rollout-check/sandboxes"
export SWE_AGENT_TIME_BUDGET_SEC=1200 SWE_EVAL_TIMEOUT_SEC=180 SWE_BOOT_CONCURRENCY=2
export SLIME_AGENT_CC_EXTRA_ARGS='--max-turns 40 --tools Bash,Read,Edit,Write,Glob,Grep --disable-slash-commands'
export SLIME_AGENT_CC_EXTRA_ENVS='{"MAX_THINKING_TOKENS":"4096","CLAUDE_CODE_MAX_OUTPUT_TOKENS":"8192","CLAUDE_CODE_MAX_CONTEXT_TOKENS":"65536","CLAUDE_AUTOCOMPACT_PCT_OVERRIDE":"70"}'
python -m examples.coding_agent_rl.check_rollout \
  --data "$DATA_ROOT/train.jsonl" --output "$DATA_ROOT/rollout-check/samples" \
  --hf-checkpoint "$HF_CHECKPOINT" --sglang-url http://127.0.0.1:18100 \
  --apply-chat-template-kwargs '{"reasoning_effort":"medium"}'
```

This uses the training `generate()` entry point, including the coding CLI,
patch capture and fresh-sandbox official grader. It saves tokens, loss masks,
log-probabilities and per-task outcomes, and exits unsuccessfully if no sample
earns reward 1. No trainers are started. Establish a failing baseline on the
unmodified image when validating a task's reward.

## Modules and data contract

- `prepare_mimo.py` converts the official code parquet and image mapping to slime JSONL.
- `prepare_sunabako_images.py` imports task images and writes the provider's image mapping.
- `sandbox.py` selects sunabako for both agent and evaluator sandboxes; `sunabako_sandbox.py` adapts the installed SDK to `slime.agent.sandbox.Sandbox`.
- `generate.py` opens an `AnthropicAdapter` session, runs the coding harness, captures the diff, obtains the reward, and exports training `Sample`s.
- `swe.py` prepares the task workspace and runs the independent evaluator.
- `slime.agent.harness` installs and drives the coding CLI; the shared adapter sends model requests to SGLang and records token provenance.

Prepared rows use `--input-key prompt --label-key label --metadata-key metadata`:

```json
{
  "prompt": [{"role": "user", "content": "<public problem statement>"}],
  "label": "format-code-task-002661",
  "metadata": {
    "instance_id": "format-code-task-002661",
    "dataset": "XiaomiMiMo/MiMo-V2.6-RL-oss",
    "image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-002661",
    "workdir": "/testbed",
    "problem_statement": "<public problem statement>",
    "eval_cmd": "<converter-generated official test command>",
    "eval_user": "root"
  }
}
```

The generic grader also accepts existing SWE-bench Pro and `f2p_script` rows.
E2B remains an optional provider: set `SWE_SANDBOX_PROVIDER=e2b`, configure
`E2B_API_KEY` and `SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY` for that service,
and use a launcher that propagates those variables. The supplied 2+2 launcher
selects sunabako explicitly.

## Environment settings

These are the 2+2 launcher's defaults; direct calls to `generate()` use the
module defaults unless the corresponding variables are exported.

| Variable | Launcher default | Meaning |
| --- | --- | --- |
| `SWE_SANDBOX_PROVIDER` | `sunabako` | Provider used for both agent and evaluator. |
| `SUNABAKO_CLUSTER` / `SUNABAKO_IMAGES` | `$DATA_ROOT/cluster.json` / `images.json` | Sandbox nodes and imported task images. |
| `SUNABAKO_MEMORY_MB` | `2048` | Requested memory per sandbox. |
| `SUNABAKO_ALLOW_TEST_MEMORY` | Must be set explicitly | `0` for cgroups; `1` only for explicit RSS functional tests. |
| `SUNABAKO_ARTIFACTS` | `$RUN_ROOT/sandboxes` | Saved commands, CLI trajectories and grading output. |
| `ADAPTER_PUBLIC_HOST` | `auto` | Routable address advertised to sandbox nodes. |
| `ADAPTER_BIND_HOST` / `ADAPTER_PORT` | `0.0.0.0` / `18091` | Adapter listening address. |
| `SLIME_AGENT_NODE_TARBALL` / `SLIME_AGENT_CC_TARBALL` | Files in `$DATA_ROOT/toolchain` | Host-side Node 22 and Claude Code tarballs. |
| `SLIME_AGENT_CC_EXTRA_ARGS` | Six code tools, max 40 turns | Claude CLI flags. |
| `SLIME_AGENT_CC_EXTRA_ENVS` | Context/output/thinking limits | JSON overrides merged into the CLI environment. |
| `SWE_AGENT_TIME_BUDGET_SEC` | `1200` | Agent CLI wall-clock budget. |
| `SWE_EVAL_TIMEOUT_SEC` | `180` | Official grader timeout. |
| `SWE_ROLLOUT_GUARD_SEC` | `agent+eval+300` | Whole-rollout deadline, including sandbox setup. |
| `SWE_BOOT_CONCURRENCY` | `2` | Concurrent sandbox boots. |
| `SWE_CC_PROMPT` | Read the problem, edit source, test, summarize | Agent instruction; official tests remain in the grader. |

`SLIME_AGENT_*` belongs to the reusable agent library, `SWE_*` to this task
example, `SUNABAKO_*` to its provider, and `ADAPTER_*` to the serving endpoint.
`--rollout-max-response-len` limits each SGLang request; the adapter clamps it
to the remaining multi-turn context budget. The parser flags must match the
served model.

## Token-in, token-out model calls

The coding-agent environment is string/message based: claude-code sends
Anthropic Messages requests and Codex sends Responses requests. They receive
text, reasoning and tool calls, then send back rendered tool observations. Training must stay
token based. A trajectory is only a valid RL target when the optimized tokens
are the same tokens the rollout model actually sampled.

The shared adapter accepts messages from the agent and uses **token in, token out**
at the model and training boundary:

- Each incoming message history is rendered with the served model's chat
  template and sent to SGLang as `input_ids`.
- Parallel Anthropic tool results are matched by `tool_use_id` and restored to
  call order before rendering. Qwen's template omits those IDs, so preserving
  asynchronous completion order would associate file contents with the wrong
  `Read` call.
- SGLang is called with `return_logprob=True`; the adapter records the exact
  `prompt_ids`, sampled `output_ids`, and per-token rollout logprobs for that
  turn. Missing logprobs, inconsistent token counts/IDs and non-finite logprobs
  fail the request instead of producing an incomplete training trajectory.
- At training export time, samples are assembled from those saved token ids.
  The decoded `response` field is only a readable sidecar; it is not
  re-tokenized to recover the training sequence.

Multi-turn agents still force the adapter to tokenize later message
histories, because tool observations and the agent's compacted messages
arrive as strings. `slime.agent.trajectory.TrajectoryManager` routes
those later prompts against the saved token stream:

- New prompt suffixes that are tool/user/environment context are appended with
  `loss_mask=0`.
- Fresh model outputs from SGLang are appended with `loss_mask=1`.
- If a later prompt no longer token-matches an earlier sampled output, the
  example starts a new training segment. The earlier output keeps its original
  prompt, token IDs and logprobs. Re-rendered history in the new segment is
  context with `loss_mask=0`.
- Generated prefixes shared by sibling branches contribute loss only once.

This example defaults to `SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=0`, including on
remote Ray workers. It preserves every captured generated turn when history
changes. Setting a positive threshold enables the manager's optional short-turn
rewrite merging, which can discard earlier training signal to reduce segment count.
The agent's message protocol itself is still text based; only the exact captured
model tokens are used as optimization targets.
The unit tests in `tests/test_agent/test_trajectory_manager_branching.py` cover matched
prefixes, skipped turns, split-output drift, changed token counts, and
prompt-base restarts.

## Fan-out Semantics

- `generate()` returns `list[Sample]`. A root-to-leaf chain can produce multiple
  segments if the token prefix changes.
- Segments retain the original `group_index` (prompt group), `index` (agent run)
  and `rollout_id` (the unit used for optimizer scheduling and loss averaging).
- Every segment receives the agent run's full outcome reward. GRPO computes
  mean/std within each `group_index` over distinct `rollout_id` values, then
  broadcasts the resulting advantage to their segments. A run with 20 segments
  therefore contributes one outcome to these statistics, just like a run with one.
- Shared generated prefixes are loss-masked on subsequent branches. All segments
  of a rollout stay in one optimizer step, and their loss denominators use the
  total number of unmasked tokens in that rollout, including across microbatches.
- Sub-agent dispatch and auto-compaction increase the number of segments, so the
  flattened sample count can exceed `rollout_batch_size * n_samples_per_prompt`.

The short GPU CI retains and audits every real model turn, then trains the first
and last segments to bound runtime. It checks exact token/logprob identity again
in the trainer's saved tensors and confirms real parameter updates. CPU tests
cover unequal fork counts, shuffled prompt groups, shared prefixes and singleton
groups; the single-rollout GPU smoke test disables reward normalization so it can
exercise a nonzero optimizer update. Normal training consumes all segments.

## Porting to a New Sandbox Backend

`slime.agent.sandbox.Sandbox` exposes the shared sandbox contract, and
`slime.agent.sandbox.E2BSandbox` is the E2B implementation:

```python
await sb.exec(cmd, user=..., check=..., timeout=...)
await sb.write_file(sandbox_path, content_or_host_path, user=...)
await sb.read_file(sandbox_path, user=...)
async with create_sandbox(image) as sb: ...
```

Implement this contract and register the provider in this example's `sandbox.py`.
Both `generate.py` and `swe.py` use that factory.
