# Coding-Agent RL

This directory provides an example of running end-to-end **SWE (Software-Engineering) coding-agent RL** with slime: a real coding agent (claude-code CLI) drives `Read/Edit/Grep/Bash/Agent` tools inside a fresh sandbox per sample, the model produces a `git diff`, and the diff is graded against the dataset's test harness in a second clean sandbox (no test-cheating).

Three files implement the loop:

- `generate.py` — per-sample `generate()` registered via `--custom-generate-function-path`. Boots the sandbox, runs claude-code, captures the diff, scores it, and emits one or more `Sample`s back to slime.
- `middleware.py` — Anthropic Messages API ↔ SGLang `/generate` shim. claude-code talks to it as if it were Anthropic; the shim renders chat with raw-token splice, masks model-generated tokens (`loss_mask=1`) vs template/observation (`0`), TITO-verifies each turn, and emits **three kinds of segments** per trajectory: `subagent` (completed `Task/Agent` dispatch), `wipe` (chain frozen by auto-compact), `final` (tail of the main chain).
- `sandbox.py` — E2B sandbox helpers (boot/kill, exec, file I/O, install bootstraps, agent spawn, diff capture, fresh-sandbox evaluator). Public 5-primitive contract on `E2BSandbox`: `exec / upload / write_text / read_text / __a*` — reimplement these on Docker / Modal / local VM and the rest plugs in unchanged.

## Environment Setup

The slime training stack itself follows the standard setup. On top of that you need:

1. **An E2B-compatible sandbox cluster** (or any provider that speaks the E2B SDK). Configure via `E2B_API_KEY` (e.g. the standard `e2b_xxx` key from https://e2b.dev, or any internal endpoint that accepts the same SDK).
2. **Host-side tarballs** that get uploaded into each sandbox at boot:
   - Node 22 (`node-v22.x-linux-x64.tar.xz`) — exported as `SWE_HOST_NODE_TARBALL`.
   - Claude Code CLI npm tarball (`anthropic-ai-claude-code-local-linux-x64.tgz`) — exported as `SWE_HOST_CC_TARBALL`.
3. **A sandbox metadata file** (`SWE_SANDBOX_METADATA_FILE`) — JSON dict whose keys are passed as routing tags when booting an E2B sandbox. Must contain the image key referenced by `SWE_SANDBOX_IMAGE_METADATA_KEY` (e.g. `image`).
4. **Network reachability**: each sandbox dials back to the slime head node's middleware over `http://${SLIME_HEAD_HOST}:${SHIM_PORT}`. The head host must be reachable from inside the sandboxes (set `SLIME_HEAD_HOST` to a routable IP, not `127.0.0.1`).

## Dataset Format

Standard slime JSONL with three keys:

```jsonc
{
  "prompt": "<falls back here if metadata.problem_statement is missing>",
  "label": "<instance_id or grader label>",
  "metadata": {
    "image": "swedev/scaleswe.oh.34:<tag>",   // sandbox image reference
    "workdir": "/workspace/<repo>",            // repo path inside the sandbox
    "problem_statement": "<issue body>",
    // exactly one of the following two graders:
    "swepro": { /* SWE-bench Pro test harness — preferred */ },
    "eval_cmd": "pytest -x tests/..."          // last-resort: exit 0 = solved
    // sweb-style rows: metadata.remote_env_info.f2p_script (Python file
    // ending in `sys.exit(pytest.main(...))`) is auto-wrapped into eval_cmd.
  }
}
```

Wire it up with `--input-key prompt --label-key label --metadata-key metadata`.

## Running the Script

Override the paths at the top of the launcher, then run from a long-lived shell on the Ray head node (do **not** wrap in `nohup` — Ray child processes get cleaned up with it):

```bash
cd slime/

export HF_CHECKPOINT=/path/to/Qwen3.6-35B-A3B
export REF_MODEL_PATH=/path/to/Qwen3.6-35B-A3B_torch_dist
export PROMPT_DATA=/path/to/swe_train.jsonl
export SANDBOX_METADATA_FILE=/path/to/sandbox_metadata.json
export SWE_HOST_NODE_TARBALL=/path/to/node-v22.20.0-linux-x64.tar.xz
export SWE_HOST_CC_TARBALL=/path/to/anthropic-ai-claude-code-local-linux-x64.tgz

bash examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
```

The launcher brings up Ray across all hosts in `/root/mpi_rack_hostfile`, dumps every rollout to `runs/${EXP_TAG}_${STAMP}/rollout_dumps/`, and tees stdout into `runs/${EXP_TAG}_${STAMP}/run.log`.

## New Arguments

`generate.py` is wired in through slime's standard custom-generate hook:

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len 96000
   --rollout-max-response-len 32768
   --rollout-stop-token-ids 248046 248044
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
```

The SGLang server must expose Qwen3.6's tool-call and reasoning parsers so claude-code's tool invocations are parsed correctly:

```bash
SGLANG_ARGS=(
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   ...
)
```

## SWE-specific Environment Knobs

All set in the launcher; tune per cluster.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLIME_HEAD_HOST` | `${MASTER_ADDR}` | Public IP the sandbox uses to reach the middleware. **Must be routable from inside the sandbox.** |
| `SHIM_BIND_HOST` / `SHIM_PORT` | `0.0.0.0` / `18001` | Bind address of the middleware shim on the head node. |
| `E2B_API_KEY` | — | E2B (or compatible) API key. |
| `SWE_SANDBOX_METADATA_FILE` | — | JSON dict of routing metadata passed at sandbox boot. |
| `SWE_SANDBOX_IMAGE_METADATA_KEY` | — | Which key in the metadata file holds the image reference (e.g. `image`). |
| `SWE_HOST_NODE_TARBALL` | — | Host path to Node 22 tarball uploaded into each sandbox. |
| `SWE_HOST_CC_TARBALL` | — | Host path to the Claude Code CLI npm tarball. |
| `SWE_TIME_BUDGET_SEC` | `1800` | Wallclock budget for one agent run. |
| `SWE_EVAL_TIMEOUT_SEC` | `600` | Wallclock cap on the evaluator sandbox. |
| `SWE_BOOT_CONCURRENCY` | `6` | Cap on simultaneous sandbox boots (eases h2/SSL long-tail). |
| `SWE_MAX_RESPONSE_TOKENS` | `32768` | Per-segment response cap. Total trajectory can reach `K * SWE_MAX_RESPONSE_TOKENS`. |
| `SWE_MAX_SEGMENT_TOKENS` | `MAX_CONTEXT_LEN` | Drop any segment whose `prompt+response` exceeds the trainer's DP budget. |
| `SWE_LIST_TRAJECTORY` | `1` | Emit one `Sample` per segment (reducer splits `reward / K`). `0` collapses to the final segment only. |
| `SWE_SAVE_TRAJECTORY_TREE` | `1` | Persist tree metadata so sub-agent fan-out shows up in the trace viewer. |
| `SWE_TOOL_PARSER` / `SWE_REASONING_PARSER` | `qwen3_coder` / `qwen3` | Must match the parsers loaded by SGLang. |
| `SWE_CLAUDE_EXTRA_ARGS` | (see launcher) | Extra flags appended to the `claude` CLI invocation — registers the read-only `investigator` sub-agent, disables `WebFetch`/`WebSearch`, disables slash commands. |
| `SWE_CC_PROMPT` | unset | Optional override for the user-turn prompt. Setting this to require sub-agent dispatch is the most reliable way to maximize fan-out. |

## Fan-out Semantics (`SWE_LIST_TRAJECTORY=1`)

- `generate()` returns `list[Sample]` — one Sample per trajectory **segment** (`subagent` / `wipe` / `final`).
- Per-trajectory reward is split as `reward / K` across segments; `rollout_id` is shared so the per-rollout-mean loss reducer still counts the trajectory once.
- Sub-agent dispatch increases `K` (each completed `Agent` turn block becomes its own segment), so the effective batch after flatten can be much larger than `rollout_batch_size * n_samples_per_prompt`.

## Porting to a New Sandbox Backend

`sandbox.py`'s `E2BSandbox` class exposes a 5-primitive contract:

```python
await sb.exec(cmd, user=..., check=..., timeout=...)
await sb.upload(host_path, sandbox_path, user=...)
await sb.write_text(sandbox_path, text, user=...)
await sb.read_text(sandbox_path, user=...)
async with E2BSandbox(...) as sb: ...
```

Reimplement those on Docker / Modal / a local VM and everything in `generate.py` and `middleware.py` keeps working unchanged.
