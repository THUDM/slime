# Coding-Agent RL

This directory provides an example of running end-to-end **SWE (Software-Engineering) coding-agent RL** with slime: a real coding agent (claude-code CLI) drives `Read/Edit/Grep/Bash/Agent` tools inside a fresh sandbox per sample, the model produces a `git diff`, and the diff is graded against the dataset's test harness in a second clean sandbox (no test-cheating).

Two example files and one shared adapter implement the loop:

- `generate.py` — per-sample `generate()` registered via `--custom-generate-function-path`. Boots the sandbox, runs claude-code, captures the diff, scores it, and emits one or more `Sample`s back to slime.
- `slime.agent.adapters.AnthropicAdapter` — the shared Anthropic Messages adapter. claude-code talks to it as if it were Anthropic; the adapter tokenizes the current message history each turn, records prompt/output token snapshots, preserves model-generated tokens (`loss_mask=1`) only while later prompts stitch onto them, masks template/observation tokens (`0`), and emits **three kinds of segments** per trajectory: `subagent` (completed `Task/Agent` dispatch), `wipe` (chain frozen by auto-compact), `final` (tail of the main chain).
- `sandbox.py` — coding-agent/SWE helpers built on `slime.agent.sandbox`: install bootstraps, spawn claude-code, capture patches, and run the fresh-sandbox evaluator. The shared sandbox contract lives in `slime.agent.sandbox.Sandbox`.

`generate.py` owns one `AnthropicAdapter` instance. For each sample it calls
`adapter.open_session(...)` before starting claude-code, serves `adapter.app` as
the Anthropic-compatible endpoint, and drains trainable `TokenSegment`s with
`await adapter.finish_session(...)` when the trajectory ends.

## Environment Setup

The slime training stack itself follows the standard setup. On top of that you need:

1. **An E2B-compatible sandbox cluster** (or any provider that speaks the E2B SDK). Configure via `E2B_API_KEY` (e.g. the standard `e2b_xxx` key from https://e2b.dev, or any internal endpoint that accepts the same SDK). The official SDK validates this value locally, so internal gateways that ignore auth still need a syntactically valid `e2b_` + 40 hex-character placeholder.
2. **Host-side tarballs** that get uploaded into each sandbox at boot:
   - Node 22 (`node-v22.x-linux-x64.tar.xz`) — exported as `SWE_HOST_NODE_TARBALL`.
   - Claude Code CLI npm tarball (`anthropic-ai-claude-code-local-linux-x64.tgz`) — exported as `SWE_HOST_CC_TARBALL`.
3. **A sandbox metadata file** (`SWE_SANDBOX_METADATA_FILE`, or the generic `SLIME_AGENT_SANDBOX_METADATA_FILE`) — JSON dict whose keys are passed as routing tags when booting an E2B sandbox. Must contain the image key referenced by `SWE_SANDBOX_IMAGE_METADATA_KEY` / `SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY` (e.g. `image`).
4. **Network reachability**: each sandbox dials back to the slime head node's Anthropic adapter over `http://${SLIME_HEAD_HOST}:${SHIM_PORT}`. The head host must be reachable from inside the sandboxes (set `SLIME_HEAD_HOST` to a routable IP, not `127.0.0.1`).

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
| `SLIME_HEAD_HOST` | `${MASTER_ADDR}` | Public IP the sandbox uses to reach the Anthropic adapter. **Must be routable from inside the sandbox.** |
| `SHIM_BIND_HOST` / `SHIM_PORT` | `0.0.0.0` / `18001` | Bind address of the adapter shim on the head node. |
| `E2B_API_KEY` | — | E2B (or compatible) API key. |
| `SWE_SANDBOX_METADATA_FILE` / `SLIME_AGENT_SANDBOX_METADATA_FILE` | — | JSON dict of routing metadata passed at sandbox boot. |
| `SWE_SANDBOX_IMAGE_METADATA_KEY` / `SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY` | — | Which key in the metadata file holds the image reference (e.g. `image`). |
| `SWE_HOST_NODE_TARBALL` | — | Host path to Node 22 tarball uploaded into each sandbox. |
| `SWE_HOST_CC_TARBALL` | — | Host path to the Claude Code CLI npm tarball. |
| `SWE_TIME_BUDGET_SEC` | `1800` | Wallclock budget for one agent run. |
| `SWE_EVAL_TIMEOUT_SEC` | `600` | Wallclock cap on the evaluator sandbox. |
| `SWE_BOOT_CONCURRENCY` | `6` | Cap on simultaneous sandbox boots (eases h2/SSL long-tail). |
| `SWE_CLAUDE_EXTRA_ARGS` | (see launcher) | Extra flags appended to the `claude` CLI invocation — registers the read-only `investigator` sub-agent, disables `WebFetch`/`WebSearch`, disables slash commands. |
| `SWE_CC_PROMPT` | unset | Optional override for the user-turn prompt. Setting this to require sub-agent dispatch is the most reliable way to maximize fan-out. |

`--rollout-max-response-len` is the per-turn generation cap passed to each
SGLang `/generate` call as `max_new_tokens`. `--rollout-max-context-len` is the
multi-turn prompt+response budget enforced only during generation: each turn
clamps `max_new_tokens` to the remaining context. Trajectory merge/export keeps
the emitted segments and does not drop them for length.
The Anthropic adapter reuses `--sglang-tool-call-parser` and
`--sglang-reasoning-parser` for output parsing, so those flags must match the
served model.

## String-in, Token-out Trajectories

The coding-agent environment is string/message based: claude-code sends
Anthropic Messages requests, receives streamed text/thinking/tool-use blocks,
and later sends back rendered tool observations. Training, however, must stay
token based. A trajectory is only a valid RL target when the optimized tokens
are the same tokens the rollout model actually sampled.

The Anthropic adapter therefore follows a **string in, token out** contract:

- Each incoming message history is rendered with the served model's chat
  template and sent to SGLang as `input_ids`.
- SGLang is called with `return_logprob=True`; the adapter records the exact
  `prompt_ids`, sampled `output_ids`, and per-token rollout logprobs for that
  turn.
- At training export time, samples are assembled from those saved token ids.
  The decoded `response` field is only a readable sidecar; it is not
  re-tokenized to recover the training sequence.

Multi-turn agents still force the adapter to tokenize later message
histories, because tool observations and claude-code's own compacted messages
arrive as strings. `slime.agent.trajectory.merge_turns` stitches those later
prompts against the saved token stream:

- New prompt suffixes that are tool/user/environment context are appended with
  `loss_mask=0`.
- Fresh model outputs from SGLang are appended with `loss_mask=1`.
- If a later prompt no longer token-matches an earlier sampled output, the
  unmatched suffix is dropped. If the drift cuts through the middle of a
  previous model output, the retained prefix of that whole output turn is also
  assigned `loss_mask=0`.

That last case is the important correctness guard. A re-tokenization mismatch
can make a string-level conversation look continuous while token-level
provenance is broken. slime keeps the context needed to continue the agent, but
does not backprop through tokens whose sampled origin can no longer be proven.
The unit tests in `tests/test_agent_trajectory.py` cover matched prefixes,
skipped turns, split-output drift, changed token counts, and prompt-base
restarts.

## Fan-out Semantics

- `generate()` returns `list[Sample]` — one Sample per trajectory **segment** (`subagent` / `wipe` / `final`).
- Per-trajectory reward is split as `reward / K` across segments; `rollout_id` is shared so the per-rollout-mean loss reducer still counts the trajectory once.
- Sub-agent dispatch increases `K` (each completed `Agent` turn block becomes its own segment), so the effective batch after flatten can be much larger than `rollout_batch_size * n_samples_per_prompt`.

## Porting to a New Sandbox Backend

`slime.agent.sandbox.Sandbox` exposes the shared sandbox contract, and
`slime.agent.sandbox.E2BSandbox` is the E2B implementation:

```python
await sb.exec(cmd, user=..., check=..., timeout=...)
await sb.write_file(sandbox_path, content_or_host_path, user=...)
await sb.read_file(sandbox_path, user=...)
async with E2BSandbox(...) as sb: ...
```

Reimplement those on Docker / Modal / a local VM and everything in `generate.py` keeps working unchanged.
