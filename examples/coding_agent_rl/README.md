# coding_agent_rl

A minimal, readable example of **coding agent + sandbox execution + test
reward** in slime. One sample of training looks like:

> spin up a sandbox &nbsp;→&nbsp; run a coding agent (Claude Code) inside it
> &nbsp;→&nbsp; capture the model-produced `git diff` &nbsp;→&nbsp; spin up a
> SECOND clean sandbox, apply the diff, run the dataset's tests &nbsp;→&nbsp;
> 0/1 reward &nbsp;→&nbsp; feed the actual generated tokens (with loss-mask)
> back to slime, no re-tokenization.

The whole pipeline is ~1500 LoC across 4 files; everything is async and
sandbox-agnostic. Wire it up with one CLI flag:

```bash
--custom-generate-function-path examples.coding_agent_rl.generate.generate
```

slime's default `sglang_rollout.generate_rollout` outer loop is reused; only
the per-sample `generate()` is swapped.

## Files

| File | Role |
|---|---|
| `generate.py` | Per-sample entrypoint slime calls. Reads top-to-bottom: provision sandbox → drop `PROBLEM_STATEMENT.md` → run agent → `git diff` → eval in a fresh sandbox → fill `Sample`. |
| `sandbox.py` | Sandbox backend. Owns boot/kill, exec/upload, `install_node22` + `install_claude_code`, the long-running agent spawn (done-marker poll), `git diff`, and the fresh-sandbox eval runner (`swepro` / `f2p_script` / `eval_cmd`). |
| `middleware.py` | head-node aiohttp shim. Translates the agent's Anthropic Messages API into slime's SGLang `/generate` (token-native + logprobs) and keeps `(prompt_ids, response_ids, loss_mask)` per session so the trainer skips re-tokenization. Model-agnostic. |
| `run_glm47_355b.sh` | Reference launch script: GLM-4.7-355B-A32B, 8 nodes / 64 GPUs, colocate, E2B sandbox. |

## How a sample flows

```
┌──────────┐  apply_chat_template + /generate    ┌───────────┐
│middleware│ ───────────────────────────────────►│  SGLang   │
│ (head)   │ ◄───────────────────────────────────│ (rollout) │
└────▲─────┘     output_token_logprobs           └───────────┘
     │
     │ Anthropic Messages SSE
     │ Authorization: Bearer <session_id>
     │
┌────┴───────────────────────────────┐
│  Claude Code  (inside sandbox)     │   1) work sandbox
│  reads PROBLEM_STATEMENT.md        │      ↓ git diff
│  edits source, runs tests          │   2) eval sandbox (fresh)
└────────────────────────────────────┘      apply diff + run tests
                                            → 0/1 reward
```

The session_id (passed as `ANTHROPIC_AUTH_TOKEN`) is what lets the middleware
demux concurrent requests from many parallel rollouts and assemble the right
per-sample `(prompt_ids, response_ids, loss_mask)`.

The eval sandbox is always **fresh**. Reward depends only on the diff the
model produced — the agent cannot affect reward by mutating tests or env
state inside the work sandbox.

## Dataset schema

Each row needs `prompt`, `label`, `metadata`. `metadata` supports two layouts
(see `_metadata()` in `generate.py`):

```jsonc
{
  "prompt": "<issue body or chat-style list>",
  "label":  "<instance_id>",
  "metadata": {
    // sandbox + repo
    "image":             "registry/repo:tag",   // sandbox image
    "workdir":           "/workspace/<repo>",    // repo path inside sandbox
    "problem_statement": "...",                  // optional; falls back to prompt

    // test harness — pick ONE (priority: swepro > f2p_script > eval_cmd)
    "swepro": {                                  // SWE-bench Pro style
      "run_script_path":      "<host path to run.sh>",
      "parser_script_path":   "<host path to parser.py>",
      "fail_to_pass":         ["..."],
      "pass_to_pass":         ["..."],
      "selected_test_files":  ["..."],
      "before_repo_set_cmd":  "pip install -e ."
    },
    "f2p_script": "<a pytest script string; exit 0 == solved>",
    "eval_cmd":   "<any shell command; exit 0 == solved>",

    // optional setup commands run before evaluation (list[str] or && string)
    "pre_commands": ["..."]
  }
}
```

The alternative layout nests `image_url`, `workdir`, `f2p_script`,
`pre_commands` under `metadata.remote_env_info` — both work.

## Env knobs

The launch script is a TEMPLATE; you must fill in cluster-specific values. The
following env vars are required:

| Var | What it is |
|---|---|
| `E2B_API_KEY` | Your E2B cloud API key. |
| `SLIME_HEAD_HOST` | Public host/IP the sandboxes use to reach the middleware. Required when sandboxes can't resolve the trainer's hostname (always required for E2B cloud). |
| `SWE_HOST_NODE_TARBALL` | Host path to a Node 22 tarball — uploaded into every sandbox. Debian-12-based images often ship Node 16, which can't run Claude Code's `cli.js`. |
| `SWE_HOST_CC_TARBALL` | Host path to the Claude Code npm tarball — also uploaded into every sandbox. |
| `HF_CHECKPOINT` | HF dir / id used as actor init (and tokenizer source). |
| `REF_LOAD` | Megatron `torch_dist` ref checkpoint dir. |
| `PROMPT_DATA` | jsonl dataset path (one row per problem; schema above). |

Optional tuning knobs:

| Var | Default | Notes |
|---|---|---|
| `SWE_TIME_BUDGET_SEC` | `900` | Per-agent wallclock budget. |
| `SWE_EVAL_TIMEOUT_SEC` | `600` | Per-eval test execution timeout. |
| `SWE_TOOL_PARSER` | `glm47` | sglang `FunctionCallParser` name (empty = disable). |
| `SWE_REASONING_PARSER` | `glm45` | sglang `ReasoningParser` name (empty = disable). |
| `SWE_BOOT_CONCURRENCY` | `16` | Caps concurrent sandbox `boot + install`. E2B's per-sandbox h2 upload path is fragile at high concurrency (see `_provision_sandbox` docstring). |
| `SWE_BOOT_RETRIES` | `2` | Retry the whole `boot + install` on transient h2 / SSL errors. |
| `SWE_RPC_RETRIES` | `3` | Per-RPC retry count for transient httpx/h2 failures. |
| `SWE_SANDBOX_LIFETIME_SEC` | `3600` | Upper-bound sandbox lifetime (E2B kills regardless of activity). |
| `SWE_SANDBOX_METADATA_JSON` | `""` | Optional JSON object passed verbatim into `AsyncSandbox.create(metadata=...)`. Use this if your sandbox backend reads routing/size tags from metadata (e.g. `'{"my-platform/size": "lg"}'`). |
| `SHIM_BIND_HOST` / `SHIM_PORT` | `0.0.0.0` / `18001` | middleware bind. |
| `E2B_ENV_FILE` | — | Optional path to a `.env` the launch script will `source` (handy for `E2B_API_KEY`). |

## Run it

```bash
# minimal setup
export E2B_API_KEY=...
export SLIME_HEAD_HOST=<host/IP reachable from sandboxes>
export SWE_HOST_NODE_TARBALL=/path/to/node-v22.20.0-linux-x64.tar.xz
export SWE_HOST_CC_TARBALL=/path/to/anthropic-ai-claude-code.tgz
export HF_CHECKPOINT=/path/to/hf/checkpoint
export REF_LOAD=/path/to/megatron/torch_dist/checkpoint
export PROMPT_DATA=/path/to/dataset.jsonl

bash examples/coding_agent_rl/run_glm47_355b.sh
```

The launch script is a reference — adapt parallelism (TP/PP/EP), node count,
SGLang DP/EP sizes, optimizer / GRPO knobs, and the host-side tarball paths to
your environment.

## Swap things out

**Swap the model.** Set `SWE_TOOL_PARSER` / `SWE_REASONING_PARSER` to the
sglang parser names matching your model (e.g. `qwen25` / `qwen3`) and point
`--hf-checkpoint` at the new HF dir. As long as the HF tokenizer's chat
template accepts `tools=`, no code change is needed.

**Swap the agent (Claude Code → Codex / your own).** Replace
`install_claude_code` and the `shell_command + env` block inside
`run_claude_code` in `sandbox.py`. If the agent speaks OpenAI Chat
Completions instead of Anthropic Messages, also replace `_handle_messages`
in `middleware.py` — the session token store + prefix-diff over
`apply_chat_template` is reusable as-is, only the inbound/outbound API
shape changes.

**Swap the sandbox backend (E2B → local docker / modal / VM).** Rewrite
`E2BSandbox` to expose the same 5 async primitives (`__aenter__` /
`__aexit__` / `exec` / `upload` / `write_text` / `read_text`). Everything
above it (`install_*`, `run_claude_code`, `evaluate`, ...) only uses those
primitives — no E2B-specific code.

**Add an eval mode.** Extend `evaluate()` in `sandbox.py` with another
branch alongside the existing `swepro` / `f2p_script` / `eval_cmd` handlers.

## Design notes

- **No re-tokenization.** The middleware captures the exact `output_token_logprobs`
  returned by `/generate` and stitches them onto `prompt_ids` (re-rendered
  via `apply_chat_template` every turn). The trainer reads
  `(tokens, response_length, loss_mask)` from `Sample` directly.
- **Reasoning round-trip.** Models with a `<think>...</think>` block need
  their reasoning fed back as `reasoning_content` so the next turn's
  template re-render is byte-equivalent. The empty-thinking case has a
  subtle workaround (see comment in `_handle_messages`).
- **Done-marker poll.** E2B's gateway resets streaming HTTP/2 responses
  around 6.5 minutes, so the agent run can't be a long-lived foreground
  exec. The launcher writes its exit code into a marker file; the host polls
  every 5 s with short-lived RPCs. The 5 s cadence also acts as a sandbox-
  side keep-alive (the platform GC-s sandboxes whose only activity is a
  detached `setsid` process).
- **Boot semaphore + retry.** E2B's per-sandbox h2 uploads degrade fast under
  concurrency (~3% ProtocolError at N=64, ~67% at N=256). We cap concurrent
  boot+install at 16 and retry the whole provision on transient errors;
  agent run + eval don't need the cap (they're sparse RPC).
- **Minimal `Sample`.** Only `tokens / response_length / response /
  loss_mask / reward / status` are filled. Debug fields live in logs, not
  on the sample.
