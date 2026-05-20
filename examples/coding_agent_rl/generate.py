"""Coding-Agent RL: per-sample generate() function for slime.

Wire-up:

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

``generate()`` below IS the agent. Read it top-to-bottom to see what one SWE
rollout sample does. All sandbox-side details live in ``sandbox.py``; the LLM
plumbing (Anthropic <-> SGLang /generate, token capture) lives in ``middleware.py``.

Per-sample steps:

    1. Boot a fresh sandbox from the dataset image.
    2. Install Node 22 + Claude Code CLI.
    3. Create the agent user, drop PROBLEM_STATEMENT.md.
    4. Run claude-code pointed at the head-node middleware (the middleware
       captures tokens by session_id, passed via the Bearer token).
    5. ``git diff`` to capture the model-produced patch.
    6. Boot a SECOND, fresh sandbox; apply diff; run the dataset's tests for
       reward. (No-test-cheating guarantee: reward only depends on the diff.)
    7. Pull (prompt_ids, response_ids, loss_mask) from the middleware and fill
       the Sample. No re-tokenization.

Dataset row ``metadata`` schema::

    image:             str        # sandbox image
    workdir:           str        # repo path inside the sandbox
    problem_statement: str        # issue body (falls back to sample.prompt)
    swepro:            dict|None  # SWE-bench Pro test harness (preferred)
    f2p_script:        str|None   # fallback: pytest script (exit 0 = solved)
    eval_cmd:          str|None   # last-resort: shell command (exit 0 = solved)

Env knobs (set in run.sh):

    SWE_HOST_NODE_TARBALL    host path to a Node 22 tarball (REQUIRED)
    SWE_HOST_CC_TARBALL      host path to the Claude Code npm tarball (REQUIRED)
    SWE_TIME_BUDGET_SEC      900   per agent run, wallclock
    SWE_EVAL_TIMEOUT_SEC     600   per eval test execution
    SWE_TOOL_PARSER          glm47           (sglang FunctionCallParser name)
    SWE_REASONING_PARSER     glm45           (sglang ReasoningParser name)
    SWE_SAVE_TRAJECTORY_TREE 0               save full per-turn request/response tree in metadata
    SHIM_BIND_HOST           0.0.0.0
    SHIM_PORT                18001
    SLIME_HEAD_HOST          public host the sandboxes use to reach the middleware
                              (REQUIRED for E2B — set to a host/IP that the
                              sandboxes can route back to)

---

Forking this file (each step above is one or two lines below; modify in place):

* Swap claude_code for codex: edit ``_provision_sandbox`` (the
  ``install_claude_code`` line) and replace ``sandbox.run_claude_code(...)``
  in ``generate()`` with your own helpers in sandbox.py (or another module).
  Most likely you also need a new ``middleware.py`` if your agent speaks OpenAI
  Chat Completions instead of Anthropic.

* Swap E2B for local docker: re-import a different sandbox module (e.g.
  ``from . import docker_backend as sandbox``). The orchestrator is sandbox-agnostic.

* Swap GLM-4.7 for another model: set ``SWE_TOOL_PARSER`` + ``SWE_REASONING_PARSER``
  env (and obviously change the HF checkpoint path in run.sh). No code change here.

* Add a custom prompt or rollout-time guardrail: edit ``CC_PROMPT`` or add a
  step between ``sandbox.run_claude_code`` and ``sandbox.git_diff``.

* Tune sandbox boot concurrency / retries: ``SWE_BOOT_CONCURRENCY`` (default 16)
  caps how many tasks may be boot+install-ing at once;  ``SWE_BOOT_RETRIES``
  (default 2) is how many times each sample's provision is retried on
  transient h2/SSL errors. See ``_provision_sandbox`` for context.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import middleware, sandbox

logger = logging.getLogger(__name__)


# Env knobs used here (sandbox-side knobs live in sandbox.py).
# Tarball paths must point at host-local files; the defaults below are
# placeholders — set SWE_HOST_NODE_TARBALL / SWE_HOST_CC_TARBALL in the launch
# script. Both files are uploaded into every sandbox at provisioning time.
SWE_HOST_NODE_TARBALL = Path(os.environ.get(
    "SWE_HOST_NODE_TARBALL",
    "/path/to/node-v22.20.0-linux-x64.tar.xz",
))
SWE_HOST_CC_TARBALL = Path(os.environ.get(
    "SWE_HOST_CC_TARBALL",
    "/path/to/anthropic-ai-claude-code.tgz",
))
SWE_TIME_BUDGET_SEC = int(os.environ.get("SWE_TIME_BUDGET_SEC", "900"))
SWE_EVAL_TIMEOUT_SEC = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
SWE_TOOL_PARSER = os.environ.get("SWE_TOOL_PARSER", "") or None
SWE_REASONING_PARSER = os.environ.get("SWE_REASONING_PARSER", "") or None
SWE_SAVE_TRAJECTORY_TREE = os.environ.get("SWE_SAVE_TRAJECTORY_TREE", "0") == "1"
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18001"))

# Cap concurrent sandbox boot+install. E2B's envd file-upload path goes through
# per-sandbox h2 connections; at N>=64 we observed ~3% h2 ProtocolError, at
# N=256 ~67% (server-side GOAWAY under load, then httpx h2 state machine
# explodes — encode/httpx#2761 family of bugs). N=16 ran 80/80 uploads with 0
# ProtocolErrors in repro. Run-agent + git_diff + eval don't need this cap
# (they're sparse RPC).
SWE_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
SWE_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))
_BOOT_SEM: asyncio.Semaphore | None = None  # lazy-init: must bind to the right loop

CC_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)


# ---------------------------------------------------------------------------
# Singleton: tokenizer + in-process middleware handle (one per worker process)
# ---------------------------------------------------------------------------
class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        public_host = (
            os.environ.get("SLIME_HEAD_HOST")
            or os.environ.get("MLP_WORKER_0_HOST")
            or socket.gethostname()
        )
        self.middleware = middleware.start(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=SWE_TOOL_PARSER,
            reasoning_parser=SWE_REASONING_PARSER,
            host=SHIM_BIND_HOST,
            port=SHIM_PORT,
            public_host=public_host,
        )
        logger.info("[coding_agent_rl] tokenizer=%s middleware=%s", args.hf_checkpoint, self.middleware.public_url)


# ---------------------------------------------------------------------------
# Sandbox provisioning: boot + Node22 + Claude Code, semaphore-gated + retry
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _provision_sandbox(image: str):
    """Yield a sandbox with Node22 + Claude Code already installed.

    Two layers of protection against the boot+upload failure modes observed
    in production (see SWE_BOOT_CONCURRENCY docstring above):

    * concurrency cap via ``_BOOT_SEM`` — only N tasks may be boot+install-ing
      at once. After install, the semaphore is released, and the multi-minute
      agent run proceeds without contention.

    * one retry on any boot/install exception (ProtocolError, SSLError,
      transient SandboxException). The retry creates a NEW sandbox; the old
      one is killed via __aexit__ first.
    """
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)

    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = sandbox.E2BSandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await sandbox.install_node22(cand, SWE_HOST_NODE_TARBALL)
                    await sandbox.install_claude_code(cand, SWE_HOST_CC_TARBALL)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[coding_agent_rl] provision attempt %d/%d failed: %s: %s",
                attempt + 1, SWE_BOOT_RETRIES, type(e).__name__, str(e)[:200],
            )
            await asyncio.sleep(1 + attempt)  # tiny backoff before retry
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Main per-sample agent function
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    state = _State(args)
    md = _metadata(sample)
    if not md["image"] or not md["workdir"]:
        return _abort(sample, "missing_image_or_workdir")

    session_id = sample.session_id or f"cagent-{md['instance_id']}-{secrets.token_hex(4)}"
    sample.session_id = session_id
    state.middleware.open_session(
        session_id,
        sampling_defaults=sampling_params,
        record_tree=SWE_SAVE_TRAJECTORY_TREE,
    )

    t0 = time.time()
    diff_text = ""
    try:
        # --- 1) work sandbox: provision (concurrency-capped + retried) ------
        #     -> ensure_agent_user / drop PROBLEM_STATEMENT.md / run agent
        async with _provision_sandbox(md["image"]) as sb:
            await sandbox.ensure_agent_user(sb, md["workdir"])
            await sb.write_text(
                f"{md['workdir']}/PROBLEM_STATEMENT.md",
                md["problem_statement"] or "", user="agent",
            )
            if md["swepro"]:
                await sandbox.apply_before_repo_set_cmd(sb, md["workdir"], md["swepro"])

            await sandbox.run_claude_code(
                sb, workdir=md["workdir"], session_id=session_id,
                middleware_url=state.middleware.public_url, prompt=CC_PROMPT,
                time_budget_sec=SWE_TIME_BUDGET_SEC,
            )
            diff_text = await sandbox.git_diff(sb, md["workdir"])

        # --- 2) eval sandbox: fresh checkout, apply diff, run tests ---------
        reward, is_solved, applied_cleanly = await sandbox.evaluate(
            image=md["image"], workdir=md["workdir"], diff_text=diff_text,
            swepro=md["swepro"], f2p_script=md["f2p_script"],
            eval_cmd=md["eval_cmd"], pre_commands=md["pre_commands"],
            timeout_sec=SWE_EVAL_TIMEOUT_SEC,
        )
    except Exception as e:
        logger.error("[coding_agent_rl] %s: rollout failed: %s\n%s",
                     md["instance_id"], e, traceback.format_exc())
        return _abort(sample, f"exception:{type(e).__name__}")

    # --- 3) pull tokens from middleware, fill sample -----------------------
    prompt_ids, response_ids, loss_mask, trajectory_tree = state.middleware.pop_session(session_id)
    if not prompt_ids:
        return _abort(sample, "middleware_session_empty")

    sample.tokens = prompt_ids + response_ids
    sample.response_length = len(response_ids)
    sample.loss_mask = loss_mask
    sample.response = state.tokenizer.decode(response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        "instance_id": md["instance_id"],
        "is_solved": bool(is_solved),
        "applied_cleanly": bool(applied_cleanly),
        "elapsed_sec": time.time() - t0,
    }
    if SWE_SAVE_TRAJECTORY_TREE:
        sample.metadata["trajectory_tree"] = trajectory_tree
    logger.info(
        "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs tokens=%d",
        md["instance_id"], reward, is_solved, applied_cleanly,
        time.time() - t0, len(response_ids),
    )
    return sample


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def _metadata(sample: Sample) -> dict[str, Any]:
    """Normalize the two dataset schemas (flat vs ``remote_env_info``) we see."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    return {
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "swepro": m.get("swepro"),
        "f2p_script": m.get("f2p_script") or rem.get("f2p_script"),
        "eval_cmd": m.get("eval_cmd"),
        "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
    }


def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(p.get("text", "") for p in c
                                     if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _abort(sample: Sample, reason: str) -> Sample:
    # Dummy 1-prompt + 1-response token so megatron's get_batch can compute
    # `prompt_length = total_length - response_length >= 1` (else F.pad with
    # negative length crashes the entire training job — see
    # megatron_utils/data.py:141). loss_mask=[0] means this sample
    # contributes no gradient, which is what we want for an abort.
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[coding_agent_rl] aborted: %s", reason)
    return sample
