"""Coding-Agent RL: per-sample generate() function for slime.

Wire-up:

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

``generate()`` below IS the agent. Read it top-to-bottom to see what one SWE
rollout sample does. All sandbox-side details live in ``sandbox.py``; the LLM
plumbing (Anthropic <-> SGLang /generate, token capture, 3-kind segment split)
lives in ``middleware.py``.

Per-sample steps:

    1. Boot a fresh sandbox from the dataset image.
    2. Install Node 22 + Claude Code CLI.
    3. Create the agent user, drop PROBLEM_STATEMENT.md.
    4. Run claude-code pointed at the head-node middleware (the middleware
       captures tokens by session_id, passed via the Bearer token).
    5. ``git diff`` to capture the model-produced patch.
    6. Boot a SECOND, fresh sandbox; apply diff; run the dataset's tests for
       reward. (No-test-cheating guarantee: reward only depends on the diff.)
    7. Pull (prompt_ids, response_ids, loss_mask, ...) segments from the
       middleware (one segment per chain reset; >=1 per trajectory).
    8. Either collapse to one Sample (final segment, default) or fan out
       one Sample per segment with reward/K. Fan-out is fail-soft: any bug
       aborts THIS sample only, never blocks the training step.

Dataset row ``metadata`` schema::

    image:             str        # sandbox image
    workdir:           str        # repo path inside the sandbox
    problem_statement: str        # issue body (falls back to sample.prompt)
    swepro:            dict|None  # SWE-bench Pro test harness (preferred)
    eval_cmd:          str|None   # last-resort: shell command (exit 0 = solved)

Also accepted (sweb-style rows): ``metadata.remote_env_info.f2p_script`` —
a self-contained Python test file ending in ``sys.exit(pytest.main(...))``.
When ``eval_cmd`` is absent, ``_metadata`` wraps this script into a base64
materialize-and-run shell command so the existing eval path stays unchanged.

Env knobs (set in run.sh):

    SWE_HOST_NODE_TARBALL    host path to a Node 22 tarball (REQUIRED)
    SWE_HOST_CC_TARBALL      host path to the Claude Code npm tarball (REQUIRED)
    SWE_TIME_BUDGET_SEC      1800  per agent run, wallclock
    SWE_EVAL_TIMEOUT_SEC     600   per eval test execution
    SWE_MAX_RESPONSE_TOKENS  0     optional smoke-test cap before training (0 = off)
    SWE_TOOL_PARSER          glm47           (sglang FunctionCallParser name)
    SWE_REASONING_PARSER     glm45           (sglang ReasoningParser name)
    SHIM_BIND_HOST           0.0.0.0
    SHIM_PORT                18001
    SLIME_HEAD_HOST          public host the sandboxes use to reach the middleware (REQUIRED)
"""

from __future__ import annotations

import asyncio
import base64
import copy
import logging
import os
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from slime.agent.sandbox import E2BSandbox
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import middleware, sandbox
from .aiohttp_threaded import run_app_in_thread

logger = logging.getLogger(__name__)


SWE_HOST_NODE_TARBALL = Path(
    os.environ.get(
        "SWE_HOST_NODE_TARBALL",
        "/path/to/node-v22.20.0-linux-x64.tar.xz",
    )
)
SWE_HOST_CC_TARBALL = Path(
    os.environ.get(
        "SWE_HOST_CC_TARBALL",
        "/path/to/anthropic-ai-claude-code.tgz",
    )
)
SWE_TIME_BUDGET_SEC = int(os.environ.get("SWE_TIME_BUDGET_SEC", "1800"))
SWE_EVAL_TIMEOUT_SEC = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
# Wall-clock guard for the entire generate() call. Defaults to
# SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180 (buffer for sandbox boot,
# diff capture, etc). When exceeded, the in-flight sample is aborted with
# reason `wall_clock_timeout` and the rest of the rollout continues -- this
# isolates a single hung trajectory (e.g. stuck in sandbox.evaluate) so it
# does not kill the whole training step.
SWE_GENERATE_GUARD_SEC = int(os.environ.get("SWE_GENERATE_GUARD_SEC", "0") or 0) or (
    SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180
)
SWE_MAX_RESPONSE_TOKENS = int(os.environ.get("SWE_MAX_RESPONSE_TOKENS", "0") or 0)
# SWE_LIST_TRAJECTORY: 0 (default) = collapse segments into 1 Sample
# (avoids fan-out sample-count explosion that triggers host pinned-memory
# pressure and GPU wake_up OOM). Single-sample mode uses the FINAL segment
# (reward-bearing segment, post-final-compact-reset) as the trajectory
# tokens. 1 = enable fan-out (one Sample per segment).
SWE_LIST_TRAJECTORY = os.environ.get("SWE_LIST_TRAJECTORY", "0") == "1"
SWE_TOOL_PARSER = os.environ.get("SWE_TOOL_PARSER", "") or None
SWE_REASONING_PARSER = os.environ.get("SWE_REASONING_PARSER", "") or None
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18001"))

SWE_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
SWE_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))
_BOOT_SEM: asyncio.Semaphore | None = None

CC_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)


# ---------------------------------------------------------------------------
# Singleton: tokenizer + in-process middleware handle + reducer
# ---------------------------------------------------------------------------
class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        public_host = os.environ.get("SLIME_HEAD_HOST")
        if not public_host:
            raise RuntimeError(
                "SLIME_HEAD_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for reverse-connection to the middleware. "
                "Without it the sandbox cannot dial back and the rollout will "
                "silently abort."
            )
        app, self.store = middleware.start(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=SWE_TOOL_PARSER,
            reasoning_parser=SWE_REASONING_PARSER,
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request inside the
        # middleware. Without it a cancelled client leaves an inflight sglang
        # /generate that races with the next release_memory_occupation and
        # trips sglang's "server is idle" assertion.
        self.app_handle = run_app_in_thread(
            app,
            host=SHIM_BIND_HOST,
            port=SHIM_PORT,
            thread_name="anthropic-middleware",
            runner_kwargs={"handler_cancellation": True},
        )
        self.middleware_url = f"http://{public_host}:{self.app_handle.port}"
        logger.info(
            "[coding_agent_rl] tokenizer=%s middleware=%s",
            args.hf_checkpoint,
            self.middleware_url,
        )


# ---------------------------------------------------------------------------
# Sandbox boot + agent toolchain install
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _boot_agent_sandbox(image: str):
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)

    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = E2BSandbox(image)
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
                attempt + 1,
                SWE_BOOT_RETRIES,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Segment -> Sample conversion
#
# A "segment" is (prompt_ids, response_ids, loss_mask, seg_meta) produced by
# middleware.pop_session_split(). One trajectory yields >=1 segments because
# the agent may compact + reset mid-run.
#
# The collapse path (SWE_LIST_TRAJECTORY=0, default) is inlined in generate()
# because it is only 4 lines. The fan-out path (SWE_LIST_TRAJECTORY=1) lives
# in _fan_out_to_samples below because it has subtle metadata-sharing logic.
# ---------------------------------------------------------------------------
Segment = tuple[list[int], list[int], list[int], dict]


def _write_segment_to_sample(sample: Sample, seg: Segment, reward: float, tokenizer) -> None:
    """Populate the token / loss_mask / response / reward fields of `sample`
    from one segment. Shared by both the collapse and fan-out paths."""
    prompt_ids, response_ids, loss_mask, _ = seg
    sample.tokens = list(prompt_ids) + list(response_ids)
    sample.response_length = len(response_ids)
    sample.loss_mask = list(loss_mask)
    sample.response = tokenizer.decode(response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED


def _fan_out_to_samples(
    sample: Sample,
    segments: list[Segment],
    reward: float,
    tokenizer,
    instance_id: str,
) -> list[Sample]:
    """SWE_LIST_TRAJECTORY=1 path. Emit one Sample per segment, splitting the
    rollout reward uniformly (reward/K per segment).

    All K samples share the same `rollout_id` so the loss reducer counts
    this trajectory once (per-rollout mean) instead of K times
    (per-sample mean). The dataset row id (`sample.index`) is reused as the
    rollout_id.

    The first segment reuses the input `sample` object; later ones get a
    shallow copy -- avoids a copy in the common single-segment case."""
    K = len(segments)
    per_segment_reward = float(reward) / max(1, K)
    rollout_id = getattr(sample, "index", None)

    out: list[Sample] = []
    for i, seg in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        _write_segment_to_sample(sub, seg, per_segment_reward, tokenizer)
        sub.rollout_id = rollout_id
        sub.metadata = {
            **(sub.metadata or {}),
            "instance_id": instance_id,
            **seg[3],
            "segment_idx": i,
            "num_segments": K,
        }
        out.append(sub)
    return out


# ---------------------------------------------------------------------------
# Main per-sample agent function
#
# Read top-to-bottom. The [N] section comments below correspond to the 8-step
# recipe in the module docstring at the top of this file.
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any]):
    """Per-sample agent function with wall-clock guard. See
    SWE_GENERATE_GUARD_SEC docstring above."""
    state = _State(args)
    md = _metadata(sample)
    if not md["image"] or not md["workdir"]:
        return _abort_result(sample, "missing_image_or_workdir")

    # [1] Open a middleware session. claude-code inside the sandbox dials
    #     back to the middleware with this session_id (passed as the Bearer
    #     token) so its turns are grouped under one chain history.
    #     Build sid from (instance_id, index, group_index) so it's unique by
    #     construction within a rollout step; fall back to random hex if either
    #     index is missing. open_session raises on duplicate as a safety net.
    if sample.session_id:
        session_id = sample.session_id
    elif sample.index is not None and sample.group_index is not None:
        session_id = f"cagent-{md['instance_id']}-{sample.index}-{sample.group_index}"
    else:
        session_id = f"cagent-{md['instance_id']}-{secrets.token_hex(8)}"
    sample.session_id = session_id
    middleware.open_session(state.store, session_id, sampling_defaults=sampling_params)

    instance_id = md["instance_id"]
    t0 = time.time()
    try:
        async with asyncio.timeout(SWE_GENERATE_GUARD_SEC):
            # [2-3] Boot a fresh sandbox, install Node + Claude Code, create
            #       the agent user, drop PROBLEM_STATEMENT.md, run the agent,
            #       capture the resulting git diff.
            async with _boot_agent_sandbox(md["image"]) as sb:
                await sandbox.ensure_agent_user(sb, md["workdir"])
                if md["swepro"]:
                    await sandbox.apply_before_repo_set_cmd(sb, md["workdir"], md["swepro"])
                if md["pre_commands"]:
                    await sandbox.apply_pre_commands(sb, md["workdir"], md["pre_commands"])
                await sb.write_file(
                    f"{md['workdir']}/PROBLEM_STATEMENT.md",
                    md["problem_statement"] or "",
                    user="agent",
                )
                await sandbox.run_claude_code(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    middleware_url=state.middleware_url,
                    prompt=CC_PROMPT,
                    time_budget_sec=SWE_TIME_BUDGET_SEC,
                )
                diff_text = await sandbox.git_diff(sb, md["workdir"])

            # [4] Second fresh sandbox runs the dataset's tests against the
            #     captured diff. No-test-cheating guarantee: reward depends
            #     only on the diff, never on what the agent sandbox did.
            reward, is_solved, applied_cleanly = await sandbox.evaluate(
                image=md["image"],
                workdir=md["workdir"],
                diff_text=diff_text,
                swepro=md["swepro"],
                eval_cmd=md["eval_cmd"],
                pre_commands=md["pre_commands"],
                timeout_sec=SWE_EVAL_TIMEOUT_SEC,
            )

            # [5] Pull (prompt_ids, response_ids, loss_mask, seg_meta)
            #     segments from the middleware. Drop empty-response segments
            #     and apply the optional per-segment training cap in one go
            #     (SWE_MAX_RESPONSE_TOKENS=0 disables the cap).
            cap = SWE_MAX_RESPONSE_TOKENS
            segments: list[Segment] = [
                (p, r[:cap], m[:cap], meta) if cap and len(r) > cap else (p, r, m, meta)
                for (p, r, m, meta) in (middleware.pop_session_split(state.store, session_id) or [])
                if r
            ]
            if not segments:
                return _abort_result(sample, "middleware_session_empty")

            # [6] Top-level metadata that every output Sample will inherit.
            elapsed = time.time() - t0
            sample.metadata = {
                **(sample.metadata or {}),
                "instance_id": instance_id,
                "is_solved": bool(is_solved),
                "applied_cleanly": bool(applied_cleanly),
                "elapsed_sec": elapsed,
            }

            # [7] segments -> Sample(s). Two modes:
            #
            #   collapse (SWE_LIST_TRAJECTORY=0, default): keep ONLY the
            #     final (reward-bearing, post-final-compact-reset) segment
            #     as a single Sample. The middle K-1 segments are dropped
            #     intentionally -- avoids the sample-count explosion that
            #     bloats ray.put + host pinned memory and can trigger GPU
            #     wake_up OOM at large batch sizes.
            #
            #   fan-out (SWE_LIST_TRAJECTORY=1): emit one Sample per
            #     segment, splitting reward uniformly. All K share the
            #     same rollout_id so the loss reducer counts the
            #     trajectory once. Fail-soft: any bug here aborts THIS
            #     sample only, never the whole training step.
            if not SWE_LIST_TRAJECTORY:
                final_seg = segments[-1]
                _write_segment_to_sample(sample, final_seg, reward, state.tokenizer)
                sample.metadata = {
                    **sample.metadata,
                    **final_seg[3],
                    "num_segments_collapsed": len(segments),
                }
                logger.info(
                    "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs "
                    "single-sample collapsed_segments=%d",
                    instance_id,
                    reward,
                    is_solved,
                    applied_cleanly,
                    elapsed,
                    len(segments),
                )
                return sample

            try:
                fanned = _fan_out_to_samples(
                    sample,
                    segments,
                    reward,
                    state.tokenizer,
                    instance_id,
                )
                if not fanned:
                    raise ValueError("fan-out produced no samples")
            except Exception as e:
                logger.warning(
                    "[coding_agent_rl] fan-out failed for instance=%s: %s -- sample aborted",
                    instance_id,
                    e,
                )
                return [_abort(sample, reason=f"reducer_failure:{type(e).__name__}")]
            logger.info(
                "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs segments=%d",
                instance_id,
                reward,
                is_solved,
                applied_cleanly,
                elapsed,
                len(fanned),
            )
            return fanned

    except asyncio.TimeoutError:
        _log_timeout_diagnostic(t0)
        return _abort_result(sample, "wall_clock_timeout")
    except Exception as e:
        logger.error(
            "[coding_agent_rl] %s: rollout failed: %s\n%s",
            instance_id,
            e,
            traceback.format_exc(),
        )
        return _abort_result(sample, f"exception:{type(e).__name__}")


def _log_timeout_diagnostic(t0: float) -> None:
    """Dump pending-task names when the wall-clock guard fires so future
    debugging can see which await was stuck. Must never crash."""
    try:
        elapsed = time.time() - t0
        pending = [t for t in asyncio.all_tasks() if not t.done()]
        stuck = []
        for t in pending[:5]:  # cap to avoid log spam
            coro = getattr(t, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[coding_agent_rl] generate() wall_clock_timeout after %.1fs "
            "(guard=%ds); %d tasks pending; sample of stuck: %s",
            elapsed,
            SWE_GENERATE_GUARD_SEC,
            len(pending),
            stuck,
        )
    except Exception:  # pragma: no cover - diag must never crash
        pass


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def _wrap_f2p_script(script: str | None) -> str | None:
    # Materialize a self-contained pytest script (typical sweb f2p_script:
    # ends with `sys.exit(pytest.main([...]))`) into the sandbox via base64
    # so we sidestep all shell quoting; python's exit code carries the
    # pytest pass/fail signal that `_run_eval_cmd` turns into reward.
    if not script:
        return None
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return f"echo {b64} | base64 -d > /tmp/slime_f2p.py && python /tmp/slime_f2p.py"


def _metadata(sample: Sample) -> dict[str, Any]:
    """Normalize the two dataset schemas (flat vs ``remote_env_info``)."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    return {
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "swepro": m.get("swepro"),
        "eval_cmd": m.get("eval_cmd") or _wrap_f2p_script(rem.get("f2p_script")),
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
                    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _abort(sample: Sample, reason: str) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[coding_agent_rl] aborted: %s", reason)
    return sample


def _abort_result(sample: Sample, reason: str):
    """Return abort result matching the active fan-out mode so the framework
    sees a uniform shape (`list[Sample]` when fan-out is on, bare `Sample`
    otherwise). Mixing the two breaks `_get_rollout_data`'s flatten loop."""
    s = _abort(sample, reason)
    return [s] if SWE_LIST_TRAJECTORY else s
