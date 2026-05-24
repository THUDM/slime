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
       middleware (D4 default = list mode = one Sample per chain segment).
    8. Fan out the rollout reward across segments via the configured reducer
       (default uniform reward/K; user override via --swe-segment-reducer-path).
       Failure is fail-soft (U4): sample marked abort, never blocks step.

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
    SWE_MAX_RESPONSE_TOKENS  0     optional smoke-test cap before training (0 = off)
    SWE_TOOL_PARSER          glm47           (sglang FunctionCallParser name)
    SWE_REASONING_PARSER     glm45           (sglang ReasoningParser name)
    SWE_DUMP_RAW_TRAJECTORY  0               1 = attach trajectory_raw_dump to seg 0
    SHIM_BIND_HOST           0.0.0.0
    SHIM_PORT                18001
    SLIME_HEAD_HOST          public host the sandboxes use to reach the middleware

CLI args:

    --swe-segment-reducer-path  dotted.path.to.reducer (overrides default uniform)
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import logging
import os
import secrets
import socket
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import middleware, sandbox

try:
    from slime.rollout.sglang_rollout import GenerateState as _SlimeGenerateState
except Exception:  # pragma: no cover
    _SlimeGenerateState = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
# Wall-clock guard for the entire generate() call. Defaults to
# SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180 (buffer for sandbox boot,
# diff capture, etc). When exceeded, the in-flight sample is aborted with
# reason `wall_clock_timeout` and the rest of the rollout continues -- this
# replaces the external builtin TimeoutError observed in run
# planD_e2_pr1933_fanout_20260524_073011 (see r5 doc) that killed the whole
# step when a single trajectory hung in sandbox.evaluate.
SWE_GENERATE_GUARD_SEC = int(
    os.environ.get("SWE_GENERATE_GUARD_SEC", "0") or 0
) or (SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180)
SWE_MAX_RESPONSE_TOKENS = int(os.environ.get("SWE_MAX_RESPONSE_TOKENS", "0") or 0)
# SWE_LIST_TRAJECTORY: 0 (default) = collapse segments into 1 Sample
# (main-repo behavior; avoids fan-out sample-count explosion that triggers
# host pinned-memory pressure and GPU wake_up OOM). Single-sample mode
# uses the FINAL segment (reward-bearing segment, post-final-compact-reset)
# as the trajectory tokens. 1 = enable fan-out (one Sample per segment).
SWE_LIST_TRAJECTORY = os.environ.get("SWE_LIST_TRAJECTORY", "0") == "1"
SWE_TOOL_PARSER = os.environ.get("SWE_TOOL_PARSER", "") or None
SWE_REASONING_PARSER = os.environ.get("SWE_REASONING_PARSER", "") or None
# Q5 / SPEC §6.2: renamed from SWE_SAVE_TRAJECTORY_TREE; semantics narrowed
# to "attach trajectory_raw_dump to segment 0 metadata".
SWE_DUMP_RAW_TRAJECTORY = os.environ.get("SWE_DUMP_RAW_TRAJECTORY", "0") == "1"
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
        self.segment_reducer: Callable = _load_reducer(args)
        if _SlimeGenerateState is not None:
            try:
                slime_state = _SlimeGenerateState(args)
                self.middleware.install_abort_poll(
                    lambda s=slime_state: bool(getattr(s, "aborted", False)),
                    interval_sec=float(os.environ.get("SWE_ABORT_POLL_INTERVAL", "0.5")),
                    max_wait_sec=float(os.environ.get("SWE_ABORT_MAX_WAIT_SEC", "1800")),
                )
                logger.info("[coding_agent_rl] abort-poll wired to GenerateState.aborted")
            except Exception as e:
                logger.warning("[coding_agent_rl] could not wire abort-poll: %s", e)
        logger.info(
            "[coding_agent_rl] tokenizer=%s middleware=%s reducer=%s",
            args.hf_checkpoint, self.middleware.public_url,
            getattr(self.segment_reducer, "__qualname__", repr(self.segment_reducer)),
        )


# ---------------------------------------------------------------------------
# Sandbox provisioning
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _provision_sandbox(image: str):
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
            await asyncio.sleep(1 + attempt)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Segment fan-out (D4 / U4: default uniform + fail-soft reducer wrapper)
# ---------------------------------------------------------------------------
def _collapse_to_final_segment(
    sample: Sample,
    segments: list[tuple[list[int], list[int], list[int], dict]],
    reward: float,
    tokenizer,
) -> Sample:
    """Stage 14 OOM fix: mutate `sample` to carry only the FINAL segment
    (reward-bearing post-final-compact-reset segment).

    Activated by ``SWE_LIST_TRAJECTORY=0`` (default). Avoids fan-out
    sample-count explosion (16 -> ~80) that bloats ray.put + host pinned
    mem and triggers GPU wake_up OOM. See OOM root-cause analysis
    2026-05-24 in stage14_oom_rootcause_fanout.md.

    The middle K-1 segments are intentionally dropped here. Long-term
    alternative is per-segment fan-out + PR #1933 per-rollout reducer;
    that is the Plan-D long-arm. This collapse is the short-arm.
    """
    prompt_ids, response_ids, loss_mask, seg_meta = segments[-1]
    sample.tokens = list(prompt_ids) + list(response_ids)
    sample.response_length = len(response_ids)
    sample.loss_mask = list(loss_mask)
    sample.response = tokenizer.decode(response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        **seg_meta,
        "num_segments_collapsed": len(segments),
    }
    return sample


def _default_uniform_fan_out(
    segments: list[tuple[list[int], list[int], list[int], dict]],
    reward: float,
    sample_proto: Sample,
    tokenizer,
    instance_id: str,
) -> list[Sample]:
    """Default reducer (D4 / Q7). Splits reward uniformly: reward/K per
    segment. Returns one Sample per non-empty segment, each carrying the
    SPEC §6.1 metadata fields (segment_kind, finish_reason, num_aborts,
    tito_masked_turns, segment_idx, num_segments, ...)."""
    K = len(segments)
    # PR #1933 port: all K segments from one trajectory must share the same
    # rollout_id so the loss reducer counts the trajectory once (per-rollout
    # mean) instead of K times (per-sample mean). sample_proto.index is the
    # dataset row id used as the per-rollout unique identifier in the default
    # rollout shape; reuse it as rollout_id. This also makes the
    # _validate_rollout_id_annotated check (depth>=2 leaf) happy if/when SWE
    # output is wrapped into a list-of-list-of-sample shape.
    trajectory_rollout_id = getattr(sample_proto, "index", None)
    out: list[Sample] = []
    for i, (prompt_ids, response_ids, loss_mask, seg_meta) in enumerate(segments):
        sub = sample_proto if i == 0 else copy.copy(sample_proto)
        # I1 mirrored at the Sample layer: response_length == len(loss_mask)
        sub.tokens = list(prompt_ids) + list(response_ids)
        sub.response_length = len(response_ids)
        sub.loss_mask = list(loss_mask)
        sub.response = tokenizer.decode(response_ids, skip_special_tokens=False)
        sub.reward = float(reward) / max(1, K)
        sub.status = Sample.Status.COMPLETED
        sub.rollout_id = trajectory_rollout_id
        # Merge segment meta fields (segment_kind, finish_reason, etc.).
        merged: dict[str, Any] = {**(sub.metadata or {}),
                                  "instance_id": instance_id,
                                  **seg_meta,
                                  "segment_idx": i,
                                  "num_segments": K}
        sub.metadata = merged
        out.append(sub)
    return out


def _load_reducer(args) -> Callable:
    """Load the segment reducer with import-time fail-soft (U4)."""
    path = getattr(args, "swe_segment_reducer_path", None)
    if not path:
        return _default_uniform_fan_out
    try:
        module_name, attr = path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), attr)
        if not callable(fn):
            raise TypeError(f"{path} is not callable")
        return fn
    except Exception as e:
        logger.error(
            "[coding_agent_rl] could not import segment reducer %r: %s "
            "- falling back to default uniform", path, e,
        )
        return _default_uniform_fan_out


def _fan_out_with_fail_soft(
    state: "_State",
    segments: list[tuple[list[int], list[int], list[int], dict]],
    reward: float,
    sample_proto: Sample,
    instance_id: str,
) -> list[Sample]:
    """U4 - wrap the reducer so bad reducers don't kill the rollout step.

    4 fail paths:
      1. import path bad      -> _load_reducer logs error + falls back to default
      2. reducer raises       -> log warning + sample abort + metric bump
      3. returns non-list/None -> same as (2)
      4. returns Sample missing required field -> trainer rejects later (not here)
    """
    reducer = state.segment_reducer
    try:
        out = reducer(segments, reward, sample_proto, state.tokenizer, instance_id)
        if not isinstance(out, list) or not out:
            raise ValueError(
                f"reducer returned non-list or empty: {type(out).__name__}"
            )
        return out
    except Exception as e:
        logger.warning(
            "[coding_agent_rl] reducer failed for instance=%s: %s - sample marked abort",
            instance_id, e,
        )
        try:
            from slime.utils.metric_utils import METRICS  # type: ignore
            METRICS.reducer_failure_count.labels(reason=type(e).__name__).inc()
        except Exception:
            pass
        return [_abort(sample_proto, reason=f"reducer_failure:{type(e).__name__}")]


# ---------------------------------------------------------------------------
# Main per-sample agent function
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any]):
    """Wall-clock-guarded wrapper around the inner generate logic. See
    SWE_GENERATE_GUARD_SEC docstring above."""
    t0 = time.time()
    try:
        return await asyncio.wait_for(
            _generate_inner(args, sample, sampling_params),
            timeout=SWE_GENERATE_GUARD_SEC,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        # Diagnostic: dump current pending tasks so future debugging can
        # see which await was stuck. Do not raise; let _abort handle it.
        try:
            pending = [
                t for t in asyncio.all_tasks() if not t.done()
            ]
            stuck_summary = []
            for t in pending[:5]:  # cap to avoid log spam
                coro = getattr(t, "_coro", None)
                name = getattr(coro, "__qualname__", repr(coro))
                stuck_summary.append(name)
            logger.warning(
                "[coding_agent_rl] generate() wall_clock_timeout after %.1fs "
                "(guard=%ds); %d tasks pending; sample of stuck: %s",
                elapsed, SWE_GENERATE_GUARD_SEC, len(pending), stuck_summary,
            )
        except Exception:  # pragma: no cover - diag must never crash
            pass
        return _abort(sample, "wall_clock_timeout")
    # Note: builtin TimeoutError (e.g., from concurrent.futures) was the
    # observed failure mode in run 073011. wait_for raises asyncio.TimeoutError
    # which is a subclass of builtin TimeoutError in Python >= 3.11 (PEP 657
    # made them the same class); the except above catches both. Other
    # exceptions still bubble up so _generate_inner's own try-except can
    # handle them.


async def _generate_inner(args, sample: Sample, sampling_params: dict[str, Any]):
    state = _State(args)
    md = _metadata(sample)
    if not md["image"] or not md["workdir"]:
        return _abort(sample, "missing_image_or_workdir")

    session_id = sample.session_id or f"cagent-{md['instance_id']}-{secrets.token_hex(4)}"
    sample.session_id = session_id
    state.middleware.open_session(
        session_id,
        sampling_defaults=sampling_params,
        record_raw_dump=SWE_DUMP_RAW_TRAJECTORY,
    )

    t0 = time.time()
    diff_text = ""
    try:
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

    # D4 / list mode is always on now (Q1 decision):
    segments, raw_dump = state.middleware.pop_session_split(session_id)
    if not segments:
        return _abort(sample, "middleware_session_empty")
    segments = [seg for seg in segments if seg[1]]
    if not segments:
        return _abort(sample, "middleware_session_empty")

    # Apply per-sample training cap to each segment's response.
    segments = [_cap_segment(seg) for seg in segments]

    instance_id = md["instance_id"]
    elapsed = time.time() - t0
    sample.metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "is_solved": bool(is_solved),
        "applied_cleanly": bool(applied_cleanly),
        "elapsed_sec": elapsed,
    }

    # SWE_LIST_TRAJECTORY=0 (default): collapse to single Sample using the
    # FINAL segment. Avoids fan-out sample-count explosion (16 -> ~80) that
    # bloats ray.put + host pinned mem and triggers GPU wake_up OOM. See
    # OOM root-cause analysis 2026-05-24.
    if not SWE_LIST_TRAJECTORY:
        _collapse_to_final_segment(sample, segments, reward, state.tokenizer)
        if SWE_DUMP_RAW_TRAJECTORY:
            sample.metadata["trajectory_raw_dump"] = raw_dump
        logger.info(
            "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs "
            "single-sample collapsed_segments=%d",
            instance_id, reward, is_solved, applied_cleanly, elapsed, len(segments),
        )
        return sample

    fanned = _fan_out_with_fail_soft(state, segments, reward, sample, instance_id)
    # Attach raw_dump only to segment 0 (avoid duplication).
    if SWE_DUMP_RAW_TRAJECTORY and fanned and isinstance(fanned[0], Sample):
        md0 = fanned[0].metadata or {}
        md0["trajectory_raw_dump"] = raw_dump
        fanned[0].metadata = md0

    logger.info(
        "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs segments=%d",
        instance_id, reward, is_solved, applied_cleanly, elapsed, len(fanned),
    )
    return fanned


def _cap_segment(seg: tuple[list[int], list[int], list[int], dict]
                 ) -> tuple[list[int], list[int], list[int], dict]:
    p, r, m, meta = seg
    if SWE_MAX_RESPONSE_TOKENS <= 0 or len(r) <= SWE_MAX_RESPONSE_TOKENS:
        return seg
    return p, r[:SWE_MAX_RESPONSE_TOKENS], m[:SWE_MAX_RESPONSE_TOKENS], meta


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
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
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[coding_agent_rl] aborted: %s", reason)
    return sample
