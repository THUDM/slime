"""Coding-Agent RL: per-sample generate() function for slime.

Wire-up:

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

``generate()`` is intentionally a small four-stage orchestrator:

    1. ``sandbox.run_claude_code`` prepares the agent sandbox and runs claude-code.
    2. ``sandbox.git_diff`` captures the model-produced patch.
    3. ``sandbox.evaluate`` scores that patch in a second clean sandbox.
    4. ``_merge_samples`` combines reward + adapter ``TokenSegment``s,
       delegating segment-to-``Sample`` fan-out to ``slime.agent.trajectory``.

All sandbox-side details live in ``sandbox.py``; the LLM plumbing
(Anthropic <-> SGLang /generate, token capture, 3-kind segment split) uses
``slime.agent.adapters.AnthropicAdapter``.

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
    SHIM_BIND_HOST           0.0.0.0
    SHIM_PORT                18001
    SLIME_HEAD_HOST          public host the sandboxes use to reach the adapter (REQUIRED)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import time
import traceback
from dataclasses import dataclass
from typing import Any

from slime.agent.adapters import AnthropicAdapter
from slime.agent.trajectory import TokenSegment, fan_out_sample_segments
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import sandbox
from .aiohttp_threaded import run_app_in_thread

logger = logging.getLogger(__name__)


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
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18001"))


# ---------------------------------------------------------------------------
# Singleton: tokenizer + in-process Anthropic adapter + reducer
# ---------------------------------------------------------------------------
class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        public_host = os.environ.get("SLIME_HEAD_HOST")
        if not public_host:
            raise RuntimeError(
                "SLIME_HEAD_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for reverse-connection to the Anthropic adapter. "
                "Without it the sandbox cannot dial back and the rollout will "
                "silently abort."
            )
        self.adapter = AnthropicAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request inside the
        # adapter. Without it a cancelled client leaves an inflight sglang
        # /generate that races with the next release_memory_occupation and
        # trips sglang's "server is idle" assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=SHIM_BIND_HOST,
            port=SHIM_PORT,
            thread_name="anthropic-adapter",
            runner_kwargs={"handler_cancellation": True},
        )
        self.adapter_url = f"http://{public_host}:{self.app_handle.port}"
        logger.info(
            "[coding_agent_rl] tokenizer=%s adapter=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


# ---------------------------------------------------------------------------
# Trajectory -> Sample conversion
# adapter.finish_session() returns TokenSegments. One trajectory yields >=1
# segments because the agent may compact + reset mid-run; trajectory.py handles
# the mechanical segment -> Sample fan-out.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardResult:
    reward: float
    is_solved: bool
    applied_cleanly: bool


def _start_session(
    state: _State,
    sample: Sample,
    md: dict[str, Any],
    sampling_params: dict[str, Any],
) -> str:
    # claude-code inside the sandbox dials back to the adapter with this
    # session_id (passed as the Bearer token) so its turns are grouped under
    # one chain history. Build from (instance_id, index, group_index) when
    # possible; fall back to random hex if either index is missing.
    if sample.session_id:
        session_id = sample.session_id
    elif sample.index is not None and sample.group_index is not None:
        session_id = f"cagent-{md['instance_id']}-{sample.index}-{sample.group_index}"
    else:
        session_id = f"cagent-{md['instance_id']}-{secrets.token_hex(8)}"
    sample.session_id = session_id
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    return session_id


def _merge_samples(
    *,
    sample: Sample,
    state: _State,
    segments: list[TokenSegment],
    reward_result: RewardResult,
    elapsed_sec: float,
    instance_id: str,
):
    if not segments:
        return _abort_result(sample, "adapter_session_empty")

    trajectory_metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "is_solved": reward_result.is_solved,
        "applied_cleanly": reward_result.applied_cleanly,
        "elapsed_sec": elapsed_sec,
    }

    # All K samples share group_id so the loss reducer counts this trajectory once.
    fanned = fan_out_sample_segments(
        sample,
        segments,
        reward_result.reward,
        state.tokenizer,
        metadata=trajectory_metadata,
    )
    if not fanned:
        raise ValueError("fan-out produced no samples")

    logger.info(
        "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s elapsed=%.1fs segments=%d",
        instance_id,
        reward_result.reward,
        reward_result.is_solved,
        reward_result.applied_cleanly,
        elapsed_sec,
        len(fanned),
    )
    return fanned


# ---------------------------------------------------------------------------
# Main per-sample agent function
#
# The four calls inside the timeout are the high-level rollout recipe:
# run_claude_code -> git_diff -> sandbox.evaluate -> merge_samples.
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any]):
    """Per-sample agent function with wall-clock guard. See
    SWE_GENERATE_GUARD_SEC docstring above."""
    state = _State(args)
    md = _metadata(sample)
    if not md["image"] or not md["workdir"]:
        return _abort_result(sample, "missing_image_or_workdir")

    instance_id = md["instance_id"]
    session_id = _start_session(state, sample, md, sampling_params)
    t0 = time.time()
    try:
        async with asyncio.timeout(SWE_GENERATE_GUARD_SEC):
            async with sandbox.boot_agent_sandbox(md["image"]) as sb:
                await sandbox.run_claude_code(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=SWE_TIME_BUDGET_SEC,
                    problem_statement=md["problem_statement"],
                    swepro=md["swepro"],
                    pre_commands=md["pre_commands"],
                )
                diff_text = await sandbox.git_diff(sb, md["workdir"])

            reward, is_solved, applied_cleanly = await sandbox.evaluate(
                image=md["image"],
                workdir=md["workdir"],
                diff_text=diff_text,
                swepro=md["swepro"],
                eval_cmd=md["eval_cmd"],
                pre_commands=md["pre_commands"],
                timeout_sec=SWE_EVAL_TIMEOUT_SEC,
            )
            reward_result = RewardResult(
                reward=float(reward),
                is_solved=bool(is_solved),
                applied_cleanly=bool(applied_cleanly),
            )
            segments = await state.adapter.finish_session(session_id)
            return _merge_samples(
                sample=sample,
                state=state,
                segments=segments,
                reward_result=reward_result,
                elapsed_sec=time.time() - t0,
                instance_id=instance_id,
            )

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
    finally:
        # Close the sid before next train step's release_memory_occupation;
        # stragglers from this trajectory would otherwise race its idle assert.
        await state.adapter.finish_session(session_id)  # idempotent


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
    """Return a uniform list shape for this fan-out generate function."""
    return [_abort(sample, reason)]
