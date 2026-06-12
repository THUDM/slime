"""Coding-Agent RL: per-sample generate() function for slime.

Wire-up:

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

``generate()`` is intentionally a small four-stage orchestrator:

    1. ``swe.prepare_workspace`` + ``CLAUDE_CODE.run`` prepare the agent sandbox
       and run claude-code.
    2. ``swe.git_diff`` captures the model-produced patch.
    3. ``swe.evaluate`` scores that patch in a second clean sandbox.
    4. ``adapter.finish_session`` drains the session tree into reward-weighted
       ``Sample`` objects with ``.response`` already decoded; ``generate`` logs.

Sandbox-side details split across three layers: the provider-agnostic sandbox
contract (``slime.agent.sandbox``), the harness lifecycle
(``slime.agent.harness`` -- swappable coding agent), and the SWE task layer
(``examples.coding_agent_rl.swe`` -- dataset-row parsing, workspace prep, diff
capture, eval). The LLM plumbing (Anthropic <-> SGLang /generate, token capture,
3-kind segment split) uses ``slime.agent.adapters.AnthropicAdapter``.

The dataset row ``metadata`` schema (and the two accepted shapes: flat vs
``remote_env_info``) is documented in ``swe.metadata``, which produces the ``md``
dict the orchestration below consumes.

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
import logging
import os
import secrets
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import CLAUDE_CODE
from slime.agent.sandbox import E2BSandbox
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import swe

logger = logging.getLogger(__name__)


SWE_TIME_BUDGET_SEC = int(os.environ.get("SWE_TIME_BUDGET_SEC", "1800"))
SWE_EVAL_TIMEOUT_SEC = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
# Wall-clock guard for the entire generate() call. Defaults to
# SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180 (buffer for sandbox boot,
# diff capture, etc). When exceeded, the in-flight sample is aborted with
# reason `wall_clock_timeout` and the rest of the rollout continues -- this
# isolates a single hung trajectory (e.g. stuck in swe.evaluate) so it
# does not kill the whole training step.
SWE_GENERATE_GUARD_SEC = int(os.environ.get("SWE_GENERATE_GUARD_SEC", "0") or 0) or (
    SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180
)
SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18001"))

# Boot tuning. The Node 22 + CLI tarball host paths (SWE_HOST_NODE_TARBALL /
# SWE_HOST_CC_TARBALL, REQUIRED env in run.sh) are read inside
# CLAUDE_CODE.install_cli, not here.
SWE_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
SWE_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))

_BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)


@asynccontextmanager
async def boot_agent_sandbox(image: str) -> AsyncIterator[E2BSandbox]:
    """Boot a fresh E2B sandbox and install the Claude Code toolchain.

    Create the sandbox from the dataset image, install Node 22 + the harness CLI
    from host tarballs, retry transient boot/install failures, and close the
    sandbox when the caller leaves the context.
    """
    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = E2BSandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await CLAUDE_CODE.install_cli(cand)
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
        fork_merge_threshold = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        self.adapter = AnthropicAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=fork_merge_threshold,
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
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
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
# Session setup
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main per-sample agent function
#
# The four calls inside the timeout are the high-level rollout recipe:
# swe.prepare_workspace + CLAUDE_CODE.run -> swe.git_diff -> swe.evaluate
# -> finish_session.
# ---------------------------------------------------------------------------
async def generate(args, sample: Sample, sampling_params: dict[str, Any]):
    """Per-sample agent function with wall-clock guard. See
    SWE_GENERATE_GUARD_SEC docstring above."""
    state = _State(args)
    md = swe.metadata(sample)
    if not md["image"] or not md["workdir"]:
        return _abort_result(sample, "missing_image_or_workdir")

    instance_id = md["instance_id"]
    session_id = _start_session(state, sample, md, sampling_params)
    t0 = time.time()
    try:
        async with asyncio.timeout(SWE_GENERATE_GUARD_SEC):
            async with boot_agent_sandbox(md["image"]) as sb:
                await swe.prepare_workspace(sb, md["workdir"], md)
                agent_rc = await CLAUDE_CODE.run(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=SWE_TIME_BUDGET_SEC,
                    prompt=swe.SWE_PROMPT,
                )
                diff_text = await swe.git_diff(sb, md["workdir"])

            reward, is_solved, applied_cleanly = await swe.evaluate(
                image=md["image"],
                workdir=md["workdir"],
                diff_text=diff_text,
                swepro=md["swepro"],
                eval_cmd=md["eval_cmd"],
                f2p_script=md["f2p_script"],
                pre_commands=md["pre_commands"],
                timeout_sec=SWE_EVAL_TIMEOUT_SEC,
            )
            samples = await state.adapter.finish_session(
                session_id,
                base_sample=sample,
                reward=float(reward),
            )
            if not samples:
                return _abort_result(sample, "adapter_session_empty")

            # finish_session already linearized, reward-weighted and decoded
            # each segment's .response; here we only log a summary. agent_rc is
            # the harness exit code (0=clean, -2=time budget exceeded, -1=marker
            # parse fail, else CLI crash) -- kept on metadata so a reward=0 run
            # can be triaged into "ran but unsolved" vs "timed out" vs "crashed".
            for s in samples:
                s.metadata = {**(s.metadata or {}), "agent_rc": agent_rc}
            logger.info(
                "[coding_agent_rl] %s: reward=%.2f solved=%s applied=%s agent_rc=%d elapsed=%.1fs segments=%d",
                instance_id,
                float(reward),
                bool(is_solved),
                bool(applied_cleanly),
                agent_rc,
                time.time() - t0,
                len(samples),
            )
            return samples

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


def _abort_result(sample: Sample, reason: str) -> list[Sample]:
    """Mark ``sample`` aborted in place and return it in the list shape this
    fan-out generate function always yields."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[coding_agent_rl] aborted: %s", reason)
    return [sample]
