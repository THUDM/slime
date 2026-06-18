"""Coding-Agent RL: per-sample generate() function for slime.

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

generate() is a four-stage orchestrator: swe.prepare_workspace + ClaudeCodeHarness.run
-> swe.git_diff -> swe.evaluate -> adapter.finish_session. Sandbox-side work is
split across three layers: the provider-agnostic sandbox contract
(slime.agent.sandbox), the swappable harness lifecycle (slime.agent.harness), and
the SWE task layer (examples.coding_agent_rl.swe -- dataset parsing, workspace
prep, diff, eval). LLM plumbing (Anthropic <-> SGLang /generate, token capture,
segment split) is slime.agent.adapters.AnthropicAdapter. swe.get_metadata documents
the dataset row schema and produces the md dict consumed below.
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
from dataclasses import dataclass
from typing import Any

from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import ClaudeCodeHarness
from slime.agent.sandbox import E2BSandbox
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import swe

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)


@dataclass(frozen=True)
class SweConfig:
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int

    @classmethod
    def from_env(cls) -> SweConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        guard = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (agent_time_budget + eval_timeout + 180)
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SWE_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("SWE_BOOT_RETRIES", "2")),
        )


CONFIG = SweConfig.from_env()

_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)


@asynccontextmanager
async def boot_agent_sandbox(image: str, instance_id: str) -> AsyncIterator[E2BSandbox]:
    """Boot a fresh E2B sandbox and install the Claude Code toolchain.

    Create the sandbox from the dataset image, install Node 22 + the harness CLI
    from host tarballs, retry transient boot/install failures, and close the
    sandbox when the caller leaves the context.
    """
    sb = None
    last_err: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        cand = E2BSandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await ClaudeCodeHarness().install_cli(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[coding_agent_rl] %s: provision attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                CONFIG.boot_retries,
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


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        if not CONFIG.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. Export it to the host IP that "
                "sandboxes can reach for reverse-connection to the adapter; "
                "without it the sandbox cannot dial back and the rollout aborts."
            )
        self.adapter = AnthropicAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request in the adapter.
        # Otherwise a cancelled client leaves an inflight sglang /generate that
        # races the next release_memory_occupation and trips its idle assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="anthropic-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = f"http://{CONFIG.adapter_public_host}:{self.app_handle.port}"
        logger.info(
            "[coding_agent_rl] tokenizer=%s adapter=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any]):
    """Per-sample agent function with wall-clock guard (see rollout_guard_sec)."""
    state = _AdapterService(args)
    md = swe.get_metadata(base_sample)
    instance_id = md["instance_id"]
    if not md["image"] or not md["workdir"]:
        return _abort_result(base_sample, "missing_image_or_workdir", instance_id)

    session_id = base_sample.session_id = _session_id(base_sample, instance_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    t0 = time.time()
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_agent_sandbox(md["image"], instance_id) as sb:
                await swe.prepare_workspace(sb, md["workdir"], md)
                agent_exit_code = await ClaudeCodeHarness().run(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=CONFIG.agent_time_budget_sec,
                    prompt=swe.SWE_PROMPT,
                )
                diff_text = await swe.git_diff(sb, md["workdir"])

            reward, applied_cleanly = await swe.evaluate(
                image=md["image"],
                workdir=md["workdir"],
                diff_text=diff_text,
                swepro=md["swepro"],
                eval_cmd=md["eval_cmd"],
                f2p_script=md["f2p_script"],
                pre_commands=md["pre_commands"],
                timeout_sec=CONFIG.eval_timeout_sec,
            )
            samples = await state.adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=float(reward),
            )
            if not samples:
                return _abort_result(base_sample, "adapter_session_empty", instance_id)

            for s in samples:
                s.metadata = {**(s.metadata or {}), "agent_exit_code": agent_exit_code}
            if agent_exit_code != 0:
                reason = "time budget exceeded" if agent_exit_code < 0 else f"CLI error (exit {agent_exit_code})"
                logger.warning(
                    "[coding_agent_rl] %s: agent_exit_code=%d (%s)",
                    instance_id,
                    agent_exit_code,
                    reason,
                )
            logger.info(
                "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs segments=%d",
                instance_id,
                float(reward),
                bool(applied_cleanly),
                agent_exit_code,
                time.time() - t0,
                len(samples),
            )
            return samples

    except asyncio.TimeoutError:
        _log_timeout_diagnostic(t0, instance_id)
        return _abort_result(base_sample, "wall_clock_timeout", instance_id)
    except Exception as e:
        logger.warning(
            "[coding_agent_rl] %s: rollout failed: %s\n%s",
            instance_id,
            e,
            traceback.format_exc(),
        )
        return _abort_result(base_sample, f"exception:{type(e).__name__}", instance_id)
    finally:
        await state.adapter.finish_session(session_id)  # idempotent


def _log_timeout_diagnostic(t0: float, instance_id: str) -> None:
    # Dump pending-task names when the wall-clock guard fires. Must not crash.
    try:
        elapsed = time.time() - t0
        pending = [t for t in asyncio.all_tasks() if not t.done()]
        stuck = []
        for t in pending[:5]:  # cap to avoid log spam
            coro = getattr(t, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[coding_agent_rl] %s: wall_clock_timeout after %.1fs "
            "(guard=%ds); %d tasks pending; sample of stuck: %s",
            instance_id,
            elapsed,
            CONFIG.rollout_guard_sec,
            len(pending),
            stuck,
        )
    except Exception:  # pragma: no cover - diag must never crash
        pass


def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"cagent-{instance_id}-{sample.index}-{sample.group_index}"
    return f"cagent-{instance_id}-{secrets.token_hex(8)}"


def _abort_result(sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    """Mark ``sample`` aborted in place and return it in the list shape this
    fan-out generate function always yields."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[coding_agent_rl] %s aborted: %s", instance_id, reason)
    return [sample]
