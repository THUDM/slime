"""CPU-only agent rollout test: the whole pipeline, no GPU / E2B / sglang.

This walks a real agent rollout end to end on CPU. Only four external edges are
faked (see ``tests/test_agent/_fakes.py``): the tokenizer, the E2B sandbox, the
sglang ``/generate`` server, and the agent CLI process inside the sandbox.
Everything between -- ``generate.generate`` orchestration, the in-thread adapter
HTTP app, wire translation, ``record_turn`` / tree building, ``finish_session``
linearization, ``swe`` workspace-prep / diff / eval, the harness lifecycle and
its detached-launch transport -- is the real code.

The "agent" is a coroutine standing in for ``claude -p`` / ``codex exec``: the
sandbox fake invokes it on launch, and it dials the adapter back over real HTTP
loopback (``trust_env=False`` so the cluster proxy can't hijack 127.0.0.1),
firing a couple of turns the way the real CLI would.

The production selector and protocol chains are covered:

  * ``test_generate_*`` -- the production Claude Code path.
  * ``test_one_worker_switches_*`` -- real ``generate.generate()`` selects
    Codex / Responses and Claude Code / Anthropic per sample in one worker.
  * ``test_codex_openai_rollout_closes_loop`` -- the same loop for the
    CodexHarness + OpenAIAdapter pair, hand-wired to cover Chat Completions.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing generate pulls slime.utils.processing_utils at module load, which
# eagerly imports transformers -- a heavy dep deliberately absent from the
# CPU-only CI env for this test. We never touch a real tokenizer (load_tokenizer
# is patched with FakeTokenizer below), so stub transformers before the import
# so the chain resolves without it.
if "transformers" not in sys.modules and importlib.util.find_spec("transformers") is None:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

# generate.generate() uses asyncio.timeout(), a 3.11+ API. CI runs the agent
# tests on 3.10, so shim it onto wait_for. The wall-clock guard never fires in
# these tests (every case finishes well under the guard), so a thin pass-through
# context manager is enough.
if not hasattr(asyncio, "timeout"):

    @contextlib.asynccontextmanager
    async def _timeout_shim(_delay):
        yield

    asyncio.timeout = _timeout_shim

import examples.coding_agent_rl.generate as gen  # noqa: E402
import examples.coding_agent_rl.swe as swe  # noqa: E402
from examples.coding_agent_rl.sunabako_sandbox import SunabakoSandbox  # noqa: E402
from tests.test_agent._fakes import FakeSandbox, FakeTokenizer, fake_call_sglang_generate  # noqa: E402

from slime.agent.adapters import OpenAIAdapter  # noqa: E402
from slime.agent.adapters import common as adapters_common  # noqa: E402
from slime.agent.aiohttp_threaded import run_app_in_thread  # noqa: E402
from slime.agent.harness import ClaudeCodeHarness, CodexHarness  # noqa: E402
from slime.agent.harness import common as harness_common  # noqa: E402
from slime.utils.misc import SingletonMeta  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0

_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):  # collapse run_command's 5s poll loop
    await _REAL_SLEEP(0)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint="unused",  # load_tokenizer is patched
        rollout_max_context_len=0,
        sglang_tool_call_parser=None,
        sglang_reasoning_parser=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,  # never dialed (call_sglang_generate is patched)
    )


def _base_sample(**md) -> Sample:
    meta = {
        "instance_id": "demo-1",
        "image": "fake-image",
        "workdir": "/workspace/repo",
        "problem_statement": "fix the bug",
        "eval_cmd": "true",
        **md,
    }
    return Sample(index=0, group_index=0, prompt="fix the bug", metadata=meta)


async def _anthropic_agent(env: dict, *, n_turns: int = 2, compact_history: bool = False) -> int:
    """Stand-in for ``claude -p``: dial the adapter back over real HTTP and fire
    a few Anthropic turns, reading wiring from the env the harness exported."""
    base_url = env["ANTHROPIC_BASE_URL"]
    token = env["ANTHROPIC_AUTH_TOKEN"]
    history = [{"role": "user", "content": [{"type": "text", "text": "solve the issue"}]}]
    async with aiohttp.ClientSession(trust_env=False) as sess:
        for turn in range(n_turns):
            if turn and compact_history:
                history = [{"role": "user", "content": [{"type": "text", "text": "summary of earlier work"}]}]
            async with sess.post(
                f"{base_url}/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "m", "max_tokens": 64, "messages": history},
            ) as r:
                data = await r.json()
            history.append({"role": "assistant", "content": data["content"]})
            history.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})
    return 0


def _patch_generate(monkeypatch, tokenizer: FakeTokenizer, sandbox_factory) -> None:
    """Wire generate.generate()'s four external edges to CPU fakes."""
    # in-thread adapter must bind to loopback and need no public host check.
    monkeypatch.setattr(
        gen,
        "CONFIG",
        dataclasses.replace(
            gen.CONFIG,
            adapter_public_host="127.0.0.1",
            adapter_bind_host="127.0.0.1",
            adapter_port=0,
            rollout_guard_sec=60,
            agent_time_budget_sec=30,
            eval_timeout_sec=30,
            boot_retries=1,
        ),
    )
    monkeypatch.setattr(gen, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(gen, "create_sandbox", sandbox_factory)  # boot sandbox
    monkeypatch.setattr(swe, "create_sandbox", sandbox_factory)  # eval sandbox
    monkeypatch.setattr(ClaudeCodeHarness, "install_cli", _noop_install)
    monkeypatch.setattr(harness_common.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(
        adapters_common, "call_sglang_generate", fake_call_sglang_generate(_two_turn_script(), tokenizer)
    )
    # _AdapterService is a SingletonMeta singleton; drop any cached instance so
    # each test builds a fresh adapter + app thread.
    SingletonMeta.clear_instances(gen._AdapterService)


async def _noop_install(self, sb) -> None:
    return None


@pytest.mark.parametrize("limit,expected_boots", [(0, 4), (2, 2)])
def test_sandbox_boot_limit_can_be_disabled(monkeypatch, limit, expected_boots):
    async def run_case():
        booted = 0
        cleaned = 0
        installed = asyncio.Event()
        release = asyncio.Event()

        class Sandbox:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                nonlocal cleaned
                cleaned += 1

        class Harness:
            async def install_cli(self, sandbox):
                nonlocal booted
                booted += 1
                if booted == expected_boots:
                    installed.set()
                await release.wait()

        monkeypatch.setattr(gen, "create_sandbox", lambda _: Sandbox())
        monkeypatch.setattr(gen, "_BOOT_SEM", asyncio.Semaphore(limit) if limit else None)

        async def boot():
            async with gen.boot_agent_sandbox("image", "task", harness=Harness()):
                pass

        tasks = [asyncio.create_task(boot()) for _ in range(4)]
        try:
            await asyncio.wait_for(installed.wait(), timeout=2)
            await asyncio.sleep(0)
            assert booted == expected_boots
        finally:
            release.set()
            await asyncio.gather(*tasks)
        assert booted == cleaned == 4

    asyncio.run(run_case())


def test_sandbox_boot_default_is_unlimited(monkeypatch):
    monkeypatch.delenv("SWE_BOOT_CONCURRENCY", raising=False)
    assert gen.SweConfig.from_env().boot_concurrency == 0


def _two_turn_script():
    # (response_text, finish_reason, logprobs) per sglang call; encoded by the
    # FakeTokenizer so the adapter's decode round-trips it.
    return [
        ("let me look at the code", "stop", None),
        ("the fix is applied done", "stop", None),
    ]


# ===========================================================================
# §1 production path: real generate() over ClaudeCode + Anthropic
# ===========================================================================


@pytest.mark.parametrize("compact_history", [False, True])
def test_generate_produces_trained_samples(tmp_path, compact_history):
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        sandbox_factory = FakeSandbox.factory(
            on_launch=lambda env: _anthropic_agent(env, compact_history=compact_history)
        )
        _patch_generate(monkeypatch, tok, sandbox_factory)
        monkeypatch.setattr(gen, "CONFIG", dataclasses.replace(gen.CONFIG, fork_merge_threshold=0))
        monkeypatch.setenv("SWE_TRACE_DIR", str(tmp_path))

        base = _base_sample()
        samples = await gen.generate(_args(), base, sampling_params={"max_new_tokens": 32})

        assert samples, "rollout produced no samples"
        for s in samples:
            assert s.status == Sample.Status.COMPLETED
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0  # at least one trained token
            assert s.metadata.get("agent_exit_code") == 0
            assert s.trace is base.trace
        # Every segment retains the same trajectory outcome. The trace counts
        # that outcome once, including when compaction creates a new segment.
        assert all(s.reward == 1.0 for s in samples)
        assert len(samples) == (2 if compact_history else 1)
        events = base.trace["events"]
        starts = {event["span_id"]: event for event in events if event["type"] == "span_start"}
        ends = {event["span_id"]: event for event in events if event["type"] == "span_end"}
        assert starts.keys() == ends.keys()
        assert all(ends[key]["ts"] >= event["ts"] for key, event in starts.items())
        names = {event["name"] for event in starts.values()}
        assert {
            "coding_rollout",
            "sandbox_create",
            "harness_install",
            "workspace_prepare",
            "agent_run",
            "agent_model_turn",
            "agent_tokenize",
            "patch_extract",
            "evaluation_sandbox_create",
            "evaluation_prepare",
            "evaluation_tests",
            "evaluation_sandbox_cleanup",
            "trajectory_finalize",
            "adapter_cleanup",
        } <= names
        root = next(event["span_id"] for event in starts.values() if event["name"] == "coding_rollout")
        turns = [event for event in starts.values() if event["name"] == "agent_model_turn"]
        assert len(turns) == 2
        assert all(event["parent_span_id"] == root for event in turns)
        record = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
        assert record["reward"] == 1.0
        assert record["metadata"]["segments"] == len(samples)
        assert record["trace"] == base.trace
        assert record["response_length"] == sum(sum(s.loss_mask) for s in samples)
        assert "prompt" not in record and "response" not in record

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


@pytest.mark.parametrize(
    "limit_reached,exit_code,truncation_reason",
    [(False, 0, None), (True, 143, "compact_limit"), (False, -1, "time_budget")],
)
def test_budget_stops_truncate_every_training_segment(monkeypatch, limit_reached, exit_code, truncation_reason):
    async def run_case():
        async def agent(env):
            await _anthropic_agent(env, compact_history=True)
            return exit_code

        sandbox_factory = FakeSandbox.factory(on_launch=agent)
        _patch_generate(monkeypatch, FakeTokenizer(), sandbox_factory)
        monkeypatch.setattr(gen, "CONFIG", dataclasses.replace(gen.CONFIG, fork_merge_threshold=0))

        async def collect_metrics(self, sandbox, *, workdir):
            return {"compact_count": 1, "compact_attempts": 1, "compact_limit_reached": int(limit_reached)}

        monkeypatch.setattr(ClaudeCodeHarness, "collect_metrics", collect_metrics)
        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})
        assert len(samples) == 2
        truncated = truncation_reason is not None
        expected = Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED
        assert all(sample.status == expected for sample in samples)
        assert all(sample.reward == 1.0 for sample in samples)
        assert all(sample.metadata.get("truncation_reason") == truncation_reason for sample in samples)

        from slime.observability.rollout_metrics import compute_metrics_from_samples

        metrics = compute_metrics_from_samples(
            SimpleNamespace(advantage_estimator="grpo", reward_key=None, log_reward_category=None), samples
        )
        assert metrics["compact_count"] == 1.0
        assert "truncated_ratio" not in metrics
        assert not any(key.startswith("compact_") and key != "compact_count" for key in metrics)

    asyncio.run(run_case())


@pytest.mark.parametrize("agent_name", ["claude_code", "codex"])
@pytest.mark.parametrize("evaluation", [False, True])
def test_terminated_sandbox_returns_zero_reward_trajectory(monkeypatch, caplog, agent_name, evaluation):
    class SandboxError(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "sunabako", SimpleNamespace(SandboxError=SandboxError))
    monkeypatch.delenv("SUNABAKO_ARTIFACTS", raising=False)
    clients = []

    class TerminatedSandbox(FakeSandbox):
        async def __aexit__(self, *exc):
            self._sandbox = SimpleNamespace(kill=AsyncMock())
            clients.append(self._sandbox)
            await SunabakoSandbox.__aexit__(self, *exc)

    async def agent(env):
        if agent_name == "claude_code":
            await _anthropic_agent(env, compact_history=True)
        else:
            body = {"input": [{"role": "user", "content": "solve the issue"}], "store": False}
            async with aiohttp.ClientSession(trust_env=False) as session:
                for _ in range(2):
                    async with session.post(
                        env["OPENAI_BASE_URL"] + "/responses",
                        headers={"Authorization": "Bearer " + env["OPENAI_API_KEY"]},
                        json=body,
                    ) as response:
                        assert response.status == 200
                        await response.json()
                    body["input"] = [{"role": "user", "content": "summary of earlier work"}]
        raise SandboxError('{"error":"sandbox must be running"}')

    async def run_case():
        _patch_generate(monkeypatch, FakeTokenizer(), TerminatedSandbox.factory(on_launch=agent))
        monkeypatch.setattr(gen, "CONFIG", dataclasses.replace(gen.CONFIG, fork_merge_threshold=0))
        monkeypatch.setattr(CodexHarness, "install_cli", _noop_install)
        samples = await gen.generate(
            _args(), _base_sample(agent=agent_name), sampling_params={"max_new_tokens": 32}, evaluation=evaluation
        )
        assert samples
        assert all(sample.reward == 0.0 for sample in samples)
        assert all(sample.metadata["agent_error"] == "sandbox must be running" for sample in samples)
        assert all(sample.metadata["agent_error_type"] == "sandbox_terminated" for sample in samples)
        if not evaluation:
            assert len(samples) == 2
            for sample in samples:
                assert sample.status == Sample.Status.COMPLETED
                assert not sample.remove_sample
                assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
                assert sum(sample.loss_mask) > 0
                assert sample.tokens != [0, 0]
                assert "abort_reason" not in sample.metadata
        assert len(clients) == 1  # no retry or evaluation sandbox for this failed trajectory
        clients[0].kill.assert_awaited_once()
        assert "rollout failed" not in caplog.text
        assert "Traceback" not in caplog.text

    asyncio.run(run_case())


@pytest.mark.parametrize("generated", [False, True])
def test_infrastructure_errors_remain_retryable(monkeypatch, generated):
    async def agent(env):
        if generated:
            await _anthropic_agent(env)
        raise OSError("transport unavailable")

    async def run_case():
        _patch_generate(monkeypatch, FakeTokenizer(), FakeSandbox.factory(on_launch=agent))
        samples = await gen.generate(_args(), _base_sample(), sampling_params={})
        assert len(samples) == 1
        assert samples[0].status == Sample.Status.ABORTED
        assert samples[0].remove_sample
        assert not any(samples[0].loss_mask)
        assert samples[0].metadata["abort_reason"] == "exception:OSError"

    asyncio.run(run_case())


def test_terminated_sandbox_without_model_actions_remains_retryable(monkeypatch):
    from slime.agent.sandbox import SandboxTerminatedError

    async def agent(_env):
        raise SandboxTerminatedError("sandbox must be running")

    async def run_case():
        _patch_generate(monkeypatch, FakeTokenizer(), FakeSandbox.factory(on_launch=agent))
        samples = await gen.generate(_args(), _base_sample(), sampling_params={})
        assert len(samples) == 1
        assert samples[0].status == Sample.Status.ABORTED
        assert samples[0].remove_sample
        assert samples[0].metadata["abort_reason"] == "sandbox_terminated_before_generation"

    asyncio.run(run_case())


def test_generate_aborts_on_empty_trajectory():
    """If the agent never drives a turn, the session is empty and generate()
    returns a single ABORTED sample (the fan-out shape) rather than crashing."""

    async def silent_agent(_env) -> int:
        return 0  # never contacts the adapter -> empty trajectory

    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        sandbox_factory = FakeSandbox.factory(on_launch=silent_agent)
        _patch_generate(monkeypatch, tok, sandbox_factory)

        samples = await gen.generate(_args(), _base_sample(), sampling_params={})

        assert len(samples) == 1
        assert samples[0].status == Sample.Status.ABORTED
        assert samples[0].metadata.get("abort_reason") == "adapter_session_empty"

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


def test_generate_aborts_on_missing_image():
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        _patch_generate(monkeypatch, tok, FakeSandbox.factory(on_launch=_anthropic_agent))
        # blank image -> early abort before any sandbox boot.
        samples = await gen.generate(_args(), _base_sample(image=""), sampling_params={})
        assert len(samples) == 1
        assert samples[0].status == Sample.Status.ABORTED
        assert samples[0].metadata.get("abort_reason") == "missing_image_or_workdir"

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ===========================================================================
# §2 the Codex + OpenAI pair closes the same loop (hand-wired)
# ===========================================================================


async def _codex_agent(env: dict, *, n_turns: int = 2) -> int:
    base_url = env["OPENAI_BASE_URL"]  # already includes /v1
    token = env["OPENAI_API_KEY"]
    history = [{"role": "user", "content": "solve the issue"}]
    async with aiohttp.ClientSession(trust_env=False) as sess:
        for _ in range(n_turns):
            async with sess.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "m", "max_tokens": 64, "messages": history},
            ) as r:
                data = await r.json()
            msg = data["choices"][0]["message"]
            history.append({"role": "assistant", "content": msg.get("content") or ""})
            history.append({"role": "user", "content": "continue"})
    return 0


def test_codex_openai_rollout_closes_loop(monkeypatch):
    """CodexHarness drives an in-thread OpenAIAdapter through a FakeSandbox; the
    loop produces trained samples just like the production Anthropic path."""

    async def run_case():
        tok = FakeTokenizer()
        monkeypatch.setattr(
            adapters_common, "call_sglang_generate", fake_call_sglang_generate(_two_turn_script(), tok)
        )
        monkeypatch.setattr(harness_common.asyncio, "sleep", _fast_sleep)

        adapter = OpenAIAdapter(tokenizer=tok, sglang_url="http://unused")
        handle = run_app_in_thread(adapter.app, host="127.0.0.1", port=0, thread_name="test-openai-adapter")
        adapter_url = f"http://127.0.0.1:{handle.port}"
        sid = "codex-sess"
        adapter.open_session(sid)
        try:
            sb = FakeSandbox(on_launch=_codex_agent)
            rc = await CodexHarness().run(
                sb,
                workdir="/workspace/repo",
                session_id=sid,
                adapter_url=adapter_url,
                time_budget_sec=30,
                prompt="fix it",
            )
            samples = await adapter.finish_session(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
        finally:
            handle.stop()

        assert rc == 0
        assert samples
        for s in samples:
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0

    asyncio.run(run_case())


def test_one_worker_switches_from_codex_to_claude_without_mutating_global_config(monkeypatch):
    async def responses_agent(env):
        body = {"input": [{"role": "user", "content": "solve the issue"}], "store": False}
        async with aiohttp.ClientSession(trust_env=False) as session:
            for _ in range(2):
                async with session.post(
                    env["OPENAI_BASE_URL"] + "/responses",
                    headers={"Authorization": "Bearer " + env["OPENAI_API_KEY"]},
                    json=body,
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                body["input"] += data["output"] + [{"role": "user", "content": "continue"}]
        return 0

    async def launch(env):
        return await (responses_agent(env) if "OPENAI_API_KEY" in env else _anthropic_agent(env))

    async def run():
        tok = FakeTokenizer()
        _patch_generate(monkeypatch, tok, FakeSandbox.factory(on_launch=launch))
        monkeypatch.setattr(CodexHarness, "install_cli", _noop_install)
        monkeypatch.setattr(
            adapters_common, "call_sglang_generate", fake_call_sglang_generate(_two_turn_script() * 2, tok)
        )
        default = gen.AGENT_NAME
        for index, name in enumerate(("codex", "claude_code")):
            base = _base_sample(agent=name)
            base.index = index
            samples = await gen.generate(_args(), base, sampling_params={"max_new_tokens": 32})
            assert samples and {s.metadata.get("agent") for s in samples} == {name}
            assert all(s.metadata["agent_exit_code"] == 0 for s in samples)
        state = gen._AdapterService(_args())
        assert state.endpoint("codex")[0] is not state.endpoint("claude_code")[0]
        assert state.endpoint("codex")[1] != state.endpoint("claude_code")[1]
        assert gen.AGENT_NAME == default

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
