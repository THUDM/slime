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

Two protocol chains are covered:

  * ``test_generate_*`` -- the production path: real ``generate.generate()``,
    which is hardwired to ClaudeCodeHarness + AnthropicAdapter.
  * ``test_codex_openai_rollout_closes_loop`` -- the same loop for the
    CodexHarness + OpenAIAdapter pair, hand-wired (generate() does not select it).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
if "transformers" not in sys.modules:
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


async def _anthropic_agent(env: dict, *, n_turns: int = 2) -> int:
    """Stand-in for ``claude -p``: dial the adapter back over real HTTP and fire
    a few Anthropic turns, reading wiring from the env the harness exported."""
    base_url = env["ANTHROPIC_BASE_URL"]
    token = env["ANTHROPIC_AUTH_TOKEN"]
    history = [{"role": "user", "content": [{"type": "text", "text": "solve the issue"}]}]
    async with aiohttp.ClientSession(trust_env=False) as sess:
        for _ in range(n_turns):
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
    monkeypatch.setattr(gen, "E2BSandbox", sandbox_factory)  # boot sandbox
    monkeypatch.setattr(swe, "E2BSandbox", sandbox_factory)  # eval sandbox
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


def test_generate_produces_trained_samples():
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        sandbox_factory = FakeSandbox.factory(on_launch=_anthropic_agent)
        _patch_generate(monkeypatch, tok, sandbox_factory)

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert samples, "rollout produced no samples"
        for s in samples:
            assert s.status == Sample.Status.COMPLETED
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0  # at least one trained token
            assert s.metadata.get("agent_exit_code") == 0
        # eval_cmd "true" applied cleanly on a clean (empty) diff -> reward 1.0,
        # split evenly across the emitted samples.
        assert abs(sum(s.reward for s in samples) - 1.0) < 1e-9

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
