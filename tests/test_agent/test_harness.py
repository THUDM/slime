"""Unit tests for the coding-agent harness + sandbox layers.

These cover the parts a happy-path rollout can't pin down precisely: that each
harness writes the right CLI config and launches with the right command + env,
that ``run_agent``'s detached-launch / poll-marker handshake returns the right
exit code (and times out correctly), and that ``ensure_agent_user`` issues the
expected provisioning command. A :class:`tests.test_agent._fakes.FakeSandbox`
records every ``exec`` / ``write_file`` so we assert on the issued commands
without a real sandbox or any root privilege.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSandbox  # noqa: E402

from slime.agent import sandbox as sandbox_mod  # noqa: E402
from slime.agent.harness import ClaudeCodeHarness, CodexHarness, HarnessContext  # noqa: E402
from slime.agent.harness import common as hc  # noqa: E402

NUM_GPUS = 0

# Run the 5s poll loop instantly without recursing into the patched function.
_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _ctx(workdir="/workspace/repo", sid="sess-1", url="http://host:18001") -> HarnessContext:
    return HarnessContext(workdir=workdir, session_id=sid, adapter_url=url)


def _find(exec_log, needle):
    return [cmd for cmd, _user in exec_log if needle in cmd]


# ===========================================================================
# §1 run_agent handshake (the E2B detached-launch transport)
# ===========================================================================


def test_run_agent_returns_marker_exit_code():
    async def run_case():
        seen = {}

        async def fake_agent(env):
            seen["env"] = env
            return 0  # exit code written into the done marker

        sb = FakeSandbox(on_launch=fake_agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc = await hc.run_agent(
                sb, workdir="/workspace/repo", start_cmd="claude -p hi", env={"A": "1"}, time_budget_sec=30
            )
        assert rc == 0
        assert seen["env"] == {"A": "1"}
        # launcher script + detached setsid launch all issued, exit code captured.
        assert any("run.sh" in p for p in sb.files)
        assert _find(sb.exec_log, "setsid")
        assert any("echo $?" in v for v in sb.files.values())

    asyncio.run(run_case())


def test_run_agent_propagates_nonzero_exit():
    async def run_case():
        async def fail_agent(_env):
            return 7

        sb = FakeSandbox(on_launch=fail_agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc = await hc.run_agent(sb, workdir="/w", start_cmd="x", env={}, time_budget_sec=30)
        assert rc == 7

    asyncio.run(run_case())


def test_run_agent_times_out_when_marker_never_appears():
    async def run_case():
        sb = FakeSandbox(on_launch=None)  # no agent -> marker never written
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc = await hc.run_agent(sb, workdir="/w", start_cmd="x", env={}, time_budget_sec=0)
        assert rc == sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED

    asyncio.run(run_case())


# ===========================================================================
# §2 ClaudeCodeHarness config + launch
# ===========================================================================


def test_claude_code_write_config_preacks_bypass_permissions():
    async def run_case():
        sb = FakeSandbox()
        await ClaudeCodeHarness().write_config(sb, _ctx())
        joined = " ".join(cmd for cmd, _ in sb.exec_log)
        assert "/home/agent/.claude/settings.json" in joined
        assert "bypassPermissionsModeAccepted" in joined
        assert "hasCompletedOnboarding" in joined

    asyncio.run(run_case())


def test_claude_code_launch_command_and_env():
    async def run_case():
        captured = {}

        async def agent(env):
            captured["env"] = env
            return 0

        capturing = FakeSandbox(on_launch=agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc = await ClaudeCodeHarness().launch_and_wait(
                capturing, _ctx(sid="sess-cc", url="http://host:18001"), prompt="solve it", time_budget_sec=30
            )
        assert rc == 0
        # the prompt + flags land in the launcher script body.
        body = next(v for k, v in capturing.files.items() if k.endswith("run.sh"))
        assert "claude -p 'solve it'" in body
        assert "--permission-mode bypassPermissions" in body
        # env carries the adapter wiring under the Anthropic var names.
        env = captured["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://host:18001"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sess-cc"
        assert env["ANTHROPIC_MODEL"] == "slime-actor"
        assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"

    asyncio.run(run_case())


# ===========================================================================
# §3 CodexHarness config + launch
# ===========================================================================


def test_codex_write_config_base64_roundtrips_inline_base_url():
    async def run_case():
        sb = FakeSandbox()
        await CodexHarness().write_config(sb, _ctx(url="http://host:18001"))
        # config written via base64 round-trip; decode the captured payload.
        cmd = next(c for c, _ in sb.exec_log if "base64 -d > /home/agent/.codex/config.toml" in c)
        b64 = cmd.split("echo ")[1].split(" | base64")[0].strip("'")
        toml = base64.b64decode(b64).decode()
        assert 'base_url = "http://host:18001/v1"' in toml  # MUST be inline
        assert 'wire_api = "chat"' in toml
        assert 'model_provider = "slime"' in toml

    asyncio.run(run_case())


def test_codex_launch_command_and_env():
    async def run_case():
        captured = {}

        async def agent(env):
            captured["env"] = env
            return 0

        sb = FakeSandbox(on_launch=agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc = await CodexHarness().launch_and_wait(
                sb, _ctx(sid="sess-cx", url="http://host:18001"), prompt="do work", time_budget_sec=30
            )
        assert rc == 0
        body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
        assert "codex exec" in body and "do work" in body and "--skip-git-repo-check" in body
        env = captured["env"]
        assert env["OPENAI_API_KEY"] == "sess-cx"
        assert env["OPENAI_BASE_URL"] == "http://host:18001/v1"

    asyncio.run(run_case())


# ===========================================================================
# §4 ensure_agent_user (sandbox infra)
# ===========================================================================


def test_ensure_agent_user_provisions_user_and_git_safe_dir():
    async def run_case():
        sb = FakeSandbox()
        await sandbox_mod.ensure_agent_user(sb, "/workspace/repo")
        cmd = next(c for c, _ in sb.exec_log if "useradd" in c)
        assert "id agent" in cmd
        assert "chown -R agent:agent" in cmd and "/workspace/repo" in cmd
        assert "git config --system --add safe.directory '*'" in cmd

    asyncio.run(run_case())


# ===========================================================================
# §5 harness.run wires the steps in order
# ===========================================================================


def test_base_harness_run_calls_steps_in_order():
    async def run_case():
        async def agent(_env):
            return 0

        sb = FakeSandbox(on_launch=agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc = await ClaudeCodeHarness().run(
                sb,
                workdir="/workspace/repo",
                session_id="sess-run",
                adapter_url="http://host:18001",
                time_budget_sec=30,
                prompt="go",
            )
        assert rc == 0
        joined = " ".join(c for c, _ in sb.exec_log)
        # ensure_agent_user (useradd) -> write_config (settings.json) -> launch (setsid)
        order = [k for k in ("useradd", "settings.json", "setsid") if k in joined]
        assert order == ["useradd", "settings.json", "setsid"]

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
