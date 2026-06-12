"""Claude Code harness."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, spawn_detached


class ClaudeCodeHarness(BaseHarness):
    name = "claude_code"

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ["SWE_HOST_NODE_TARBALL"]),
            npm_package=Path(os.environ["SWE_HOST_CC_TARBALL"]),
            check_cmd="ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Pre-ack bypass-permissions so claude-code starts headless."""
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        await sb.exec(
            "mkdir -p /home/agent/.claude && "
            f"echo {shlex.quote(settings)} "
            "| tee /home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
            "chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        extra = os.environ.get("SWE_CLAUDE_EXTRA_ARGS", "").strip()
        cmd = (
            f"/usr/local/bin/claude -p {shlex.quote(prompt)} "
            "--permission-mode bypassPermissions "
            "--output-format stream-json --include-partial-messages "
            "--include-hook-events --verbose"
        )
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }
        return await spawn_detached(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)


CLAUDE_CODE = ClaudeCodeHarness()
