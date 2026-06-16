"""Claude Code harness."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_command


class ClaudeCodeHarness(BaseHarness):
    name = "claude_code"

    # host paths + CLI knobs, all under the agent-layer SLIME_AGENT_* prefix
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_CC_TARBALL"
    extra_args_env = "SLIME_AGENT_CC_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CC_EXTRA_ENVS"

    launch_flags = (
        "--permission-mode bypassPermissions "
        "--output-format stream-json --include-partial-messages "
        "--include-hook-events --verbose"
    )

    static_env = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
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
        cmd = f"/usr/local/bin/claude -p {shlex.quote(prompt)} {self.launch_flags}"
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            **self.static_env,
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_command(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
