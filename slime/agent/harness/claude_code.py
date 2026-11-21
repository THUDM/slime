"""Claude Code harness."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class ClaudeCodeHarness(BaseHarness):
    name = "claude_code"

    # host paths + CLI knobs, all under the agent-layer SLIME_AGENT_* prefix
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_CC_TARBALL"
    native_tarball_env = "SLIME_AGENT_CC_NATIVE_TARBALL"
    extra_args_env = "SLIME_AGENT_CC_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CC_EXTRA_ENVS"
    max_compactions_env = "SLIME_AGENT_CC_MAX_COMPACTIONS"

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
        if archive := os.environ.get(self.native_tarball_env):
            await sb.write_file("/tmp/claude-native.tgz", Path(archive))
            await sb.exec(
                "set -e; mkdir -p /opt/claude-code /usr/local/bin; "
                "tar xzf /tmp/claude-native.tgz -C /opt/claude-code --strip-components=1; "
                "ln -sf /opt/claude-code/claude /usr/local/bin/claude; claude --version",
                user="root",
                timeout=180,
                check=True,
            )
            return
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Pre-ack bypass-permissions so claude-code starts headless."""
        limit = int(os.environ.get(self.max_compactions_env, "-1"))
        if limit < -1:
            raise ValueError(f"{self.max_compactions_env} must be -1 or nonnegative")
        directory = f"{ctx.workdir}/.harness"
        hook_path = "/tmp/slime-claude-compaction.py"
        await sb.write_file(hook_path, Path(__file__).with_name("compaction.py"))
        command = f"python3 {hook_path} {shlex.quote(directory)} {limit}"
        settings = json.dumps(
            {
                "hasCompletedOnboarding": True,
                "bypassPermissionsModeAccepted": True,
                "hooks": {
                    event: [{"hooks": [{"type": "command", "command": command, "timeout": 10}]}]
                    for event in ("PreCompact", "PostCompact")
                },
            }
        )
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
        directory = shlex.quote(f"{ctx.workdir}/.harness")
        # The detached launcher stays alive to write its normal completion
        # marker when the compaction hook stops only the CLI child.
        cmd = (
            f'printf \'%s\\n\' \'{{"attempts":0,"count":0,"limit_reached":false}}\' '
            f"> {directory}/compaction.json; {cmd} &\n"
            f"slime_cli_pid=$!\nprintf '%s\\n' \"$slime_cli_pid\" > {directory}/cli.pid\n"
            'wait "$slime_cli_pid"'
        )
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)

    async def collect_metrics(self, sb: Sandbox, *, workdir: str) -> dict[str, int | float]:
        contents = await sb.read_file(f"{workdir}/.harness/compaction.json")
        if not contents.strip():
            return {}
        state = json.loads(contents)
        return {
            "compact_count": int(state.get("count", 0)),
            "compact_attempts": int(state.get("attempts", 0)),
            "compact_limit_reached": int(state.get("limit_reached", False)),
        }
