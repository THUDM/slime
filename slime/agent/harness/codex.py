"""Codex harness.

Implementation mirrors the standalone Codex runner in
``tests/test_agent/test_codex_agent/_runner.py`` (read-only reference). The two
non-obvious bits: the provider ``base_url`` MUST be inline in the TOML (Codex
only honours env vars for the default OpenAI provider), and the config is written
via a base64 round-trip to dodge shell-quoting traps.
"""

from __future__ import annotations

import base64
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, spawn_detached


class CodexHarness(BaseHarness):
    name = "codex"

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ["SWE_HOST_NODE_TARBALL"]),
            npm_package=Path(os.environ["SWE_HOST_CODEX_TARBALL"]),
            check_cmd="codex --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        toml = (
            f'model = "{ctx.model_label}"\n'
            'model_provider = "slime"\n'
            "\n"
            "[model_providers.slime]\n"
            'name = "slime"\n'
            f'base_url = "{ctx.adapter_url}/v1"\n'
            'env_key = "OPENAI_API_KEY"\n'
            'wire_api = "chat"\n'
        )
        toml_b64 = base64.b64encode(toml.encode("utf-8")).decode("ascii")
        await sb.exec(
            "mkdir -p /home/agent/.codex && "
            # base64 round-trip avoids any single-quote / heredoc shell-quoting trap.
            f"echo {shlex.quote(toml_b64)} | base64 -d > /home/agent/.codex/config.toml && "
            "chown -R agent:agent /home/agent/.codex",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        # ``codex exec`` is the non-interactive entrypoint. --skip-git-repo-check
        # lets it run in workdirs whose git check is brittle (shallow clones).
        cmd = f"codex exec --skip-git-repo-check {shlex.quote(prompt)}"
        env = {
            # Codex CLI propagates OPENAI_API_KEY into ``Authorization: Bearer``;
            # the slime adapter resolves the sid from that header.
            "OPENAI_API_KEY": ctx.session_id,
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
        }
        return await spawn_detached(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)


CODEX = CodexHarness()
