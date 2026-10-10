"""Codex harness.

Two non-obvious bits: the provider base_url must be inline in the TOML (Codex
only honours env vars for the default OpenAI provider), and the config is written
via a base64 round-trip to dodge shell-quoting traps.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class CodexHarness(BaseHarness):
    name = "codex"

    # host paths + CLI knobs, all under the agent-layer SLIME_AGENT_* prefix
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_CODEX_TARBALL"
    native_tarball_env = "SLIME_AGENT_CODEX_NATIVE_TARBALL"
    extra_args_env = "SLIME_AGENT_CODEX_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CODEX_EXTRA_ENVS"

    # static flags after ``codex exec``; --skip-git-repo-check lets it run in
    # workdirs whose git check is brittle (e.g. shallow clones)
    # The outer Sandbox owns isolation. Codex's nested bwrap cannot run inside
    # an unprivileged PRoot sandbox. Never use this harness directly on the host.
    exec_flags = "--skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --json"

    # config.toml written into the sandbox. base_url MUST be inline here (Codex
    # only honours env vars for the default OpenAI provider). {model} / {base_url}
    # are filled per run in write_config; the rest is fixed wiring.
    config_toml = (
        'model = "{model}"\n'
        'model_provider = "slime"\n'
        'web_search = "disabled"\n'
        'model_reasoning_effort = "medium"\n'
        "\n"
        "[model_providers.slime]\n"
        'name = "slime"\n'
        'base_url = "{base_url}"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        "supports_websockets = false\n"
        "\n[features]\n"
        "multi_agent = false\n"
    )

    async def install_cli(self, sb: Sandbox) -> None:
        if archive := os.environ.get(self.native_tarball_env):
            # Official @openai/codex platform npm archive: keep the sibling
            # resources/tools beside bin/codex. This is an offline install and
            # needs neither Node nor an npm optional-dependency download.
            await sb.write_file("/tmp/codex-native.tgz", Path(archive))
            await sb.exec(
                "mkdir -p /opt/codex /usr/local/bin && "
                "tar xzf /tmp/codex-native.tgz -C /opt/codex --strip-components=3 && "
                "ln -sf /opt/codex/bin/codex /usr/local/bin/codex && "
                "ln -sf /opt/codex/codex-path/rg /usr/local/bin/rg && codex --version",
                user="root",
                timeout=180,
                check=True,
            )
            return
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="codex --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        toml = self.config_toml.format(model=ctx.model_label, base_url=f"{ctx.adapter_url}/v1")
        toml_b64 = base64.b64encode(toml.encode("utf-8")).decode("ascii")
        await sb.exec(
            "mkdir -p /home/agent/.codex && "
            # base64 round-trip avoids any single-quote / heredoc shell-quoting trap
            f"echo {shlex.quote(toml_b64)} | base64 -d > /home/agent/.codex/config.toml && "
            "chown -R agent:agent /home/agent/.codex",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        # ``codex exec`` is the non-interactive entrypoint
        cmd = f"codex exec {self.exec_flags} {shlex.quote(prompt)}"
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"
        env = {
            # Codex propagates OPENAI_API_KEY into Authorization: Bearer; the
            # slime adapter resolves the sid from that header.
            "OPENAI_API_KEY": ctx.session_id,
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
        }
        # extra env vars as a JSON object, merged last so callers can override
        # the defaults above
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
