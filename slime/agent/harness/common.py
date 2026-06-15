"""Harness-agnostic coding-agent lifecycle in a sandbox.

A *harness* is a swappable coding agent (Claude Code, Codex, ...). Each one
installs a CLI, drops its own config, and runs the agent against a prompt; the
shared parts (create the agent user, the ``run`` skeleton, the E2B
launch-detached-and-poll transport) live here. Adding a CLI-style harness means
subclassing ``BaseHarness`` and implementing the three differing steps
(``install_cli`` + ``write_config`` + ``launch_and_wait``) -- no edits to this
file. Two module-level helpers absorb the common cases: ``install_npm_cli`` for
npm-packaged CLIs, and ``run_command`` for the run-one-command-to-completion
case (run one command detached, poll a done-marker). A harness with a different
install path (pip, curl, pre-baked image) or execution model (interactive / a
long-running server fed turn by turn) just writes those two methods directly.

The base does NOT know anything about SWE: ``run()`` takes only generic fields
(workdir / session_id / adapter_url / prompt). Task-specific workspace prep and
scoring live in the example layer (``examples/coding_agent_rl/swe.py``).
"""

from __future__ import annotations

import asyncio
import lzma
import os
import shlex
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from slime.agent import sandbox as _sandbox
from slime.agent.sandbox import Sandbox

# Sentinel for run_command's int return: the time budget expired before the
# command finished, so there is no real exit code. Process exit codes are POSIX
# 0-255, so a negative value never collides; anything >=0 is the command's real
# PIPESTATUS[0] and carries diagnostic value (1=error, 137=OOM-killed, etc.).
EXIT_TIME_BUDGET_EXCEEDED = -1


@dataclass(frozen=True)
class HarnessContext:
    """Generic inputs a harness needs to write config + build a launch command.

    Deliberately free of any task (SWE) fields. ``model_label`` is a fixed
    constant: the model name the harness advertises to its CLI. The slime
    adapter ignores it and serves whatever upstream sglang has loaded, so it is
    not a ``run()`` parameter -- no caller ever needs to vary it.
    """

    workdir: str
    session_id: str
    adapter_url: str
    model_label: str = "slime-actor"


class BaseHarness(ABC):
    """Base lifecycle for a sandbox-resident coding agent.

    Subclasses set ``name`` and implement the three differing steps
    (``install_cli`` / ``write_config`` / ``launch_and_wait``); everything else
    (agent user, the ``run`` skeleton) is shared. npm-packaged CLIs implement
    ``install_cli`` by delegating to ``install_npm_cli``; non-interactive CLIs
    implement ``launch_and_wait`` by delegating to ``run_command``.
    """

    #: Short identifier; also names the launcher metadata dir owner.
    name: str = ""

    @abstractmethod
    async def install_cli(self, sb: Sandbox) -> None:
        """Install the harness CLI into the sandbox (harness-specific).
        npm-packaged harnesses delegate to ``install_npm_cli``."""

    @abstractmethod
    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Write any CLI config files into the sandbox (harness-specific)."""

    @abstractmethod
    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        """Run the agent to completion and return its exit code (harness-specific).

        This is the execution-model step. A non-interactive CLI builds one shell
        command and hands it to ``run_command`` (which provides the E2B
        transport: run detached, poll a done-marker) -- see the existing
        harnesses. An interactive or long-running harness drives its own loop
        here instead, and may still reuse ``run_command`` for individual commands.
        """

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
    ) -> int:
        """Run the harness in ``sb`` and return its exit code.

        Steps: ensure the agent user (sandbox infra) -> write config (subclass)
        -> launch and wait for completion (subclass). Workspace prep (writing
        PROBLEM_STATEMENT.md etc.) is the caller's job and must run before this.
        """
        await _sandbox.ensure_agent_user(sb, workdir)
        ctx = HarnessContext(
            workdir=workdir,
            session_id=session_id,
            adapter_url=adapter_url,
        )
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(sb, ctx, prompt, time_budget_sec)


async def run_command(sb: Sandbox, *, workdir: str, start_cmd: str, env: dict[str, str], time_budget_sec: int) -> int:
    """Run ``start_cmd`` to completion in ``sb`` and return its exit code.

    The E2B transport mechanism, independent of any harness: its gateway resets
    HTTP/2 around 6.5 min, so we can't keep a long-lived foreground exec. Instead
    the command runs detached (``setsid``), piping output to a trajectory log and
    writing the *command's* exit code (``PIPESTATUS[0]``, not tee's) into a marker
    file; we poll that marker every 5s via short RPCs (which also keeps the
    sandbox alive against idle GC). All metadata goes under ``{workdir}/.harness/``
    so diff capture only has to exclude one directory. Returns
    ``EXIT_TIME_BUDGET_EXCEEDED`` if the time budget is exceeded before the
    command finishes.
    """
    meta_dir = f"{workdir}/.harness"
    done = f"{meta_dir}/done"
    launcher = f"{meta_dir}/run.sh"
    traj = f"{meta_dir}/trajectory.jsonl"

    launcher_body = (
        "#!/bin/bash\n"
        f"cd {workdir}\n"
        "export HOME=/home/agent\n"
        f"{start_cmd} 2>&1 | tee {shlex.quote(traj)}\n"
        f"echo ${{PIPESTATUS[0]}} > {done}\n"
    )
    await sb.exec(f"mkdir -p {meta_dir} && chown agent:agent {meta_dir}", user="root", check=True, timeout=30)
    await sb.write_file(launcher, launcher_body, user="agent")
    await sb.exec(f"chmod +x {launcher}", user="agent", timeout=30)

    env_keys = ",".join(env.keys())
    await sb.exec(
        f"runuser -u agent --whitelist-environment={env_keys}"
        f" -- bash -c 'setsid {launcher} < /dev/null > /dev/null 2>&1 &'",
        user="root",
        env=env,
        timeout=30,
        check=True,
    )

    deadline = time.time() + time_budget_sec
    exit_code = EXIT_TIME_BUDGET_EXCEEDED  # until the marker yields a real code
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(
            f"test -f {done} && cat {done}",
            user="agent",
            timeout=15,
            check=False,
        )
        if ec == 0:
            exit_code = int((out or "").strip())
            break
    return exit_code


async def install_npm_cli(sb: Sandbox, *, node_runtime: Path, npm_package: Path, check_cmd: str) -> None:
    """Install an npm-packaged CLI into the sandbox: the Node 22 runtime (from
    ``node_runtime``) as a prerequisite, then the CLI's npm package (from
    ``npm_package``, global install + self-check via ``check_cmd``). Shared by
    npm-based harnesses; non-npm harnesses write their own ``install_cli``."""
    await install_node22(sb, node_runtime)
    await sb.write_file("/tmp/harness-cli.tgz", npm_package)
    await sb.exec(
        f"npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/harness-cli.tgz && {check_cmd}",
        user="root",
        timeout=300,
        check=True,
    )


async def install_node22(sb: Sandbox, host_tarball: Path) -> None:
    """Node 22 over the base image (Debian 12 ships 16; cli.js needs >= 20).
    Decompresses .xz on the host (cached) so sandboxes without xz-utils can
    still run plain `tar xf`. npm prefix=/usr/local required for sweap-images."""
    host_tarball = Path(host_tarball)
    if host_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"coding_agent_rl.{host_tarball.stem}.tar"
        if not plain.exists():
            tmp = plain.with_suffix(".tar.partial")
            with lzma.open(host_tarball, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, plain)
        host_tarball = plain
    await sb.write_file("/tmp/node22.tar", host_tarball)
    await sb.exec(
        "set -e && mkdir -p /opt/node22 && "
        "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
        "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
        "ln -sf /opt/node22/bin/npm  /usr/local/bin/npm && "
        "ln -sf /opt/node22/bin/npx  /usr/local/bin/npx && "
        "hash -r 2>/dev/null || true && node --version && npm --version",
        user="root",
        timeout=180,
        check=True,
    )
