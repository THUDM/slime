"""Coding-agent sandbox helpers.

The provider-agnostic sandbox contract and E2B backend live in
``slime.agent.sandbox``. This module keeps the coding-agent/SWE-specific
bootstrap, Claude Code runner, diff capture, and fresh-sandbox evaluator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import lzma
import os
import shlex
import shutil
import tempfile
import time
from pathlib import Path

from slime.agent.sandbox import E2BSandbox, Sandbox


logger = logging.getLogger(__name__)

# Paths inside the sandbox (avoid clashes with image-shipped paths).
_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_SWEPRO_DIR = "/workspace/swepro_eval"


# ---------------------------------------------------------------------------
# Sandbox bootstrap (Node + Claude Code + agent user)
# ---------------------------------------------------------------------------
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


async def install_claude_code(sb: Sandbox, host_tarball: Path) -> None:
    await sb.write_file("/tmp/claude-code.tgz", host_tarball)
    await sb.exec(
        "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/claude-code.tgz "
        "&& ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
        user="root",
        timeout=300,
        check=True,
    )


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff.
    Settings file pre-acks bypass-permissions so claude-code starts headless."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent && "
        f"mkdir -p /home/agent/.claude && "
        f'echo \'{{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}}\' '
        f"| tee /home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
        f"chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
        user="root",
        check=True,
        timeout=60,
    )


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd") if swepro else None
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup", user="root", check=True
    )
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)


# ---------------------------------------------------------------------------
# Agent run (claude-code spawn + done-marker poll)
# ---------------------------------------------------------------------------
async def run_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    middleware_url: str,
    prompt: str,
    time_budget_sec: int,
) -> int:
    """Spawn claude-code detached + poll a done-marker file.

    E2B's gateway resets HTTP/2 around 6.5 min, so we can't keep a long-lived
    foreground exec. The launcher writes the exit code into a marker file
    and we poll it every 5s via short RPCs (which also keeps the sandbox
    alive against idle GC)."""
    done = f"{workdir}/.cagent_done"
    launcher = f"{workdir}/.cagent_run.sh"
    traj = f"{workdir}/claude_code_trajectory.jsonl"

    launcher_body = (
        "#!/bin/bash\n"
        f"cd {workdir}\n"
        "export HOME=/home/agent\n"
        f"/usr/local/bin/claude -p {json.dumps(prompt)} "
        f"--permission-mode bypassPermissions "
        f"--output-format stream-json --include-partial-messages "
        f"--include-hook-events --verbose "
        f"{os.environ.get('SWE_CLAUDE_EXTRA_ARGS', '').strip()} "
        f"2>&1 | tee {shlex.quote(traj)}\n"
        f"echo $? > {done}\n"
    )
    await sb.write_file(launcher, launcher_body, user="agent")
    await sb.exec(f"chmod +x {launcher}", user="agent", timeout=30)

    env = {
        "ANTHROPIC_BASE_URL": middleware_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": "slime-actor",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
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
    exit_code = -2  # convention: -2 = budget exceeded
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(
            f"test -f {done} && cat {done}",
            user="agent",
            timeout=15,
            check=False,
        )
        if ec == 0:
            try:
                exit_code = int((out or "").strip() or "-1")
            except ValueError:
                exit_code = -1
            break
    return exit_code


async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = (
        f"cd {workdir} && git add -N . && "
        f"git diff -- . ':(exclude)PROBLEM_STATEMENT.md' "
        f"':(exclude)claude_code_trajectory.jsonl' "
        f"':(exclude).cagent_done' ':(exclude).cagent_run.sh'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


# ---------------------------------------------------------------------------
# Eval (fresh sandbox, apply diff, run dataset tests)
# ---------------------------------------------------------------------------
async def evaluate(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    swepro: dict | None = None,
    eval_cmd: str | None = None,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool, bool]:
    """Returns (reward, solved, applied_cleanly).

    No-test-cheating guarantee: the eval sandbox is built from the same image
    but starts CLEAN, so only the model-produced diff affects reward."""
    if not (swepro or eval_cmd):
        logger.warning("[e2b.evaluate] no swepro/eval_cmd; reward=0")
        return 0.0, False, True

    async with E2BSandbox(image) as ev:
        await ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            return 0.0, False, False

        if swepro:
            r, s = await _run_swepro(ev, workdir, swepro, timeout_sec)
            return r, s, True
        r, s = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        return r, s, True


async def _setup_swepro_assets(ev: Sandbox, swepro: dict) -> None:
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            text = Path(host_p).read_text()
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", text, user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}", user="root", check=True)


async def apply_pre_commands(ev: Sandbox, workdir: str, pre: list[str] | str) -> None:
    # Public: also called by generate.py to keep the work sandbox baseline
    # aligned with eval (sweb-style pre_commands typically `git checkout
    # <base_sha> -f`, so skipping in work sandbox makes the model's diff
    # context mismatch the eval base -> 100% apply failure).
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await ev.write_file(_PRE, "set -e\n" + body, user="agent")
    await ev.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}", user="agent", check=False, timeout=600)


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user="agent")
    for cmd in [
        f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ]:
        ec, _, _ = await ev.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict, timeout: int) -> tuple[float, bool]:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh "
        f"{json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent",
        check=False,
        timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        user="agent",
        check=False,
        timeout=120,
    )
    raw = await ev.read_file(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return (1.0 if solved else 0.0), solved


async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> tuple[float, bool]:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    return (1.0 if ec == 0 else 0.0), ec == 0
