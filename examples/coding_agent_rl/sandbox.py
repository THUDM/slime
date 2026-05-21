"""E2B sandbox helpers for the coding-agent RL demo.

This file is the SANDBOX BACKEND. It owns boot/kill, exec, file I/O, the
``install_*`` bootstraps, the long-running agent spawn (with the E2B-specific
done-marker poll pattern), patch capture, and the fresh-sandbox test runner.

Public surface used by ``generate.py``::

    async with E2BSandbox(image) as sb:
        await install_node22(sb, host_tarball)
        await install_claude_code(sb, host_tarball)
        await ensure_agent_user(sb, workdir)
        await sb.write_text(...); await sb.exec(...)
        await run_claude_code(sb, ...)
        diff = await git_diff(sb, workdir)

    reward, solved, applied = await evaluate(
        image=..., workdir=..., diff_text=..., swepro=..., f2p_script=...,
        eval_cmd=..., pre_commands=..., timeout_sec=...,
    )

---

Forking this file:

* Swap E2B for local docker / modal / your own VM cloud: rewrite ``E2BSandbox``
  to expose the same 5 async primitives (exec / upload / write_text / read_text
  / __aenter__ / __aexit__). Everything below it (install_node22, run_claude_code,
  evaluate, etc.) only uses those primitives -- no E2B-specific code.

* Swap claude_code for codex: replace ``install_claude_code`` and the
  shell_command + env block inside ``run_claude_code``. The done-marker poll
  pattern is generic and worth keeping for any long-running agent.

* Add a new evaluation mode: extend ``evaluate()`` with another elif branch.
  The 3 existing modes (swepro / f2p_script / eval_cmd) are self-contained
  inner functions and easy to copy.
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
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (env-driven so the same code runs across clusters)
# ---------------------------------------------------------------------------
# Sandbox metadata passed verbatim into ``AsyncSandbox.create``'s ``metadata=``
# kwarg. The metadata content is cluster-specific (E2B-style backends use it
# for routing tags like ``glm-platform/image``, ``provider/size``, etc.) so it
# MUST NOT be hardcoded here. Two ways to provide it; the file path wins:
#
#   * ``SWE_SANDBOX_METADATA_FILE`` — absolute path to a JSON file with a
#     top-level object of string keys to string values. Example::
#
#         {
#           "glm-platform/group": "rl-rollout",
#           "glm-platform/size":  "lg"
#         }
#
#   * ``SWE_SANDBOX_METADATA_JSON`` — inline JSON object string (legacy /
#     small-config path); same shape as above.
#
# If both are set, the file path wins. Either may be empty/missing, in which
# case we send an empty metadata dict (the only key we then attach is the
# image-routing key derived from the dataset row; see ``__aenter__`` below).
def _parse_sandbox_metadata() -> dict[str, str]:
    file_path = os.environ.get("SWE_SANDBOX_METADATA_FILE", "").strip()
    raw = ""
    if file_path:
        try:
            raw = Path(file_path).read_text()
        except OSError as e:
            logger.warning("[sandbox] SWE_SANDBOX_METADATA_FILE=%s unreadable, ignoring: %s",
                           file_path, e)
            raw = ""
    if not raw:
        raw = os.environ.get("SWE_SANDBOX_METADATA_JSON", "").strip()
    if not raw:
        return {}
    try:
        md = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("[sandbox] sandbox metadata not valid JSON, ignoring: %s", e)
        return {}
    if not isinstance(md, dict):
        logger.warning("[sandbox] sandbox metadata must be a JSON object, got %s", type(md).__name__)
        return {}
    return {str(k): str(v) for k, v in md.items()}

SANDBOX_METADATA = _parse_sandbox_metadata()

# Key used to route to the right image on E2B-style backends. The dataset row
# specifies the image string; this env var lets the cluster pick which metadata
# key carries it (default matches the GLM internal E2B gateway).
SANDBOX_IMAGE_METADATA_KEY = os.environ.get(
    "SWE_SANDBOX_IMAGE_METADATA_KEY", "glm-platform/image",
)

# Sandbox wall-clock lifetime in seconds. E2B kills the sandbox `timeout` seconds
# after creation, regardless of activity (it is NOT an idle timeout — commands.run
# does not auto-extend). SDK default is 300s; with claude-code agent runs > 5min
# we need much more. We still kill the sandbox immediately via __aexit__ when the
# context block exits, so this is just an upper-bound safety net to prevent leaks
# if the host process crashes mid-run. Pro account cap is 86_400s (24h).
SANDBOX_LIFETIME_SEC = int(os.environ.get("SWE_SANDBOX_LIFETIME_SEC", "3600"))

# RPC retry knobs. The e2b SDK rides on httpx + h2; long-lived sandboxes
# accumulate transient h2 ProtocolErrors / connection resets (server-side
# GOAWAY then httpx state-machine bug — see encode/httpx#2761). Each RPC is
# idempotent, so we retry inline. Tune via SWE_RPC_RETRIES (default 3) and
# SWE_RPC_RETRY_BASE_SEC (default 1.0; exponential).
SWE_RPC_RETRIES = int(os.environ.get("SWE_RPC_RETRIES", "3"))
SWE_RPC_RETRY_BASE_SEC = float(os.environ.get("SWE_RPC_RETRY_BASE_SEC", "1.0"))

# Paths inside the sandbox (avoid clashes with whatever the image ships).
_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_F2P = "/workspace/__cagent_f2p__.py"
_SWEPRO_DIR = "/workspace/swepro_eval"


def _is_transient_rpc_error(e: BaseException) -> bool:
    """True if ``e`` looks like a transient client-side / network RPC failure
    safe to retry. Includes httpx/h2 ProtocolError / WriteError / ReadError /
    SSLError, and timeouts. Does NOT include CommandExitException (real cmd
    failure) or SandboxException 'resource does not exist' (sandbox already
    GC'd — retry won't bring it back)."""
    name = type(e).__name__
    if name in {"ProtocolError", "LocalProtocolError",
                "WriteError", "ReadError", "ConnectError", "ConnectTimeout",
                "ReadTimeout", "WriteTimeout", "PoolTimeout",
                "RemoteProtocolError", "SSLError"}:
        return True
    # e2b SandboxException with "resource does not exist" = sandbox dead; bail.
    # Other SandboxException variants (e.g. network 5xx) may be retryable.
    msg = str(e)
    if name == "SandboxException":
        if "does not exist" in msg or "STOPPED state" in msg:
            return False
        return True
    return False


async def _rpc_retry(op_name: str, coro_factory):
    """Run ``coro_factory()`` with up to SWE_RPC_RETRIES attempts; only retry
    on _is_transient_rpc_error. coro_factory must be a 0-arg callable that
    returns a fresh awaitable each call (coroutines can only be awaited once)."""
    last_err = None
    for attempt in range(SWE_RPC_RETRIES):
        try:
            return await coro_factory()
        except Exception as e:
            if not _is_transient_rpc_error(e):
                raise
            last_err = e
            if attempt + 1 < SWE_RPC_RETRIES:
                backoff = SWE_RPC_RETRY_BASE_SEC * (2 ** attempt)
                logger.debug("[e2b] %s transient %s, retry %d/%d in %.1fs: %s",
                             op_name, type(e).__name__, attempt + 1, SWE_RPC_RETRIES,
                             backoff, str(e)[:120])
                await asyncio.sleep(backoff)
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Sandbox primitives
# ---------------------------------------------------------------------------
class E2BSandbox:
    """Async context manager around ``e2b.AsyncSandbox`` with the few primitives
    this demo actually uses."""

    def __init__(self, image: str, *, timeout: int | None = None) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else SANDBOX_LIFETIME_SEC
        self._sb = None
        self.sandbox_id = ""

    async def __aenter__(self) -> "E2BSandbox":
        from e2b import AsyncSandbox  # type: ignore
        # Internal E2B routing reads the image from metadata, not template.
        # The metadata schema is cluster-specific (key name + extra routing
        # tags) so the base dict comes from SWE_SANDBOX_METADATA_FILE /
        # SWE_SANDBOX_METADATA_JSON — never hardcoded.
        md = dict(SANDBOX_METADATA)
        md.setdefault(SANDBOX_IMAGE_METADATA_KEY, self.image)
        self._sb = await AsyncSandbox.create(timeout=self.timeout, metadata=md)
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[e2b] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> tuple[int, str, str]:
        # e2b's commands.run raises CommandExitException on non-zero exit by
        # default (no flag to suppress). Catch it so callers can rely on the
        # `check` flag for raise/no-raise control. Transient h2/network errors
        # are retried inline by _rpc_retry (does not include CommandExitException).
        from e2b.sandbox.commands.command_handle import CommandExitException
        try:
            res = await _rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd, user=user, envs=env, timeout=timeout,
                    on_stdout=lambda s: None, on_stderr=lambda s: None,
                ),
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def upload(self, host_path: str | Path, sandbox_path: str, *, user: str = "root") -> None:
        # Retry the *whole* upload; e2b SDK uses a single POST so re-opening
        # the file each attempt is required.
        async def _do():
            with open(host_path, "rb") as fp:
                await self._sb.files.write(
                    sandbox_path, fp, user=user,
                    gzip=False, use_octet_stream=True, request_timeout=600,
                )
        await _rpc_retry(f"upload({Path(host_path).name})", _do)

    async def write_text(self, sandbox_path: str, content: str, *, user: str = "root") -> None:
        await _rpc_retry(
            f"write_text({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_text(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await _rpc_retry(
                f"read_text({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Sandbox bootstrap (Node + Claude Code + agent user)
# ---------------------------------------------------------------------------
async def install_node22(sb: E2BSandbox, host_tarball: Path) -> None:
    """Node 22 over the base image's Node (Debian 12 ships 16 which can't run
    claude-code's cli.js). ``npm prefix=/usr/local`` is required for the
    sweap-images family (see e2b_sweap_image_node_and_npm_quirks).

    Decompresses ``.xz`` on the host (cached) so sandboxes without ``xz-utils``
    (e.g. sweap-images / ScaleSWE) can still run plain ``tar xf``.
    """
    host_tarball = Path(host_tarball)
    if host_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"coding_agent_rl.{host_tarball.stem}.tar"
        if not plain.exists():
            tmp = plain.with_suffix(".tar.partial")
            with lzma.open(host_tarball, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, plain)
        host_tarball = plain
    await sb.upload(host_tarball, "/tmp/node22.tar")
    await sb.exec(
        "set -e && mkdir -p /opt/node22 && "
        "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
        "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
        "ln -sf /opt/node22/bin/npm  /usr/local/bin/npm && "
        "ln -sf /opt/node22/bin/npx  /usr/local/bin/npx && "
        "hash -r 2>/dev/null || true && node --version && npm --version",
        user="root", timeout=180, check=True,
    )


async def install_claude_code(sb: E2BSandbox, host_tarball: Path) -> None:
    await sb.upload(host_tarball, "/tmp/claude-code.tgz")
    await sb.exec(
        "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/claude-code.tgz "
        "&& ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
        user="root", timeout=300, check=True,
    )


async def ensure_agent_user(sb: E2BSandbox, workdir: str) -> None:
    """Unprivileged 'agent' user that owns the workdir + can ``git diff``."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent && "
        f"mkdir -p /home/agent/.claude && "
        f"echo '{{\"hasCompletedOnboarding\": true, \"bypassPermissionsModeAccepted\": true}}' "
        f"| tee /home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
        f"chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
        user="root", check=True, timeout=60,
    )


async def apply_before_repo_set_cmd(sb: E2BSandbox, workdir: str, swepro: dict) -> None:
    before = swepro.get("before_repo_set_cmd") if swepro else None
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec("mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup",
                  user="root", check=True)
    await sb.write_text("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh",
                  user="agent", check=False, timeout=600)


# ---------------------------------------------------------------------------
# Agent run (claude-code spawn + done-marker poll)
# ---------------------------------------------------------------------------
async def run_claude_code(
    sb: E2BSandbox,
    *,
    workdir: str,
    session_id: str,
    middleware_url: str,
    prompt: str,
    time_budget_sec: int,
) -> int:
    """Spawn claude-code detached + poll a done-marker file.

    E2B's gateway resets streaming HTTP/2 responses around 6.5 minutes, so we
    can't keep a long-lived foreground exec. We write the exit code into a
    marker file and check it every 15 s via short-lived RPCs."""
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
    await sb.write_text(launcher, launcher_body, user="agent")
    await sb.exec(f"chmod +x {launcher}", user="agent", timeout=30)

    env = {
        "ANTHROPIC_BASE_URL": middleware_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,    # middleware reads this as session id
        "ANTHROPIC_MODEL": "slime-actor",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
    env_keys = ",".join(env.keys())
    await sb.exec(
        f"runuser -u agent --whitelist-environment={env_keys}"
        f" -- bash -c 'setsid {launcher} < /dev/null > /dev/null 2>&1 &'",
        user="root", env=env, timeout=30, check=True,
    )

    # Poll cadence: short enough to act as a sandbox-side keep-alive (the E2B
    # platform GC-s sandboxes that look idle; setsid'd claude-code is
    # invisible to it). 5s is well under any plausible eviction timeout while
    # still being cheap compared to the multi-minute agent run.
    deadline = time.time() + time_budget_sec
    exit_code = -2  # convention: -2 = budget exceeded
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(
            f"test -f {done} && cat {done}",
            user="agent", timeout=15, check=False,
        )
        if ec == 0:
            try:
                exit_code = int((out or "").strip() or "-1")
            except ValueError:
                exit_code = -1
            break
    return exit_code


async def git_diff(sb: E2BSandbox, workdir: str) -> str:
    cmd = (
        f"cd {workdir} && git add -N . && "
        f"git diff -- . ':(exclude)PROBLEM_STATEMENT.md' "
        f"':(exclude)claude_code_trajectory.jsonl' "
        f"':(exclude).cagent_done' ':(exclude).cagent_run.sh'"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


# ---------------------------------------------------------------------------
# Eval-side test runners (run inside a fresh sandbox)
# ---------------------------------------------------------------------------
async def evaluate(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    swepro: dict | None = None,
    f2p_script: str | None = None,
    eval_cmd: str | None = None,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool, bool]:
    """Boot a fresh sandbox, apply the diff, run the dataset's tests. Returns
    ``(reward, solved, applied_cleanly)``. The "no test cheating" guarantee:
    the eval sandbox is built from the same image but starts CLEAN, so the
    only thing that affects reward is the model-produced diff."""
    if not (swepro or f2p_script or eval_cmd):
        logger.warning("[e2b.evaluate] no swepro/f2p_script/eval_cmd; reward=0")
        return 0.0, False, True

    async with E2BSandbox(image) as ev:
        await ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await _apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            return 0.0, False, False

        if swepro:
            r, s = await _run_swepro(ev, workdir, swepro, timeout_sec)
            return r, s, True
        if f2p_script:
            r, s = await _run_f2p(ev, workdir, f2p_script, timeout_sec)
            return r, s, True
        r, s = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        return r, s, True


async def _setup_swepro_assets(ev: E2BSandbox, swepro: dict) -> None:
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            text = Path(host_p).read_text()
            await ev.write_text(f"{_SWEPRO_DIR}/{dst}", text, user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}",
                  user="root", check=True)


async def _apply_pre_commands(ev: E2BSandbox, workdir: str, pre: list[str] | str) -> None:
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await ev.write_text(_PRE, "set -e\n" + body, user="agent")
    await ev.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}",
                  user="agent", check=False, timeout=600)


async def _apply_diff(ev: E2BSandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_text(_PATCH, diff_text, user="agent")
    for cmd in [
        f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ]:
        ec, _, _ = await ev.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _run_swepro(ev: E2BSandbox, workdir: str, swepro: dict, timeout: int) -> tuple[float, bool]:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh "
        f"{json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent", check=False, timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}",
        user="agent", check=False, timeout=120,
    )
    raw = await ev.read_text(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return (1.0 if solved else 0.0), solved


async def _run_f2p(ev: E2BSandbox, workdir: str, script: str, timeout: int) -> tuple[float, bool]:
    await ev.write_text(_F2P, script, user="agent")
    ec, _, _ = await ev.exec(
        f"cd {workdir} && python3 -m pytest -xvs {_F2P}",
        user="agent", check=False, timeout=timeout,
    )
    return (1.0 if ec == 0 else 0.0), ec == 0


async def _run_eval_cmd(ev: E2BSandbox, workdir: str, cmd: str, timeout: int) -> tuple[float, bool]:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    return (1.0 if ec == 0 else 0.0), ec == 0
