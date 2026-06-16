"""SWE task layer: workspace prep, diff capture, and fresh-sandbox eval.

Harness-agnostic on purpose -- nothing here is Claude-specific. ``SWE_PROMPT`` is
the task instruction (semantics, not CLI syntax); ``prepare_workspace`` /
``git_diff`` / ``evaluate`` work with any harness. The only place a task meets a
harness is the prompt, which the orchestrator passes into ``harness.run()``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from slime.agent import sandbox as agent_sandbox
from slime.agent.sandbox import E2BSandbox, Sandbox
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Paths inside the sandbox (avoid clashes with image-shipped paths).
_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_F2P = "/workspace/__cagent_f2p__.py"
_SWEPRO_DIR = "/workspace/swepro_eval"

SWE_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit.",
)


# ---------------------------------------------------------------------------
# Dataset row -> SWE metadata
#
# ``get_metadata(sample)`` defines the ``md`` dict schema consumed by
# ``prepare_workspace`` / ``evaluate``. Two dataset shapes are normalized:
#
#     image:             str        # sandbox image
#     workdir:           str        # repo path inside the sandbox
#     problem_statement: str        # issue body (falls back to sample.prompt)
#     swepro:            dict|None  # SWE-bench Pro test harness (preferred)
#     eval_cmd:          str|None   # shell command (exit 0 = solved)
#     f2p_script:        str|None   # sweb pytest file (exit 0 = solved)
#     pre_commands:      list|str|None
#
# This layer is pure data: it only *extracts* fields, it never decides how they
# run in the sandbox. ``f2p_script`` (a self-contained pytest file ending in
# ``sys.exit(pytest.main(...))``) is carried verbatim; ``evaluate`` materializes
# and runs it via ``write_file`` so no shell-quoting workaround is needed here.
# ---------------------------------------------------------------------------
def get_metadata(sample: Sample) -> dict[str, Any]:
    """Normalize the two dataset schemas (flat vs ``remote_env_info``)."""
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    return {
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "swepro": m.get("swepro"),
        "eval_cmd": m.get("eval_cmd"),
        "f2p_script": rem.get("f2p_script"),
        "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
    }


def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


# ---------------------------------------------------------------------------
# Workspace prep (agent sandbox, before harness.run)
# ---------------------------------------------------------------------------
async def prepare_workspace(sb: Sandbox, workdir: str, md: dict) -> None:
    """Apply swepro setup + pre_commands, then drop PROBLEM_STATEMENT.md.

    Assumes the agent user already owns ``workdir`` (the harness's ``run()`` calls
    ``ensure_agent_user``; the orchestrator runs this before ``run()`` and the
    agent user is created lazily there). To stay independent of call order we
    create the agent user here too -- it is idempotent.
    """
    await agent_sandbox.ensure_agent_user(sb, workdir)
    swepro = md.get("swepro")
    if swepro:
        await apply_before_repo_set_cmd(sb, workdir, swepro)
    pre_commands = md.get("pre_commands")
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        md.get("problem_statement") or "",
        user="agent",
    )


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict) -> None:
    """Run swepro['before_repo_set_cmd'] in the sandbox if present (no-op if not)."""
    before = swepro.get("before_repo_set_cmd")
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup", user="root", check=True
    )
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)


async def apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:
    # Public: also called for the work sandbox to keep its baseline aligned with
    # eval (sweb-style pre_commands typically `git checkout <base_sha> -f`, so
    # skipping in the work sandbox makes the model's diff context mismatch the
    # eval base -> 100% apply failure).
    if isinstance(pre, str):
        body = pre.replace("\\n", "\n")
    else:
        body = "\n".join(c for c in (pre or []) if c)
    await sb.write_file(_PRE, "set -e\n" + body, user="agent")
    await sb.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}", user="agent", check=False, timeout=600)


# ---------------------------------------------------------------------------
# Diff capture (agent sandbox, after harness.run)
# ---------------------------------------------------------------------------
async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = f"cd {workdir} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
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
    f2p_script: str | None = None,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool]:
    """Returns (reward, applied_cleanly).

    Three mutually-exclusive grading paths, in priority order: swepro test
    harness, a shell ``eval_cmd``, or a self-contained ``f2p_script`` pytest
    file. All resolve to "exit 0 == solved", and reward is 1.0 iff solved.

    No-test-cheating guarantee: the eval sandbox is built from the same image
    but starts CLEAN, so only the model-produced diff affects reward."""
    if not (swepro or eval_cmd or f2p_script):
        logger.warning("[e2b.evaluate] no swepro/eval_cmd/f2p_script; reward=0")
        return 0.0, True

    async with E2BSandbox(image) as ev:
        await agent_sandbox.ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            return 0.0, False

        if swepro:
            r, _ = await _run_swepro(ev, workdir, swepro, timeout_sec)
        elif eval_cmd:
            r, _ = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        else:
            r, _ = await _run_f2p_script(ev, workdir, f2p_script, timeout_sec)
        return r, True


async def _setup_swepro_assets(ev: Sandbox, swepro: dict) -> None:
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for k, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_p = swepro.get(k)
        if host_p:
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", Path(host_p), user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}", user="root", check=True)


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


async def _run_f2p_script(ev: Sandbox, workdir: str, script: str, timeout: int) -> tuple[float, bool]:
    # sweb f2p_script is a self-contained pytest file ending in
    # `sys.exit(pytest.main([...]))`; write it verbatim (no shell quoting) and
    # let python's exit code carry the pass/fail signal.
    await ev.write_file(_F2P, script, user="agent")
    ec, _, _ = await ev.exec(f"cd {workdir} && python {_F2P}", user="agent", check=False, timeout=timeout)
    return (1.0 if ec == 0 else 0.0), ec == 0
