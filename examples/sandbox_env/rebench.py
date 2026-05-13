"""SWE-rebench eval logic + sandbox runtime helpers.

Two layers in one file (slime-free):

- **Pure layer** (no IO): metadata unwraps, eval-script rendering, log-parser
  loading, ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` scoring.
- **Sandbox layer** (IO): the per-sample lifecycle that runs inside an
  already-started Inspire sandbox — workspace prep, candidate patch
  extraction, end-to-end eval invocation.

The pure layer used to live in ``rebench_eval.py`` and the sandbox layer in
``rebench_runtime.py``; both were small and tightly coupled, so they sit
together here.  The section banner halfway down marks the layer break.
"""
from __future__ import annotations

import importlib
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from .sandbox_runtime import LiveLog, run_sandbox_command, truncate_text, write_sandbox_file


# ---------------------------------------------------------------------------
# Pure layer: metadata, eval-script rendering, log parsing, scoring
# ---------------------------------------------------------------------------


DEFAULT_REBENCH_REPO_ROOT = str(
    Path(__file__).resolve().parents[4] / "data" / "raw_data" / "single" / "swe_rebench_v2" / "SWE-rebench-V2"
)

_REBENCH_LOG_PARSERS = None
_REBENCH_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def ensure_rebench_repo_on_path(rebench_repo_root: str) -> None:
    """Adds the SWE-rebench repo and its lib/ directory to sys.path so log parsers can be imported."""
    repo_root = Path(rebench_repo_root).resolve()
    lib_root = repo_root / "lib"
    for candidate in (repo_root, lib_root):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def normalize_rebench_test_name(name: str) -> str:
    """Strips timing annotations from test names so parsed output names match the FAIL_TO_PASS/PASS_TO_PASS lists."""
    normalized = str(name or "")
    for pattern in _REBENCH_TIMING_NORMALIZE_RES:
        normalized = pattern.sub("", normalized)
    return normalized.strip()


def normalize_rebench_test_cmds(value: Any) -> list[str]:
    """Normalizes install_config.test_cmd to a list of non-empty strings whether it was a string or list."""
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def resolve_rebench_workdir(metadata: dict[str, Any]) -> str:
    """Returns the in-sandbox repo working directory from repo_workdir or derived from the repo field."""
    workdir = str(metadata.get("repo_workdir") or "").strip()
    if workdir:
        return workdir
    repo = str(metadata.get("repo") or "").strip()
    parts = repo.split("/", 1)
    if len(parts) == 2 and parts[1]:
        return f"/{parts[1]}"
    raise RuntimeError("rebench metadata missing repo_workdir/repo")


def resolve_rebench_base_commit(metadata: dict[str, Any]) -> str:
    """Returns the base git commit hash from task metadata; raises if missing."""
    base_commit = str(metadata.get("base_commit") or "").strip()
    if not base_commit:
        raise RuntimeError("rebench metadata missing base_commit")
    return base_commit


def remote_rebench_patch_path(instance_id: str, kind: str) -> str:
    """Returns the /tmp/... path used to stage a patch file (test_patch or candidate_patch) inside the sandbox."""
    safe_instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_id or "task"))
    return f"/tmp/{safe_instance_id}.{kind}.diff"


def sandbox_default_user(metadata: dict[str, Any]) -> str | None:
    """Returns the non-root user to run sandbox commands as, from docker_image_default_user in metadata."""
    return str(metadata.get("docker_image_default_user") or "").strip() or None


def resolve_template_alias(metadata: dict[str, Any]) -> str:
    """Returns the Inspire sandbox template alias for this task instance; raises if missing."""
    alias = str(metadata.get("inspire_template") or metadata.get("template_alias") or "").strip()
    if not alias:
        instance_id = str(metadata.get("instance_id") or "unknown")
        raise RuntimeError(f"missing inspire_template/template_alias for instance_id={instance_id!r}")
    return alias


def render_rebench_eval_script(
    metadata: dict[str, Any],
    *,
    test_patch_file: str | None = None,
) -> str:
    """Builds the eval shell script: applies test_patch via git apply then runs all test_cmd entries."""
    install_config = metadata.get("install_config") or {}
    if not isinstance(install_config, dict):
        raise RuntimeError("rebench metadata.install_config must be an object")

    test_cmds = normalize_rebench_test_cmds(install_config.get("test_cmd"))
    if not test_cmds:
        raise RuntimeError("rebench metadata.install_config.test_cmd is empty")

    workdir = resolve_rebench_workdir(metadata)
    instance_id = str(metadata.get("instance_id") or "task")
    test_patch = str(metadata.get("test_patch") or "")
    test_patch_file = test_patch_file or remote_rebench_patch_path(instance_id, "test_patch")
    lines = [
        "set -e",
        f"cd {shlex.quote(workdir)}",
    ]

    if test_patch:
        lines.extend(
            [
                'echo "[stage] git_apply_test_patch"',
                f"if git apply -v --3way --recount --ignore-space-change --whitespace=nowarn {shlex.quote(test_patch_file)}",
                "then",
                "  :",
                "else",
                "  apply_rc=$?",
                "  exit ${apply_rc}",
                "fi",
            ]
        )

    for idx, test_cmd in enumerate(test_cmds, start=1):
        lines.append(f"printf '%s\\n' {shlex.quote(f'[stage] test_cmd_{idx}: {test_cmd}')}")
        lines.append(test_cmd)
    return "\n".join(lines)


def _get_rebench_log_parsers_module():
    """Lazily imports and caches the SWE-rebench agent.log_parsers module from the rebench repo."""
    global _REBENCH_LOG_PARSERS
    if _REBENCH_LOG_PARSERS is not None:
        return _REBENCH_LOG_PARSERS
    ensure_rebench_repo_on_path(os.environ.get("SWE_REBENCH_REPO_ROOT", DEFAULT_REBENCH_REPO_ROOT))
    _REBENCH_LOG_PARSERS = importlib.import_module("agent.log_parsers")
    return _REBENCH_LOG_PARSERS


def _get_rebench_log_parser(parser_name: str):
    """Looks up and returns a specific log parser callable by name from the SWE-rebench module."""
    module = _get_rebench_log_parsers_module()
    parser = getattr(module, "NAME_TO_PARSER", {}).get(parser_name)
    if parser is None:
        parser = getattr(module, parser_name, None)
    if parser is None:
        raise RuntimeError(f"Unknown SWE-rebench log parser: {parser_name}")
    return parser


def evaluate_rebench_result(
    metadata: dict[str, Any],
    *,
    eval_exit_code: int | None,
    eval_output: str,
) -> dict[str, Any]:
    """Parses test output, scores FAIL_TO_PASS/PASS_TO_PASS, and returns binary reward, dense_reward, and details."""
    install_config = metadata.get("install_config") or {}
    if not isinstance(install_config, dict):
        raise RuntimeError("rebench metadata.install_config must be an object")

    parser_name = str(install_config.get("log_parser") or "").strip()
    if not parser_name:
        raise RuntimeError("rebench metadata.install_config.log_parser is missing")

    parser = _get_rebench_log_parser(parser_name)
    parsed = parser(eval_output or "")
    normalized = {normalize_rebench_test_name(name): status for name, status in parsed.items()}
    passed = {name for name, status in normalized.items() if status == "PASSED"}
    failed = {name for name, status in normalized.items() if status in {"FAILED", "ERROR"}}

    fail_to_pass_expected = {
        normalize_rebench_test_name(name) for name in (metadata.get("FAIL_TO_PASS") or [])
    }
    pass_to_pass_expected = {
        normalize_rebench_test_name(name) for name in (metadata.get("PASS_TO_PASS") or [])
    }

    from_fail_to_pass = passed & fail_to_pass_expected
    failed_from_pass_to_pass = pass_to_pass_expected - passed

    fail_ratio = 1.0 if not fail_to_pass_expected else len(from_fail_to_pass) / len(fail_to_pass_expected)
    pass_ratio = 1.0 if not pass_to_pass_expected else (
        (len(pass_to_pass_expected) - len(failed_from_pass_to_pass)) / len(pass_to_pass_expected)
    )
    dense_reward = (fail_ratio + pass_ratio) / 2.0
    # Use FAIL_TO_PASS as the binary RL signal. Some SWE-rebench environments
    # have PASS_TO_PASS false negatives from unrelated CLI warnings/flakes; keep
    # those in metadata and dense_reward, but do not zero out a gold patch.
    solved = bool(fail_to_pass_expected) and len(from_fail_to_pass) == len(fail_to_pass_expected)

    return {
        "reward": 1.0 if solved else 0.0,
        "dense_reward": dense_reward,
        "solved": solved,
        "parser_name": parser_name,
        "passed_actual": sorted(passed),
        "failed_actual": sorted(failed),
        "from_fail_to_pass": sorted(from_fail_to_pass),
        "failed_from_pass_to_pass": sorted(failed_from_pass_to_pass),
        "fail_to_pass_expected": sorted(fail_to_pass_expected),
        "pass_to_pass_expected": sorted(pass_to_pass_expected),
    }


# ---------------------------------------------------------------------------
# Sandbox layer: workspace prep, patch extraction, end-to-end eval
# ---------------------------------------------------------------------------


async def prepare_workspace(
    sandbox: Any,
    metadata: dict[str, Any],
    *,
    user: str | None,
    wait_timeout: int,
    log: LiveLog,
) -> str:
    """Resets the sandbox repo to base_commit via git reset --hard + git clean, called before the agent runs."""
    workdir = resolve_rebench_workdir(metadata)
    base_commit = resolve_rebench_base_commit(metadata)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(workdir)}",
            'echo "=== REBENCH TARGET BASE COMMIT ==="',
            f"printf '%s\\n' {shlex.quote(base_commit)}",
            'git cat-file -e "${BASE_COMMIT}^{commit}"',
            'git reset --hard "${BASE_COMMIT}"',
            "git clean -fd",
            "git status --short",
        ]
    )
    result = await run_sandbox_command(
        sandbox,
        f"BASE_COMMIT={shlex.quote(base_commit)} bash -lc {shlex.quote(script)}",
        timeout=wait_timeout,
        user=user,
        cwd=workdir,
        log=log,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"failed to prepare rebench workspace: {result.output}".strip())
    return result.output


async def extract_candidate_patch(
    sandbox: Any,
    metadata: dict[str, Any],
    *,
    user: str | None,
    wait_timeout: int,
    log: LiveLog,
) -> str:
    """Extracts the agent's changes as a git diff from base_commit; stored in metadata only, not used for eval."""
    workdir = resolve_rebench_workdir(metadata)
    base_commit = resolve_rebench_base_commit(metadata)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(workdir)}",
            "git add -N -A",
            f"git diff --binary --no-color {shlex.quote(base_commit)}",
        ]
    )
    result = await run_sandbox_command(
        sandbox,
        f"bash -lc {shlex.quote(script)}",
        timeout=wait_timeout,
        user=user,
        cwd=workdir,
        log=log,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"failed to extract rebench candidate patch: {result.output}".strip())
    return result.stdout


async def run_rebench_eval(
    sandbox: Any,
    metadata: dict[str, Any],
    *,
    user: str | None,
    eval_log_path: Path | None,
    aggregate_log: LiveLog | None = None,
    wait_timeout: int,
    preview_limit: int,
    reached_turn_limit: bool,
    last_generation_finish_reason: str,
) -> tuple[float, dict[str, Any]]:
    """Writes test_patch to sandbox, runs the eval script on the agent-modified workspace, returns reward + metadata."""
    log = LiveLog(eval_log_path, mirror=aggregate_log)
    instance_id = str(metadata.get("instance_id") or "task")
    test_patch_remote_path = remote_rebench_patch_path(instance_id, "test_patch")
    test_patch = str(metadata.get("test_patch") or "")
    if test_patch:
        mkdir_result = await run_sandbox_command(
            sandbox,
            "mkdir -p /tmp",
            timeout=30,
            user=user,
            log=log,
        )
        if mkdir_result.exit_code != 0:
            raise RuntimeError(f"failed to create /tmp for test patch: {mkdir_result.output}")
        await write_sandbox_file(sandbox, test_patch_remote_path, test_patch.encode("utf-8"), user=user, log=log)

    rendered_eval_script = render_rebench_eval_script(metadata, test_patch_file=test_patch_remote_path)
    eval_shell_script = "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false\n" + rendered_eval_script
    result = await run_sandbox_command(
        sandbox,
        f"/bin/bash -c {shlex.quote(eval_shell_script)}",
        timeout=wait_timeout,
        user=user,
        cwd=resolve_rebench_workdir(metadata),
        log=log,
    )
    eval_output = result.output
    rebench_eval = evaluate_rebench_result(
        metadata,
        eval_exit_code=result.exit_code,
        eval_output=eval_output,
    )
    reward = float(rebench_eval["reward"])
    return reward, {
        "dense_reward": float(rebench_eval["dense_reward"]),
        "solved": bool(rebench_eval["solved"]),
        "log_parser": rebench_eval["parser_name"],
        "passed_actual": rebench_eval["passed_actual"],
        "failed_actual": rebench_eval["failed_actual"],
        "from_fail_to_pass": rebench_eval["from_fail_to_pass"],
        "failed_from_pass_to_pass": rebench_eval["failed_from_pass_to_pass"],
        "fail_to_pass_expected": rebench_eval["fail_to_pass_expected"],
        "pass_to_pass_expected": rebench_eval["pass_to_pass_expected"],
        "eval_exit_code": result.exit_code,
        "eval_output_preview": truncate_text(eval_output, preview_limit),
        "eval_script_path": f"/tmp/{instance_id}.eval.sh",
        "eval_script_preview": truncate_text(rendered_eval_script, preview_limit),
        "raw_reward": reward,
        "reached_turn_limit": reached_turn_limit,
        "last_generation_finish_reason": last_generation_finish_reason,
    }
