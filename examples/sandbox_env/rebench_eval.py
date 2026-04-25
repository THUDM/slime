from __future__ import annotations

import importlib
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any


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
    repo_root = Path(rebench_repo_root).resolve()
    lib_root = repo_root / "lib"
    for candidate in (repo_root, lib_root):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def normalize_rebench_test_name(name: str) -> str:
    normalized = str(name or "")
    for pattern in _REBENCH_TIMING_NORMALIZE_RES:
        normalized = pattern.sub("", normalized)
    return normalized.strip()


def normalize_rebench_test_cmds(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def resolve_rebench_workdir(metadata: dict[str, Any]) -> str:
    workdir = str(metadata.get("repo_workdir") or "").strip()
    if workdir:
        return workdir
    repo = str(metadata.get("repo") or "").strip()
    parts = repo.split("/", 1)
    if len(parts) == 2 and parts[1]:
        return f"/{parts[1]}"
    raise RuntimeError("rebench metadata missing repo_workdir/repo")


def resolve_rebench_base_commit(metadata: dict[str, Any]) -> str:
    base_commit = str(metadata.get("base_commit") or "").strip()
    if not base_commit:
        raise RuntimeError("rebench metadata missing base_commit")
    return base_commit


def remote_rebench_patch_path(instance_id: str, kind: str) -> str:
    safe_instance_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_id or "task"))
    return f"/tmp/{safe_instance_id}.{kind}.diff"


def render_rebench_eval_script(
    metadata: dict[str, Any],
    *,
    test_patch_file: str | None = None,
) -> str:
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
    global _REBENCH_LOG_PARSERS
    if _REBENCH_LOG_PARSERS is not None:
        return _REBENCH_LOG_PARSERS
    ensure_rebench_repo_on_path(os.environ.get("ROCK_SWE_REBENCH_REPO_ROOT", DEFAULT_REBENCH_REPO_ROOT))
    _REBENCH_LOG_PARSERS = importlib.import_module("agent.log_parsers")
    return _REBENCH_LOG_PARSERS


def _get_rebench_log_parser(parser_name: str):
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
    solved = (
        eval_exit_code == 0
        and len(from_fail_to_pass) == len(fail_to_pass_expected)
        and not failed_from_pass_to_pass
    )

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
