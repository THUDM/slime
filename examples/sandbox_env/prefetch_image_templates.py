#!/usr/bin/env python3
"""Prefetch SWE-rebench public images as Inspire templates.

Each template starts from the task's published `image_name` and preinstalls the
agent/runtime pieces described by the Rock agent yaml so rollout can later use
`installed: true` and skip reinstalling Node/Python/model-service dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from _inspire_sandbox_bootstrap import bootstrap_inspire_sandbox_path

bootstrap_inspire_sandbox_path()
os.environ.setdefault("SBX_API_URL", "https://qz-sbx-api.sii.edu.cn")

from inspire_sandbox import SandboxSpecCode, Template


AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
ROCK_ROOT = AVALANCHE_ROOT / "ROCK"
DEFAULT_TASKS_JSON = AVALANCHE_ROOT / "data" / "raw_data" / "single" / "swe_rebench_v2" / "data" / "train-00000-of-00001.json"
DEFAULT_OUT_DIR = AVALANCHE_ROOT / "data" / "raw_data" / "single" / "swe_rebench_v2" / "data"
DEFAULT_SUCCESS = DEFAULT_OUT_DIR / "prefetch_image_template_success.jsonl"
DEFAULT_FAILURE = DEFAULT_OUT_DIR / "prefetch_image_template_failure.jsonl"
DEFAULT_AGENT_CONFIG = Path(__file__).with_name("rock_agent_qwen_rebench_template.yaml")
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "data_output"
DEFAULT_SPEC = "G_C4"
AGENT_RUNTIME_ROOT = "/home/user/.rock/preinstalled/agent-runtime"
MODEL_SERVICE_RUNTIME_ROOT = "/home/user/.rock/preinstalled/model-service-runtime"
ALIAS_PREFIX = "rebench-"
POLL_INTERVAL_SECONDS = 5
BUILD_TIMEOUT_SECONDS = 3600
RETRYABLE_IMAGE_ACCESS_MAX_ATTEMPTS = 5
RETRYABLE_IMAGE_ACCESS_SLEEP_SECONDS = 10
SUPPORTED_SPECS = {"G_C1", "G_C2", "G_C4"}
SUPPORTED_PYTHON_VERSIONS = {"default", "3.11", "3.12"}
SUPPORTED_NODE_VERSIONS = {"default", "22.18.0"}


class _TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, message: str) -> None:
        text = str(message)
        with self._lock:
            print(text, flush=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")


LOGGER: _TeeLogger | None = None
TemplateCommand = dict[str, str | None]


def _emit(message: str) -> None:
    global LOGGER
    if LOGGER is None:
        print(message, flush=True)
        return
    LOGGER.emit(message)


def _emit_json(payload: dict[str, Any]) -> None:
    _emit(json.dumps(payload, ensure_ascii=False))


def _default_log_path(log_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"prefetch_image_templates_{timestamp}.log"


def _ensure_rock_root_on_path() -> None:
    value = str(ROCK_ROOT.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


_ensure_rock_root_on_path()
from rock import env_vars  # noqa: E402


def _parse_spec(value: str) -> str:
    spec = value.strip().upper()
    if spec not in SUPPORTED_SPECS:
        raise ValueError(f"Unsupported --spec={value!r}; expected one of G_C1/G_C2/G_C4")
    return spec


def _normalize_instance_alias(instance_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    return f"{ALIAS_PREFIX}{normalized}"


def _instance_id(row: dict[str, Any]) -> str:
    return str(row.get("instance_id", "")).strip()


def _log_file_path(log_file: str, log_dir: str) -> Path:
    if log_file.strip():
        return Path(log_file)
    return _default_log_path(Path(log_dir))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def _load_manifest_instance_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    instance_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = _instance_id(row)
            if instance_id:
                instance_ids.add(instance_id)
    return instance_ids


def _api_client():
    from inspire_sandbox.api.client_sync import get_api_client
    from inspire_sandbox.connection_config import ConnectionConfig

    config = ConnectionConfig()
    return get_api_client(config, require_api_key=True, require_access_token=False)


def _delete_template_by_alias(alias: str) -> bool:
    from inspire_sandbox.api.client.api.templates.delete_v_1_templates_template_id import sync_detailed as _delete_template
    from inspire_sandbox.template_sync.build_api import get_v1_templates_aliases_alias

    try:
        client = _api_client()
        res = get_v1_templates_aliases_alias.sync_detailed(alias=alias, client=client)
        if res.status_code == 404:
            return False
        if res.parsed is None:
            return False
        template_id = getattr(res.parsed, "template_id", None)
        if not template_id:
            return False
        deleted = _delete_template(template_id=template_id, client=client)
        return deleted.status_code == 204
    except Exception as exc:
        _emit(f"[warn] failed to delete template alias={alias!r}: {exc}")
        return False


def _load_agent_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return _expand_runtime_placeholders(payload)


def _runtime_install_root(kind: str) -> str:
    if kind == "agent":
        return AGENT_RUNTIME_ROOT
    if kind == "model_service":
        return MODEL_SERVICE_RUNTIME_ROOT
    raise ValueError(f"Unsupported runtime kind: {kind}")


def _runtime_bin_dir(kind: str) -> str:
    return f"{_runtime_install_root(kind)}/runtime-env/bin"


def _runtime_path_vars() -> dict[str, str]:
    agent_bin = _runtime_bin_dir("agent")
    model_bin = _runtime_bin_dir("model_service")
    return {
        "AGENT_RUNTIME_ROOT": AGENT_RUNTIME_ROOT,
        "MODEL_SERVICE_RUNTIME_ROOT": MODEL_SERVICE_RUNTIME_ROOT,
        "AGENT_RUNTIME_BIN_DIR": agent_bin,
        "MODEL_SERVICE_RUNTIME_BIN_DIR": model_bin,
        "AGENT_RUNTIME_NODE": f"{agent_bin}/node",
        "AGENT_RUNTIME_NPM": f"{agent_bin}/npm",
        "AGENT_RUNTIME_QWEN": f"{agent_bin}/qwen",
        "AGENT_RUNTIME_IFLOW": f"{agent_bin}/iflow",
        "MODEL_SERVICE_RUNTIME_PYTHON": f"{model_bin}/python",
        "MODEL_SERVICE_RUNTIME_PIP": f"{model_bin}/pip",
        "MODEL_SERVICE_RUNTIME_ROCK": f"{model_bin}/rock",
    }


def _expand_runtime_placeholders(value: Any) -> Any:
    replacements = _runtime_path_vars()
    if isinstance(value, str):
        text = value
        for key, replacement in replacements.items():
            text = text.replace(f"${{{key}}}", replacement)
        return text
    if isinstance(value, list):
        return [_expand_runtime_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_runtime_placeholders(v) for k, v in value.items()}
    return value


def _quote(value: str) -> str:
    return shlex.quote(value)


def _bin_path(bin_dir: str, executable: str) -> str:
    return f"{bin_dir}/{executable}"


def _quoted_bin_path(bin_dir: str, executable: str) -> str:
    return _quote(_bin_path(bin_dir, executable))


def _step(command: str, *, user: str | None = None) -> TemplateCommand:
    return {"command": command, "user": user}


def _bash_command(name: str, command: str, *, user: str | None = None) -> TemplateCommand:
    return _step(_bash_step(name, command), user=user)


def _bash_step(name: str, command: str) -> str:
    script = f"set -euxo pipefail\nprintf '[prefetch-step] %s\\n' {_quote(name)}\n{command}"
    return f"bash -lc {_quote(script)}"


def _append_path_expr(bin_dir: str) -> str:
    return f'export PATH="$PATH":{_quote(bin_dir)}'


def _with_python_runtime(bin_dir: str, command: str) -> str:
    python_bin = _quoted_bin_path(bin_dir, "python")
    pip_bin = _quoted_bin_path(bin_dir, "pip")
    return (
        f"{_append_path_expr(bin_dir)} && "
        f'python() {{ {python_bin} "$@"; }} && '
        f'pip() {{ {pip_bin} "$@"; }} && '
        f"{command}"
    )


def _with_node_runtime(bin_dir: str, command: str) -> str:
    node_bin = _quoted_bin_path(bin_dir, "node")
    npm_bin = _quoted_bin_path(bin_dir, "npm")
    return (
        f"{_append_path_expr(bin_dir)} && "
        f'node() {{ {node_bin} "$@"; }} && '
        f'npm() {{ {npm_bin} "$@"; }} && '
        f"{command}"
    )


def _python_verify_command(bin_dir: str) -> str:
    return _with_python_runtime(
        bin_dir,
        f"test -x {_quoted_bin_path(bin_dir, 'python')} && "
        f"test -x {_quoted_bin_path(bin_dir, 'pip')} && "
        "python --version && pip --version",
    )


def _node_verify_command(bin_dir: str) -> str:
    return _with_node_runtime(
        bin_dir,
        f"test -x {_quoted_bin_path(bin_dir, 'node')} && "
        f"test -x {_quoted_bin_path(bin_dir, 'npm')} && "
        "node --version && npm --version",
    )


def _make_verbose(command: str) -> str:
    text = str(command)
    text = text.replace("apt-get update -qq", "apt-get update")
    text = text.replace("apt-get install -qq -y", "apt-get install -y")
    text = text.replace("wget -q -O", "wget -O")
    return text


def _apt_prereq_command() -> str:
    packages = "git wget xz-utils ca-certificates"
    raw = (
        "command -v git >/dev/null 2>&1 && command -v wget >/dev/null 2>&1 && command -v xz >/dev/null 2>&1"
        f" || (command -v apt-get >/dev/null 2>&1 && apt-get update -qq && apt-get install -qq -y {packages} && rm -rf /var/lib/apt/lists/*)"
    )
    return _bash_step("ensure_prereqs", _make_verbose(raw))


def _normalize_python_version(value: Any) -> str:
    version = str(value).strip() if value is not None else "default"
    if not version:
        version = "default"
    if version not in SUPPORTED_PYTHON_VERSIONS:
        raise ValueError(f"Unsupported Python runtime version for template preinstall: {version}")
    return version


def _build_python_runtime_commands(config: dict[str, Any], *, install_root: str) -> list[TemplateCommand]:
    version = _normalize_python_version(config.get("version"))
    if version in {"default", "3.11"}:
        install_cmd = env_vars.ROCK_RTENV_PYTHON_V31114_INSTALL_CMD
    else:
        install_cmd = env_vars.ROCK_RTENV_PYTHON_V31212_INSTALL_CMD

    bin_dir = f"{install_root}/runtime-env/bin"
    commands: list[TemplateCommand] = [
        _bash_command("python_runtime_mkdir", f"mkdir -p {_quote(install_root)}"),
        _bash_command("python_runtime_install", f"cd {_quote(install_root)} && {_make_verbose(install_cmd)}"),
        _bash_command("python_runtime_verify", _python_verify_command(bin_dir)),
    ]

    pip_index_url = str(config.get("pip_index_url") or env_vars.ROCK_PIP_INDEX_URL).strip()

    pip_packages = config.get("pip")
    if isinstance(pip_packages, list) and pip_packages:
        packages = " ".join(_quote(str(pkg)) for pkg in pip_packages if str(pkg).strip())
        if packages:
            pip_install_cmd = f"pip install {packages}"
            pip_install_cmd = _inject_pip_index_into_install_cmd(pip_install_cmd, pip_index_url)
            commands.append(_bash_command("python_runtime_pip_install", _with_python_runtime(bin_dir, pip_install_cmd)))
    elif isinstance(pip_packages, str) and pip_packages.strip():
        raise ValueError("Template prefetch does not support runtime_env_config.pip requirements files yet.")

    custom_install_cmd = str(config.get("custom_install_cmd", "")).strip()
    if custom_install_cmd:
        commands.append(
            _bash_command(
                "python_runtime_custom_install",
                f"cd {_quote(install_root)} && {_with_python_runtime(bin_dir, custom_install_cmd)}",
            )
        )
    return commands


def _build_node_runtime_commands(config: dict[str, Any], *, install_root: str) -> list[TemplateCommand]:
    version = str(config.get("version", "default")).strip() or "default"
    if version not in SUPPORTED_NODE_VERSIONS:
        raise ValueError(f"Unsupported Node runtime version for template preinstall: {version}")

    install_cmd = env_vars.ROCK_RTENV_NODE_V22180_INSTALL_CMD
    bin_dir = f"{install_root}/runtime-env/bin"
    commands: list[TemplateCommand] = [
        _bash_command("node_runtime_mkdir", f"mkdir -p {_quote(install_root)}"),
        _bash_command("node_runtime_install", f"cd {_quote(install_root)} && {_make_verbose(install_cmd)}"),
        _bash_command("node_runtime_verify", _node_verify_command(bin_dir)),
    ]

    npm_registry = str(config.get("npm_registry", "")).strip()

    custom_install_cmd = str(config.get("custom_install_cmd", "")).strip()
    if custom_install_cmd:
        if npm_registry:
            custom_install_cmd = f"export NPM_CONFIG_REGISTRY={_quote(npm_registry)} && {custom_install_cmd}"
        commands.append(
            _bash_command(
                "node_runtime_custom_install",
                f"cd {_quote(install_root)} && {_with_node_runtime(bin_dir, custom_install_cmd)}",
            )
        )
    return commands


def _build_runtime_commands(config: dict[str, Any], *, kind: str) -> list[TemplateCommand]:
    runtime_type = str(config.get("type", "")).strip().lower()
    install_root = _runtime_install_root(kind)
    if runtime_type == "python":
        return _build_python_runtime_commands(config, install_root=install_root)
    if runtime_type == "node":
        return _build_node_runtime_commands(config, install_root=install_root)
    raise ValueError(f"Unsupported runtime_env_config.type for template preinstall: {runtime_type!r}")


def _build_model_service_commands(config: dict[str, Any]) -> list[TemplateCommand]:
    if not config or not config.get("enabled"):
        return []

    runtime_env_config = config.get("runtime_env_config", {})
    if not isinstance(runtime_env_config, dict):
        raise ValueError("model_service_config.runtime_env_config must be an object")
    pip_index_url = str(runtime_env_config.get("pip_index_url") or env_vars.ROCK_PIP_INDEX_URL).strip()

    bin_dir = _runtime_bin_dir("model_service")
    commands = _build_runtime_commands(runtime_env_config, kind="model_service")

    install_cmd = str(config.get("install_cmd") or env_vars.ROCK_MODEL_SERVICE_INSTALL_CMD).strip()
    if install_cmd:
        install_cmd = _inject_pip_index_into_install_cmd(install_cmd, pip_index_url)
        commands.append(
            _bash_command(
                "model_service_install",
                f'cd {_quote(_runtime_install_root("model_service"))} && {_with_python_runtime(bin_dir, install_cmd)}',
            )
        )
    return commands


def _build_template_commands(agent_config: dict[str, Any]) -> list[TemplateCommand]:
    commands: list[TemplateCommand] = [_step(_apt_prereq_command(), user="root")]

    runtime_env_config = agent_config.get("runtime_env_config", {})
    if not isinstance(runtime_env_config, dict):
        raise ValueError("runtime_env_config must be an object")
    commands.extend(_build_runtime_commands(runtime_env_config, kind="agent"))
    commands.extend(_build_model_service_commands(agent_config.get("model_service_config", {})))
    return commands


def _inject_pip_index_into_install_cmd(command: str, pip_index_url: str) -> str:
    command = command.strip()
    if not command or not pip_index_url:
        return command

    url = _quote(pip_index_url)
    replacements = (
        ("python -m pip install ", f"python -m pip install --index-url {url} "),
        ("python3 -m pip install ", f"python3 -m pip install --index-url {url} "),
        ("pip install ", f"pip install --index-url {url} "),
        ("pip3 install ", f"pip3 install --index-url {url} "),
    )
    for prefix, rewritten in replacements:
        if command.startswith(prefix):
            return rewritten + command[len(prefix) :]
    return command


def _make_template_from_image(image_name: str, commands: list[TemplateCommand]) -> Template:
    template = Template().from_image(image_name)
    for step in commands:
        template.run_cmd([str(step["command"])], user=step.get("user"))
    return template


def _poll_template_build(info: Any, *, row: dict[str, Any], alias: str) -> dict[str, Any]:
    offset = 0
    started_at = time.time()
    while time.time() - started_at < BUILD_TIMEOUT_SECONDS:
        status = Template.get_build_status(info, logs_offset=offset)
        for entry in status.log_entries:
            _emit_json(
                {
                    "status": "builder_log",
                    "instance_id": row["instance_id"],
                    "alias": alias,
                    "build_id": info.build_id,
                    "log_entry": str(entry),
                }
            )
        offset += len(status.log_entries)
        if status.status.value == "ready":
            return {
                "instance_id": _instance_id(row),
                "source_name": "swe_rebench_v2",
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "image_name": row["image_name"],
                "inspire_template": alias,
                "template_alias": alias,
                "template_id": info.template_id,
                "build_id": info.build_id,
                "status": "ready",
                "elapsed_s": round(time.time() - started_at, 2),
            }
        if status.status.value == "error":
            reason = getattr(status, "reason", None)
            if reason is None:
                message = "template build failed"
            else:
                message = str(getattr(reason, "message", reason))
            raise RuntimeError(message)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Template build timed out after {BUILD_TIMEOUT_SECONDS}s")


def _build_template(row: dict[str, Any], *, alias: str, commands: list[TemplateCommand], spec: str) -> dict[str, Any]:
    template = _make_template_from_image(str(row["image_name"]).strip(), commands)
    info = Template.build_in_background(template, alias, spec_code=getattr(SandboxSpecCode, spec))
    return _poll_template_build(info, row=row, alias=alias)


def _is_retryable_image_access_error(message: str) -> bool:
    text = str(message).strip().lower()
    if not text:
        return False
    return (
        "access denied to 'docker.io/swerebenchv2/" in text
        and ("authentication required" in text or "insufficient permissions" in text)
    )


def _write_manifest_line(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_manifests(
    *,
    success_manifest: Path,
    failure_manifest: Path,
    force: bool,
) -> set[str]:
    success_manifest.parent.mkdir(parents=True, exist_ok=True)
    failure_manifest.parent.mkdir(parents=True, exist_ok=True)

    if force:
        success_manifest.write_text("", encoding="utf-8")
        failure_manifest.write_text("", encoding="utf-8")
        return set()

    return _load_manifest_instance_ids(success_manifest)


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    requested_ids: set[str],
    max_instances: int,
) -> list[dict[str, Any]]:
    selected = rows
    if requested_ids:
        selected = [row for row in selected if _instance_id(row) in requested_ids]
    if max_instances >= 0:
        selected = selected[:max_instances]
    return selected


def _collect_candidates(rows: list[dict[str, Any]], *, existing_success_ids: set[str], force: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        instance_id = _instance_id(row)
        if not instance_id:
            continue
        if (not force) and instance_id in existing_success_ids:
            _emit_json({"status": "skip_manifest", "instance_id": instance_id})
            continue
        candidates.append(row)
    return candidates


def _failure_row(instance_id: str, error: Exception) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "source_name": "swe_rebench_v2",
        "status": "error",
        "reason": str(error),
    }


def _record_failure(instance_id: str, error: Exception, *, failure_manifest: Path, failure_lock: threading.Lock) -> None:
    row = _failure_row(instance_id, error)
    _write_manifest_line(failure_manifest, row, failure_lock)
    _emit_json({"status": "failed", "instance_id": instance_id, "error": str(error)})


def _record_success(result: dict[str, Any], *, success_manifest: Path, success_lock: threading.Lock) -> None:
    _write_manifest_line(success_manifest, result, success_lock)
    _emit_json(
        {
            "status": "ok",
            "instance_id": result.get("instance_id"),
            "template": result.get("inspire_template"),
        }
    )


def _build_one(
    row: dict[str, Any],
    *,
    spec: str,
    commands: list[TemplateCommand],
) -> dict[str, Any]:
    instance_id = _instance_id(row)
    alias = _normalize_instance_alias(instance_id)
    if Template.alias_exists(alias):
        _emit_json({"status": "deleting_stale", "instance_id": instance_id, "alias": alias})
        _delete_template_by_alias(alias)

    _emit_json(
        {
            "status": "building",
            "instance_id": instance_id,
            "template_alias": alias,
            "image_name": row["image_name"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "spec": spec,
        }
    )
    attempts = 0
    while True:
        attempts += 1
        try:
            return _build_template(row, alias=alias, commands=commands, spec=spec)
        except Exception as exc:
            if (not _is_retryable_image_access_error(str(exc))) or attempts >= RETRYABLE_IMAGE_ACCESS_MAX_ATTEMPTS:
                raise
            _emit_json(
                {
                    "status": "retry_image_access",
                    "instance_id": instance_id,
                    "alias": alias,
                    "attempt": attempts,
                    "max_attempts": RETRYABLE_IMAGE_ACCESS_MAX_ATTEMPTS,
                    "sleep_s": RETRYABLE_IMAGE_ACCESS_SLEEP_SECONDS,
                    "reason": str(exc),
                }
            )
            if Template.alias_exists(alias):
                _delete_template_by_alias(alias)
            time.sleep(RETRYABLE_IMAGE_ACCESS_SLEEP_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch SWE-rebench public images as Inspire templates.")
    parser.add_argument("--tasks-json", default=str(DEFAULT_TASKS_JSON))
    parser.add_argument("--agent-config", default=str(DEFAULT_AGENT_CONFIG))
    parser.add_argument("--success-manifest", default=str(DEFAULT_SUCCESS))
    parser.add_argument("--failure-manifest", default=str(DEFAULT_FAILURE))
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    global LOGGER
    log_file = _log_file_path(args.log_file, args.log_dir)
    LOGGER = _TeeLogger(log_file)
    _emit(f"[info] writing build log to {log_file}")

    spec = _parse_spec(args.spec)
    rows = _load_rows(Path(args.tasks_json))
    agent_config = _load_agent_config(Path(args.agent_config))
    install_commands = _build_template_commands(agent_config)

    requested_ids = {value.strip() for value in args.instance_id if value.strip()}
    rows = _select_rows(rows, requested_ids=requested_ids, max_instances=args.max_instances)

    success_manifest = Path(args.success_manifest)
    failure_manifest = Path(args.failure_manifest)
    success_lock = threading.Lock()
    failure_lock = threading.Lock()
    existing_success_ids = _prepare_manifests(
        success_manifest=success_manifest,
        failure_manifest=failure_manifest,
        force=args.force,
    )
    candidates = _collect_candidates(rows, existing_success_ids=existing_success_ids, force=args.force)

    if not candidates:
        _emit("No candidates to build.")
        return

    _emit(f"Preparing {len(candidates)} SWE-rebench image templates with agent config: {args.agent_config}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as executor:
        future_to_instance = {
            executor.submit(_build_one, row, spec=spec, commands=install_commands): row["instance_id"]
            for row in candidates
        }
        for future in concurrent.futures.as_completed(future_to_instance):
            row_instance_id = future_to_instance[future]
            try:
                result = future.result()
            except Exception as exc:
                _record_failure(
                    row_instance_id,
                    exc,
                    failure_manifest=failure_manifest,
                    failure_lock=failure_lock,
                )
                continue

            _record_success(result, success_manifest=success_manifest, success_lock=success_lock)


if __name__ == "__main__":
    main()
