"""Convert supported Harbor task directories to Slime coding-agent JSONL.

Call :func:`convert_task` for one row or use the module CLI for an atomic batch.
Unsupported Harbor semantics raise :class:`UnsupportedHarborTaskError` with
stable feature names and detailed explanations.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_SAFE_WORKDIR_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
_TASK_CONFIG_NAME = "task.toml"
_HARBOR_SETUP_ROOT = "/setup_files"
_DEFAULT_SCHEMA_VERSION = "1.2"
_REWARD_JSON_NAME = "reward.json"
_REWARD_TEXT_NAME = "reward.txt"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", "1.2"})
_COMPOSE_NAMES = frozenset(
    {
        "compose.yaml",
        "compose.yml",
        "docker-compose.yaml",
        "docker-compose.yml",
        "singularity-compose.yaml",
        "singularity-compose.yml",
    }
)


class HarborConversionError(ValueError):
    """A malformed task or invalid converter invocation."""


class UnsupportedHarborTaskError(HarborConversionError):
    """A valid Harbor task uses behavior Slime cannot preserve."""

    def __init__(self, task_dir: Path, reasons: Iterable[str]) -> None:
        self.task_dir = task_dir
        self.reasons = tuple(reasons)
        self.features = tuple(reason.partition(":")[0] for reason in self.reasons)
        detail = "\n".join(f"  - {reason}" for reason in self.reasons)
        super().__init__(f"Harbor task {task_dir} is not convertible:\n{detail}")


@dataclass(frozen=True)
class ConversionOverrides:
    """Per-task immutable sandbox and workspace overrides."""

    image: str | None = None
    snapshot: str | None = None
    workdir: str | None = None


def strip_canary(text: str) -> str:
    """Match Harbor's removal of leading provenance-marker lines."""

    lines = text.split("\n")
    index = 0
    while index < len(lines) and _CANARY_LINE_RE.match(lines[index].strip()):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


def discover_task_dirs(paths: Iterable[Path]) -> list[Path]:
    """Resolve task directories and dataset roots in deterministic order."""

    tasks: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_file() and path.name == _TASK_CONFIG_NAME:
            tasks.add(path.parent)
        elif (path / _TASK_CONFIG_NAME).is_file():
            tasks.add(path)
        elif path.is_dir():
            tasks.update(config.parent for config in path.rglob(_TASK_CONFIG_NAME))
        else:
            raise HarborConversionError(f"No Harbor task found at {raw_path}")
    if not tasks:
        raise HarborConversionError(f"No Harbor {_TASK_CONFIG_NAME} files were found")
    return sorted(tasks, key=lambda task: task.as_posix())


def convert_task(task_dir: Path, overrides: ConversionOverrides | None = None) -> dict[str, Any]:
    """Convert one Harbor task directory to a Slime JSONL row.

    Unsupported Harbor semantics are reported together instead of being
    silently dropped.  ``solution/`` is deliberately never read.
    """

    task_dir = task_dir.resolve()
    overrides = overrides or ConversionOverrides()
    config = _read_config(task_dir)
    name = _task_name(config, task_dir)
    environment = _table(config, "environment")
    workdir = overrides.workdir or environment.get("workdir")
    image = overrides.image or environment.get("docker_image")
    snapshot = overrides.snapshot

    reasons = _unsupported_reasons(task_dir, config)
    if overrides.image and overrides.snapshot:
        reasons.append("sandbox: image and snapshot overrides are mutually exclusive")
    if snapshot:
        image = None
    if image is not None and (not isinstance(image, str) or not image.strip()):
        reasons.append("environment.docker_image: image references must be non-empty strings")
    if snapshot is not None and (not isinstance(snapshot, str) or not snapshot.strip()):
        reasons.append("sandbox snapshot: snapshot references must be non-empty strings")
    if not image and not snapshot:
        reasons.append("environment: provide environment.docker_image or a --snapshot/--image override")
    if not isinstance(workdir, str) or not _safe_workdir(workdir):
        reasons.append(
            "environment.workdir: provide an absolute path containing only letters, digits, '.', '_', '-', and '/'"
        )

    instruction_path = task_dir / "instruction.md"
    test_script = task_dir / "tests" / "test.sh"
    if not instruction_path.is_file():
        reasons.append("instruction.md: a single-step instruction is required")
    if not test_script.is_file():
        reasons.append("tests/test.sh: a Linux Harbor verifier is required")

    for source_name in ("setup_files", "tests"):
        source = task_dir / source_name
        if source.exists():
            reasons.extend(_unsupported_entries(source, source_name))
    if reasons:
        raise UnsupportedHarborTaskError(task_dir, reasons)

    assert isinstance(workdir, str)
    prompt = strip_canary(instruction_path.read_text())
    setup_dir = f"{workdir}/.slime-harbor/setup_files"
    tests_dir = f"{workdir}/.slime-harbor/tests"
    logs_dir = f"{workdir}/.slime-harbor/logs"
    prompt = prompt.replace(_HARBOR_SETUP_ROOT, setup_dir)

    metadata: dict[str, Any] = {
        "instance_id": name,
        "workdir": workdir,
        "problem_statement": prompt,
        "eval_cmd": _build_eval_command(
            task_dir / "tests",
            tests_dir=tests_dir,
            logs_dir=logs_dir,
            timeout_sec=_verifier_timeout(config),
        ),
        "harbor": {
            "schema_version": str(config.get("schema_version", config.get("version", _DEFAULT_SCHEMA_VERSION))),
            "metadata": _json_value(config.get("metadata", {})),
        },
    }
    if image:
        metadata["image"] = image
    else:
        metadata["snapshot"] = snapshot

    setup_files = task_dir / "setup_files"
    if setup_files.is_dir() and any(setup_files.iterdir()):
        metadata["pre_commands"] = _build_materialize_command(
            setup_files,
            setup_dir,
            replacements={_HARBOR_SETUP_ROOT.encode(): setup_dir.encode()},
        )

    return {"prompt": prompt, "label": name, "metadata": metadata}


def _read_config(task_dir: Path) -> dict[str, Any]:
    config_path = task_dir / _TASK_CONFIG_NAME
    if not config_path.is_file():
        raise HarborConversionError(f"Missing {config_path}")
    try:
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HarborConversionError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise HarborConversionError(f"{config_path} must contain a TOML table")
    return config


def _table(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HarborConversionError(f"{key} must be a TOML table")
    return value


def _task_name(config: dict[str, Any], task_dir: Path) -> str:
    value = _table(config, "task").get("name", task_dir.name)
    if not isinstance(value, str) or not value.strip():
        raise HarborConversionError("task.name must be a non-empty string")
    return value


def _safe_workdir(workdir: str) -> bool:
    path = PurePosixPath(workdir)
    return bool(_SAFE_WORKDIR_RE.fullmatch(workdir)) and ".." not in path.parts and workdir != "/"


def _verifier_timeout(config: dict[str, Any]) -> float:
    value = _table(config, "verifier").get("timeout_sec", 600.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise HarborConversionError("verifier.timeout_sec must be a positive number")
    return float(value)


def _json_value(value: Any) -> Any:
    """Normalize TOML date/time values for JSONL output."""

    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    return value


def _unsupported_reasons(task_dir: Path, config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    schema_version = str(config.get("schema_version", config.get("version", _DEFAULT_SCHEMA_VERSION)))
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        reasons.append(f"schema_version: {schema_version!r} is not one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)}")
    if config.get("steps"):
        reasons.append("steps: multi-step tasks require Harbor's stateful trial loop")
    if config.get("multi_step_reward_strategy") is not None:
        reasons.append("multi_step_reward_strategy: Slime's coding grader produces one binary reward")
    if config.get("artifacts"):
        reasons.append("artifacts: Slime's coding sandbox does not collect Harbor artifacts")

    environment = _table(config, "environment")
    if str(environment.get("os", "linux")).lower() != "linux":
        reasons.append("environment.os: only Linux sandboxes are supported")
    for field in ("cpus", "memory_mb", "memory", "storage_mb", "storage"):
        if environment.get(field) is not None:
            reasons.append(f"environment.{field}: per-task resource enforcement is unavailable")
    if environment.get("gpus") not in (None, 0):
        reasons.append("environment.gpus: accelerator sandboxes are unsupported")
    if environment.get("gpu_types"):
        reasons.append("environment.gpu_types: accelerator selection is unsupported")
    if environment.get("tpu"):
        reasons.append("environment.tpu: TPU sandboxes are unsupported")
    if environment.get("allow_internet") is False:
        reasons.append("environment.allow_internet: per-task network isolation is unavailable")
    for field, description in (
        ("mcp_servers", "Harbor MCP injection is unavailable"),
        ("env", "host environment interpolation is unavailable"),
        ("skills_dir", "Harbor skill injection is unavailable"),
        ("healthcheck", "Harbor retrying healthchecks are unavailable"),
    ):
        if environment.get(field):
            reasons.append(f"environment.{field}: {description}")

    agent = _table(config, "agent")
    if agent.get("timeout_sec") is not None:
        reasons.append("agent.timeout_sec: Slime configures one rollout timeout for the dataset")
    if agent.get("user") is not None:
        reasons.append("agent.user: Slime's coding harness runs as the fixed 'agent' user")

    verifier = _table(config, "verifier")
    if verifier.get("env"):
        reasons.append("verifier.env: host environment interpolation is unavailable")
    if verifier.get("user") is not None:
        reasons.append("verifier.user: Slime's coding grader runs as the fixed 'agent' user")
    if verifier.get("environment_mode") == "separate" or verifier.get("environment"):
        reasons.append("verifier.environment: separate verifier containers are unsupported")
    if verifier.get("collect"):
        reasons.append("verifier.collect: pre-artifact collection commands are unsupported")

    environment_dir = task_dir / "environment"
    if environment_dir.is_dir() and any(child.name.lower() in _COMPOSE_NAMES for child in environment_dir.iterdir()):
        reasons.append("environment compose file: multi-container tasks are unsupported")
    return reasons


def _unsupported_entries(root: Path, label: str) -> list[str]:
    reasons: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            reasons.append(f"{label}/{path.relative_to(root)}: symbolic links cannot be embedded safely")
        elif not path.is_file() and not path.is_dir():
            reasons.append(f"{label}/{path.relative_to(root)}: only regular files and directories are supported")
    return reasons


def _build_payload(source: Path, replacements: dict[bytes, bytes]) -> str:
    files: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        content = path.read_bytes()
        for old, new in replacements.items():
            content = content.replace(old, new)
        files.append(
            {
                "path": path.relative_to(source).as_posix(),
                "mode": path.stat().st_mode & 0o777,
                "content": base64.b64encode(content).decode("ascii"),
            }
        )
    packed = gzip.compress(json.dumps(files, separators=(",", ":")).encode(), mtime=0)
    return base64.b64encode(packed).decode("ascii")


def _build_materialize_command(source: Path, destination: str, replacements: dict[bytes, bytes]) -> str:
    payload = _build_payload(source, replacements)
    return f"""python3 - <<'SLIME_HARBOR_FILES'
import base64
import gzip
import json
from pathlib import Path

root = Path({destination!r})
root.mkdir(parents=True, exist_ok=True)
files = json.loads(gzip.decompress(base64.b64decode({payload!r})))
for item in files:
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(item["content"]))
    path.chmod(item["mode"])
SLIME_HARBOR_FILES"""


def _build_eval_command(source: Path, *, tests_dir: str, logs_dir: str, timeout_sec: float) -> str:
    materialize = _build_materialize_command(
        source,
        tests_dir,
        replacements={b"/tests": tests_dir.encode(), b"/logs": logs_dir.encode()},
    )
    runner = f"""python3 - <<'SLIME_HARBOR_VERIFY'
import json
import os
import subprocess
import sys
from pathlib import Path

tests_dir = Path({tests_dir!r})
logs_dir = Path({logs_dir!r})
verifier_dir = logs_dir / "verifier"
verifier_dir.mkdir(parents=True, exist_ok=True)
for stale_reward in (verifier_dir / {_REWARD_JSON_NAME!r}, verifier_dir / {_REWARD_TEXT_NAME!r}):
    stale_reward.unlink(missing_ok=True)
env = os.environ.copy()
env["TEST_DIR"] = str(tests_dir)
try:
    subprocess.run(["bash", str(tests_dir / "test.sh")], env=env, timeout={timeout_sec!r}, check=False)
except subprocess.TimeoutExpired:
    print("Harbor verifier timed out after {timeout_sec:g} seconds", file=sys.stderr)
    raise SystemExit(1)

reward_json = verifier_dir / {_REWARD_JSON_NAME!r}
reward_text = verifier_dir / {_REWARD_TEXT_NAME!r}
try:
    if reward_json.exists():
        value = json.loads(reward_json.read_text()).get("reward")
    elif reward_text.exists():
        value = float(reward_text.read_text())
    else:
        raise ValueError("verifier wrote neither reward.json nor reward.txt")
    reward = float(value)
except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
    print(f"Invalid Harbor verifier reward: {{exc}}", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0 if reward == 1.0 else 1)
SLIME_HARBOR_VERIFY"""
    return f"{materialize}\n{runner}"


def _parse_assignments(values: list[str], option: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name or not setting:
            raise HarborConversionError(f"{option} expects TASK=VALUE, got {value!r}")
        if name in assignments:
            raise HarborConversionError(f"{option} was provided more than once for {name!r}")
        assignments[name] = setting
    return assignments


def _write_rows(rows: list[dict[str, Any]], output: str) -> None:
    content = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows)
    if output == "-":
        sys.stdout.write(content)
    else:
        output_path = Path(output)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as stream:
                stream.write(content)
                temporary_path = Path(stream.name)
            os.replace(temporary_path, output_path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise HarborConversionError(f"Could not write {output_path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", type=Path, help="Harbor task directories, task.toml files, or roots")
    parser.add_argument("-o", "--output", default="-", help="JSONL output path (default: stdout)")
    parser.add_argument(
        "--image", action="append", default=[], metavar="TASK=IMAGE", help="override an image by task name"
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        metavar="TASK=SNAPSHOT",
        help="use an existing sandbox snapshot by task name",
    )
    parser.add_argument(
        "--workdir", action="append", default=[], metavar="TASK=PATH", help="override a workdir by task name"
    )
    parser.add_argument(
        "--skip-unsupported", action="store_true", help="report unsupported tasks and write the remaining rows"
    )
    args = parser.parse_args(argv)

    try:
        maps = {
            "image": _parse_assignments(args.image, "--image"),
            "snapshot": _parse_assignments(args.snapshot, "--snapshot"),
            "workdir": _parse_assignments(args.workdir, "--workdir"),
        }
        task_dirs = discover_task_dirs(args.tasks)
        task_names = [_task_name(_read_config(task), task) for task in task_dirs]
        duplicate_names = sorted({name for name in task_names if task_names.count(name) > 1})
        if duplicate_names:
            raise HarborConversionError(f"Task names are not unique: {', '.join(duplicate_names)}")
        known_names = set(task_names)
        unknown = sorted(set().union(*(mapping.keys() for mapping in maps.values())) - known_names)
        if unknown:
            raise HarborConversionError(f"Overrides refer to unknown tasks: {', '.join(unknown)}")

        rows: list[dict[str, Any]] = []
        failures: list[UnsupportedHarborTaskError] = []
        for task_dir, name in zip(task_dirs, task_names, strict=True):
            overrides = ConversionOverrides(
                image=maps["image"].get(name),
                snapshot=maps["snapshot"].get(name),
                workdir=maps["workdir"].get(name),
            )
            try:
                rows.append(convert_task(task_dir, overrides))
            except UnsupportedHarborTaskError as exc:
                failures.append(exc)
        if failures and not args.skip_unsupported:
            raise HarborConversionError("\n".join(str(failure) for failure in failures))
        for failure in failures:
            print(failure, file=sys.stderr)
        _write_rows(rows, args.output)
        return 0
    except HarborConversionError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
