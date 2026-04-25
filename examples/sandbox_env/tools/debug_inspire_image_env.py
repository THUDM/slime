#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

# Make sandbox_env siblings importable when this script lives under tools/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _inspire_sandbox_bootstrap import bootstrap_inspire_sandbox_path

bootstrap_inspire_sandbox_path()

from inspire_sandbox import Sandbox, SandboxSpecCode, Template


BUILD_TIMEOUT_SECONDS = 1800
POLL_INTERVAL_SECONDS = 5

DEFAULT_COMMANDS = [
    "whoami",
    "id",
    "echo HOME=$HOME",
    "pwd",
    "env | sort | grep -E '^(HOME|PATH|USER|LOGNAME|SHELL|RUSTUP_HOME|CARGO_HOME)=' || true",
    "command -v rustup || true",
    "command -v cargo || true",
    "ls -la /root || true",
    "ls -la /root/.cargo || true",
    "ls -la /root/.rustup || true",
    "cat /root/.rustup/settings.toml || true",
    "rustup show || true",
    "cargo --version || true",
]


def _parse_env_assignments(values: list[str]) -> dict[str, str]:
    envs: dict[str, str] = {}
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"invalid env assignment {raw!r}; expected KEY=VALUE")
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid env assignment {raw!r}; empty key")
        envs[key] = value
    return envs


def _sanitize_template_name(image: str) -> str:
    digest = hashlib.sha1(image.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^a-z0-9-]+", "-", image.lower()).strip("-")
    stem = stem[-40:] if len(stem) > 40 else stem
    stem = stem or "image"
    return f"debug-{stem}-{digest}"


def _wait_for_build(info) -> None:
    offset = 0
    deadline = time.time() + BUILD_TIMEOUT_SECONDS
    while time.time() < deadline:
        status = Template.get_build_status(info, logs_offset=offset)
        for entry in status.log_entries:
            print(f"[build] {entry}", flush=True)
        offset += len(status.log_entries)
        if status.status.value == "ready":
            return
        if status.status.value == "error":
            reason = getattr(status, "reason", None)
            message = str(getattr(reason, "message", "") or reason or "template build failed")
            raise RuntimeError(message)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Template build timed out after {BUILD_TIMEOUT_SECONDS}s")


def _resolve_template(args: argparse.Namespace) -> str:
    if args.template:
        return args.template
    image = str(args.image).strip()
    alias = args.alias.strip() if args.alias else _sanitize_template_name(image)
    print(f"[info] building template from image={image} alias={alias} spec={args.spec}", flush=True)
    template = Template().from_image(image)
    build_info = Template.build_in_background(
        template,
        alias,
        spec_code=getattr(SandboxSpecCode, args.spec),
    )
    print(
        f"[info] build started template_id={build_info.template_id} build_id={build_info.build_id}",
        flush=True,
    )
    _wait_for_build(build_info)
    print(f"[info] build ready alias={alias}", flush=True)
    return alias


def _run_commands(
    sandbox: Sandbox,
    *,
    user: str | None,
    commands: list[str],
    envs: dict[str, str] | None,
) -> None:
    for command in commands:
        print(f"\n===== command user={user or '<default>'} =====", flush=True)
        print(command, flush=True)
        if envs:
            print(f"[command_envs] {envs}", flush=True)
        result = sandbox.commands.run(command, user=user, timeout=300, envs=envs or None)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        print(f"[exit_code] {result.exit_code}", flush=True)
        if stdout:
            print("[stdout]", flush=True)
            print(stdout, flush=True)
        if stderr:
            print("[stderr]", flush=True)
            print(stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Inspire sandbox image/template runtime environment.")
    parser.add_argument("--image", default="", help="Docker image to turn into a temporary Inspire template.")
    parser.add_argument("--template", default="", help="Existing Inspire template alias/id to start directly.")
    parser.add_argument("--alias", default="", help="Template alias to use when building from --image.")
    parser.add_argument("--spec", default="G_C4", choices=["G_C1", "G_C2", "G_C4"])
    parser.add_argument("--timeout", type=int, default=3600, help="Sandbox timeout in seconds.")
    parser.add_argument("--user", default="root", help="User to run diagnostics as. Use empty string for default user.")
    parser.add_argument("--keep-sandbox", action="store_true", help="Do not stop sandbox after diagnostics.")
    parser.add_argument(
        "--sandbox-env",
        action="append",
        default=[],
        help="Sandbox-level env assignment KEY=VALUE, applied at Sandbox.create(..., envs=...).",
    )
    parser.add_argument(
        "--command-env",
        action="append",
        default=[],
        help="Command-level env assignment KEY=VALUE, applied to every commands.run(..., envs=...).",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Extra command to run. If omitted, a default Rust environment diagnostic set is used.",
    )
    args = parser.parse_args()
    if not args.image and not args.template:
        parser.error("one of --image or --template is required")
    return args


def main() -> int:
    args = parse_args()
    template = _resolve_template(args)
    run_user = args.user if str(args.user).strip() else None
    commands = args.command or list(DEFAULT_COMMANDS)
    sandbox_envs = _parse_env_assignments(args.sandbox_env)
    command_envs = _parse_env_assignments(args.command_env)
    print(f"[info] creating sandbox template={template} timeout={args.timeout}", flush=True)
    if sandbox_envs:
        print(f"[info] sandbox envs={sandbox_envs}", flush=True)
    if command_envs:
        print(f"[info] command envs={command_envs}", flush=True)
    sandbox = Sandbox.create(template=template, timeout=args.timeout, envs=sandbox_envs or None)
    print(f"[info] sandbox ready sandbox_id={sandbox.sandbox_id}", flush=True)
    try:
        _run_commands(sandbox, user=run_user, commands=commands, envs=command_envs)
    finally:
        if args.keep_sandbox:
            print(f"[info] keeping sandbox sandbox_id={sandbox.sandbox_id}", flush=True)
        else:
            print(f"[info] stopping sandbox sandbox_id={sandbox.sandbox_id}", flush=True)
            sandbox.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
