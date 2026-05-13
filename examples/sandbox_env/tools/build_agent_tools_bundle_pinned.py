#!/usr/bin/env python3
"""Build a pinned agent-tools bundle for SWE-rebench Inspire templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from agentic_protocol.command_factory.claude import ClaudeCodeCommandFactory
from agentic_protocol.command_factory.codex import CodexCommandFactory
from agentic_protocol.command_factory.layout import AGENTIC_PROTOCOL_ROOT, framework_bin_dir, shell_command
from agentic_protocol.command_factory.node import NODE_VERSION
from agentic_protocol.command_factory.opencode import OpenCodeCommandFactory
from agentic_protocol.command_factory.openhands import OpenHandsCommandFactory
from agentic_protocol.command_factory.qwen import QwenCodeCommandFactory
from agentic_protocol.command_factory.uv import uv_bin_path, uv_template_install_lines
from agentic_protocol.command_factory.wstunnel import WSTUNNEL_VERSION, wstunnel_template_install_command
from agentic_protocol.template_build.inspire_template_build import NPM_REGISTRY


PINNED_VERSIONS = {
    "claude_code": "2.1.126",
    "codex": "0.128.0",
    "open_code": "1.14.33",
    "openai_compatible": "2.0.45",
    "qwen_code": "0.15.6",
    "openhands": "1.15.0",
    "openhands_sdk": "1.17.0",
    "openhands_agent_server": "1.19.1",
    "openhands_tools": "1.17.0",
}


def run(command: str, *, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, shell=True, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt >= attempts:
                raise
            time.sleep(30 * attempt)


def pinned_openhands_install_command(
    *,
    python_version: str,
    constraints_path: Path | None = None,
) -> str:
    factory = OpenHandsCommandFactory()
    wrapper = "\n".join(
        [
            "#!/bin/sh",
            f'export HOME="{factory.home_path}"',
            f'export TMPDIR="{factory.tmp_path}"',
            f'export UV_CACHE_DIR="{factory.cache_path}/uv"',
            f'export UV_PYTHON_INSTALL_DIR="{factory.agent_root}/python"',
            'mkdir -p "$HOME" "$TMPDIR" "$UV_CACHE_DIR"',
            f'exec {shlex.quote(factory.tool_binary_path)} "$@"',
        ]
    )
    install_cmd = " ".join(
        [
            shlex.quote(uv_bin_path()),
            "tool",
            "install",
            "--python",
            shlex.quote(python_version),
            "--link-mode",
            "copy",
            "--force",
            *(
                [
                    "--constraints",
                    shlex.quote(str(constraints_path)),
                ]
                if constraints_path is not None
                else []
            ),
            "--with",
            shlex.quote(f"openhands-sdk=={PINNED_VERSIONS['openhands_sdk']}"),
            "--with",
            shlex.quote(f"openhands-agent-server=={PINNED_VERSIONS['openhands_agent_server']}"),
            "--with",
            shlex.quote(f"openhands-tools=={PINNED_VERSIONS['openhands_tools']}"),
            shlex.quote(f"openhands=={PINNED_VERSIONS['openhands']}"),
        ]
    )
    lines = [
        *uv_template_install_lines(),
        f"FRAMEWORK_ROOT={shlex.quote(factory.agent_root)}",
        f"FRAMEWORK_BIN_DIR={shlex.quote(framework_bin_dir(factory.name))}",
        f"FRAMEWORK_HOME={shlex.quote(factory.home_path)}",
        f"FRAMEWORK_TMP={shlex.quote(factory.tmp_path)}",
        f"FRAMEWORK_CACHE={shlex.quote(factory.cache_path)}",
        f"FRAMEWORK_PYTHON_DIR={shlex.quote(factory.agent_root + '/python')}",
        "mkdir -p "
        '"$FRAMEWORK_ROOT/tool-bin" "$FRAMEWORK_BIN_DIR" "$FRAMEWORK_PYTHON_DIR" '
        '"$FRAMEWORK_HOME" "$FRAMEWORK_TMP" "$FRAMEWORK_CACHE"',
        'chmod -R a+rwX "$FRAMEWORK_HOME" "$FRAMEWORK_TMP" "$FRAMEWORK_CACHE"',
        'export UV_CACHE_DIR="$FRAMEWORK_CACHE/uv"',
        'export UV_PYTHON_INSTALL_DIR="$FRAMEWORK_PYTHON_DIR"',
        'export UV_TOOL_DIR="$FRAMEWORK_ROOT/tools"',
        'export UV_TOOL_BIN_DIR="$FRAMEWORK_ROOT/tool-bin"',
        install_cmd,
        f"test -x {shlex.quote(factory.tool_binary_path)}",
        f"cat > {shlex.quote(factory.binary_path)} <<'EOF'",
        wrapper,
        "EOF",
        f"chmod 0755 {shlex.quote(factory.binary_path)}",
        f"{shlex.quote(factory.binary_path)} --help >/dev/null || true",
    ]
    return shell_command(lines)


def npm_commands(*, npm_registry: str, node_version: str) -> list[tuple[str, str]]:
    open_code = OpenCodeCommandFactory()
    open_code.extra_npm_packages = (
        f"@ai-sdk/openai-compatible@{PINNED_VERSIONS['openai_compatible']}",
    )
    factories = [
        ("claude_code", ClaudeCodeCommandFactory(), PINNED_VERSIONS["claude_code"]),
        ("codex", CodexCommandFactory(), PINNED_VERSIONS["codex"]),
        ("open_code", open_code, PINNED_VERSIONS["open_code"]),
        ("qwen_code", QwenCodeCommandFactory(), PINNED_VERSIONS["qwen_code"]),
    ]
    return [
        (
            name,
            factory.template_install_command(
                npm_registry=npm_registry,
                node_version=node_version,
                agent_package_version=version,
            ),
        )
        for name, factory, version in factories
    ]


def validate_versions() -> None:
    root = Path(AGENTIC_PROTOCOL_ROOT)
    node = root / "linux" / "bin" / "node"
    package_paths = {
        "claude_code": root
        / "frameworks/claude_code/npm-prefix/node_modules/@anthropic-ai/claude-code/package.json",
        "codex": root / "frameworks/codex/npm-prefix/node_modules/@openai/codex/package.json",
        "open_code": root / "frameworks/open_code/npm-prefix/node_modules/opencode-ai/package.json",
        "openai_compatible": root
        / "frameworks/open_code/npm-prefix/node_modules/@ai-sdk/openai-compatible/package.json",
        "qwen_code": root / "frameworks/qwen_code/npm-prefix/node_modules/@qwen-code/qwen-code/package.json",
    }
    for key, path in package_paths.items():
        version = json.loads(path.read_text(encoding="utf-8"))["version"]
        expected = PINNED_VERSIONS[key]
        if version != expected:
            raise RuntimeError(f"{key} version mismatch: {version} != {expected}")

    py = root / "frameworks/openhands/tools/openhands/bin/python"
    script = "\n".join(
        [
            "import importlib.metadata as md",
            "expected = {",
            f"  'openhands': {PINNED_VERSIONS['openhands']!r},",
            f"  'openhands-sdk': {PINNED_VERSIONS['openhands_sdk']!r},",
            f"  'openhands-agent-server': {PINNED_VERSIONS['openhands_agent_server']!r},",
            f"  'openhands-tools': {PINNED_VERSIONS['openhands_tools']!r},",
            "}",
            "for name, want in expected.items():",
            "    got = md.version(name)",
            "    if got != want:",
            "        raise SystemExit(f'{name} version mismatch: {got} != {want}')",
        ]
    )
    subprocess.run([str(py), "-c", script], check=True)
    py_target = py.resolve(strict=True)
    if root != py_target and root not in py_target.parents:
        raise RuntimeError(f"OpenHands Python resolves outside bundle root: {py_target}")
    subprocess.run([str(node), "--version"], check=True)


def prune_bundle_contents() -> None:
    root = Path(AGENTIC_PROTOCOL_ROOT)
    for pyc in root.rglob("*.pyc"):
        pyc.unlink()
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    opencode_modules = root / "frameworks/open_code/npm-prefix/node_modules"
    cached_opencode = opencode_modules / "opencode-ai/bin/.opencode"
    if cached_opencode.is_file():
        for package_dir in opencode_modules.glob("opencode-*"):
            if package_dir.name != "opencode-ai":
                shutil.rmtree(package_dir, ignore_errors=True)


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest.hexdigest()}  {path}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--node-version", default=NODE_VERSION)
    parser.add_argument("--openhands-python-version", default="3.12")
    parser.add_argument("--openhands-constraints", type=Path, default=None)
    parser.add_argument("--npm-registry", default=NPM_REGISTRY)
    args = parser.parse_args()

    root = Path(AGENTIC_PROTOCOL_ROOT)
    if root.exists():
        run(f"rm -rf {shlex.quote(str(root))}")

    commands = [
        ("wstunnel", wstunnel_template_install_command(version=WSTUNNEL_VERSION)),
        *npm_commands(npm_registry=args.npm_registry, node_version=args.node_version),
        (
            "openhands",
            pinned_openhands_install_command(
                python_version=args.openhands_python_version,
                constraints_path=args.openhands_constraints,
            ),
        ),
    ]
    for label, command in commands:
        print(f"=== install {label} ===", flush=True)
        run(command)

    validate_versions()
    prune_bundle_contents()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tar_cmd = " ".join(
        [
            "GZIP=-9",
            "tar",
            "-C",
            "/",
            "--sort=name",
            "--mtime=" + shlex.quote("UTC 2026-01-01"),
            "--numeric-owner",
            "--exclude=__avaeval_agentic_protocol_v1__/state/cache",
            "--exclude=__avaeval_agentic_protocol_v1__/state/logs",
            "--exclude=__avaeval_agentic_protocol_v1__/state/tmp",
            "-czf",
            shlex.quote(str(tmp)),
            "__avaeval_agentic_protocol_v1__",
        ]
    )
    run(tar_cmd)
    tmp.replace(args.output)
    write_sha256(args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
