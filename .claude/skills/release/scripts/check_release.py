#!/usr/bin/env python3
"""Check local slime release metadata and Docker/conda patch alignment."""

import argparse
import ast
import re
import sys
from pathlib import Path


def _setup_version(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "version":
                return ast.literal_eval(keyword.value)
    raise ValueError(f"setup version not found in {path}")


def _assigned_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def _shell_exports(text: str) -> dict[str, str]:
    return dict(re.findall(r'^export ([A-Z][A-Z0-9_]*)="([^"]+)"$', text, re.MULTILINE))


def _docker_args(text: str) -> dict[str, str]:
    return dict(re.findall(r"^ARG ([A-Z][A-Z0-9_]*)=(\S+)$", text, re.MULTILINE))


def _loop_items(text: str, variable: str) -> list[list[str]]:
    return [items.split() for items in re.findall(rf"for {variable} in ([^;]+); do", text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    repo = args.repo.resolve()
    errors: list[str] = []

    setup_version = _setup_version(repo / "setup.py")
    docs_version = _assigned_string(repo / "docs/conf.py", "__version__")
    if setup_version != docs_version:
        errors.append(f"setup.py={setup_version} but docs/conf.py={docs_version}")
    if args.expected_version and setup_version != args.expected_version:
        errors.append(f"release version is {setup_version}, expected {args.expected_version}")

    docker_text = (repo / "docker/Dockerfile").read_text()
    conda_text = (repo / "build_conda.sh").read_text()
    readme_text = (repo / "docker/README.md").read_text()
    justfile_text = (repo / "docker/justfile").read_text()
    docker_args = _docker_args(docker_text)
    conda_exports = _shell_exports(conda_text)

    image_tag = docker_args.get("SGLANG_IMAGE_TAG", "")
    docker_sglang_version = re.sub(r"-cu\d+$", "", image_tag)
    conda_sglang_version = conda_exports.get("SGLANG_VERSION", "")
    if docker_sglang_version != conda_sglang_version:
        errors.append(
            f"Docker SGLang={docker_sglang_version or '<missing>'}, "
            f"conda SGLang={conda_sglang_version or '<missing>'}"
        )

    stable_match = re.search(r"current stable version is:\s*\n- sglang (v\S+)", readme_text)
    readme_sglang_version = stable_match.group(1) if stable_match else ""
    if readme_sglang_version != conda_sglang_version:
        errors.append(
            f"README stable SGLang={readme_sglang_version or '<missing>'}, "
            f"conda SGLang={conda_sglang_version or '<missing>'}"
        )

    for tag in re.findall(r"SGLANG_IMAGE_TAG=(v[^'\"\s]+)", justfile_text):
        if re.sub(r"-cu\d+$", "", tag) != conda_sglang_version:
            errors.append(f"docker/justfile uses inconsistent SGLang tag {tag}")

    for pin in ("MEGATRON_COMMIT", "TMS_COMMIT", "FLASH_QLA_COMMIT"):
        if docker_args.get(pin) != conda_exports.get(pin):
            errors.append(
                f"{pin}: Docker={docker_args.get(pin, '<missing>')}, conda={conda_exports.get(pin, '<missing>')}"
            )

    patch_version = conda_exports.get("PATCH_VERSION", "")
    if patch_version != conda_sglang_version:
        errors.append(f"PATCH_VERSION={patch_version or '<missing>'}, expected {conda_sglang_version}")

    latest_dir = repo / "docker/patch/latest"
    stable_dir = repo / f"docker/patch/{patch_version}"
    latest = {path.name: path.read_bytes() for path in latest_dir.glob("*.patch")}
    stable = {path.name: path.read_bytes() for path in stable_dir.glob("*.patch")}
    if latest.keys() != stable.keys():
        errors.append(
            "stable patch filenames differ from latest: "
            f"only_latest={sorted(latest.keys() - stable.keys())}, "
            f"only_stable={sorted(stable.keys() - latest.keys())}"
        )
    for name in latest.keys() & stable.keys():
        if latest[name] != stable[name]:
            errors.append(f"stable patch differs from latest: {name}")

    docker_sglang_loops = _loop_items(docker_text, "patch")
    conda_loops = _loop_items(conda_text, "patch_name")
    docker_sglang_order = docker_sglang_loops[0] if docker_sglang_loops else []
    conda_sglang_order = conda_loops[0] if conda_loops else []
    expected_sglang = {name for name in latest if name.startswith("sglang")}
    if docker_sglang_order != conda_sglang_order:
        errors.append("Docker and conda SGLang patch order differs")
    if set(docker_sglang_order) != expected_sglang:
        errors.append("Docker/conda SGLang patch loop does not cover the latest patch set")

    docker_megatron_order = re.findall(r"git apply (megatron[^ ]*\.patch)", docker_text)
    conda_megatron_order = conda_loops[1] if len(conda_loops) > 1 else []
    expected_megatron = {name for name in latest if name.startswith("megatron")}
    if docker_megatron_order != conda_megatron_order:
        errors.append("Docker and conda Megatron patch order differs")
    if set(docker_megatron_order) != expected_megatron:
        errors.append("Docker/conda Megatron patch logic does not cover the latest patch set")

    docker_version = (repo / "docker/version.txt").read_text().strip()
    if not re.fullmatch(r"nightly-dev-\d{8}[a-z]", docker_version):
        errors.append(f"unexpected docker/version.txt format: {docker_version}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"release={setup_version}, sglang={conda_sglang_version}, docker={docker_version}, patches={len(latest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
