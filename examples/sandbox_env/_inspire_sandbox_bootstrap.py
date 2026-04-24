from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_site_packages_root() -> Path:
    return Path(__file__).resolve().parents[4] / ".local" / "share" / "inspire_sandbox_site_packages"


def iter_inspire_sandbox_paths() -> list[str]:
    raw_root = os.environ.get("INSPIRE_SANDBOX_SITE_PACKAGES", "").strip()
    root = Path(raw_root) if raw_root else _default_site_packages_root()

    candidates: list[Path] = [root]
    if root.exists():
        candidates.extend(sorted(root.glob("lib/python*/site-packages")))

    seen: set[str] = set()
    resolved: list[str] = []
    for candidate in candidates:
        value = str(candidate)
        if candidate.exists() and value not in seen:
            resolved.append(value)
            seen.add(value)
    return resolved


def bootstrap_inspire_sandbox_path() -> list[str]:
    added: list[str] = []
    for value in reversed(iter_inspire_sandbox_paths()):
        if value not in sys.path:
            sys.path.insert(0, value)
            added.append(value)
    return added


def build_pythonpath(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    entries: list[str] = []
    seen: set[str] = set()

    for value in iter_inspire_sandbox_paths():
        if value not in seen:
            entries.append(value)
            seen.add(value)

    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        for value in existing.split(os.pathsep):
            value = value.strip()
            if value and value not in seen:
                entries.append(value)
                seen.add(value)

    return os.pathsep.join(entries)
