"""B2 regression test (U5): the example modules must be importable as a
real package (`examples.coding_agent_rl.*`) from an *external* cwd with
only the worktree root on PYTHONPATH.

Why this exists
---------------
Ray rollout workers load the user-provided generate function via
`importlib.import_module("examples.coding_agent_rl.generate")` from a
process whose cwd is NOT the example directory. Stage 8B's real-cluster
run failed because middleware.py used a bare `import aiohttp_threaded`
that only resolves when `coding_agent_rl/` itself is on sys.path -- the
case the in-tree smoke tests happen to set up but ray workers never do.

The other smoke tests prepend `coding_agent_rl/` to sys.path before
importing, so they exercise the top-level-module mode and would never
catch this class of regression. This test deliberately runs in a fresh
subprocess in `/tmp` to assert the package-mode import path stays green.

If a module touched here grows a real import-time side effect (network,
GPU init, missing optional dep), prefer fixing the side effect rather
than relaxing this test -- ray workers will hit the same wall.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def _run_import(module_names: list[str]) -> subprocess.CompletedProcess:
    """Spawn a fresh python process in /tmp with PYTHONPATH=worktree root,
    then import the given modules sequentially.

    Returncode 0 + empty stderr ModuleNotFoundError == pass."""
    code = "; ".join(f"import {m}" for m in module_names)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKTREE_ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_middleware_import_from_external_cwd():
    """middleware.py must not depend on `coding_agent_rl/` being on
    sys.path -- it ships as `examples.coding_agent_rl.middleware`."""
    proc = _run_import(["examples.coding_agent_rl.middleware"])
    assert proc.returncode == 0, (
        f"middleware import failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr


def test_sandbox_import_from_external_cwd():
    """sandbox.py same contract as middleware."""
    proc = _run_import(["examples.coding_agent_rl.sandbox"])
    assert proc.returncode == 0, (
        f"sandbox import failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr


def test_generate_import_from_external_cwd():
    """generate.py is the entrypoint ray actually loads via
    `importlib.import_module("examples.coding_agent_rl.generate")`.
    This is the exact failure mode stage 8B hit -- keep it covered."""
    proc = _run_import(["examples.coding_agent_rl.generate"])
    assert proc.returncode == 0, (
        f"generate import failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr


def test_all_three_in_one_process():
    """And together, to catch order-dependent module-cache surprises."""
    proc = _run_import([
        "examples.coding_agent_rl.generate",
        "examples.coding_agent_rl.middleware",
        "examples.coding_agent_rl.sandbox",
    ])
    assert proc.returncode == 0, (
        f"combined import failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
