#!/usr/bin/env bash
# Sourced inside the existing privileged CI container; no Docker daemon needed.
set -euo pipefail
if ! command -v skopeo >/dev/null || ! command -v umoci >/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends skopeo umoci
fi
# Refresh the wheel even when the CI image already includes sunabako.
python -m pip install --upgrade --no-deps --only-binary=sunabako --break-system-packages sunabako
python -m pip install --break-system-packages -r examples/coding_agent_rl/requirements-sunabako.txt
export SLIME_AGENT_TEST_CACHE=/data/slime_ci/agent-e2e/cache
export TILELANG_CACHE_DIR="$SLIME_AGENT_TEST_CACHE/tilelang"
export TRITON_CACHE_DIR="$SLIME_AGENT_TEST_CACHE/triton"
export SLIME_AGENT_TEST_RUN_DIR="$PWD/.agent-e2e/run"
# This bounded functional test does not certify aggregate hard RAM enforcement.
# Production still requires a delegated writable cgroup and fails closed.
export SUNABAKO_ALLOW_TEST_MEMORY=1
export SUNABAKO_RUNTIME=native
