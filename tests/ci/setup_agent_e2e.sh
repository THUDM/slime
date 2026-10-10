#!/usr/bin/env bash
# Sourced inside the existing privileged CI container; no Docker daemon needed.
set -euo pipefail
# Keep downloads on the configured proxy and local Ray/node traffic direct.
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-$http_proxy}}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export all_proxy="${all_proxy:-${ALL_PROXY:-}}" ALL_PROXY="${all_proxy:-${ALL_PROXY:-}}"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-${NO_PROXY:-}}"
export NO_PROXY="$no_proxy"
if ! command -v skopeo >/dev/null || ! command -v umoci >/dev/null; then
    apt-get update -qq
    apt-get install -y --no-install-recommends skopeo umoci
fi
# Install the released wheel pinned by the example requirements.
python -m pip install --only-binary=sunabako --break-system-packages \
    -r examples/coding_agent_rl/requirements-sunabako.txt
export SLIME_AGENT_TEST_CACHE="${SLIME_AGENT_TEST_CACHE:-/data/slime_ci/agent-e2e/cache}"
export TILELANG_CACHE_DIR="$SLIME_AGENT_TEST_CACHE/tilelang"
export TRITON_CACHE_DIR="$SLIME_AGENT_TEST_CACHE/triton"
export SLIME_AGENT_TEST_RUN_DIR="${SLIME_AGENT_TEST_RUN_DIR:-$PWD/.agent-e2e/run}"
# This bounded functional test does not certify aggregate hard RAM enforcement.
# Production still requires a delegated writable cgroup and fails closed.
export SUNABAKO_ALLOW_TEST_MEMORY=1
export SUNABAKO_RUNTIME=native
# Cold downloads may take much longer than training. Populate persistent caches
# before gpu_lock_exec so another job can use the GPUs while we fetch assets.
python tests/test_agent_sunabako_codex_e2e.py --prepare-only
