#!/usr/bin/env bash
# Runs the synthetic train-only smoke that verifies both linear samples and
# samples carrying trajectory_tree metadata are accepted by the trainer.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

cd "${SLIME_DIR}"

export SLIME_RUN_CODING_AGENT_RL_DYNAMIC=1
pytest -q examples/coding_agent_rl/test_dynamic_trajectory_tree_train.py "$@"

