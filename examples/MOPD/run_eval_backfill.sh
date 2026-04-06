#!/bin/bash
# MOPD eval backfill: run official benchmark evals on existing checkpoints.
# Reuses shared examples/eval_backfill.py with MOPD's reward router.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

WANDB_PROJECT="${WANDB_PROJECT:-slime-mopd}"
MODEL_DIR="${MODEL_DIR:-}"
SGLANG_CLEANUP_SLEEP_SECONDS="${SGLANG_CLEANUP_SLEEP_SECONDS:-2}"
EVAL_DATASET_DISCOVERY_MODE="${EVAL_DATASET_DISCOVERY_MODE:-glob}"
EVAL_REWARD_MODULE="${EVAL_REWARD_MODULE:-reward_mopd_eval_router.reward_func}"
EVAL_BACKFILL_PY="${EVAL_BACKFILL_PY:-${PROJECT_ROOT}/slime/examples/eval_backfill.py}"
EVAL_BACKFILL_PYTHONPATH_PREFIX="${EVAL_BACKFILL_PYTHONPATH_PREFIX:-${SCRIPT_DIR}:${PROJECT_ROOT}/slime/examples:${PROJECT_ROOT}/slime}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../_shared/eval_backfill_utils.sh"

run_eval_backfill_main
