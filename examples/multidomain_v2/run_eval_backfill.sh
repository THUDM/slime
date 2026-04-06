#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints/iter_0001432_hf}"
EVAL_DATASET_DISCOVERY_MODE="${EVAL_DATASET_DISCOVERY_MODE:-canonical}"
EVAL_BACKFILL_PY="${EVAL_BACKFILL_PY:-${SCRIPT_DIR}/eval_backfill.py}"
EVAL_BACKFILL_PYTHONPATH_PREFIX="${EVAL_BACKFILL_PYTHONPATH_PREFIX:-${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PROJECT_ROOT}/slime}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../_shared/eval_backfill_utils.sh"

run_eval_backfill_main
