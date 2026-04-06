#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

source "${PROJECT_ROOT}/login.sh"

RESOURCE="${RESOURCE:-8xH100}"
PRIORITY="${PRIORITY:-10}"
MAX_TIME="${MAX_TIME:-24}"
IMAGE="${IMAGE:-${INSP_IMAGE:-}}"
WORKSPACE_ID="${WORKSPACE_ID:-ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6}"
REMOTE_ROOT="${INSPIRE_TARGET_DIR:-${PROJECT_ROOT}}"
RUN_SCRIPT="slime/examples/run_convert_ckpt_to_hf.sh"
MODEL_DIR="${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints/iter_0001432_hf}"
FORCE_RECONVERT="${FORCE_RECONVERT:-0}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/submit_inspire_utils.sh"

submit_one() {
  local exp_name="$1"
  local experiment_dir="$2"

  local job_name="v2-ckpt2hf-${exp_name}-$(date '+%m%d-%H%M')"
  local run_cmd="cd ${REMOTE_ROOT} && export EXPERIMENT_DIR='${experiment_dir}' && export MODEL_DIR='${MODEL_DIR}' && export FORCE_RECONVERT='${FORCE_RECONVERT}' && bash ${RUN_SCRIPT}"

  echo "=== Submitting v2 ckpt->hf convert: ${exp_name} ==="
  submit_inspire_command_job "${job_name}" 1 "${run_cmd}"
  echo ""
}

submit_one "main-retry" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_main_retry"

submit_one "main-plus-xlam" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_main_plus_xlam"

submit_one "nk-ns" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_nk_ns"

submit_one "no-jsonschema-only-ns" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_no_jsonschema_only_ns"
