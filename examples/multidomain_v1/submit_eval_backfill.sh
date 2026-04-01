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
RUN_SCRIPT="slime/examples/multidomain_v1/run_eval_backfill.sh"

submit_one() {
  local exp_name="$1"
  local experiment_dir="$2"
  local wandb_run_id="$3"
  local wandb_group="$4"

  local job_name="eval-backfill-${exp_name}-$(date '+%m%d-%H%M')"
  local run_cmd="cd ${REMOTE_ROOT} && export WANDB_API_KEY='${WANDB_API_KEY:-}' && export WANDB_BASE_URL='${WANDB_BASE_URL:-}' && export EXPERIMENT_DIR='${experiment_dir}' && export WANDB_RUN_ID='${wandb_run_id}' && export WANDB_GROUP='${wandb_group}' && bash ${RUN_SCRIPT}"

  local cmd=(
    "$INSPIRE_CLI" job create
    --name "${job_name}"
    --resource "${RESOURCE}"
    --nodes 1
    --priority "${PRIORITY}"
    --max-time "${MAX_TIME}"
    --no-auto
    --command "${run_cmd}"
  )

  if [[ -n "${IMAGE}" ]]; then
    cmd+=(--image "${IMAGE}")
  fi

  if [[ -n "${WORKSPACE_ID}" ]]; then
    cmd+=(--workspace-id "${WORKSPACE_ID}")
  fi

  echo "=== Submitting eval backfill: ${exp_name} ==="
  "${cmd[@]}"
  echo ""
}

submit_one "tool15-struct55-stem30" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/qwen3_30b_a3b_mdv1_3node_tool15_struct55_stem30_0330-233539" \
  "irgu6dsr" \
  "qwen3-30b-a3b-mdv1-3node-tool15-struct55-stem30-0330-233539"

submit_one "tool45-struct25-stem30" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/qwen3_30b_a3b_mdv1_3node_tool45_struct25_stem30_0330-233543" \
  "0hgesrzt" \
  "qwen3-30b-a3b-mdv1-3node-tool45-struct25-stem30-0330-233543"

submit_one "tool50-struct35-stem15" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/qwen3_30b_a3b_mdv1_3node_tool50_struct35_stem15_0330-234805" \
  "akzm0qu9" \
  "qwen3-30b-a3b-mdv1-3node-tool50-struct35-stem15-0330-234805"
