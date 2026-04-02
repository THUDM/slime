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
RUN_SCRIPT="slime/examples/multidomain_v2/run_eval_backfill.sh"

submit_one() {
  local exp_name="$1"
  local experiment_dir="$2"
  local wandb_run_id="$3"
  local wandb_group="$4"

  local job_name="v2-eval-backfill-${exp_name}-$(date '+%m%d-%H%M')"
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

  echo "=== Submitting v2 eval backfill: ${exp_name} ==="
  "${cmd[@]}"
  echo ""
}

# --- v2 experiments to backfill ---

submit_one "main-retry" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_main_retry" \
  "k2dtse1o" \
  "mdv2-main-retry-0331-2159"

submit_one "main-plus-xlam" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_main_plus_xlam" \
  "xcphh05q" \
  "toolbench_v1,apibench,apigen,agent,jsonschemabench,nemotron_structured_outputs,xlam_function_calling_60k"

submit_one "nk-ns" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_nk_ns" \
  "m4blb0d9" \
  "mdv2-nk-ns-retry-0331-2242"

submit_one "no-jsonschema-only-ns" \
  "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/multidomain_v2_no_jsonschema_only_ns" \
  "22lh8kda" \
  "toolbench_v1,apibench,apigen,agent,nemotron_structured_outputs"
