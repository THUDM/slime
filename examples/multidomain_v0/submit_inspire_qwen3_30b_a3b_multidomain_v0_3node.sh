#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

source "${PROJECT_ROOT}/login.sh"

RESOURCE="${RESOURCE:-8xH200}"
PRIORITY="${PRIORITY:-10}"
MAX_TIME="${MAX_TIME:-24}"
JOB_NAME="${JOB_NAME:-multidomain-v0-3node-$(date '+%m%d-%H%M')}"
IMAGE="${IMAGE:-${INSP_IMAGE:-}}"
WORKSPACE_ID="${WORKSPACE_ID:-${INSPIRE_WORKSPACE_GPU_ID:-}}"
REMOTE_ROOT="${INSPIRE_TARGET_DIR:-${PROJECT_ROOT}}"
UV_VENV_DIR="${UV_VENV_DIR:-${REMOTE_ROOT}/.nemo_gym_venvs}"
RUN_SCRIPT="slime/examples/multidomain_v0/run_qwen3_30b_a3b_multidomain_v0_3node.sh"
RUN_CMD="cd ${REMOTE_ROOT} && export WANDB_API_KEY='${WANDB_API_KEY:-}' && export WANDB_BASE_URL='${WANDB_BASE_URL:-}' && export UV_VENV_DIR='${UV_VENV_DIR}' && bash ${RUN_SCRIPT}"

CMD=(
  "$INSPIRE_CLI" job create
  --name "${JOB_NAME}"
  --resource "${RESOURCE}"
  --nodes 3
  --priority "${PRIORITY}"
  --max-time "${MAX_TIME}"
  --command "${RUN_CMD}"
)

if [[ -n "${IMAGE}" ]]; then
  CMD+=(--image "${IMAGE}")
fi

if [[ -n "${WORKSPACE_ID}" ]]; then
  CMD+=(--workspace-id "${WORKSPACE_ID}")
fi

"${CMD[@]}"
