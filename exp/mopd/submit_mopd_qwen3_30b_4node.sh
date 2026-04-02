#!/usr/bin/env bash
# Submit MOPD 4-node job to Inspire.
#
# Usage:
#   bash exp/mopd/submit_mopd_qwen3_30b_4node.sh
#
# Override env vars as needed:
#   JOB_NAME=mopd-test RESOURCE=8xH100 bash exp/mopd/submit_mopd_qwen3_30b_4node.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -f "${PROJECT_ROOT}/login.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/login.sh"
fi

INSPIRE_BIN="${INSPIRE_CLI:-inspire}"

RESOURCE="${RESOURCE:-8xH100}"
PRIORITY="${PRIORITY:-10}"
MAX_TIME="${MAX_TIME:-24}"
JOB_NAME="${JOB_NAME:-mopd-4node-$(date '+%m%d-%H%M')}"
SUBMIT_NODES="${SUBMIT_NODES:-4}"
IMAGE="${IMAGE:-${INSP_IMAGE:-}}"
WORKSPACE_ID="${WORKSPACE_ID:-ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6}"
REMOTE_ROOT="${INSPIRE_TARGET_DIR:-${PROJECT_ROOT}}"
RUN_SCRIPT="slime/exp/mopd/run_mopd_qwen3_30b_4node.sh"

FORWARDED_ENV_VARS=(
  JOB_NAME
  WORK_ROOT
  MODEL_DIR
  TORCH_DIST_DIR
  STUDENT_CKPT_STEP
  NUM_NODES
  ACTOR_NUM_NODES
  ACTOR_GPUS_PER_NODE
  ROLLOUT_GPUS_TOTAL
  TRAIN_POOL_ROOT
  TRAIN_POOL_INCLUDE_DOMAINS
  TRAIN_POOL_EXCLUDE_PATTERNS
  TRAIN_SOURCE_LIST_BASENAME
  SGLANG_MULTIMODEL_CONFIG
  OPD_DOMAIN_MODEL_MAP
  TOOLCALL_ROLLOUT_BATCH_SIZE
  TOOLCALL_SAMPLES_PER_PROMPT
  TOOLCALL_GLOBAL_BATCH_SIZE
  TOOLCALL_STEPS_PER_ROLLOUT
  TOOLCALL_MAX_CONTEXT_LEN
  TOOLCALL_MAX_RESPONSE_LEN
  TOOLCALL_PARSER_TYPE
  TOOLCALL_LR
  TOOLCALL_ADAM_BETA2
  OPD_KL_COEF
  KL_LOSS_COEF
  TOOLCALL_RESUME_TRAINING
  TOOLCALL_RESUME_NO_OPTIM
  TOOLCALL_RESUME_NO_RNG
  TOOLCALL_RESUME_FINETUNE
  TOOL_CALL_WANDB_PROJECT
  TOOL_CALL_WANDB_GROUP
  WANDB_API_KEY
  WANDB_BASE_URL
)

RUN_ENV_EXPORTS=""
for var_name in "${FORWARDED_ENV_VARS[@]}"; do
  if [[ -n "${!var_name+x}" ]]; then
    RUN_ENV_EXPORTS+=" export ${var_name}=$(printf '%q' "${!var_name}");"
  fi
done
RUN_CMD="cd ${REMOTE_ROOT} &&${RUN_ENV_EXPORTS} bash ${RUN_SCRIPT}"

CMD=(
  "${INSPIRE_BIN}" job create
  --name "${JOB_NAME}"
  --resource "${RESOURCE}"
  --nodes "${SUBMIT_NODES}"
  --priority "${PRIORITY}"
  --max-time "${MAX_TIME}"
  --no-auto
  --command "${RUN_CMD}"
)

if [[ -n "${IMAGE}" ]]; then
  CMD+=(--image "${IMAGE}")
fi

if [[ -n "${WORKSPACE_ID}" ]]; then
  CMD+=(--workspace-id "${WORKSPACE_ID}")
fi

"${CMD[@]}"
