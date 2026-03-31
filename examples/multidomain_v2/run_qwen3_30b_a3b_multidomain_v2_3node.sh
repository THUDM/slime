#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

export TRAIN_POOL_ROOT="${TRAIN_POOL_ROOT:-${AVALANCHE_ROOT}/data/pool}"
export WORK_ROOT="${WORK_ROOT:-${AVALANCHE_ROOT}/experiments/multidomain_v2_3node}"
export TRAIN_DATA_BASENAME="${TRAIN_DATA_BASENAME:-multidomain_v2_train.normalized.jsonl}"
export TOOL_CALL_WANDB_PROJECT="${TOOL_CALL_WANDB_PROJECT:-slime-multidomain-v2}"
export TOOL_CALL_WANDB_GROUP="${TOOL_CALL_WANDB_GROUP:-${JOB_NAME:-qwen3-30b-a3b-mdv2-3node}}"

exec "${SCRIPT_DIR}/../multidomain_v1/run_qwen3_30b_a3b_multidomain_v1_3node.sh" "$@"
