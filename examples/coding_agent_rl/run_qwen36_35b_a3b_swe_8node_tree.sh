#!/usr/bin/env bash
# End-to-end SWE RL case: save full per-turn trajectory_tree metadata.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SWE_SAVE_TRAJECTORY_TREE=1
export RUN_ROOT="${RUN_ROOT:-/mnt/jingshenghang/storage/slime_swe_runs/qwen36_cagent_tree_e2e_$(date +%Y%m%d_%H%M%S)}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export SWE_MAX_RESPONSE_TOKENS="${SWE_MAX_RESPONSE_TOKENS:-1024}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8}"
export SWE_TIME_BUDGET_SEC="${SWE_TIME_BUDGET_SEC:-120}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-120}"

exec bash "${SCRIPT_DIR}/run_qwen36_35b_a3b_swe_8node.sh" "$@"
