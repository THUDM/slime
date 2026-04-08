#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

source "${PROJECT_ROOT}/login.sh"

RESOURCE="${RESOURCE:-1xH200}"
PRIORITY="${PRIORITY:-10}"
MAX_TIME="${MAX_TIME:-72}"
IMAGE="${IMAGE:-${INSP_IMAGE:-}}"
WORKSPACE_ID="${WORKSPACE_ID:-${INSPIRE_WORKSPACE_GPU_ID:-ws-be832b74-6ac9-45f0-8839-5b0cfcb81496}}"
REMOTE_ROOT="${REMOTE_ROOT:-${INSPIRE_TARGET_DIR:-${PROJECT_ROOT}}}"
RUN_SCRIPT="${RUN_SCRIPT:-slime/examples/run_eval_backfill.sh}"
SUBMIT_CWD="${SUBMIT_CWD:-${PROJECT_ROOT}}"
JOB_LOG_ROOT="${JOB_LOG_ROOT:-${PROJECT_ROOT}/../experiments}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/submit_inspire_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/checkpoint_utils.sh"

csv_contains() {
  local needle="$1"
  local csv="${2:-}"
  local item=""

  [[ -z "${csv}" ]] && return 1
  IFS=',' read -r -a _items <<< "${csv}"
  for item in "${_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "${item}" ]] || continue
    if [[ "$((10#${item}))" -eq "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

join_by_comma() {
  local joined=""
  local item=""
  for item in "$@"; do
    [[ -n "${item}" ]] || continue
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${item}"
  done
  printf '%s' "${joined}"
}

build_iteration_list() {
  local ckpt_dir="$1"
  local requested="${ONLY_ITERATIONS:-}"
  local excluded="${EXCLUDE_ITERATIONS:-}"
  local iter=""
  local selected=()

  while IFS= read -r iter; do
    [[ -n "${iter}" ]] || continue
    if [[ -n "${requested}" ]] && ! csv_contains "${iter}" "${requested}"; then
      continue
    fi
    if csv_contains "${iter}" "${excluded}"; then
      continue
    fi
    selected+=("${iter}")
  done < <(list_checkpoint_iterations "${ckpt_dir}")

  printf '%s\n' "${selected[@]}"
}

submit_sharded_backfill() {
  local experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
  local wandb_run_id="${WANDB_RUN_ID:?Set WANDB_RUN_ID}"
  local ckpt_dir="${CKPT_DIR:-${experiment_dir}/checkpoints}"
  local shard_count="${SHARD_COUNT:-4}"
  local exp_name="${EXP_NAME:-$(basename "${experiment_dir}")}"
  local job_name_prefix="${JOB_NAME_PREFIX:-mopd-eval-backfill-${exp_name}}"
  local batch_size="${BATCH_SIZE:-4}"
  local mem_fraction="${SGLANG_MEM_FRACTION_STATIC:-0.82}"
  local sglang_tp="${SGLANG_TP:-1}"
  local sglang_ep="${SGLANG_EP:-1}"
  local wandb_project="${WANDB_PROJECT:-slime-mopd}"
  local wandb_group="${WANDB_GROUP:-}"
  local wandb_dir_base="${WANDB_DIR_BASE:-${REMOTE_ROOT}/wandb_backfill}"
  local hf_cache_base="${HF_CACHE_BASE:-${experiment_dir}/hf_cache_backfill_jobs}"
  local eval_config_base="${EVAL_CONFIG_BASE:-${experiment_dir}/data_cache}"
  local log_base="${BACKFILL_LOG_BASE:-${experiment_dir}/logs}"
  local keep_hf_cache="${KEEP_HF_CACHE:-0}"
  local reward_module="${EVAL_REWARD_MODULE:-}"
  local eval_pythonpath_prefix="${EVAL_BACKFILL_PYTHONPATH_PREFIX:-}"
  local model_dir="${MODEL_DIR:?Set MODEL_DIR}"
  local port_base="${SGLANG_PORT_BASE:-31200}"
  local timestamp
  timestamp="$(date '+%m%d-%H%M')"

  mapfile -t iterations < <(build_iteration_list "${ckpt_dir}")
  if [[ ${#iterations[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoints selected under ${ckpt_dir}" >&2
    return 1
  fi

  declare -a shard_lists
  local idx=0
  local iter=""
  for iter in "${iterations[@]}"; do
    local shard_idx=$((idx % shard_count))
    shard_lists[${shard_idx}]="${shard_lists[${shard_idx}]:-} ${iter}"
    idx=$((idx + 1))
  done

  echo "=== Submitting sharded MOPD eval backfill ==="
  echo "Experiment: ${experiment_dir}"
  echo "Run ID:     ${wandb_run_id}"
  echo "Steps:      ${iterations[*]}"
  echo "Shards:     ${shard_count}"

  local shard_id=""
  for (( shard_id=0; shard_id<shard_count; shard_id++ )); do
    local shard_steps_raw="${shard_lists[${shard_id}]:-}"
    [[ -n "${shard_steps_raw// }" ]] || continue

    read -r -a shard_steps <<< "${shard_steps_raw}"
    local only_iterations
    only_iterations="$(join_by_comma "${shard_steps[@]}")"
    local shard_num=$((shard_id + 1))
    local shard_tag
    shard_tag="$(printf 's%02d' "${shard_num}")"
    local shard_job_name="${job_name_prefix}-${shard_tag}-${timestamp}"
    local shard_port=$((port_base + shard_id))
    local shard_hf_cache="${hf_cache_base}/${shard_tag}"
    local shard_eval_config="${eval_config_base}/eval_config.backfill.${shard_tag}.yaml"
    local shard_sglang_log="${log_base}/sglang_eval.${shard_tag}.log"
    local shard_wandb_dir="${wandb_dir_base}/${shard_tag}"

    local run_cmd="cd ${REMOTE_ROOT}"
    run_cmd+=" && export EXPERIMENT_DIR='${experiment_dir}'"
    run_cmd+=" && export MODEL_DIR='${model_dir}'"
    run_cmd+=" && export WANDB_RUN_ID='${wandb_run_id}'"
    run_cmd+=" && export WANDB_PROJECT='${wandb_project}'"
    run_cmd+=" && export WANDB_GROUP='${wandb_group}'"
    run_cmd+=" && export WANDB_API_KEY='${WANDB_API_KEY:-}'"
    run_cmd+=" && export WANDB_BASE_URL='${WANDB_BASE_URL:-}'"
    run_cmd+=" && export WANDB_DIR='${shard_wandb_dir}'"
    run_cmd+=" && export ONLY_ITERATIONS='${only_iterations}'"
    run_cmd+=" && export KEEP_HF_CACHE='${keep_hf_cache}'"
    run_cmd+=" && export HF_CACHE='${shard_hf_cache}'"
    run_cmd+=" && export EVAL_CONFIG_PATH='${shard_eval_config}'"
    run_cmd+=" && export SGLANG_LOG_PATH='${shard_sglang_log}'"
    run_cmd+=" && export SGLANG_PORT='${shard_port}'"
    run_cmd+=" && export SGLANG_TP='${sglang_tp}'"
    run_cmd+=" && export SGLANG_EP='${sglang_ep}'"
    run_cmd+=" && export SGLANG_MEM_FRACTION_STATIC='${mem_fraction}'"
    run_cmd+=" && export BATCH_SIZE='${batch_size}'"
    if [[ -n "${EVAL_DATASETS:-}" ]]; then
      run_cmd+=" && export EVAL_DATASETS='${EVAL_DATASETS}'"
    fi
    if [[ -n "${EVAL_DATASETS_EXTRA:-}" ]]; then
      run_cmd+=" && export EVAL_DATASETS_EXTRA='${EVAL_DATASETS_EXTRA}'"
    fi
    if [[ -n "${EVAL_PATHS:-}" ]]; then
      run_cmd+=" && export EVAL_PATHS='${EVAL_PATHS}'"
    fi
    if [[ -n "${EVAL_PATHS_EXTRA:-}" ]]; then
      run_cmd+=" && export EVAL_PATHS_EXTRA='${EVAL_PATHS_EXTRA}'"
    fi
    if [[ -n "${reward_module}" ]]; then
      run_cmd+=" && export EVAL_REWARD_MODULE='${reward_module}'"
    fi
    if [[ -n "${eval_pythonpath_prefix}" ]]; then
      run_cmd+=" && export EVAL_BACKFILL_PYTHONPATH_PREFIX='${eval_pythonpath_prefix}'"
    fi
    run_cmd+=" && bash ${RUN_SCRIPT}"

    echo "  ${shard_tag}: steps=${only_iterations}"
    submit_inspire_command_job "${shard_job_name}" 1 "${run_cmd}"
  done
}

submit_sharded_backfill
