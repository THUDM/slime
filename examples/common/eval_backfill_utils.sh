#!/usr/bin/env bash

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_QUERIES_PY="${SCRIPT_QUERIES_PY:-${COMMON_DIR}/script_queries.py}"

# shellcheck source=/dev/null
source "${COMMON_DIR}/checkpoint_utils.sh"

discover_eval_data_dir() {
  local experiment_dir="$1"
  local mode="${2:-glob}"
  local candidate=""

  for candidate in "${experiment_dir}/data_cache" "${experiment_dir}/runtime_data"; do
    if [[ "${mode}" == "canonical" ]]; then
      if [[ -f "${candidate}/bfcl_v3_eval.normalized.jsonl" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    elif ls "${candidate}"/*.normalized.jsonl >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

discover_eval_datasets() {
  local data_cache="$1"
  local mode="${2:-glob}"
  if [[ "${mode}" == "canonical" ]]; then
    local name_file=""
    for name_file in \
      "bfcl_v3_eval:${data_cache}/bfcl_v3_eval.normalized.jsonl" \
      "bfcl_multi_turn_eval:${data_cache}/bfcl_multi_turn_eval.normalized.jsonl" \
      "ifeval_eval:${data_cache}/ifeval_eval.normalized.jsonl" \
      "jsonschemabench_eval:${data_cache}/jsonschemabench_eval.normalized.jsonl" \
      "ifbench_test_eval:${data_cache}/ifbench_test_eval.normalized.jsonl" \
      "mmlu_pro_eval:${data_cache}/mmlu_pro_eval.normalized.jsonl" \
      "gpqa_main_eval:${data_cache}/gpqa_main_eval.normalized.jsonl"; do
      local path="${name_file#*:}"
      if [[ -f "${path}" ]]; then
        printf '%s\n' "${name_file}"
      else
        echo "WARN: Eval file not found, skipping: ${path}"
      fi
    done
    return 0
  fi

  local f=""
  for f in "${data_cache}"/*_eval.normalized.jsonl; do
    [[ -f "${f}" ]] || continue
    local name
    name="$(basename "${f}" .normalized.jsonl)"
    printf '%s:%s\n' "${name}" "${f}"
  done
}

load_eval_specs_from_config() {
  local eval_config_path="$1"
  python3 "${SCRIPT_QUERIES_PY}" load-eval-config --path "${eval_config_path}"
}

generate_eval_specs_from_mopd_config() {
  local avalanche_root="${AVALANCHE_ROOT:-$(cd -- "${PROJECT_ROOT}/.." && pwd)}"
  local pool_root="${POOL_ROOT:-${avalanche_root}/data/pool}"
  local eval_config_path="${EVAL_CONFIG_PATH:-${EXPERIMENT_DIR}/data_cache/eval_config.backfill.yaml}"
  local write_eval_config_py="${PROJECT_ROOT}/slime/examples/MOPD/write_eval_config.py"
  local eval_args=(
    --pool-root "${pool_root}"
    --output "${eval_config_path}"
    --max-response-len "${MAX_TOKENS}"
  )
  local item=""

  IFS=',' read -r -a _eval_datasets <<< "${EVAL_DATASETS:-}"
  for item in "${_eval_datasets[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--dataset "${item}")
    fi
  done
  IFS=',' read -r -a _eval_dataset_extras <<< "${EVAL_DATASETS_EXTRA:-}"
  for item in "${_eval_dataset_extras[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--dataset-extra "${item}")
    fi
  done
  IFS=',' read -r -a _eval_paths <<< "${EVAL_PATHS:-}"
  for item in "${_eval_paths[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--source "${item}")
    fi
  done
  IFS=',' read -r -a _eval_path_extras <<< "${EVAL_PATHS_EXTRA:-}"
  for item in "${_eval_path_extras[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--source-extra "${item}")
    fi
  done

  PYTHONPATH="${PROJECT_ROOT}/slime/examples:${PROJECT_ROOT}/slime:${PYTHONPATH:-}" \
    python3 "${write_eval_config_py}" "${eval_args[@]}" >&2

  load_eval_specs_from_config "${eval_config_path}"
}

cleanup_sglang_eval_server() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep "${SGLANG_CLEANUP_SLEEP_SECONDS:-3}"
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
}

start_sglang_eval_server() {
  local hf_dir="$1"
  cleanup_sglang_eval_server
  local sglang_log_path="${SGLANG_LOG_PATH:-${EXPERIMENT_DIR}/logs/sglang_eval.log}"

  echo "Starting sglang server for ${hf_dir} ..."
  python3 -m sglang.launch_server \
    --model-path "${hf_dir}" \
    --port "${SGLANG_PORT}" \
    --tp "${SGLANG_TP}" \
    --ep "${SGLANG_EP}" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.85}" \
    &>"${sglang_log_path}" &

  local deadline=$((SECONDS + ${SGLANG_START_TIMEOUT_SECONDS:-600}))
  while (( SECONDS < deadline )); do
    if curl -sf "http://localhost:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
      echo "sglang server ready."
      return 0
    fi
    sleep 5
  done

  echo "ERROR: sglang did not start in ${SGLANG_START_TIMEOUT_SECONDS:-600}s" >&2
  tail -50 "${sglang_log_path}" || true
  return 1
}

run_eval_backfill_for_checkpoint() {
  local hf_dir="$1"
  local rollout_id="$2"
  local eval_args=()
  local ds=""

  while IFS= read -r ds; do
    [[ -n "${ds}" ]] || continue
    eval_args+=(--eval-data "${ds}")
  done <<< "${EVAL_DATASETS_TEXT}"

  local pythonpath_prefix="${EVAL_BACKFILL_PYTHONPATH_PREFIX:-${PROJECT_ROOT}/slime}"
  local reward_module="${EVAL_REWARD_MODULE:-}"
  local reward_args=()
  if [[ -n "${reward_module}" ]]; then
    reward_args+=(--reward-module "${reward_module}")
  fi

  PYTHONPATH="${pythonpath_prefix}:${PYTHONPATH:-}" \
  python3 "${EVAL_BACKFILL_PY}" \
    --sglang-url "http://localhost:${SGLANG_PORT}" \
    --model-path "${hf_dir}" \
    "${reward_args[@]}" \
    "${eval_args[@]}" \
    --rollout-id "${rollout_id}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-host "${WANDB_BASE_URL:-}" \
    --wandb-key "${WANDB_API_KEY:-}" \
    --wandb-group "${WANDB_GROUP}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}"
}

should_run_iteration() {
  local iter_num="$1"
  [[ -z "${ONLY_ITERATIONS:-}" ]] && return 0

  local item=""
  IFS=',' read -r -a selected <<< "${ONLY_ITERATIONS}"
  for item in "${selected[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "${item}" ]] || continue
    if [[ "$((10#${item}))" -eq "${iter_num}" ]]; then
      return 0
    fi
  done
  return 1
}

run_eval_backfill_main() {
  export PYTHONUNBUFFERED=1

  EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
  WANDB_RUN_ID="${WANDB_RUN_ID:?Set WANDB_RUN_ID}"
  WANDB_PROJECT="${WANDB_PROJECT:-slime-multidomain-v2}"
  WANDB_GROUP="${WANDB_GROUP:-}"

  SGLANG_PORT="${SGLANG_PORT:-30000}"
  SGLANG_TP="${SGLANG_TP:-4}"
  SGLANG_EP="${SGLANG_EP:-2}"
  MAX_TOKENS="${MAX_TOKENS:-8192}"
  BATCH_SIZE="${BATCH_SIZE:-32}"

  CKPT_DIR="${CKPT_DIR:-${EXPERIMENT_DIR}/checkpoints}"
  HF_CACHE="${HF_CACHE:-${EXPERIMENT_DIR}/hf_cache}"

  local data_cache=""
  if [[ -n "${EVAL_DATASETS:-}" || -n "${EVAL_DATASETS_EXTRA:-}" || -n "${EVAL_PATHS:-}" || -n "${EVAL_PATHS_EXTRA:-}" ]]; then
    echo "Using MOPD-style eval dataset selection: ${EVAL_DATASETS:-<custom>}"
    EVAL_DATASETS_TEXT="$(generate_eval_specs_from_mopd_config)"
  else
    data_cache="$(discover_eval_data_dir "${EXPERIMENT_DIR}" "${EVAL_DATASET_DISCOVERY_MODE:-canonical}")" || {
      echo "ERROR: No eval data found in data_cache or runtime_data" >&2
      return 1
    }
    echo "Using eval data from: ${data_cache}"

    EVAL_DATASETS_TEXT="$(
      discover_eval_datasets "${data_cache}" "${EVAL_DATASET_DISCOVERY_MODE:-canonical}"
    )"
  fi
  if [[ -z "${EVAL_DATASETS_TEXT}" ]]; then
    echo "ERROR: No eval datasets resolved" >&2
    return 1
  fi

  echo "=== Eval backfill ==="
  echo "Experiment: ${EXPERIMENT_DIR}"
  echo "wandb run:  ${WANDB_RUN_ID}"
  echo "Eval sets:"
  while IFS= read -r ds; do
    [[ -n "${ds}" ]] || continue
    echo "  ${ds%%:*}"
  done <<< "${EVAL_DATASETS_TEXT}"

  mkdir -p "${HF_CACHE}" "${EXPERIMENT_DIR}/logs"

  local iterations=()
  local iter_num=""
  while IFS= read -r iter_num; do
    [[ -n "${iter_num}" ]] || continue
    iterations+=("${iter_num}")
  done < <(list_checkpoint_iterations "${CKPT_DIR}")
  if [[ ${#iterations[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoints found under ${CKPT_DIR}" >&2
    return 1
  fi

  echo "Found ${#iterations[@]} checkpoints: ${iterations[*]}"

  local prev_hf_dir=""
  for iter_num in "${iterations[@]}"; do
    if ! should_run_iteration "${iter_num}"; then
      continue
    fi

    local iter_name
    iter_name="$(printf 'iter_%07d' "${iter_num}")"
    local hf_dir="${HF_CACHE}/${iter_name}_hf"

    echo ""
    echo "====== Checkpoint ${iter_name} (rollout_id=${iter_num}) ======"

    convert_torch_dist_checkpoint_to_hf "${CKPT_DIR}/${iter_name}" "${hf_dir}"
    start_sglang_eval_server "${hf_dir}"
    run_eval_backfill_for_checkpoint "${hf_dir}" "${iter_num}"
    cleanup_sglang_eval_server

    if [[ "${KEEP_HF_CACHE:-0}" != "1" ]] && [[ -n "${prev_hf_dir}" ]] && [[ -d "${prev_hf_dir}" ]]; then
      echo "  Cleaning up previous HF cache: ${prev_hf_dir}"
      rm -rf "${prev_hf_dir}"
    fi
    prev_hf_dir="${hf_dir}"
  done

  if [[ "${KEEP_HF_CACHE:-0}" != "1" ]] && [[ -n "${prev_hf_dir}" ]] && [[ -d "${prev_hf_dir}" ]]; then
    rm -rf "${prev_hf_dir}"
  fi

  echo ""
  echo "=== Eval backfill complete ==="
}
