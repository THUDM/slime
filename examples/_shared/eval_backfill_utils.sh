#!/usr/bin/env bash

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

cleanup_sglang_eval_server() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep "${SGLANG_CLEANUP_SLEEP_SECONDS:-3}"
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
}

start_sglang_eval_server() {
  local hf_dir="$1"
  cleanup_sglang_eval_server

  echo "Starting sglang server for ${hf_dir} ..."
  python3 -m sglang.launch_server \
    --model-path "${hf_dir}" \
    --port "${SGLANG_PORT}" \
    --tp "${SGLANG_TP}" \
    --ep "${SGLANG_EP}" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.85}" \
    &>"${EXPERIMENT_DIR}/logs/sglang_eval.log" &

  local deadline=$((SECONDS + ${SGLANG_START_TIMEOUT_SECONDS:-600}))
  while (( SECONDS < deadline )); do
    if curl -sf "http://localhost:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
      echo "sglang server ready."
      return 0
    fi
    sleep 5
  done

  echo "ERROR: sglang did not start in ${SGLANG_START_TIMEOUT_SECONDS:-600}s" >&2
  tail -50 "${EXPERIMENT_DIR}/logs/sglang_eval.log" || true
  return 1
}

convert_checkpoint_to_hf() {
  local iter_dir="$1"
  local hf_out="$2"

  if [[ -f "${hf_out}/config.json" ]]; then
    echo "  HF checkpoint already exists: ${hf_out}"
    return 0
  fi

  echo "  Converting torch_dist -> HF: ${iter_dir} -> ${hf_out}"
  python3 "${PROJECT_ROOT}/slime/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${iter_dir}" \
    --output-dir "${hf_out}" \
    --origin-hf-dir "${MODEL_DIR}" \
    --force
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

  local data_cache
  data_cache="$(discover_eval_data_dir "${EXPERIMENT_DIR}" "${EVAL_DATASET_DISCOVERY_MODE:-canonical}")" || {
    echo "ERROR: No eval data found in data_cache or runtime_data" >&2
    return 1
  }
  echo "Using eval data from: ${data_cache}"

  EVAL_DATASETS_TEXT="$(
    discover_eval_datasets "${data_cache}" "${EVAL_DATASET_DISCOVERY_MODE:-canonical}"
  )"
  if [[ -z "${EVAL_DATASETS_TEXT}" ]]; then
    echo "ERROR: No eval datasets found in ${data_cache}" >&2
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
  local iter_dir=""
  for iter_dir in "${CKPT_DIR}"/iter_*; do
    [[ -d "${iter_dir}" ]] || continue
    local iter_name
    iter_name="$(basename "${iter_dir}")"
    local iter_num="${iter_name#iter_}"
    iterations+=("$((10#${iter_num}))")
  done
  if [[ ${#iterations[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoints found under ${CKPT_DIR}" >&2
    return 1
  fi
  IFS=$'\n' iterations=($(sort -n <<< "${iterations[*]}"))
  unset IFS

  echo "Found ${#iterations[@]} checkpoints: ${iterations[*]}"

  local prev_hf_dir=""
  local iter_num=""
  for iter_num in "${iterations[@]}"; do
    local iter_name
    iter_name="$(printf 'iter_%07d' "${iter_num}")"
    local hf_dir="${HF_CACHE}/${iter_name}_hf"

    echo ""
    echo "====== Checkpoint ${iter_name} (rollout_id=${iter_num}) ======"

    convert_checkpoint_to_hf "${CKPT_DIR}/${iter_name}" "${hf_dir}"
    start_sglang_eval_server "${hf_dir}"
    run_eval_backfill_for_checkpoint "${hf_dir}" "${iter_num}"
    cleanup_sglang_eval_server

    if [[ -n "${prev_hf_dir}" ]] && [[ -d "${prev_hf_dir}" ]]; then
      echo "  Cleaning up previous HF cache: ${prev_hf_dir}"
      rm -rf "${prev_hf_dir}"
    fi
    prev_hf_dir="${hf_dir}"
  done

  if [[ -n "${prev_hf_dir}" ]] && [[ -d "${prev_hf_dir}" ]]; then
    rm -rf "${prev_hf_dir}"
  fi

  echo ""
  echo "=== Eval backfill complete ==="
}
