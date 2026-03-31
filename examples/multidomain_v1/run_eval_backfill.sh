#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

# --- Required inputs ---
EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the experiment path}"
WANDB_RUN_ID="${WANDB_RUN_ID:?Set WANDB_RUN_ID}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-multidomain-v1}"
WANDB_GROUP="${WANDB_GROUP:-}"

# --- Model config ---
MODEL_DIR="${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints/iter_0001432_hf}"
NUM_GPUS="${NUM_GPUS:-8}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_TP="${SGLANG_TP:-4}"
SGLANG_EP="${SGLANG_EP:-2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# --- Derived paths ---
CKPT_DIR="${EXPERIMENT_DIR}/checkpoints"
DATA_CACHE="${EXPERIMENT_DIR}/data_cache"
HF_CACHE="${EXPERIMENT_DIR}/hf_cache"

# --- Eval datasets (skip gpqa which already ran) ---
EVAL_DATASETS=()
for name_file in \
    "bfcl_v3_eval:${DATA_CACHE}/bfcl_v3_eval.normalized.jsonl" \
    "bfcl_multi_turn_eval:${DATA_CACHE}/bfcl_multi_turn_eval.normalized.jsonl" \
    "ifeval_eval:${DATA_CACHE}/ifeval_eval.normalized.jsonl" \
    "jsonschemabench_eval:${DATA_CACHE}/jsonschemabench_eval.normalized.jsonl" \
    "ifbench_test_eval:${DATA_CACHE}/ifbench_test_eval.normalized.jsonl" \
    "mmlu_pro_eval:${DATA_CACHE}/mmlu_pro_eval.normalized.jsonl"; do
  path="${name_file#*:}"
  if [[ -f "${path}" ]]; then
    EVAL_DATASETS+=("${name_file}")
  else
    echo "WARN: Eval file not found, skipping: ${path}"
  fi
done

if [[ ${#EVAL_DATASETS[@]} -eq 0 ]]; then
  echo "ERROR: No eval datasets found in ${DATA_CACHE}" >&2
  exit 1
fi

echo "=== Eval backfill ==="
echo "Experiment: ${EXPERIMENT_DIR}"
echo "wandb run:  ${WANDB_RUN_ID}"
echo "Eval sets:  ${#EVAL_DATASETS[@]}"
for ds in "${EVAL_DATASETS[@]}"; do echo "  ${ds%%:*}"; done

# --- Helpers ---
cleanup_sglang() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep 3
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
}

start_sglang() {
  local hf_dir="$1"
  cleanup_sglang

  echo "Starting sglang server for ${hf_dir} ..."
  python3 -m sglang.launch_server \
    --model-path "${hf_dir}" \
    --port "${SGLANG_PORT}" \
    --tp "${SGLANG_TP}" \
    --ep "${SGLANG_EP}" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    &>"${EXPERIMENT_DIR}/logs/sglang_eval.log" &

  # Wait for server ready
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if curl -sf "http://localhost:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
      echo "sglang server ready."
      return 0
    fi
    sleep 5
  done
  echo "ERROR: sglang did not start in 600s" >&2
  tail -50 "${EXPERIMENT_DIR}/logs/sglang_eval.log"
  return 1
}

convert_checkpoint() {
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

run_eval_for_checkpoint() {
  local hf_dir="$1"
  local rollout_id="$2"

  EVAL_ARGS=()
  for ds in "${EVAL_DATASETS[@]}"; do
    EVAL_ARGS+=(--eval-data "${ds}")
  done

  PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}/slime:${PYTHONPATH:-}" \
  python3 "${SCRIPT_DIR}/eval_backfill.py" \
    --sglang-url "http://localhost:${SGLANG_PORT}" \
    --model-path "${hf_dir}" \
    "${EVAL_ARGS[@]}" \
    --rollout-id "${rollout_id}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-host "${WANDB_BASE_URL:-}" \
    --wandb-key "${WANDB_API_KEY:-}" \
    --wandb-group "${WANDB_GROUP}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}"
}

# --- Main loop: iterate checkpoints ---
mkdir -p "${HF_CACHE}" "${EXPERIMENT_DIR}/logs"

# Discover checkpoint iterations (sorted numerically)
ITERATIONS=()
for iter_dir in "${CKPT_DIR}"/iter_*; do
  [[ -d "${iter_dir}" ]] || continue
  iter_name="$(basename "${iter_dir}")"
  iter_num="${iter_name#iter_}"
  iter_num="$((10#${iter_num}))"  # strip leading zeros
  ITERATIONS+=("${iter_num}")
done

IFS=$'\n' ITERATIONS=($(sort -n <<<"${ITERATIONS[*]}")); unset IFS

echo "Found ${#ITERATIONS[@]} checkpoints: ${ITERATIONS[*]}"

prev_hf_dir=""
for iter_num in "${ITERATIONS[@]}"; do
  iter_name="$(printf 'iter_%07d' "${iter_num}")"
  iter_dir="${CKPT_DIR}/${iter_name}"
  hf_dir="${HF_CACHE}/${iter_name}_hf"

  echo ""
  echo "====== Checkpoint ${iter_name} (rollout_id=${iter_num}) ======"

  # Convert
  convert_checkpoint "${iter_dir}" "${hf_dir}"

  # Start sglang (reuse if same model, but checkpoints differ so must restart)
  start_sglang "${hf_dir}"

  # Run eval
  run_eval_for_checkpoint "${hf_dir}" "${iter_num}"

  # Cleanup sglang
  cleanup_sglang

  # Remove previous HF cache to save disk
  if [[ -n "${prev_hf_dir}" ]] && [[ -d "${prev_hf_dir}" ]]; then
    echo "  Cleaning up previous HF cache: ${prev_hf_dir}"
    rm -rf "${prev_hf_dir}"
  fi
  prev_hf_dir="${hf_dir}"
done

# Clean up last HF cache
if [[ -n "${prev_hf_dir}" ]] && [[ -d "${prev_hf_dir}" ]]; then
  rm -rf "${prev_hf_dir}"
fi

echo ""
echo "=== All eval backfill complete ==="
