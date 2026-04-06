#!/bin/bash
# MOPD eval backfill: run official benchmark evals on existing checkpoints.
# Reuses multidomain_v2/eval_backfill.py with MOPD's reward router.
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime"
V2_DIR="${SLIME_DIR}/examples/multidomain_v2"

EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
WANDB_RUN_ID="${WANDB_RUN_ID:?Set WANDB_RUN_ID}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-mopd}"
WANDB_GROUP="${WANDB_GROUP:-}"

MODEL_DIR="${MODEL_DIR:-}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_TP="${SGLANG_TP:-4}"
SGLANG_EP="${SGLANG_EP:-2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"

CKPT_DIR="${EXPERIMENT_DIR}/checkpoints"
HF_CACHE="${EXPERIMENT_DIR}/hf_cache"

# Locate eval data
DATA_CACHE=""
for candidate in "${EXPERIMENT_DIR}/data_cache" "${EXPERIMENT_DIR}/runtime_data"; do
  if ls "${candidate}"/*.normalized.jsonl >/dev/null 2>&1; then
    DATA_CACHE="${candidate}"
    break
  fi
done
if [[ -z "${DATA_CACHE}" ]]; then
  echo "ERROR: No eval data found in data_cache or runtime_data" >&2
  exit 1
fi
echo "Using eval data from: ${DATA_CACHE}"

# Discover eval datasets
EVAL_DATASETS=()
for f in "${DATA_CACHE}"/*_eval.normalized.jsonl; do
  [[ -f "$f" ]] || continue
  name="$(basename "$f" .normalized.jsonl)"
  EVAL_DATASETS+=("${name}:${f}")
done
if [[ ${#EVAL_DATASETS[@]} -eq 0 ]]; then
  echo "ERROR: No eval datasets found" >&2
  exit 1
fi
echo "Eval sets: ${#EVAL_DATASETS[@]}"
for ds in "${EVAL_DATASETS[@]}"; do echo "  ${ds%%:*}"; done

# Helpers
cleanup_sglang() { pkill -f "sglang.launch_server" 2>/dev/null || true; sleep 2; pkill -9 -f "sglang.launch_server" 2>/dev/null || true; }

start_sglang() {
  local hf_dir="$1"; cleanup_sglang
  echo "Starting sglang for ${hf_dir} ..."
  python3 -m sglang.launch_server --model-path "${hf_dir}" --port "${SGLANG_PORT}" --tp "${SGLANG_TP}" --ep "${SGLANG_EP}" --trust-remote-code --disable-radix-cache --mem-fraction-static 0.85 &>"${EXPERIMENT_DIR}/logs/sglang_eval.log" &
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    curl -sf "http://localhost:${SGLANG_PORT}/v1/models" >/dev/null 2>&1 && { echo "sglang ready."; return 0; }
    sleep 5
  done
  echo "ERROR: sglang did not start" >&2; return 1
}

convert_checkpoint() {
  local iter_dir="$1" hf_out="$2"
  [[ -f "${hf_out}/config.json" ]] && return 0
  echo "  Converting ${iter_dir} -> ${hf_out}"
  python3 "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" --input-dir "${iter_dir}" --output-dir "${hf_out}" --origin-hf-dir "${MODEL_DIR}" --force
}

run_eval() {
  local hf_dir="$1" rollout_id="$2"
  EVAL_ARGS=()
  for ds in "${EVAL_DATASETS[@]}"; do EVAL_ARGS+=(--eval-data "${ds}"); done

  PYTHONPATH="${SCRIPT_DIR}:${SLIME_DIR}/examples:${SLIME_DIR}:${PYTHONPATH:-}" \
  python3 "${V2_DIR}/eval_backfill.py" \
    --sglang-url "http://localhost:${SGLANG_PORT}" \
    --model-path "${hf_dir}" \
    --reward-module reward_mopd_eval_router.reward_func \
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

# Main loop
mkdir -p "${HF_CACHE}" "${EXPERIMENT_DIR}/logs"

ITERATIONS=()
for iter_dir in "${CKPT_DIR}"/iter_*; do
  [[ -d "${iter_dir}" ]] || continue
  iter_num="${iter_dir##*iter_}"; iter_num=$((10#${iter_num}))
  ITERATIONS+=("${iter_num}")
done
IFS=$'\n' ITERATIONS=($(sort -n <<<"${ITERATIONS[*]}")); unset IFS
echo "Found ${#ITERATIONS[@]} checkpoints: ${ITERATIONS[*]}"

prev_hf=""
for iter_num in "${ITERATIONS[@]}"; do
  iter_name="$(printf 'iter_%07d' "${iter_num}")"
  hf_dir="${HF_CACHE}/${iter_name}_hf"
  echo ""; echo "====== ${iter_name} ======"
  convert_checkpoint "${CKPT_DIR}/${iter_name}" "${hf_dir}"
  start_sglang "${hf_dir}"
  run_eval "${hf_dir}" "${iter_num}"
  cleanup_sglang
  [[ -n "${prev_hf}" ]] && [[ -d "${prev_hf}" ]] && rm -rf "${prev_hf}"
  prev_hf="${hf_dir}"
done
[[ -n "${prev_hf}" ]] && [[ -d "${prev_hf}" ]] && rm -rf "${prev_hf}"
echo "=== MOPD eval backfill complete ==="
