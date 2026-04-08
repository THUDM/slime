#!/usr/bin/env bash
# Run student step0 eval and log into an existing wandb run (mode=shared).
#
# Usage:
#   # Default: log into the current MOPD training run
#   bash run_student_step0_eval.sh
#
#   # Override wandb run ID (e.g. dry-test into a scratch run)
#   WANDB_RUN_ID=xxx bash run_student_step0_eval.sh
#
#   # Override student checkpoint (default: v1 iter_479 = MOPD step0)
#   STUDENT_HF_DIR=/path/to/hf bash run_student_step0_eval.sh

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
AVALANCHE_ROOT="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche"
SLIME_DIR="${AVALANCHE_ROOT}/jy_workspace/slime"
EXAMPLES_DIR="${SLIME_DIR}/examples"
MOPD_DIR="${EXAMPLES_DIR}/MOPD"

STUDENT_EXP_DIR="${AVALANCHE_ROOT}/experiments/qwen3_30b_a3b_mdv1_3node_tool45_struct25_stem30_resume0219_0331-1803-fix-cachefix-iter219-cachedatafix-waitfix"
STUDENT_HF_DIR="${STUDENT_HF_DIR:-${STUDENT_EXP_DIR}/hf_cache/iter_0000479_hf}"

MOPD_EXP_DIR="${AVALANCHE_ROOT}/experiments/mopd-3node-h200-liteeval-noeval0-dist-h200-noroutingreplay-retry-0407-0633"
EVAL_CONFIG_PATH="${MOPD_EXP_DIR}/data_cache/eval_config.yaml"
EVAL_DATA_DIR="${MOPD_EXP_DIR}/data_cache/eval"

LOG_DIR="${MOPD_EXP_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Wandb ─────────────────────────────────────────────────────────────
WANDB_HOST="${WANDB_BASE_URL:-https://wandb2.sii.edu.cn}"
WANDB_KEY="${WANDB_API_KEY:-local-c6d3e5712d547834724d8d98094f340b3c2a869c}"
WANDB_PROJECT="slime-mopd"
WANDB_GROUP="mopd-3node-h200-liteeval-noeval0-dist-h200-noroutingreplay-retry-0407-0633"
# The running training run — eval data will be merged into it at step=0
WANDB_RUN_ID="${WANDB_RUN_ID:-0vrz32be}"
WANDB_RUN_NAME="${WANDB_GROUP}-student-step0"

# ── Sglang ────────────────────────────────────────────────────────────
PORT="${STUDENT_STEP0_PORT:-31003}"
TP_SIZE=1
MEM_FRACTION=0.76

# ── Eval params ───────────────────────────────────────────────────────
MAX_CONTEXT_LEN=32768    # 40960 - 8192
MAX_TOKENS=8192
BATCH_SIZE=32

# ── Build --eval-data args from eval_config.yaml ──────────────────────
mapfile -t EVAL_DATA_SPECS < <(python3 - "${EVAL_CONFIG_PATH}" <<'INNERPY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for ds in cfg["eval"]["datasets"]:
    print(f"{ds['name']}:{ds['path']}")
INNERPY
)

if [[ ${#EVAL_DATA_SPECS[@]} -eq 0 ]]; then
    echo "ERROR: No eval datasets found in ${EVAL_CONFIG_PATH}" >&2
    exit 1
fi
echo "Eval datasets (${#EVAL_DATA_SPECS[@]}): ${EVAL_DATA_SPECS[*]}"

# ── Start sglang ──────────────────────────────────────────────────────
echo "Starting sglang for student at ${STUDENT_HF_DIR} on port ${PORT}..."
SGLANG_LOG="${LOG_DIR}/student_step0_sglang.log"
python3 -m sglang.launch_server \
    --model-path "${STUDENT_HF_DIR}" \
    --port "${PORT}" \
    --tp "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --trust-remote-code \
    > "${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!
echo "sglang PID: ${SGLANG_PID}"

cleanup() {
    echo "Stopping sglang (PID ${SGLANG_PID})..."
    kill "${SGLANG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for sglang to be ready
echo "Waiting for sglang on port ${PORT}..."
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "sglang ready (${i}s)"
        break
    fi
    sleep 5
    if [[ $i -eq 120 ]]; then
        echo "ERROR: sglang did not start in time" >&2
        exit 1
    fi
done

# ── Build eval-data args ───────────────────────────────────────────────
EVAL_ARGS=()
for spec in "${EVAL_DATA_SPECS[@]}"; do
    EVAL_ARGS+=(--eval-data "${spec}")
done

# ── Run eval_backfill.py ──────────────────────────────────────────────
echo "Running student step0 eval → wandb run ${WANDB_RUN_ID}..."
PYTHONPATH="${MOPD_DIR}:${EXAMPLES_DIR}:${SLIME_DIR}:${PYTHONPATH:-}" \
    python3 "${SLIME_DIR}/examples/eval_backfill.py" \
        --sglang-url "http://127.0.0.1:${PORT}" \
        --model-path "${STUDENT_HF_DIR}" \
        --reward-module reward_mopd_eval_router.reward_func \
        "${EVAL_ARGS[@]}" \
        --rollout-id 0 \
        --runtime-data-dir "${EVAL_DATA_DIR}" \
        --wandb-run-id "${WANDB_RUN_ID}" \
        --wandb-run-name "${WANDB_RUN_NAME}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        --wandb-host "${WANDB_HOST}" \
        --wandb-key "${WANDB_KEY}" \
        --max-context-len "${MAX_CONTEXT_LEN}" \
        --max-tokens "${MAX_TOKENS}" \
        --batch-size "${BATCH_SIZE}" \
        2>&1 | tee "${LOG_DIR}/student_step0_eval.log"

echo "Done. Check wandb run: https://wandb2.sii.edu.cn/gzy/slime-mopd/runs/${WANDB_RUN_ID}"
