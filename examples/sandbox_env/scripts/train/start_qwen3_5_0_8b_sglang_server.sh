#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SANDBOX_ENV_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_EXAMPLES_DIR="$(cd -- "${SANDBOX_ENV_DIR}/.." && pwd)"
SLIME_DIR="$(cd -- "${SLIME_EXAMPLES_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SLIME_DIR}/.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${WORKSPACE_ROOT}/.." && pwd)"

MODEL_NAME="${SGLANG_MODEL_NAME:-Qwen3.5-0.8B}"
MODEL_DIR="${SGLANG_MODEL_DIR:-${AVALANCHE_ROOT}/models/${MODEL_NAME}}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
DP_SIZE="${SGLANG_DP_SIZE:-4}"
MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.70}"
MAX_RUNNING="${SGLANG_MAX_RUNNING_REQUESTS:-512}"
MAX_QUEUED="${SGLANG_MAX_QUEUED_REQUESTS:-4096}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
PYTHON_BIN="${SGLANG_PYTHON_BIN:-python3}"
LOG_DIR="${SGLANG_LOG_DIR:-${SANDBOX_ENV_DIR}/output/sglang_qwen3_5_0_8b}"
LOG_FILE="${LOG_DIR}/server_${HOST}_${PORT}.log"
PID_FILE="${LOG_DIR}/server_${HOST}_${PORT}.pid"

mkdir -p "${LOG_DIR}"

if [[ "${SGLANG_KILL_EXISTING:-0}" == "1" ]]; then
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
fi

if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "SGLang already running: pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
  exit 0
fi

CMD=(
  "${PYTHON_BIN}" -m sglang.launch_server
  --model-path "${MODEL_DIR}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --trust-remote-code
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION}"
  --max-running-requests "${MAX_RUNNING}"
  --max-queued-requests "${MAX_QUEUED}"
)

printf 'Starting SGLang:\n  %q' "${CMD[@]}"
printf '\nlog=%s\n' "${LOG_FILE}"
setsid env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" >"${PID_FILE}"
echo "pid=${pid}"
