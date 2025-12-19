#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TAU_BENCH_OUT_DIR="${TAU_BENCH_OUT_DIR:-${SCRIPT_DIR}/../outputs}"

MODEL_DIR="${MODEL_DIR:-${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-30001}"
GPUS="${GPUS:-2,3}"
TP="${TP:-2}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Missing model directory: ${MODEL_DIR}"
  echo "Download first (example): huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir \"${MODEL_DIR}\""
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp "${TP}" \
  --mem-fraction-static "${MEM_FRACTION}"

