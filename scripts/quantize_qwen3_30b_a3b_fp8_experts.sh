#!/bin/bash
#
# Convert Qwen3-30B-A3B BF16 HF weights to FP8-experts checkpoint:
# - quantize MoE expert linears only
# - keep self_attn / LN / embed / lm_head / router in BF16
#
# Usage:
#   bash scripts/quantize_qwen3_30b_a3b_fp8_experts.sh
#   MODEL_DIR=/path/to/Qwen3-30B-A3B SAVE_DIR=/path/to/Qwen3-30B-A3B-FP8-experts \
#     bash scripts/quantize_qwen3_30b_a3b_fp8_experts.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${SLIME_ROOT}"

MODEL_DIR="${MODEL_DIR:-}"
SAVE_DIR="${SAVE_DIR:-}"
MAX_WORKERS="${MAX_WORKERS:-4}"

if [ -z "${MODEL_DIR}" ] || [ -z "${SAVE_DIR}" ]; then
  echo "Set MODEL_DIR and SAVE_DIR, e.g.:" >&2
  echo "  MODEL_DIR=/path/to/Qwen3-30B-A3B SAVE_DIR=/path/to/Qwen3-30B-A3B-FP8-experts \\" >&2
  echo "    bash scripts/quantize_qwen3_30b_a3b_fp8_experts.sh" >&2
  exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Missing MODEL_DIR: ${MODEL_DIR}" >&2
  exit 1
fi

python tools/convert_hf_to_fp8.py \
  --model-dir "${MODEL_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --strategy block \
  --block-size 128 128 \
  --scale-fmt ue8m0 \
  --skip-substrings self_attn \
  --max-workers "${MAX_WORKERS}"

echo "Saved FP8-experts checkpoint to ${SAVE_DIR}"
