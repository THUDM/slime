#!/usr/bin/env bash
# Convert avalanche v0.1 SFT 1epoch HF checkpoint -> Megatron torch_dist format
# (preparation step before launching the gc2_prod RL training).
#
# Run on a GPU notebook (>=2 GPUs, ~10 min for 35B on 4× H100).
set -euo pipefail

WORKSPACE_ROOT=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jy_workspace
AVALANCHE_ROOT=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche
SLIME_DIR="${WORKSPACE_ROOT}/slime"
MEGATRON_PATH=/root/Megatron-LM

MODEL_NAME="Qwen3.5-35B-A3B-Nemotron-Cascade-2-sft-1epoch"
HF_DIR="${AVALANCHE_ROOT}/models/avalanche_ckpts/v0.1/${MODEL_NAME}"
TORCH_DIST_DIR="${AVALANCHE_ROOT}/models/avalanche_ckpts/v0.1/${MODEL_NAME}_torch_dist"

if [[ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]] \
  || { [[ -f "${TORCH_DIST_DIR}/common.pt" ]] && [[ -f "${TORCH_DIST_DIR}/metadata.json" ]]; }; then
  echo "Found existing torch_dist at ${TORCH_DIST_DIR} — nothing to do."
  exit 0
fi

cd "${SLIME_DIR}"
# shellcheck disable=SC1091
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
[[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]] && { echo "no GPUs detected" >&2; exit 1; }

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') converting ${HF_DIR} -> ${TORCH_DIST_DIR} (${NUM_GPUS} GPU)"

PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}:${PYTHONPATH:-}" \
  python -m torch.distributed.run \
    --nproc-per-node "${NUM_GPUS}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_DIR}" \
    --save "${TORCH_DIST_DIR}"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') conversion done."
ls -la "${TORCH_DIST_DIR}" | head
