#!/bin/bash
# ============================================================
# Script 1/3: HF -> Megatron Weight Conversion
# MiniMax-M2.5 (229B MoE)
# ============================================================
set -ex

# ---- Paths (modify according to your environment) ----
HF_CKPT=${HF_CKPT:-"/root/MiniMax-M2.5"}
SAVE_DIR=${SAVE_DIR:-"/root/MiniMax-M2.5_torch_dist"}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/minimax-m2.sh"

# ---- Parallelism config (adjust based on available GPUs) ----
TP=${TP:-2}
PP=${PP:-2}
EP=${EP:-4}
WORLD_SIZE=${WORLD_SIZE:-$((TP * PP * EP))}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$((WORLD_SIZE / NNODES))}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

if (( NPROC_PER_NODE * NNODES != WORLD_SIZE )); then
    echo "NPROC_PER_NODE * NNODES must equal WORLD_SIZE (${WORLD_SIZE})." >&2
    exit 1
fi

# WORLD_SIZE must match the requested Megatron parallel layout.
torchrun \
    --nproc-per-node ${NPROC_PER_NODE} \
    --nnodes ${NNODES} \
    --node-rank ${NODE_RANK} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${HF_CKPT} \
    --save ${SAVE_DIR} \
    --megatron-to-hf-mode raw \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1
