#!/bin/bash
# ============================================================
# Script 3/3: Megatron -> HF Weight Conversion
# MiniMax-M2.5 (229B MoE)
# ============================================================
set -ex

# ---- Paths (modify according to your environment) ----
HF_CKPT=${HF_CKPT:-"/root/MiniMax-M2.5"}
MEGATRON_CKPT=${MEGATRON_CKPT:-"/root/MiniMax-M2.5_slime"}
INPUT_DIR=${INPUT_DIR:-"${MEGATRON_CKPT}/release"}
SAVE_DIR=${SAVE_DIR:-"/root/MiniMax-M2.5_hf_output"}

python tools/convert_torch_dist_to_hf.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${SAVE_DIR} \
    --origin-hf-dir ${HF_CKPT} \
    --vocab-size 200064
