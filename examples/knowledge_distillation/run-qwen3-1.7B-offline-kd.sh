#!/bin/bash

# usage: bash examples/knowledge_distillation/run-qwen3-1.7B-offline-kd.sh
#
# Offline knowledge distillation: Load pre-saved teacher data -> Qwen3-1.7B (student)
# Prerequisites: Run online KD with KD_SAVE_PATH first to generate teacher data.

set -ex

MODEL_DIR="${KD_MODEL_DIR:-/workspace}"

# Qwen3-1.7B model architecture
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)

export PYTHONPATH=/root/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
export KD_LOAD_PATH=$MODEL_DIR/kd_teacher_data/rollout_{rollout_id}.jsonl

python3 train.py \
    ${MODEL_ARGS[@]} \
    --debug-train-only \
    --rollout-function-path examples.knowledge_distillation.offline_kd.generate_rollout \
    --loss-type custom_loss \
    --custom-loss-function-path examples.knowledge_distillation.kd_loss.kd_loss_function \
    --calculate-per-token-loss \
    --prompt-data $MODEL_DIR/dapo-math-17k/dapo-math-17k.jsonl \
    --input-key prompt \
    --label-key reward_model \
    --apply-chat-template \
    --rollout-batch-size 28 \
    --n-samples-per-prompt 1 \
    --rollout-max-response-len 1024 \
    --rollout-temperature 0.8 \
    --global-batch-size 28 \
    --num-rollout 100 \
    --train-backend megatron \
    --megatron-to-hf-mode bridge \
    --hf-checkpoint $MODEL_DIR/Qwen3-1.7B \
    --ref-load $MODEL_DIR/Qwen3-1.7B \
    --save $MODEL_DIR/KD-Qwen3-1.7B-offline \
    --save-interval 20 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 7 \
    --colocate \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98
