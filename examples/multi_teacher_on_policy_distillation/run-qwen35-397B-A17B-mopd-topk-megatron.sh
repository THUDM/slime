#!/bin/bash

# Multi-Teacher On-Policy Distillation (MOPD) — Top-K KL Divergence Mode
# Model: Qwen3.5-397B-A17B (MoE, 512 experts, 10 active)
# Environment: 16 nodes × 8 L20X (143GB each), 128 GPUs total
# Teacher: Teacher model (different from student for production distillation)
# Mode: Megatron (teacher loaded into CPU memory via TensorBackuper)
# Distill Type: top_k (approximate reverse KL with top-k teacher logits + tail correction)
#
# This script is for MOPD top_k distillation with 128 GPUs.
#
# Key features of top_k mode:
#   --mopd-distill-type top_k
#     → Computes approximate D_KL(π_θ ∥ π_d) using teacher's top-k logits
#       plus tail probability correction. Much more memory-efficient than full_vocab.
#     → Stores only [R_i, k] teacher logits+indices per sample (k=1024 default),
#       vs [R_i, V] for full_vocab. ~98.7% memory reduction vs full_vocab.
#
# Prerequisites:
#   1. Convert HF checkpoint to Megatron torch_dist format before first run:
#      cd /path/to/slime
#      source scripts/models/qwen3.5-397B-A17B.sh
#
#      PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 \
#          tools/convert_hf_to_torch_dist.py \
#          ${MODEL_ARGS[@]} \
#          --hf-checkpoint /path/to/Qwen3.5-397B-A17B_teacher \
#          --save /path/to/Qwen3.5-397B-A17B_teacher_torch_dist
#
# usage: bash examples/multi_teacher_on_policy_distillation/run-qwen35-397B-A17B-mopd-topk-megatron.sh

set -ex

export PYTHONBUFFERED=16
export FLASHINFER_DISABLE_VERSION_CHECK=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SLIME_DIR="/workspace/bin/slime"
source "${SLIME_DIR}/scripts/models/qwen3.5-397B-A17B.sh"

# ============================================================================
# Paths — adjust these to your environment
# ============================================================================
BASE_DIR=/path/to/checkpoints

HF_CKPT=${BASE_DIR}/Qwen3.5-397B-A17B
TORCH_DIST_CKPT=${BASE_DIR}/Qwen3.5-397B-A17B_torch_dist
TEACHER_TORCH_DIST_CKPT=${BASE_DIR}/Qwen3.5-397B-A17B_teacher_torch_dist
SAVE_DIR=${BASE_DIR}/Qwen3.5-397B-A17B-MOPD-TopK-Output

DATA_PATH="/path/to/your/training_data.jsonl"

# MOPD teachers JSON config
export MOPD_TEACHERS_JSON='[{"name":"teacher","domain":"default"}]'

# ============================================================================
# Configure training arguments
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}/
   --ref-load ${TORCH_DIST_CKPT}/
   --load ${SAVE_DIR}/
   --save ${SAVE_DIR}/
   --save-interval 10
   --no-save-optim
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_PATH}
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --rollout-batch-size 64
   --n-samples-per-prompt 1
   --rollout-max-response-len 4096
   --rollout-temperature 0.5

   --global-batch-size 64
   --balance-data
   --num-epoch 1
)

RM_ARGS=()

EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 128
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

MOPD_ARGS=(
   --advantage-estimator grpo

   # MOPD flags — single teacher
   --use-mopd

   # token level
   # --mopd-distill-type token_level

   # top k
   --mopd-distill-type top_k
   --mopd-topk-k 1024

   # full vocab
   # --mopd-distill-type full_vocab

   --mopd-teacher-loads ${TEACHER_TORCH_DIST_CKPT}/

   # MOPD hyperparameters
   --mopd-alpha 0.0                # Pure distillation, no ORM
   --mopd-eps-low 0.2              # IS weight lower bound
   --mopd-eps-high 5.0             # IS weight upper bound
   --mopd-sampling-logprobs-key rollout_log_probs

   # Standard training flags
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7                       # Conservative LR for stability
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # CPU offload optimizer to save GPU memory for large model
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 16
    --sglang-mem-fraction-static 0.45
    --sglang-ep-size 16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --moe-token-dispatcher-type alltoall
   # --moe-enable-deepep  # DeepEP internode kernel assertion fails when EP=128 (num_topk_ranks > kNumTopkRDMARanks)
   --no-check-for-nan-in-loss-and-grad

   --colocate
)

# ============================================================================
# Launch training — multi-node setup
# ============================================================================

# --- Submit job ---
RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '/root/Megatron-LM/',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'NCCL_DEBUG': 'WARN',
    'NCCL_NVLS_ENABLE': os.environ.get('HAS_NVLINK', '0'),
    'NCCL_TIMEOUT_MS': '36000000',
    'FLASHINFER_DISABLE_VERSION_CHECK': '1',
    'MOPD_TEACHERS_JSON': os.environ.get('MOPD_TEACHERS_JSON', '')
}
print(json.dumps({'env_vars': env}))
")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ../workspace/bin/slime/train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 8 \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 4 )) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${MOPD_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

# ============================================================================
# Cleanup
# ============================================================================
pkill -9 sglang
sleep 3
pkill -9 python