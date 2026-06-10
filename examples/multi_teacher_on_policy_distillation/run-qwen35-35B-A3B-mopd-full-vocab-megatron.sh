#!/bin/bash

# Multi-Teacher On-Policy Distillation (MOPD) — Full-Vocabulary KL Divergence Mode
# Model: Qwen3.5-35B-A3B (MoE, 256 experts, 8 active)
# Environment: 8× H20 (143GB)
# Teacher: Same as student (self-distillation for connectivity validation only)
# Mode: Megatron (teacher loaded into CPU memory via TensorBackuper)
# Distill Type: full_vocab (exact full-vocabulary reverse KL D_KL(π_θ ∥ π_d))
#
# This script is for MOPD full_vocab E2E connectivity validation.
# In production, use a DIFFERENT (stronger) model as teacher.
#
# Key difference from token_level mode:
#   --mopd-distill-type full_vocab
#     → Computes exact D_KL(π_θ ∥ π_d) over full vocabulary instead of
#       approximating from sampled tokens. Requires megatron teacher mode.
#     → Uses full logits [R, V] instead of per-token log-probs, which
#       increases memory usage significantly.
#
# Parallelism: TP=2, EP=8 (matches SFT config, 256 experts / 8 = 32 per GPU)
# Colocate mode: rollout and training share all 8 GPUs with offloading
#
# usage: bash examples/multi_teacher_on_policy_distillation/run-qwen35-35B-A3B-mopd-full-vocab-megatron.sh

# ============================================================================
# Cleanup: kill existing SGLang / Ray / Python processes
# ============================================================================
pkill -9 sglang
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

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

source "/path/to/slime/scripts/models/qwen3.5-35B-A3B.sh"

# MOPD teachers JSON config
export MOPD_TEACHERS_JSON='[{"name":"self_teacher","domain":"default"}]'

# ============================================================================
# Configure training arguments
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint /path/to/checkpoints/Qwen3.5-35B-A3B/
   --ref-load /path/to/checkpoints/Qwen3.5-35B-A3B_torch_dist/
   --load /path/to/output/Qwen3.5-35B-A3B-mopd-full-vocab-test/
   --save /path/to/output/Qwen3.5-35B-A3B-mopd-full-vocab-test/
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data /path/to/dataset/train_text_user_only.jsonl
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 4
   --balance-data
   --num-epoch 1
)

RM_ARGS=(
    # Pure distillation (mopd-alpha=0): rm-type defaults to "zero" automatically.
)

EVAL_ARGS=(
   # No eval for connectivity test
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

# MOPD Configuration — Full-Vocabulary KL Divergence Mode
#
# Key changes from token_level mode:
#   1. Added --mopd-distill-type full_vocab
#      → Computes exact D_KL(π_θ ∥ π_d) = Σ_y π_θ(y)[log π_θ(y) - log π_d(y)]
#      over the full vocabulary instead of token-level approximation.
#
#   2. --mopd-teacher-loads is REQUIRED for full_vocab mode
#      → full_vocab needs megatron teacher forward pass to get full logits,
#        SGLang rollout cannot provide per-token full-vocab logits.
#
#   3. Memory considerations:
#      → full_vocab mode stores teacher logits [R_i, V_local] per sample per teacher.
#        For V=248320, TP=2 → V_local=124160, each token's logits = ~480KB in fp32.
#        With batch=4, R=4096: teacher logits per GPU ≈ 4×4096×124160×4B ≈ 7.6GB.
#        Student logits (same shape) appear during training forward pass ≈ 1.9GB/micro-batch.
#        Together with model (~9GB), optimizer (~26GB), and SGLang (40%=57GB),
#        total ≈ 102GB / 143GB, leaving ~41GB headroom.
#        If OOM: reduce rollout-batch-size, rollout-max-response-len, or sglang-mem-fraction-static.
#
#   4. Loss formula:
#      → L = L_fv_kl + alpha * L_pg (when alpha > 0)
#      → L = L_fv_kl (pure distillation, when alpha = 0)
#      where L_fv_kl = (1/D) Σ_d w_d * D_KL(π_θ ∥ π_d) (IS-corrected)
#
#   5. IS weight correction still applies (same as token_level mode)
#
# Alternative: Use top_k mode for memory-efficient approximate KL:
#   Replace --mopd-distill-type full_vocab with:
#     --mopd-distill-type top_k
#     --mopd-topk-k 1024
#   This stores only [R_i, k] teacher logits+indices per sample (k=1024 by default),
#   plus a tail probability correction. Memory per sample ≈ k*5B per token
#   (vs V*4B for full_vocab). For k=1024, V=248320: ~98.7% memory reduction.
#   Teacher logits per GPU ≈ 4×4096×1024×(4+4)B ≈ 128MB (negligible vs full_vocab).
#
# For this connectivity test, the teacher IS the same model (self-distillation).
MOPD_ARGS=(
   --advantage-estimator grpo

   # MOPD flags — single teacher
   --use-mopd
   # Pass JSON via env var MOPD_TEACHERS_JSON to avoid shell quoting issues.

   # *** KEY DIFFERENCE: full_vocab distillation type ***
   --mopd-distill-type full_vocab

   # Teacher checkpoint = same as ref model (self-distillation for validation)
   --mopd-teacher-loads /path/to/checkpoints/Qwen3.5-35B-A3B_torch_dist/

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
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-35B-mopd-full-vocab-megatron
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.4
   --sglang-ep-size 8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --moe-token-dispatcher-type flex
   --moe-enable-deepep

   --colocate
)

# ============================================================================
# Launch training
# ============================================================================

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '/root/Megatron-LM/',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'NCCL_NVLS_ENABLE': os.environ.get('HAS_NVLINK', '0'),
    'MOPD_TEACHERS_JSON': os.environ.get('MOPD_TEACHERS_JSON', '')
}
print(json.dumps({'env_vars': env}))
")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
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
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python