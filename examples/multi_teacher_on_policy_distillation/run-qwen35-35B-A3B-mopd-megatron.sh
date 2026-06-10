#!/bin/bash

# Multi-Teacher On-Policy Distillation (MOPD) — Single Teacher Connectivity Test
# Model: Qwen3.5-35B-A3B (MoE, 256 experts, 8 active)
# Environment: 8× H20 (143GB)
# Teacher: Same as student (self-distillation for connectivity validation only)
# Mode: Megatron (teacher loaded into CPU memory via TensorBackuper)
#
# This script is for MOPD E2E connectivity validation only.
# In production, use a DIFFERENT (stronger) model as teacher.
#
# Parallelism: TP=2, EP=8 (matches SFT config, 256 experts / 8 = 32 per GPU)
# Colocate mode: rollout and training share all 8 GPUs with offloading
#
# usage: bash examples/multi_teacher_on_policy_distillation/run-qwen3.5-35B-A3B-mopd-megatron.sh

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

source "/mntfn/yanyi/code/slime/scripts/models/qwen3.5-35B-A3B.sh"

# MOPD teachers JSON config
# Set as environment variable; arguments.py reads $MOPD_TEACHERS_JSON
# when --mopd-teachers is not provided on the command line.
# This avoids shell quoting issues when passing JSON through ray job submit.
export MOPD_TEACHERS_JSON='[{"name":"self_teacher","domain":"default"}]'

# ============================================================================
# Configure training arguments
# ============================================================================

# IMPORTANT: Before running this script, convert the HF checkpoint to Megatron
# torch_dist format:
#
#   cd /mntfn/yanyi/code/slime
#   source scripts/models/qwen3.5-35B-A3B.sh
#
#   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#       ${MODEL_ARGS[@]} \
#       --hf-checkpoint /mnt4/data/open_source/Qwen3.5-35B-A3B \
#       --save /mnt4/data/open_source/Qwen3.5-35B-A3B_torch_dist

CKPT_ARGS=(
   --hf-checkpoint /mnt4/data/open_source/Qwen3.5-35B-A3B/
   --ref-load /mnt4/data/open_source/Qwen3.5-35B-A3B_torch_dist/
   --load /mnt4/data/zhixiaobao/yanyi/Qwen3.5-35B-A3B-mopd-test/
   --save /mnt4/data/zhixiaobao/yanyi/Qwen3.5-35B-A3B-mopd-test/
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data /mntfn/yanyi/dataset/train_text_user_only.jsonl
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --rollout-batch-size 16
   --n-samples-per-prompt 1         # No need for multiple samples in pure distillation
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 16
   --balance-data
   --num-epoch 1
)

RM_ARGS=(
    # Pure distillation (mopd-alpha=0): rm-type defaults to "zero" automatically.
    # No reward model needed.
)

EVAL_ARGS=(
   # No eval for connectivity test
)

# Qwen3.5-35B-A3B with 8 GPUs (same parallelism as SFT config):
#   TP=2, EP=8 (256 experts / 8 = 32 experts per GPU)
#   Colocate mode: rollout and training share all 8 GPUs with offloading
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
   --max-tokens-per-gpu 8192
)

# MOPD Configuration (Megatron mode, single teacher)
# For this connectivity test, the teacher IS the same model (self-distillation).
# This validates the full MOPD pipeline: rollout → teacher log-prob → advantage → train.
#
# Key: The teacher checkpoint must be in Megatron torch_dist format.
# Since teacher = student here, we use the same torch_dist path.
#
# Memory note: The teacher model weights are backed up to CPU memory via
# TensorBackuper. For Qwen3.5-35B-A3B, expect ~70GB additional CPU RAM usage.
MOPD_ARGS=(
   --advantage-estimator grpo

   # MOPD flags — single teacher
   --use-mopd
   # Pass JSON via env var MOPD_TEACHERS_JSON to avoid shell quoting issues with ray job submit.
   # If --mopd-teachers is not set, arguments.py falls back to $MOPD_TEACHERS_JSON.

   # Teacher checkpoint = same as ref model (self-distillation for validation)
   --mopd-teacher-loads /mnt4/data/open_source/Qwen3.5-35B-A3B_torch_dist/

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
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-35B-mopd-megatron
   # --wandb-key ${WANDB_KEY}
)

# SGLang rollout config (colocate mode, shares training GPUs)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8    # All 8 GPUs for rollout
   --sglang-mem-fraction-static 0.4   # Share GPU memory with training
   --sglang-ep-size 8                 # Match EP=8 for MoE expert parallelism
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # MoE communication
   --moe-token-dispatcher-type flex
   --moe-enable-deepep

   # Colocate: rollout and training share same GPUs, with offloading
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