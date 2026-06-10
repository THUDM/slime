#!/bin/bash

# Multi-Teacher On-Policy Distillation (MOPD) — Top-K KL Divergence, SGLang Non-colocate Mode
# Model: Qwen3.5-397B-A17B (MoE, 512 experts, 10 active)
# Environment: 36 nodes × 8 GPUs (143GB each), 288 GPUs total
#   - 32 nodes (256 GPUs) for Megatron actor training (non-colocate)
#   -  4 nodes ( 32 GPUs) for SGLang rollout (2 engines × 16 GPUs each)
# Teacher: Teacher model (running on external SGLang servers)
# Mode: SGLang non-colocate (actor training and rollout on separate GPU groups)
# Distill Type: top_k (approximate reverse KL with top-k teacher logits + tail correction)
#
# Non-colocate mode separates actor training GPUs from SGLang rollout GPUs,
# avoiding GPU memory contention and allowing larger actor parallelism.
# and its top-k logprobs are collected during rollout via HTTP requests.
#
# Key differences from Megatron top_k mode:
#   - No --mopd-teacher-loads (no Megatron checkpoint needed for teacher)
#   - No --enable-weights-backuper needed for teacher
#   - Teacher can have a DIFFERENT architecture than student
#   - custom-rm-path and custom-reward-post-process-path are auto-configured
#   - MOPD_TEACHER_URLS env var specifies the SGLang teacher server endpoints
#
# Prerequisites:
#   1. Start the SGLang teacher server(s) before running this script.
#      Example for a single 397B MoE teacher on 16 GPUs:
#
#      python3 -m sglang.launch_server \
#          --model-path /personal/ckpt/Qwen3.5-397B-A17B_skin_multiturn/ \
#          --host 0.0.0.0 --port 13141 \
#          --tp 8 --ep-size 16 \
#          --chunked-prefill-size 4096 \
#          --mem-fraction-static 0.7
#
#   2. Convert student HF checkpoint to Megatron torch_dist format:
#      cd /path/to/slime
#      source scripts/models/qwen3.5-397B-A17B.sh
#
#      PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 \
#          tools/convert_hf_to_torch_dist.py \
#          ${MODEL_ARGS[@]} \
#          --hf-checkpoint /personal/ckpt/Qwen3.5-397B-A17B_Swift_SFT_Stage3b_Text1p5 \
#          --save /personal/ckpt/Qwen3.5-397B-A17B_Swift_SFT_Stage3b_Text1p5_torch_dist
#
# usage: bash examples/multi_teacher_on_policy_distillation/run-qwen35-397B-A17B-mopd-topk-sglang.sh

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
BASE_DIR=/personal/ckpt

HF_CKPT=${BASE_DIR}/Qwen3.5-397B-A17B_Swift_SFT_Stage3b_Text1p5
TORCH_DIST_CKPT=${BASE_DIR}/Qwen3.5-397B-A17B_Swift_SFT_Stage3b_Text1p5_torch_dist
SAVE_DIR=/amed/share/s1-amed-spfs-ckpt/yanyi/Qwen3.5-397B-A17B-Stage3b-Mopd-Topk-Skin-Multiturn-Enhanced

DATA_PATH="/mnt/amed-s3/dataset/14019ba0_text_report_Interpretation/a3967912440becb0d70748a478696f12b6bbf6ac/train_text_think_nothink.jsonl"

# MOPD teachers JSON config (single teacher for this example)
export MOPD_TEACHERS_JSON='[{"name":"skin-multiturn","domain":"default"}]'

# MOPD teacher SGLang server URLs
# For multi-teacher, add all domains: {"math":"https://...","code":"https://..."}
TEACHER_IP="aistudio.alipay.com/proxy/rayjob/aistudio-dvm9s0jw-tfjob-master-0"
TEACHER_PORT=8300
export MOPD_TEACHER_URLS="{\"default\":\"https://$TEACHER_IP:$TEACHER_PORT/generate\"}"

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
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 64
   --balance-data
   --num-epoch 1
)

# No RM_ARGS needed for pure distillation (alpha=0).
# custom-rm-path and custom-reward-post-process-path are auto-configured
# by the MOPD SGLang mode argument validation.
RM_ARGS=(
    --rm-url https://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 16
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
   --train-memory-margin-bytes 268435456
)

MOPD_ARGS=(
   --advantage-estimator grpo

   # MOPD flags — single teacher
   --use-mopd

   # SGLang teacher mode — teacher runs on external SGLang servers
   --mopd-teacher-mode sglang

   # top_k distillation type
   --mopd-distill-type top_k
   --mopd-topk-k 128

   # No --mopd-teacher-loads in SGLang mode!
   # Teacher data comes from SGLang server via HTTP during rollout.

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
   # --moe-enable-deepep  # DeepEP internode kernel assertion fails when EP=128
   --no-check-for-nan-in-loss-and-grad

   --recompute-loss-function
   --log-probs-chunk-size 512
   --qkv-format bshd
   --micro-batch-size 1
   # Non-colocate mode: actor training and rollout on separate GPU groups
   # Remove --colocate to use non-colocate mode
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
    'MOPD_TEACHER_URLS': os.environ.get('MOPD_TEACHER_URLS', ''),
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