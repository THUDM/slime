#!/bin/bash
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
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

# ============================================================================
# Paths — adjust these to your environment
# ============================================================================
BASE_DIR=/path/to/checkpoints

HF_CKPT=${BASE_DIR}/Qwen3.5-35B-A3B
TORCH_DIST_CKPT=${BASE_DIR}/Qwen3.5-35B-A3B-Torch-Dist-Bridge
SAVE_DIR=${BASE_DIR}/Qwen3.5-35B-A3B-Mopd-Test

# ============================================================================
# Dataset — configure your data path
# ============================================================================
# Use a local JSONL file with multimodal data
DATA_PATH="/path/to/your/multimodal_training_data.jsonl"

# Multimodal keys — passed as env var to avoid shell quoting issues with JSON
export MULTIMODAL_KEYS='{"image": "images"}'

# ============================================================================
# MOPD teachers — adjust URLs to your deployment
# ============================================================================
export MOPD_TEACHERS_JSON='[{"name":"enhanced","domain":"enhanced"},{"name":"origin","domain":"origin"}]'

# TODO: Replace with actual teacher server URLs
ENHANCED_TEACHER_IP="your-enhanced-teacher-host"
ENHANCED_TEACHER_PORT=8300
ORIGIN_TEACHER_IP="your-origin-teacher-host"
ORIGIN_TEACHER_PORT=8300

export MOPD_TEACHER_URLS="{\"enhanced\":\"https://${ENHANCED_TEACHER_IP}:${ENHANCED_TEACHER_PORT}/generate\",\"origin\":\"https://${ORIGIN_TEACHER_IP}:${ORIGIN_TEACHER_PORT}/generate\"}"

# ============================================================================
# Configure training arguments
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}/
   --load ${TORCH_DIST_CKPT}/
   --save ${SAVE_DIR}/
   --save-interval 10
   --no-save-optim
)

ROLLOUT_ARGS=(
   # --prompt-data, --multimodal-keys
   # are passed via env vars to avoid shell quoting issues with JSON in ray job submit.
   # See MULTIMODAL_KEYS above.
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-prompt-len 9216
   --rollout-max-response-len 2048
   --rollout-temperature 0.8

   --global-batch-size 16
   --balance-data
   --num-epoch 1
)

# Multimodal — dataset contains images
ROLLOUT_ARGS+=(
   --processor ${HF_CKPT}/
)

# RM_URL points to the enhanced teacher (used as default when no domain routing)
RM_ARGS=(
    --rm-url https://${ENHANCED_TEACHER_IP}:${ENHANCED_TEACHER_PORT}/generate
)

EVAL_ARGS=()

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

   # --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

MOPD_ARGS=(
   --advantage-estimator grpo

   # MOPD flags — dual teacher
   --use-mopd

   # SGLang teacher mode — teachers run on external SGLang servers
   --mopd-teacher-mode sglang

   # top_k distillation type
   --mopd-distill-type top_k
   --mopd-topk-k 16

   # No --mopd-teacher-loads in SGLang mode!
   # Teacher data comes from SGLang server via HTTP during rollout.

   # MOPD hyperparameters
   --mopd-alpha 0.0                # Pure distillation, no ORM
   --mopd-eps-low 0.2              # IS weight lower bound
   --mopd-eps-high 5.0             # IS weight upper bound
   --mopd-sampling-logprobs-key rollout_log_probs
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
    --rollout-num-gpus-per-engine 8
    --sglang-mem-fraction-static 0.25
    --sglang-ep-size 8
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
   --log-probs-chunk-size 1024
   --qkv-format bshd
   --micro-batch-size 1
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
    'NCCL_TIMEOUT_MS': '72000000',
    'FLASHINFER_DISABLE_VERSION_CHECK': '1',
    'MAX_PIXELS': '1048576',
    'MOPD_TEACHER_URLS': os.environ.get('MOPD_TEACHER_URLS', ''),
    'MOPD_TEACHERS_JSON': os.environ.get('MOPD_TEACHERS_JSON', ''),
    'MULTIMODAL_KEYS': os.environ.get('MULTIMODAL_KEYS', ''),
}
print(json.dumps({'env_vars': env}))
")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ../workspace/bin/slime/train.py \
   --actor-num-nodes 1 \
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