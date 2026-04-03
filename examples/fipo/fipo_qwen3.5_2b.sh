#!/bin/bash
#
# FIPO (Future-KL Influenced Policy Optimization) training example.
#
# Key: FIPO requires multi-step training per rollout (global-batch-size < total
# rollout samples) so theta drifts from theta_old, making Future-KL non-trivial.
#
# Usage:
#   1. Convert HF checkpoint first:
#      PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#          ${MODEL_ARGS[@]} --hf-checkpoint Qwen/Qwen3.5-2B-Base \
#          --save /path/to/Qwen3.5-2B-Base_torch_dist --no-rope-fusion
#
#   2. Run training:
#      bash examples/fipo/fipo_qwen3.5_2b.sh
#
# Reference: Ma et al., arXiv:2603.19835

set -ex

# ============================================================
# Paths — adjust these for your environment
# ============================================================
SLIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MEGATRON_ROOT="/path/to/Megatron-LM"
HF_CKPT="Qwen/Qwen3.5-2B-Base"
TORCH_DIST_CKPT="/path/to/Qwen3.5-2B-Base_torch_dist"
SAVE_DIR="/path/to/checkpoints"
TRAIN_DATA="/path/to/dapo-math-17k.jsonl"
EVAL_DATA="/path/to/aime-2024.jsonl"

export PYTHONPATH="${MEGATRON_ROOT}:${SLIME_ROOT}:${PYTHONPATH}"
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================
# Cluster setup
# ============================================================
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$( [ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0 )
export NCCL_NVLS_ENABLE=${HAS_NVLINK}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=${NUM_GPUS:-8}

pkill -9 sglang 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 2
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

# ============================================================
# Model config (Qwen3.5-2B)
# ============================================================
source "${SLIME_ROOT}/scripts/models/qwen3.5-2B.sh"

# ============================================================
# Training arguments
# ============================================================
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --load ${TORCH_DIST_CKPT}
   --save ${SAVE_DIR}
   --save-interval 20
   --rotary-base 10000000
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-max-response-len 10240
   --rollout-temperature 1.0

   # CRITICAL for FIPO: 512 total / 64 gbs = 8 training steps per rollout,
   # allowing theta to drift from theta_old so Future-KL becomes non-zero.
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${EVAL_DATA}
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# FIPO uses GRPO advantage estimator + FIPO loss type
FIPO_ARGS=(
   --loss-type fipo_loss
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   --entropy-coef 0.00

   # FIPO hyperparameters (small-model setting from paper)
   --fipo-decay-rate 32.0          # Half-life for Future-KL exponential decay
   --fipo-chunk-size 128           # Chunk size for memory-efficient computation
   --fipo-clip-ratio 0.2           # Both-side clipping [0.8, 1.2] for small models
   --fipo-safety-thresh 3.0        # Paper: 3.0 for 7B scale
   --fipo-dual-clip-c 10.0         # Dual-clip threshold
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-fipo
   --wandb-group qwen3.5-2B-fipo
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

# ============================================================
# Launch
# ============================================================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SLIME_ROOT}:${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

cd ${SLIME_ROOT}

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${FIPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
