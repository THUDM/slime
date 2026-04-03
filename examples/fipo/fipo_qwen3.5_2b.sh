#!/bin/bash
#SBATCH --job-name=fipo-qwen3.5-2b
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:H100:8
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/weka1/private/logan/experiments/fipo-qwen3.5-2b/logs/%j.out
#SBATCH --error=/mnt/weka1/private/logan/experiments/fipo-qwen3.5-2b/logs/%j.err
#
# FIPO (Future-KL Influenced Policy Optimization) training on Qwen3.5-2B-Base.
#
# Key: FIPO requires multi-step training per rollout (global-batch-size < total
# rollout samples) so theta drifts from theta_old, making Future-KL non-trivial.
#
# Reference: Ma et al., arXiv:2603.19835

set -ex

# Create log directory
mkdir -p /mnt/weka1/private/logan/experiments/fipo-qwen3.5-2b/logs

# Activate venv (shared via weka NAS)
source /mnt/weka1/private/logan/repos/slime/.venv/bin/activate

# Install flash-attn on compute node if not already present (needs CUDA)
python3 -c "import flash_attn" 2>/dev/null || pip install flash-attn --no-build-isolation 2>&1 | tail -5 || true

export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SGLANG_DISABLE_CUDNN_CHECK=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$( [ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0 )
export NCCL_NVLS_ENABLE=${HAS_NVLINK}

SLIME_ROOT="/mnt/weka1/private/logan/repos/slime"

export PYTHONPATH="/mnt/weka1/private/logan/repos/Megatron-LM:${SLIME_ROOT}:${PYTHONPATH}"

# Network config for slurm
export SLIME_HOST_IP=$(hostname -I | awk '{print $1}')
export GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | awk '$4 ~ /^10\./ {print $2; exit}')
export NCCL_SOCKET_IFNAME=$(ip -o -4 addr show | awk '$4 ~ /^10\./ {print $2; exit}')

source "${SLIME_ROOT}/scripts/models/qwen3.5-2B.sh"

# --- Kill stale processes ---
pkill -9 sglang 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 2

# --- Start Ray ---
ray start --head --node-ip-address ${SLIME_HOST_IP} --num-gpus 8 --disable-usage-stats

# --- Arguments ---
CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3.5-2B-Base
   --save /mnt/weka1/private/logan/experiments/fipo-qwen3.5-2b/checkpoints
   --save-interval 20
   --rotary-base 10000000
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/weka1/private/logan/data/dapo-math-17k.jsonl
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

   # CRITICAL for FIPO: 512 total / 64 gbs = 8 training steps per rollout
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /mnt/weka1/private/logan/data/aime-2024.jsonl
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

FIPO_ARGS=(
   --loss-type fipo_loss
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   --entropy-coef 0.00

   # FIPO hyperparameters (small-model setting from paper)
   --fipo-decay-rate 32.0
   --fipo-chunk-size 128
   --fipo-clip-ratio 0.2           # Both-side clipping [0.8, 1.2] for small models
   --fipo-safety-thresh 3.0        # Paper: 3.0 for 7B scale
   --fipo-dual-clip-c 10.0
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
   --wandb-key ${WANDB_KEY:-""}
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
)

# --- Launch ---
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/weka1/private/logan/repos/Megatron-LM:${SLIME_ROOT}:${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\"
  }
}"

cd ${SLIME_ROOT}

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${SLIME_ROOT}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
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
