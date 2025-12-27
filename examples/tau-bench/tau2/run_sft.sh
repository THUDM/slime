#!/bin/bash

# Tau2 SFT training (tau2-bench).

set -euo pipefail

if [ "${TAU2_CLEANUP:-0}" = "1" ]; then
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
fi

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
TAU_BENCH_OUT_DIR="${TAU_BENCH_OUT_DIR:-${SCRIPT_DIR}/../outputs}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/root/Megatron-LM}"

HF_DIR="${HF_DIR:-${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507_torch_dist}"
TAU2_SFT_DATA_DIR="${TAU2_SFT_DATA_DIR:-${TAU_BENCH_OUT_DIR}/tau2/data/sft1}"
SFT_DATA_JSONL="${SFT_DATA_JSONL:-${TAU2_SFT_DATA_DIR}/tau2_sft_merged_v3_rft.jsonl}"
if [ ! -f "${SFT_DATA_JSONL}" ] && [ -f "${TAU2_SFT_DATA_DIR}/seed_sft_v3.jsonl" ]; then
  SFT_DATA_JSONL="${TAU2_SFT_DATA_DIR}/seed_sft_v3.jsonl"
fi
SAVE_DIR="${SAVE_DIR:-${TAU_BENCH_OUT_DIR}/tau2/checkpoints/Qwen3-4B-tau2-sft1}"

NUM_GPUS="${NUM_GPUS:-4}"

source "${SCRIPT_DIR}/../../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_DIR}"
  --load "${TORCH_DIST_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 50
)

SFT_ARGS=(
  --prompt-data "${SFT_DATA_JSONL}"
  --input-key prompt
  --apply-chat-template
  --loss-mask-type qwen3
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout

  --num-epoch 2
  --rollout-batch-size 16
  --n-samples-per-prompt 1
  --rollout-shuffle
  --rollout-max-response-len 4096
  --global-batch-size 16
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 12288

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style cosine
  --lr-warmup-fraction 0.05
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.98
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "tau2-cookbook"
    --wandb-group "sft-qwen3-4b"
    --wandb-exp-name "tau2-sft-v1"
  )
else
  echo "NOTE: WANDB_API_KEY not set, running without WandB logging"
  WANDB_ARGS=()
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host="${RAY_DASHBOARD_HOST}" --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${REPO_ROOT}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR}:${SCRIPT_DIR}:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --debug-train-only \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${WANDB_ARGS[@]}
