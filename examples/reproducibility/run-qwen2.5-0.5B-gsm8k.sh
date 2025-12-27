#!/bin/bash
# run_qwen2p5_0p5b_gsm8k_cp_bench_4gpus.sh
#
# Usage:
#   bash run_qwen2p5_0p5b_gsm8k_cp_bench_4gpus.sh 1   # CP=1 baseline
#   bash run_qwen2p5_0p5b_gsm8k_cp_bench_4gpus.sh 2   # CP=2
#
# Assumptions:
# - 4 GPUs visible in the machine/container
# - Model + data paths exist:
#     /root/Qwen2.5-0.5B-Instruct/
#     /root/Qwen2.5-0.5B-Instruct_torch_dist/
#     /root/gsm8k/train.parquet
#     /root/gsm8k/test.parquet
#
# Notes:
# - Deterministic mode: sglang deterministic inference + Megatron deterministic-mode
# - Rollout uses 2 GPUs (tp=2 in sglang via rollout-num-gpus-per-engine=2)

set -euo pipefail
set -x

CP_SIZE="${1:-1}"
if [[ "${CP_SIZE}" != "1" && "${CP_SIZE}" != "2" ]]; then
  echo "CP_SIZE must be 1 or 2"
  exit 1
fi

# --- hard reset for rerun ---
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2

export PYTHONBUFFERED=16

# Use all 4 GPUs, but we will pin training/rollout via ray resource settings.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# qwen2.5-0.5B model spec args
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-0.5B.sh"

# --- paths ---
MODEL="/root/Qwen2.5-0.5B-Instruct"
REF_TORCH_DIST="/root/Qwen2.5-0.5B-Instruct_torch_dist"
TRAIN_DATA="/root/gsm8k/train.parquet"
EVAL_DATA="/root/gsm8k/test.parquet"

RUN_TAG="qwen2.5-0.5B-gsm8k-cp${CP_SIZE}-4gpus"
SAVE_DIR="/root/slime_runs/${RUN_TAG}"
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL}"
  --ref-load "${REF_TORCH_DIST}"
  # no saving for speed benchmark; uncomment if needed:
  # --save "${SAVE_DIR}"
  # --save-interval 50
)

# Keep rollout small for speed benchmarking.
# IMPORTANT: satisfy slime_validate_args: rollout_batch_size * n_samples_per_prompt must be multiple of global_batch_size.
# Here: 4 * 1 % 4 == 0 ✅
ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key messages
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type math

  --num-rollout 40
  --rollout-batch-size 4
  --n-samples-per-prompt 1
  --rollout-max-response-len 2048
  --rollout-temperature 0.8

  --global-batch-size 4
)

EVAL_ARGS=(
  --eval-interval 50
  --eval-prompt-data gsm8k "${EVAL_DATA}"
  --n-samples-per-eval-prompt 1
  --eval-max-response-len 512
  --eval-top-k 1
)

# Training side:
# - actor uses 2 GPUs
# - CP=1 or CP=2
PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size "${CP_SIZE}"
  # sequence-parallel requires TP>1 usually; Megatron will disable automatically if TP=1
  --sequence-parallel

  --use-dynamic-batch-size
  --max-tokens-per-gpu 8092
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

# Rollout side:
# We allocate 2 GPUs for rollout and use 1 engine with TP=2 (rollout-num-gpus-per-engine=2).
SGLANG_ARGS=(
  --rollout-num-gpus 2
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7

  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer
)

MISC_ARGS=(
  --calculate-per-token-loss
  --use-slime-router

  # deterministic on Megatron side
  --deterministic-mode

  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# --- start ray (exactly 4 GPUs) ---
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus 4 \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

# Runtime env: deterministic knobs
# NOTE: keep these as strings (Ray env_vars are strings).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_ALGO\": \"Ring\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
    \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Submit job:
# - actor-num-gpus-per-node = 2  (training uses 2 GPUs)
# - rollout-num-gpus = 2         (rollout uses 2 GPUs)
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- \
  python3 train.py \
    --train-backend megatron \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --rollout-num-gpus 2 \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]}
