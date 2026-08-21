#!/usr/bin/env bash
# Run the standard synchronous Slime GRPO loop on one GPU.
#
# Required inputs can be overridden with environment variables:
#   HF_CHECKPOINT  Hugging Face Qwen3-0.6B checkpoint
#   REF_LOAD       Converted Megatron torch_dist checkpoint
#   PROMPT_DATA    GSM8K training parquet
#   EVAL_DATA      GSM8K evaluation parquet
#   SAVE_DIR       Output checkpoint directory
#   LOAD_CHECKPOINT Checkpoint root to resume (defaults to SAVE_DIR when complete)
#   MEGATRON_ROOT  Megatron-LM checkout used by Slime
#   NUM_ROLLOUT    Absolute rollout/update target (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./scripts/models/qwen3-0.6B.sh
source "${SCRIPT_DIR}/models/qwen3-0.6B.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-/root/models/Qwen3-0.6B}"
REF_LOAD="${REF_LOAD:-/root/models/Qwen3-0.6B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/root/datasets/gsm8k/train.parquet}"
EVAL_DATA="${EVAL_DATA:-/root/datasets/gsm8k/test.parquet}"
SAVE_DIR="${SAVE_DIR:-/root/checkpoints/qwen3-0.6B-single-gpu-grpo}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

if [[ -z "${LOAD_CHECKPOINT:-}" ]]; then
   if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
      LOAD_CHECKPOINT="$SAVE_DIR"
   else
      LOAD_CHECKPOINT="$REF_LOAD"
   fi
fi

for path in "$HF_CHECKPOINT" "$REF_LOAD" "$LOAD_CHECKPOINT" "$PROMPT_DATA" "$EVAL_DATA" "$MEGATRON_ROOT"; do
   if [[ ! -e "$path" ]]; then
      echo "Missing required path: $path" >&2
      exit 2
   fi
done
mkdir -p "$SAVE_DIR"

if ! ray status >/dev/null 2>&1; then
   ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats
fi

cd "$SLIME_ROOT"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_ROOT}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   --hf-checkpoint "$HF_CHECKPOINT" \
   --ref-load "$REF_LOAD" \
   --load "$LOAD_CHECKPOINT" \
   --save "$SAVE_DIR" \
   --save-interval "$SAVE_INTERVAL" \
   --prompt-data "$PROMPT_DATA" \
   --input-key messages \
   --label-key label \
   --apply-chat-template \
   --rollout-shuffle \
   --rm-type math \
   --num-rollout "$NUM_ROLLOUT" \
   --rollout-batch-size 2 \
   --n-samples-per-prompt 4 \
   --global-batch-size 8 \
   --rollout-max-response-len 512 \
   --rollout-temperature 0.8 \
   --rollout-top-p 0.95 \
   --eval-interval "$NUM_ROLLOUT" \
   --skip-eval-before-train \
   --eval-prompt-data gsm8k "$EVAL_DATA" \
   --n-samples-per-eval-prompt 1 \
   --eval-max-response-len 512 \
   --eval-top-k 1 \
   --advantage-estimator grpo \
   --calculate-per-token-loss \
   --use-kl-loss \
   --kl-loss-coef 0.0 \
   --kl-loss-type low_var_kl \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   --optimizer adam \
   --lr 1e-6 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 4096 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.45 \
   --sglang-cuda-graph-max-bs 8 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   "${MODEL_ARGS[@]}"
