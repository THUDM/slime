#!/bin/bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

BASE_DIR=${BASE_DIR:-/root}
MODEL_NAME=${MODEL_NAME:-gemma-4-31B-it}
MODEL_DIR=${MODEL_DIR:-${BASE_DIR}/${MODEL_NAME}}
GSM8K_DIR=${GSM8K_DIR:-${BASE_DIR}/datasets/gsm8k}
NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-4}
CP_SIZE=${CP_SIZE:-1}
TORCH_DIST_CKPT=${TORCH_DIST_CKPT:-${BASE_DIR}/${MODEL_NAME}_tp${TP_SIZE}_pp${PP_SIZE}_cp${CP_SIZE}_torch_dist}
SLIME_CKPT=${SLIME_CKPT:-${BASE_DIR}/${MODEL_NAME}_tp${TP_SIZE}_pp${PP_SIZE}_cp${CP_SIZE}_slime}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gemma4-31B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${TORCH_DIST_CKPT}"
   --load "${SLIME_CKPT}"
   --save "${SLIME_CKPT}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${GSM8K_DIR}/train.parquet"
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT:-2}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-512}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.8}"
   --rollout-top-p "${ROLLOUT_TOP_P:-1.0}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
   --num-steps-per-rollout 1
   --balance-data
)

EVAL_ARGS=()
if [ "${ENABLE_EVAL:-0}" = "1" ]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL:-20}"
      --eval-prompt-data gsm8k "${GSM8K_DIR}/test.parquet"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-512}"
      --eval-top-p 1
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-2048}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef "${ENTROPY_COEF:-0.001}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [ "${USE_WANDB:-0}" = "1" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-gemma4-gsm8k}"
      --wandb-group "${WANDB_GROUP:-gemma4-31B-gsm8k}"
   )
   if [ -n "${WANDB_KEY:-}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE:-8}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.20}"
   --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-1}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-4}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --loss-mask-type gemma4
   --megatron-to-hf-mode raw
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
