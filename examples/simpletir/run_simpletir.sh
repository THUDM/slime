#!/bin/bash

set -euo pipefail

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/models/qwen2.5-3B.sh"

# Model / checkpoint locations
CKPT_ARGS=(
  --hf-checkpoint /root/Qwen2.5-3B
  --ref-load /root/Qwen2.5-3B_torch_dist
  # --load /root/Qwen2.5-3B_simpletir
  # --save /root/Qwen2.5-3B_simpletir
  # --save-interval 50
)

ROLLOUT_ARGS=(
  --prompt-data /root/datasets/simpletir/deepscaler_train.jsonl
  --input-key prompt
  --label-key label
  --metadata-key extra_info
  --apply-chat-template
  --rollout-shuffle
  --num-rollout 3000
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-prompt-len 8192
  --rollout-max-response-len 8192
  --rollout-temperature 1.0
  --global-batch-size 256
  --balance-data
  --custom-generate-function-path examples.simpletir.generate.custom_generate
  --custom-rm-path examples.simpletir.reward.async_reward
)

EVAL_ARGS=(
  --eval-prompt-data simpletir_eval /root/datasets/simpletir/deepscaler_aime.jsonl
  --n-samples-per-eval-prompt 8
  --eval-max-response-len 16384
  --eval-interval 10
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.001
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
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
  --wandb-project simpletir
  --wandb-group simpletir_qwen2.5-3B
  --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 2
  --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=()

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 4 --disable-usage-stats
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
