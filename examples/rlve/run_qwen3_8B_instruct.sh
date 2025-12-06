#!/bin/bash
# RLVE training for Qwen3-8B-Instruct on 2xH100 (TP=2)

pkill -9 sglang; sleep 3
ray stop --force; pkill -9 ray; pkill -9 python; sleep 3
pkill -9 ray; pkill -9 python

set -ex
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="${HOME:-/root}"

export RLVE_CONFIG_PATH="${SCRIPT_DIR}/configs/starter_pack.yaml"

# Generate dummy dataset (slime needs --prompt-data; RLVE replaces prompts on-the-fly)
DUMMY_PATH="${SCRIPT_DIR}/data/dummy_indices.jsonl"
mkdir -p "${SCRIPT_DIR}/data"
: > "${DUMMY_PATH}"
for i in $(seq 0 99); do echo "{\"index\": ${i}}" >> "${DUMMY_PATH}"; done

source "${SCRIPT_DIR}/../../scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${ROOT}/Qwen3-8B-Instruct/
   --ref-load ${ROOT}/Qwen3-8B-Instruct_torch_dist/
   --save ${ROOT}/Qwen3-8B-Instruct_rlve/
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data ${SCRIPT_DIR}/data/dummy_indices.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 1500
   --rollout-batch-size 24
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 0.8
   --global-batch-size 96
   --balance-data
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
   --max-tokens-per-gpu 6144
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-rlve
   # --wandb-group qwen3-8B-instruct
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.65
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.rlve.rlve_generate.generate
   --custom-rm-path examples.rlve.rlve_reward.reward_func
   --reward-key accuracy
)

# Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=2
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265 --temp-dir ${ROOT}/shared/ray_temp

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROOT}/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RLVE_CONFIG_PATH\": \"${RLVE_CONFIG_PATH}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
