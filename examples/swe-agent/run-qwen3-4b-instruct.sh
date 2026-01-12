#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export SWE_AGENT_GYM_URL="${SWE_AGENT_GYM_URL:-http://swe_env:11000}"

source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
    --hf-checkpoint /root/qwen3-4B-Instruct-2507
    --ref-load /root/qwen3-4B-Instruct-2507_torch_dist
    # --load /path/to/checkpoint/
    --save /root/qwen3-4B-Instruct-2507_slime/
    --save-interval 100
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    
    # --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
)

ROLLOUT_ARGS=(
    --prompt-data /root/swe_train.jsonl
    --input-key prompt
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout 3000
    --rollout-batch-size 8
    --n-samples-per-prompt 8
    --rollout-temperature 0.8
    --rollout-max-response-len 8192
    
    --global-batch-size 64
    --balance-data
)

EVAL_ARGS=(
    # --eval-interval 50
    # --eval-prompt-data /workspace/data/swe_gym_val.jsonl
    # --eval-input-key prompt
    # --eval-metadata-key metadata
    # --n-samples-per-eval-prompt 1
    # --eval-max-response-len 4096
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
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

WANDB_ARGS=()
if [ -n "$WANDB_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-swe-agent
        --wandb-group swe-agent-qwen2.5-3b
        --wandb-key ${WANDB_KEY}
    )
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
    # default dropout in megatron is 0.1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # should be good for model performance
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    # need to comment this when using model with MLA
    --attention-backend flash
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_swe_agent.generate
    --custom-rm-path generate_with_swe_agent.reward_func
    --rollout-function-path generate_with_swe_agent.generate_rollout
    --dynamic-sampling-filter-path generate_with_swe_agent.dynamic_filter
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "Starting Ray cluster at ${MASTER_ADDR}..."
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=8899

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SWE_AGENT_GYM_URL\": \"${SWE_AGENT_GYM_URL}\"
  }
}"
#      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",

echo "Launching training..."
echo "  SWE Agent URL: ${SWE_AGENT_GYM_URL}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}

echo "Training completed!"
