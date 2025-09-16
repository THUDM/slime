#!/bin/bash

# hf download meta-llama/Llama-3.2-3B-Instruct --local-dir /root/Llama-3.2-3B-Instruct

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -euxo pipefail

### AMD Support ###
SLIME_DIR="/home/yushensu/projects/slime" # Need to change to your own path
export SLIME_DIR=$SLIME_DIR

MODEL_DIR="/home/yushensu/projects/model" # Need to change to your own path
export MODEL_DIR=$MODEL_DIR

DATA_DIR="/home/yushensu/projects/data"  # Need to change to your own path
export DATA_DIR=$DATA_DIR

# For AMD GPU
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"} # Must set to 1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"} #You can choose which gpus to use
####################

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
# if [ "$NVLINK_COUNT" -gt 0 ]; then
#     HAS_NVLINK=1
# else
#     HAS_NVLINK=0
# fi
# echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/llama3.2-3B-Instruct-amd.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Llama-3.2-3B-Instruct
   --ref-load ${MODEL_DIR}/Llama-3.2-3B-Instruct_torch_dist
   --load ${MODEL_DIR}/Llama-3.2-3B-Instruct_slime/
   --save ${MODEL_DIR}/Llama-3.2-3B-Instruct_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-epoch 1
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group llama3.2-3B
   # --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.4
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
   ### AMD Support ###
   # disable gradient accumulation fusion: Need to add apex to enable this
   --no-gradient-accumulation-fusion
   ###################
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/workspace/Megatron-LM-amd_version/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
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
   ${MISC_ARGS[@]}


####clear after training

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python