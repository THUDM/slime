#!/usr/bin/env bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

pip install --no-build-isolation --no-deps -e /root/workspace/slime_latest/slime
pip install --no-build-isolation --no-deps -e /root/workspace/slime_latest/slime/tools/Tracer

# # 提交任务的时候需要设置
# cd /sgl-workspace/sglang 
# git apply /volume/pt-train/users/mingjie/hzl_code/code/slime/patch/scheduler.patch

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source /root/workspace/slime_latest/slime/scripts/models/qwen3-0.6B.sh

CKPT_ARGS=(
   --hf-checkpoint /root/workspace/models/HF/Qwen3-0.6B
   # --hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /root/workspace/models/Torch/Qwen3-0.6B
   --load /root/workspace/slime_latest/slime/experiments/json/qwen06b
   --save /root/workspace/slime_latest/slime/experiments/json/qwen06b
   --save-interval 1
)

ROLLOUT_ARGS=(
   --prompt-data /root/workspace/data/slime_start/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 2
   --rollout-batch-size 1
   --n-samples-per-prompt 1
   --rollout-max-response-len 1024
   --rollout-temperature 0.8

   --global-batch-size  1
   --balance-data
)

EVAL_ARGS=(
   --eval-prompt-data aime /root/workspace/data/slime_start/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 256
   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024
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

WANDB_ARGS=(
   --use-wandb
   --wandb-project 0812-0.6b
   --wandb-group 2task
)

SGLANG_ARGS=(
   --sglang-mem-fraction-static 0.8
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

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"/root/workspace/slime_latest/slime\",
    \"LOG_DIR\": \"/root/workspace/slime_latest/slime\",
    \"PERF_DIR\": \"/root/workspace/slime_latest/slime\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/workspace/slime_latest/slime/train_multi_tasks.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   --rollout-num-gpus-per-engine 1 \
   --offload \
   --update-from-disk \
   --tasks-num  1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}