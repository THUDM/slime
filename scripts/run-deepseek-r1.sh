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
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/deepseek-v3.sh"

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/DeepSeek-R1/
   #--hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16/
   --ref-load $BASE_DIR/DeepSeek-R1_torch_dist/
   --load $BASE_DIR/DeepSeek-R1_slime/
   --save $BASE_DIR/DeepSeek-R1_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 0.8

   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   --num-steps-per-rollout 4
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime $BASE_DIR/rl_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 4
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 13

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
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

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group deepseek-r1-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 64
   --sglang-mem-fraction-static 0.7
   --sglang-enable-ep-moe

   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-disable-radix-cache

   # enable deepep for sglang
   --sglang-enable-deepep-moe
   --sglang-deepep-mode auto

   # make every dp rank has 128 concurrency
   --sglang-server-concurrency 1024
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   # --attention-backend flash

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "MASTER_ADDR": "${MASTER_ADDR}",
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/sgl-workspace/nvshmem/install/lib/"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 8 \
   --colocate \
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
