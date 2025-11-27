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
export CUDA_VISIBLE_DEVICES=4,5,6,7
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
#    --ref-load /root/Qwen3-0.6B

)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type math

   --num-rollout 2
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1.0

   --global-batch-size 16
   #--balance-data
)

GRPO_ARGS=(
   --advantage-estimator gspo
#    --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

#    --fsdp-cpu-offload
)

WANDB_ARGS=(
   #--use-wandb
  #  --wandb-project slime-dev
  #  --wandb-group qwen3-30B-A3B
   #--wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.85
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --rollout-num-gpus-per-engine 8
   # --sglang-mem-fraction-static 0.6
   # --sglang-enable-dp-attention
   # --sglang-dp-size 8
   # --sglang-ep-size 8
   
   #--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)

   #--sglang-max-running-requests 512

  #  --sglang-enable-deterministic-inference
  #  --sglang-rl-on-policy-target fsdp
  #  --sglang-attention-backend fa3
  #  --attn-implementation flash_attention_3
  #  --deterministic-mode
  #  --true-on-policy-mode
)

# launch the master node of ray in container - 4 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats


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
   --rollout-num-gpus 2 \
   --colocate \
   --train-backend fsdp \
   --gradient-checkpointing \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SGLANG_ARGS[@]}

