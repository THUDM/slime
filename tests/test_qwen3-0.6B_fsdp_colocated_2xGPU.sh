#!/bin/bash

# FSDP Colocated 2GPU Training Script with Weights & Biases Support
# 
# This script runs FSDP training with wandb logging enabled.
# 
# Wandb Configuration:
# - Rank and world size are automatically detected from distributed context
# - Only rank 0 will log to wandb to avoid duplicate entries
# - Distributed coordination handled by torch.distributed in FSDP actors
# 
# To customize wandb settings:
# 1. Uncomment and set --wandb-team if you're using a team/organization (optional for personal accounts)
# 2. Set your wandb API key if needed (or use 'wandb login' beforehand)
# 3. Modify project name and group as needed
# 4. Change wandb mode to 'offline' for local logging only
# 5. Uncomment --wandb-dir to specify custom log directory

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

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-0.6B
   --ref-load /root/Qwen3-0.6B
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 256
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
   # Uncomment to use DeepSpeed CPU Adam
   # --optimizer deepspeed_cpu_adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   # Set equal to the number of GPUs per node for colocated mode
   --rollout-num-gpus-per-engine 2
   --sglang-decode-log-interval 1000
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-fsdp
   --wandb-group fsdp-2gpu-colocated
   --wandb-key ${WANDB_KEY}
)

FSDP_ARGS=(
   # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
   # --fsdp-full-params  # Uncomment this line to enable full params mode

   # Set the bucket size for weight update
   --update-weights-buffer-size $((512 * 1024 * 1024)) # 512MB
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   --train-backend fsdp \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} 
