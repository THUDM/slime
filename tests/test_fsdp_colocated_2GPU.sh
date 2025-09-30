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
export CUDA_VISIBLE_DEVICES=4,5

# Enable basic logging for OOM debugging
export PYTHONPATH=/root/william_slime:$PYTHONPATH
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-0.6B
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 64
)

GRPO_ARGS=(
   --advantage-estimator grpo
   #--use-kl-loss
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-decode-log-interval 1000
)


WANDB_ARGS=(
   --use-wandb
   # --wandb-team "your-team-name"  # Uncomment and replace with your wandb team name if using a team
   --wandb-project "gsm8k_async_rl"
   --wandb-group "fsdp-2gpu-colocated"
   --wandb-mode "online"  # Change to "offline" for local logging only
   # --wandb-key "your-api-key"  # Uncomment and set if needed (or use 'wandb login' beforehand)
   # --wandb-dir "./wandb_logs"  # Uncomment to specify custom wandb directory
)

FSDP_ARGS=(
   # FSDP-specific arguments
   # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
   # --fsdp-full-params  # Uncomment this line to enable full params mode
   # Comment out the above line to use sharded mode (default)
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "SLIME_BACKEND": "fsdp"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]}