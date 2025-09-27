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

# Clean up Ray temporary files to avoid disk space issues
echo "Cleaning up Ray temporary files..."
rm -rf /tmp/ray/*
sudo rm -rf /tmp/ray/* 2>/dev/null || true

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=3,4

# Enable basic logging for OOM debugging
export PYTHONPATH=/root/william_slime:$PYTHONPATH

# Set Ray temporary directory to avoid disk space issues
export RAY_TMPDIR=/dev/shm/ray_tmp
mkdir -p $RAY_TMPDIR

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
)

MISC_ARGS=(
   # FSDP-specific arguments
   # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
   # --fsdp-full-params  # Uncomment this line to enable full params mode
   # Comment out the above line to use sharded mode (default)
)

PROFILER_ARGS=(
   # Memory profiling for debugging OOM (using config file)
   --profiler-config ./profiler_config.yaml
)

# launch the master node of ray in container
ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats \
  --temp-dir=$RAY_TMPDIR \

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
   ${MISC_ARGS[@]} \
   ${PROFILER_ARGS[@]}



# Cleanup after job completion
echo "Job completed, cleaning up Ray temporary files..."
ray stop --force
rm -rf $RAY_TMPDIR/* 2>/dev/null || true