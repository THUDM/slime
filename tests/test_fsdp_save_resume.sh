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

# Define a temporary directory for checkpoints and ensure it's clean.
CHECKPOINT_DIR="/tmp/slime_fsdp_test_checkpoint"
rm -rf ${CHECKPOINT_DIR}
echo "--- Using checkpoint directory: ${CHECKPOINT_DIR} ---"


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
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 128
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
   --rollout-num-gpus-per-engine 1
)

# --- Stage 1: Train and Save Checkpoint ---
echo "--- Starting Ray Head Node for Save Test ---"
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

echo "--- Submitting job to train and SAVE a checkpoint to ${CHECKPOINT_DIR} ---"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "env_vars": {
            "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
            "SLIME_BACKEND": "fsdp"
        }
    }' \
    -- python3 train.py \
    --save ${CHECKPOINT_DIR} \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${SGLANG_ARGS[@]}"

echo "--- Save run finished. Stopping Ray to simulate a restart ---"
ray stop --force
sleep 3

echo "--- Restarting Ray Head Node for Load Test ---"
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 --disable-usage-stats

# Since we ran for 2 rollouts (iterations 0 and 1), we load the final checkpoint.
# The checkpointing logic saves checkpoints in subdirectories named by the iteration number.
LOAD_PATH="${CHECKPOINT_DIR}/1"

echo "--- Submitting job to LOAD from checkpoint: ${LOAD_PATH} ---"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
        "env_vars": {
            "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
            "SLIME_BACKEND": "fsdp"
        }
    }' \
    -- python3 train.py \
    --load ${LOAD_PATH} \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${SGLANG_ARGS[@]}"

echo "--- Load test finished. Cleaning up resources ---"
ray stop --force
rm -rf ${CHECKPOINT_DIR}

echo "--- Save/Load test completed successfully! ---"