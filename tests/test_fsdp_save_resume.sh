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

# --- Test Configuration ---
CHECKPOINT_DIR="/tmp/fsdp_checkpoint_test"
LOG_FILE="/tmp/fsdp_resume_log.txt"
MODEL_PATH="THUDM/Qwen2-0.5B-Instruct"
DATA_PATH="tests/data/gsm8k_sft.jsonl"




# Common arguments for both save and resume runs
FSDP_ARGS=(
    --backend fsdp
    --model_name_or_path ${MODEL_PATH}
    --train_path ${DATA_PATH}
    --rollout_batch_size 4
    --global_batch_size 8
    --fsdp_sharding_strategy SHARD_GRAD_OP
)

# Arguments for the initial run that saves the checkpoint
SAVE_RUN_ARGS=(
    --train_steps 2
    --save ${CHECKPOINT_DIR}
    --save-interval 2
)

# Arguments for the second run that resumes from the checkpoint
RESUME_RUN_ARGS=(
    --train_steps 4
    --load "${CHECKPOINT_DIR}/2"
)


rm -rf ${CHECKPOINT_DIR}
rm -f ${LOG_FILE}

# launch the master node of ray in container
ray start --head --node-ip-address "127.0.0.1" --num-gpus 2 --disable-usage-stats

echo "--- Starting FSDP Save and Resume Test ---"

# Run training for 2 steps and save a checkpoint
echo "--- Running initial training to save a checkpoint ---"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "."
     }
   }' \
   -- torchrun --standalone --nproc_per_node=2 train.py \
   ${FSDP_ARGS[@]} \
   ${SAVE_RUN_ARGS[@]}

echo "Waiting for save job to complete..."
sleep 30

# Check if the checkpoint directory was created
if [ ! -d "${CHECKPOINT_DIR}/2" ]; then
    echo "--- Test Failed: Checkpoint directory was not created. ---"
    exit 1
fi
echo "--- Checkpoint saved successfully. ---"


echo "--- Resuming training from the checkpoint ---"
# We submit the job and capture the job ID to get logs later
JOB_ID=$(ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "."
     }
   }' \
   -- torchrun --standalone --nproc_per_node=2 train.py \
   ${FSDP_ARGS[@]} \
   ${RESUME_RUN_ARGS[@]} | grep "Job ID" | awk '{print $3}')

echo "Resume job submitted with ID: $JOB_ID"
echo "Waiting for resume job to complete..."
ray job wait $JOB_ID

echo "--- Verifying resumption log ---"
ray job logs $JOB_ID > ${LOG_FILE}

if grep -q "step 2:" ${LOG_FILE}; then
    echo "--- Test Passed: Training successfully resumed from step 2. ---"
else
    echo "--- Test Failed: Could not find 'step 2:' in the log. ---"
    cat ${LOG_FILE}
    exit 1
fi

# Clean up artifacts and stop Ray
rm -rf ${CHECKPOINT_DIR}
rm -f ${LOG_FILE}
ray stop --force

echo "--- FSDP Save and Resume Test Completed Successfully ---"