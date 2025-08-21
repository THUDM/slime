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
source /root/workspace/slime_latest/slime/scripts/models/qwen3-0.6B.sh

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

export TASK_CONFIGS="/root/workspace/slime_latest/slime/slime/ray/config/tasks/task_A.yaml \
 /root/workspace/slime_latest/slime/slime/ray/config/tasks/task_B.yaml"

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_MODE\": \"offline\",
    \"WANDB_DIR\": \"/root/workspace/slime_latest/slime\",
    \"LOG_DIR\": \"/root/workspace/slime_latest/slime\",
    \"PERF_DIR\": \"/root/workspace/slime_latest/slime\",
    \"TASK_CONFIGS\": \"${TASK_CONFIGS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/workspace/slime_latest/slime/train_multi_tasks.py \
