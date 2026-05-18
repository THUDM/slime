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

# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi
# export MASTER_ADDR="127.0.0.1"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDER}/Qwen3.5-35B-A3B
   --ref-load ${BASE_FOLDER}/Qwen3.5-35B-A3B_torch_dist
   --load ${BASE_FOLDER}/Qwen3.5-35B-A3B_slime/
   --save ${BASE_FOLDER}/Qwen3.5-35B-A3B_slime/
   --save-interval 20
)

SFT_MAX_PROMPT_TOKENS=${SFT_MAX_PROMPT_TOKENS:-8192}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
CP_SIZE=${CP_SIZE:-1}
echo "SFT_MAX_PROMPT_TOKENS: ${SFT_MAX_PROMPT_TOKENS}"
echo "MAX_TOKENS_PER_GPU: ${MAX_TOKENS_PER_GPU}"
echo "CP_SIZE: ${CP_SIZE}"

if (( SFT_MAX_PROMPT_TOKENS > MAX_TOKENS_PER_GPU * CP_SIZE )); then
  echo "Invalid config: SFT_MAX_PROMPT_TOKENS (${SFT_MAX_PROMPT_TOKENS}) must be <= MAX_TOKENS_PER_GPU * CP_SIZE (${MAX_TOKENS_PER_GPU} * ${CP_SIZE} = $((MAX_TOKENS_PER_GPU * CP_SIZE)))."
  exit 1
fi

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${BASE_FOLDER}/openhermes2_5.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128
   --rollout-max-prompt-len ${SFT_MAX_PROMPT_TOKENS}

   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --use-distributed-optimizer
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-35B-sft
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

   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

SPEC_ARGS=(
#    --mtp-num-layers 1
#    --enable-mtp-training
#    --mtp-loss-scaling-factor 0.1
)

# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
done
wait


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${SPEC_ARGS[@]}
