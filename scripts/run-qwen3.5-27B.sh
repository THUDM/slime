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

# unset proxy to avoid distributed startup issues
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [ -z "${BASE_FOLDER:-}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

MASTER_ADDR=${MASTER_ADDR:-}
if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-4}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
CP_SIZE=${CP_SIZE:-4}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-27B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${BASE_FOLDER}/Qwen3.5-27B"
   --ref-load "${BASE_FOLDER}/Qwen3.5-27B_torch_dist/"
   --load "${BASE_FOLDER}/Qwen3.5-27B_slime/"
   --save "${BASE_FOLDER}/Qwen3.5-27B_slime/"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${BASE_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1.0
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${BASE_FOLDER}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 30
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
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
   # --wandb-group qwen3.5-27B-32k
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.75
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

HOSTFILE=${HOSTFILE:-}
if [ -n "${HOSTFILE}" ]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
