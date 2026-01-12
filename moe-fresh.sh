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
pkill -9 redis

set -ex

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
source "${SCRIPT_DIR}/scripts/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B
   #--hf-checkpoint /root/Qwen3-30B-A3B-FP8
   --ref-load /root/Qwen3-30B-A3B
   # --load /root/Qwen3-30B-A3B_slime/
   # --save /root/Qwen3-30B-A3B_slime/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8

   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 0.7
)

PERF_ARGS=(
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
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
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-sonicmoe
   --wandb-group qwen3-30B-A3B-sonic
   --wandb-key b6c985b417dd3a453880c3673b7035b5f0161412
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --fsdp-moe-impl sonicmoe
   --no-offload-train
)

MISC_ARGS=(
   # No debug rollout data - generate fresh rollouts each time
   --load-debug-rollout-data ./data/data_{rollout_id}.pt
   # --save-debug-rollout-data ./data/data_{rollout_id}.pt
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with nsight profiling
# Ray nsight config: https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  },
  \"nsight\": {
    \"t\": \"cuda,nvtx,osrt,cudnn,cublas\",
    \"o\": \"profile_%p\",
    \"capture-range\": \"cudaProfilerApi\",
    \"capture-range-end\": \"stop\",
    \"cuda-memory-usage\": \"true\",
    \"force-overwrite\": \"true\"
  }
}"

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- \
    python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node 8 \
      --colocate \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${GRPO_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${TRAIN_BACKEND_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${MISC_ARGS[@]}


