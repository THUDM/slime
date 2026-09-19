#!/bin/bash

# Qwen3.5-35B-A3B routed-expert INT4-QAT on 4 GPUs. The latest native slime
# bridge loads the BF16 Hugging Face checkpoint directly; no torch_dist
# pre-conversion or external Megatron-Bridge package is required.

# For rerunning inside the slime container (which has its own PID namespace).
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3.5-35B-A3B.sh"

BASE_FOLDER=${BASE_FOLDER:-/mnt/slime-qwen35}
HF_MODEL=${HF_MODEL:-${BASE_FOLDER}/Qwen3.5-35B-A3B}
INT4_MODEL=${INT4_MODEL:-${BASE_FOLDER}/Qwen3.5-35B-A3B-INT4}
SAVE_DIR=${SAVE_DIR:-${BASE_FOLDER}/Qwen3.5-35B-A3B_slime}
PROMPT_DATA=${PROMPT_DATA:-${BASE_FOLDER}/dapo-math-17k/dapo-math-17k.jsonl}

CKPT_ARGS=(
   # SGLang starts from this compressed-tensors checkpoint. Its config.json
   # also drives INT4 quantization during every online actor weight update.
   --hf-checkpoint "${INT4_MODEL}"

   # The native HF -> Megatron loader initializes actor weights from BF16. If
   # SAVE_DIR has no Megatron tracker yet, --load falls back to this HF
   # directory; later runs resume from SAVE_DIR. KL loss is intentionally off,
   # so --ref-load is only the initialization fallback here.
   --ref-load "${HF_MODEL}"
   --load "${SAVE_DIR}"
   --save "${SAVE_DIR}"
   --save-interval 2
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 2
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 1024
   --rollout-temperature 0.8

   --global-batch-size 8
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --qwen-gdn-backend fla
   --sglang-cuda-graph-bs 1 2 4 8
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 4 --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
