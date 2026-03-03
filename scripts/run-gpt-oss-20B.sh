#!/bin/bash

# GPT-OSS 20B training script — single-node 8×H100
# Model: openai/gpt-oss-20b (20B MoE, 32 experts top-4)
#
# Prerequisites:
#   1. Preprocess: python tools/preprocess_gpt_oss.py --input /path/to/gpt-oss-20b --output /path/to/gpt-oss-20b-bf16
#   2. Convert:    torchrun --nproc_per_node 8 tools/convert_hf_to_torch_dist.py \
#                    --hf-checkpoint /path/to/gpt-oss-20b-bf16 \
#                    --save /path/to/gpt-oss-20b_torch_dist \
#                    --megatron-to-hf-mode bridge \
#                    $(cat scripts/models/gpt-oss-20B.sh | grep -oP "'[^']*'|--[^ ]+( [^ -][^ ]*)?")

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

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/gpt-oss-20B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/gpt-oss-20b-bf16
   --ref-load /root/gpt-oss-20b_torch_dist
   --load /root/gpt-oss-20b_slime/
   --save /root/gpt-oss-20b_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 1536
   # PP>1 requires seq-length override (PP recv buffer is sized by seq-length).
   # Must be >= max-tokens-per-gpu. Smaller = less memory for FP32 logits.
   --seq-length 1536
   # Reduce memory margin to avoid excessive CPU↔GPU swapping in colocate mode.
   --train-memory-margin-bytes 268435456
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --num-steps-per-rollout 1
)

OPTIMIZER_ARGS=(
   --lr 5e-7
   --lr-decay-style cosine
   --min-lr 0
   --lr-warmup-fraction 0.01
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.99
   --clip-grad 1.0
   --micro-batch-size 1
)

WANDB_ARGS=(
   # --wandb-project gpt-oss-20b
   # --wandb-exp-name gpt-oss-20b-grpo
)

SGLANG_ARGS=(
   --sglang-dp-attention
   --rollout-num-gpus 8
   --sglang-tp 1
   --sglang-mem-fraction-static 0.55
   --sglang-cuda-graph-max-bs 16
   --sglang-max-running-requests 64
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --moe-token-dispatcher-type alltoall
   --megatron-to-hf-mode bridge
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
