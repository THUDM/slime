#!/bin/bash

# Megatron Colocated 4GPU Training Script for Qwen3-8B
# 
# Key differences from test_qwen3-30B-A3B.sh:
# - Uses 4 GPUs instead of 8 GPUs
# - Qwen3-8B (dense model) instead of Qwen3-30B-A3B (MoE model)
# - Tensor parallelism: TP=2 (vs TP=4 for 30B)
# - No pipeline parallelism (vs PP=1 for 30B)
# - No context parallelism (vs CP=2 for 30B)
# - No expert parallelism (8B is not MoE)
# - Smaller batch sizes and token limits
# - Less aggressive memory settings for colocated training
#
# Usage:
#   ./test_qwen3-8B_megatron_colocated_4xGPU.sh
#   
# To use specific GPUs, set CUDA_VISIBLE_DEVICES before running:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   ./test_qwen3-8B_megatron_colocated_4xGPU.sh

set -e

# Kill any existing processes for clean restart
pkill -9 sglang 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
sleep 1

export CUDA_VISIBLE_DEVICES=4,5,6,7
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
# Use the 8B model config if it exists, otherwise define inline
if [ -f "${SCRIPT_DIR}/../scripts/models/qwen3-8B.sh" ]; then
    source "${SCRIPT_DIR}/../scripts/models/qwen3-8B.sh"
fi

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist  # Uncomment if you have a reference model
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 100
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 2048
   --rollout-temperature 0.8

   --global-batch-size 16
   --balance-data
)

EVAL_ARGS=(
   # Uncomment to enable evaluation
   # --eval-interval 20
   # --eval-prompt-data test /root/datasets/test.jsonl
   # --n-samples-per-eval-prompt 1
   # --eval-max-response-len 2048
   # --eval-top-k 1
)

PERF_ARGS=(
   # Parallelism settings for 8B model on 4 GPUs
   --tensor-model-parallel-size 2    # Use TP=2 for memory efficiency
   --sequence-parallel                # Enable sequence parallelism with TP
   --pipeline-model-parallel-size 1   # No pipeline parallelism needed
   --context-parallel-size 1          # No context parallelism for 8B model
   # No expert parallelism - Qwen3-8B is not MoE

   # Gradient checkpointing to save memory
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1           # Recompute activations for 1 layer per pipeline stage

   # Dynamic batching for variable-length sequences
   --use-dynamic-batch-size
   # If OOM during training, reduce max-tokens-per-gpu to 6144 or 4096
   # If sequences are very long (>4K tokens), consider reducing to 4096
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss                     # Uncomment if using reference model
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # --use-tis                         # Uncomment for off-policy correction
   # --use-routing-replay              # Not needed for non-MoE models
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # CPU offloading can help with memory but may slow down training
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   # Use all 4 GPUs for SGLang engine
   --rollout-num-gpus-per-engine 4
   
   # Reserve less memory for inference to leave room for training
   # If OOM during training, reduce this to 0.45 or 0.50
   --sglang-mem-fraction-static 0.55
   
   # Limit concurrent requests to manage memory
   # If OOM during rollout, reduce this to 32
   --sglang-max-running-requests 64
   
   # Disable radix cache to save memory
   --sglang-disable-radix-cache
)

MISC_ARGS=(
   # Dropout settings (default in megatron is 0.1)
   --attention-dropout 0.0
   --hidden-dropout 0.0
   
   # Numerical stability
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   
   # Attention backend
   --attention-backend flash
   
   # MoE settings not needed for Qwen3-8B (not an MoE model)
   # --moe-token-dispatcher-type flex
   # --moe-enable-deepep
)

WANDB_ARGS=(
   # Uncomment to enable wandb logging
   # --use-wandb
   # --wandb-project "qwen3-8b-rl"
   # --wandb-group "megatron-4gpu-colocated"
   # --wandb-mode "online"
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"CUDA_HOME\": \"${CUDA_HOME}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

