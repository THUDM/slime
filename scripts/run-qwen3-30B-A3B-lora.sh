#!/bin/bash

# LoRA variant of run-qwen3-30B-A3B.sh: the base model stays frozen and only the
# low-rank adapters are trained, including the routed MoE experts.
#
# Differences from the full-finetune script:
#   * LORA_ARGS below is added.
#   * --ref-load is removed: LoRA computes the reference logits by disabling the
#     adapters, so a separate reference checkpoint is neither needed nor allowed.
#   * The learning rate is raised, as is usual for adapter-only training.

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
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B
   # Fresh LoRA initialization uses HF base weights. To resume, point --load
   # at the full LoRA checkpoint in --save and omit --finetune.
   --load /root/Qwen3-30B-A3B
   --save /root/Qwen3-30B-A3B_slime/
   --save-interval 20
)

LORA_ARGS=(
   --use-lora
   # moe_language_all = attention + shared experts + routed experts.
   # Use moe_language to keep the routed experts frozen.
   --lora-target-preset moe_language_all
   # --lora-rank must be divisible by --expert-tensor-parallel-size (and by
   # --tensor-model-parallel-size for the dense row-parallel targets).
   --lora-rank 32
   --lora-alpha 64
   --save-lora /root/Qwen3-30B-A3B_lora/{rollout_id}
   # To initialize a new run from an adapter (not resume a full checkpoint):
   # --finetune
   # --lora-load /root/Qwen3-30B-A3B_lora/20
   # --lora-save-merged-hf /root/Qwen3-30B-A3B_merged/{rollout_id}
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
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   # Routed-expert LoRA requires ETP to divide TP; TP=4/ETP=1 is supported.
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

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
   # Adapter-only training tolerates (and usually needs) a much larger LR than
   # full finetuning. --lora-learning-rate overrides --lr for LoRA parameters.
   --lr 1e-5
   --lr-decay-style constant
   # In LoRA mode --weight-decay is always taken from --lora-weight-decay
   # (default 0.0), so set that one rather than --weight-decay.
   --lora-weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-30B-A3B-lora
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # MLA models are not supported by this LoRA implementation.
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
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
   ${LORA_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
