#!/bin/bash

# Multi-teacher on-policy distillation (OPD) with SGLang multimodel config.
# Usage (from anywhere):
#   export HF_CHECKPOINT=...  # etc., or edit defaults below
#   bash examples/MOPD/run-opd-multidomain.sh
#
# Layout example (one 8-GPU node): --rollout-num-gpus 5 per sglang_opd_multimodel.yaml
#   (student 2 + three teachers 1 each) + --actor-num-gpus-per-node 2 for Megatron.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SGLANG_MULTIMODEL_CONFIG="${SGLANG_MULTIMODEL_CONFIG:-$SCRIPT_DIR/sglang_opd_multimodel.yaml}"

# --- Paths: set via environment or replace placeholders ---
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/student/hf-checkpoint}"
REF_LOAD="${REF_LOAD:-/path/to/student/torch-distributed-checkpoint}"
OPD_CKPT_DIR="${OPD_CKPT_DIR:-/path/to/opd/save-and-load-dir}"
PROMPT_DATA_JSONL="${PROMPT_DATA_JSONL:-/path/to/train.normalized.jsonl}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-$REPO_ROOT/scripts/models/qwen3-8B.sh}"

# Megatron-LM on PYTHONPATH inside the Ray worker (edit or export).
MEGATRON_PYTHONPATH="${MEGATRON_PYTHONPATH:-/path/to/Megatron-LM}"

# --- GPU counts: must match your YAML and hardware ---
RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-5}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "$MODEL_ARGS_SCRIPT"

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$REF_LOAD"
   --load "$OPD_CKPT_DIR"
   --save "$OPD_CKPT_DIR"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA_JSONL"
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func_route_by_domain
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 16
   # --eval-max-response-len 16384
   # --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
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
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group mopd-example
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
   --sglang-config "$SGLANG_MULTIMODEL_CONFIG"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON=$(printf '%s' "{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_PYTHONPATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
