#!/bin/bash
# FlashREINFORCE with a multi-turn tool-using agent on an MoE model: Moonlight-16B-A3B-Instruct on
# GSM8K with a calculator tool (calculator_agent.py), one rollout per prompt. Fully-async rollout
# forms batches like molt, every model turn continues across weight updates (generate_turn), and
# rollout routing replay (R3) keeps training on the experts the rollout used. One 8-GPU node:
# 4 trainer GPUs (expert parallel 4, optimizer state on CPU) and 4 SGLang engines.
#
# Prerequisites (see README.md):
#   ${MODEL_DIR}  HF checkpoint of moonshotai/Moonlight-16B-A3B-Instruct
#   ${DATA_DIR}   python prepare_data.py <zhuzilin/gsm8k> ${DATA_DIR}
#
# Environment knobs:
#   NUM_ROLLOUT=100           number of updates
#   POOL_SIZE=64              rollouts generating or waiting to be batched
#   QUEUED_BATCHES=2          batches formed ahead of training
#   TRUST_REGION_DELTA=5e-3   binary-KL trust region; inf gives plain PG-IS
#   DEBUG_ROLLOUT_DIR         dump every rollout (with routed experts) there
#   CALCULATOR_LATENCY=0      seconds per calculator call; a slow tool makes turns straddle weight updates
#   WANDB_PROJECT             enables wandb; WANDB_MODE=offline and WANDB_DIR are honoured

# clean any leftover ray/sglang from a previous run in this container
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/moonlight.sh"

MODEL_DIR=${MODEL_DIR:-/root/models/Moonlight-16B-A3B-Instruct}
DATA_DIR=${DATA_DIR:-/root/datasets/gsm8k-calculator}

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
POOL_SIZE=${POOL_SIZE:-64}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}"
)

ROLLOUT_ARGS=(
   # Generate while training, forming batches like molt; every model turn continues across weight updates.
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --fully-async-pool-size "${POOL_SIZE}"
   --fully-async-max-queued-batches "${QUEUED_BATCHES:-2}"
   # enough request slots for the whole pool
   --sglang-server-concurrency $(((POOL_SIZE + ROLLOUT_GPUS - 1) / ROLLOUT_GPUS))
   --custom-generate-function-path calculator_agent.generate

   --prompt-data "${DATA_DIR}/train.parquet"
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math

   --num-rollout "${NUM_ROLLOUT:-100}"
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   --num-steps-per-rollout 1
   # the whole trajectory's budget: model turns plus calculator results
   --rollout-max-response-len 2048
   --rollout-temperature 1.0
   --rollout-top-p 1.0
)

EVAL_ARGS=(
   --eval-prompt-data gsm8k "${DATA_DIR}/test.parquet@[0:256]"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 2048
   --eval-temperature 0.6
   --eval-interval 20
)

FLASH_REINFORCE_ARGS=(
   --advantage-estimator flash_reinforce
   --use-tis
   --custom-tis-function-path slime.backends.megatron_utils.loss.binary_kl_trust_region_function
   --tis-binary-kl-threshold "${TRUST_REGION_DELTA:-5e-3}"
   --kl-coef 0
   --entropy-coef 0
   # train on the experts SGLang routed each token to
   --use-rollout-routing-replay
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --apply-wd-to-all-params
   --adam-beta1 0.9
   --adam-beta2 0.95
   --clip-grad 1.0

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size "${ACTOR_GPUS}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

DEBUG_ARGS=()
if [ -n "${DEBUG_ROLLOUT_DIR:-}" ]; then
   DEBUG_ARGS=(--save-debug-rollout-data "${DEBUG_ROLLOUT_DIR}/rollout_{rollout_id}.pt")
fi

WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP:-flash_reinforce_moonlight_agent}"
      --wandb-mode "${WANDB_MODE:-online}"
   )
   if [ -n "${WANDB_DIR:-}" ]; then
      WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
   fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # Moonlight's multi-latent attention (qk head dim 192, v 128) has no flash kernel on A100;
   # auto picks FlashAttention 3 on H100 and Transformer Engine's unfused attention on A100
   --attention-backend auto
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}'
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

cd "${SCRIPT_DIR}/../.."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${FLASH_REINFORCE_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}"
