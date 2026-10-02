#!/bin/bash
# FlashREINFORCE (NVIDIA-NeMo/labs-molt#116) on DeepSeek-R1-Distill-Qwen-1.5B, reproducing molt's
# quick-start recipe: the FP16 sanity test sail/Sanity-Test-R1D-1.5B (1,460 MATH problems), one
# rollout per prompt, 128 rollouts per update, 8k responses, no KL, AIME 2024 + 2025 avg@32 every 128
# updates. One 8-GPU node: 1 trainer + 7 SGLang engines. Batches form as in molt's asynchronous trainer:
# a pool of 512 rollouts (generating or finished, not yet batched) refilled one per rollout taken, up to
# 8 batches formed ahead of training, rollouts continuing across weight updates, and each pass over the
# 1,460 prompts ending with a 52-rollout batch (1,000 passes x 12 updates = 12,000 updates).
#
# Prerequisites (see README.md):
#   ${MODEL_DIR}  HF checkpoint of deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   ${DATA_DIR}   hf download sail/Sanity-Test-R1D-1.5B --repo-type dataset
#
# Environment knobs (defaults reproduce the full recipe):
#   NUM_ROLLOUT=12000         number of updates; the LR schedule always spans 12,000
#   TRUST_REGION_DELTA=5e-3   binary-KL trust region; inf gives plain PG-IS (the ablation)
#   SAVE_DIR, SAVE_INTERVAL   checkpoints (about 21 GB each with optimizer state)
#   KEEP_CHECKPOINTS=N        keep only the newest N checkpoints in SAVE_DIR (default: keep all)
#   POOL_SIZE=512             rollouts kept generating or waiting to be batched (molt: vllm_generate_batch_size)
#   QUEUED_BATCHES=8          batches formed ahead of training (molt: async_queue_size)
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
# DeepSeek-R1-Distill-Qwen-1.5B is Qwen2.5-Math-1.5B with untied embeddings (rope base 10000).
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-1.5B.sh"
MODEL_ARGS+=(--untie-embeddings-and-output-weights)

MODEL_DIR=${MODEL_DIR:-/root/models/DeepSeek-R1-Distill-Qwen-1.5B}
DATA_DIR=${DATA_DIR:-/root/datasets/Sanity-Test-R1D-1.5B}
SAVE_DIR=${SAVE_DIR:-/root/checkpoints/flash_reinforce_r1d_1p5b}

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
POOL_SIZE=${POOL_SIZE:-512}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   # Starts from the HF weights; later runs resume from the latest checkpoint in SAVE_DIR.
   --ref-load "${MODEL_DIR}"
   --load "${SAVE_DIR}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-500}"
)

ROLLOUT_ARGS=(
   # Generate while training, forming batches like molt: trajectories continue across weight updates.
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --fully-async-pool-size "${POOL_SIZE}"
   --fully-async-max-queued-batches "${QUEUED_BATCHES:-8}"
   --fully-async-drain-each-epoch
   # enough request slots for the whole pool
   --sglang-server-concurrency $(((POOL_SIZE + ROLLOUT_GPUS - 1) / ROLLOUT_GPUS))

   --prompt-data "${DATA_DIR}/train/math_1460.parquet"
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 42
   --custom-rm-path reward.math_reward

   # --num-rollout, not --num-epoch: epoch boundaries would add an eval every 12 updates.
   --num-rollout "${NUM_ROLLOUT:-12000}"
   --rollout-batch-size 128
   --n-samples-per-prompt 1
   --num-steps-per-rollout 1
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-top-p 1.0
)

EVAL_ARGS=(
   --eval-prompt-data aime24 "${DATA_DIR}/test/aime_2024.parquet" aime25 "${DATA_DIR}/test/aime_2025.parquet"
   --n-samples-per-eval-prompt 32
   --eval-temperature 0.6
   --eval-top-p 0.95
   --eval-max-response-len 8192
   --eval-interval 128
   --log-passrate
)

FLASH_REINFORCE_ARGS=(
   # reward minus the rollout-batch mean; the objective defaults to --pg-loss-type reinforce
   --advantage-estimator flash_reinforce
   # per-token IS weight pi/mu, sequences outside the binary-KL trust region are dropped
   --use-tis
   --custom-tis-function-path slime.backends.megatron_utils.loss.binary_kl_trust_region_function
   --tis-binary-kl-threshold "${TRUST_REGION_DELTA:-5e-3}"
   --kl-coef 0
   --entropy-coef 0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style cosine
   --min-lr 1e-7
   --lr-warmup-iters 360
   --lr-decay-iters 12000
   --weight-decay 0.1
   # AdamW over every parameter, as molt does; Megatron exempts biases and norms by default
   --apply-wd-to-all-params
   --adam-beta1 0.9
   --adam-beta2 0.95
   --adam-eps 1e-8
   --clip-grad 1.0
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.9
)

WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP:-flash_reinforce_r1d_1p5b}"
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
   --attention-backend flash
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}'
)

if [ -n "${KEEP_CHECKPOINTS:-}" ]; then
   # Every few minutes, delete checkpoints older than the newest KEEP_CHECKPOINTS. Only ones older
   # than the checkpoint recorded as complete are touched, so a save in progress is never removed.
   (
      set +x
      while sleep 300; do
         latest=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null) || continue
         ls -d "${SAVE_DIR}"/iter_[0-9]* 2>/dev/null | sort | head -n -"${KEEP_CHECKPOINTS}" | while read -r old; do
            if [ "$((10#${old##*/iter_}))" -lt "$((10#${latest}))" ]; then
               rm -rf -- "${old}"
            fi
         done
      done
   ) &
   PRUNE_PID=$!
   trap 'kill ${PRUNE_PID} 2>/dev/null || true' EXIT
fi

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
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}"
