#!/bin/bash

# Tau2 GRPO training (tau2-bench).

set -euo pipefail

if [ "${TAU2_CLEANUP:-0}" = "1" ]; then
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
fi

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
TAU_BENCH_OUT_DIR="${TAU_BENCH_OUT_DIR:-${SCRIPT_DIR}/../outputs}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/root/Megatron-LM}"

# ---- User-configurable paths ----
HF_DIR="${HF_DIR:-${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-${TAU_BENCH_OUT_DIR}/tau2/checkpoints/Qwen3-4B-tau2-sft1}"
REF_CKPT_DIR="${REF_CKPT_DIR:-${SFT_CKPT_DIR}}"
SAVE_DIR="${SAVE_DIR:-${TAU_BENCH_OUT_DIR}/tau2/checkpoints/Qwen3-4B-tau2-grpo-v1}"

TAU2_TRAIN_TASKS_JSONL="${TAU2_TRAIN_TASKS_JSONL:-${TAU_BENCH_OUT_DIR}/tau2/tasks/tau2_train_all_tasks.jsonl}"

NUM_GPUS="${NUM_GPUS:-4}"

export TAU2_USER_MODEL="${TAU2_USER_MODEL:-openai/Qwen/Qwen3-4B-Instruct-2507}"
export TAU2_USER_API_BASE="${TAU2_USER_API_BASE:-http://127.0.0.1:30001/v1}"
export TAU2_USER_TEMPERATURE="${TAU2_USER_TEMPERATURE:-0.7}"
export TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-100}"

export TAU2_REWARD_ALPHA="${TAU2_REWARD_ALPHA:-0.25}"
export TAU2_DOMAIN_ADAPTIVE_ALPHA="${TAU2_DOMAIN_ADAPTIVE_ALPHA:-1}"
export TAU2_PARTIAL_ACTION_WEIGHT="${TAU2_PARTIAL_ACTION_WEIGHT:-0.5}"
export TAU2_PARTIAL_COMMUNICATE_WEIGHT="${TAU2_PARTIAL_COMMUNICATE_WEIGHT:-0.15}"
export TAU2_PARTIAL_ENV_ASSERTION_WEIGHT="${TAU2_PARTIAL_ENV_ASSERTION_WEIGHT:-0.35}"
export TAU2_PARTIAL_DB_WEIGHT="${TAU2_PARTIAL_DB_WEIGHT:-0.0}"

export TAU2_TELECOM_COMMUNICATION_BOOST="${TAU2_TELECOM_COMMUNICATION_BOOST:-1}"

export TAU2_USE_COMPRESSED_PROMPTS="${TAU2_USE_COMPRESSED_PROMPTS:-1}"

export TAU2_USE_CURRICULUM="${TAU2_USE_CURRICULUM:-1}"
export TAU2_CURRICULUM_MIN_ATTEMPTS="${TAU2_CURRICULUM_MIN_ATTEMPTS:-5}"
export TAU2_CURRICULUM_SOLVED_WEIGHT="${TAU2_CURRICULUM_SOLVED_WEIGHT:-0.1}"
export TAU2_CURRICULUM_HARD_WEIGHT="${TAU2_CURRICULUM_HARD_WEIGHT:-0.5}"

source "${SCRIPT_DIR}/../../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_DIR}"
  --ref-load "${REF_CKPT_DIR}"
  --load "${SFT_CKPT_DIR}"
  --no-load-optim  # Start optimizer fresh for GRPO
  --save "${SAVE_DIR}"
  --save-interval 10
)

ROLLOUT_ARGS=(
  --prompt-data "${TAU2_TRAIN_TASKS_JSONL}"
  --input-key text
  --metadata-key metadata
  --rollout-shuffle

  # Optimized for 4xH100: better task coverage per iteration
  # NOTE: rollout_batch_size * n_samples_per_prompt must be multiple of global_batch_size
  --num-rollout 200
  --rollout-batch-size 16
  --n-samples-per-prompt 4
  --rollout-max-response-len 4096
  --rollout-temperature 0.7
  # Note: --rollout-stop removed; rollout.py ensures </tool_call> is in stops

  --global-batch-size 64
  --balance-data
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --eps-clip 0.2
  --eps-clip-high 0.28
  --entropy-coef 0.001
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 20480

  # Gradient checkpointing for H100 memory efficiency
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.70  # Increased from 0.60 since user sim is on separate GPU
  # If rollouts abort, uncomment:
  # --no-offload-rollout
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-generate-function-path rollout.generate
  --custom-reward-post-process-path reward.tau2_reward_post_process
)

# WandB logging (enabled by default for cookbook reproducibility)
WANDB_ARGS=(
  --use-wandb
  --wandb-project "tau2-cookbook"
  --wandb-group "grpo-qwen3-4b"
  --wandb-exp-name "tau2-grpo-v1"
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${REPO_ROOT}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_DIR}:${SCRIPT_DIR}:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"TAU2_DATA_DIR\": \"${TAU2_DATA_DIR:-${TAU_BENCH_OUT_DIR}/_external/tau2-bench/data}\",
    \"TAU2_USER_MODEL\": \"${TAU2_USER_MODEL}\",
    \"TAU2_USER_API_BASE\": \"${TAU2_USER_API_BASE:-}\",
    \"TAU2_USER_TEMPERATURE\": \"${TAU2_USER_TEMPERATURE}\",
    \"TAU2_MAX_STEPS\": \"${TAU2_MAX_STEPS}\",
    \"TAU2_REWARD_ALPHA\": \"${TAU2_REWARD_ALPHA}\",
    \"TAU2_DOMAIN_ADAPTIVE_ALPHA\": \"${TAU2_DOMAIN_ADAPTIVE_ALPHA}\",
    \"TAU2_PARTIAL_ACTION_WEIGHT\": \"${TAU2_PARTIAL_ACTION_WEIGHT}\",
    \"TAU2_PARTIAL_COMMUNICATE_WEIGHT\": \"${TAU2_PARTIAL_COMMUNICATE_WEIGHT}\",
    \"TAU2_PARTIAL_ENV_ASSERTION_WEIGHT\": \"${TAU2_PARTIAL_ENV_ASSERTION_WEIGHT}\",
    \"TAU2_PARTIAL_DB_WEIGHT\": \"${TAU2_PARTIAL_DB_WEIGHT}\",
    \"TAU2_TELECOM_COMMUNICATION_BOOST\": \"${TAU2_TELECOM_COMMUNICATION_BOOST}\",
    \"TAU2_USE_COMPRESSED_PROMPTS\": \"${TAU2_USE_COMPRESSED_PROMPTS}\",
    \"TAU2_USE_CURRICULUM\": \"${TAU2_USE_CURRICULUM}\",
    \"TAU2_CURRICULUM_MIN_ATTEMPTS\": \"${TAU2_CURRICULUM_MIN_ATTEMPTS}\",
    \"TAU2_CURRICULUM_SOLVED_WEIGHT\": \"${TAU2_CURRICULUM_SOLVED_WEIGHT}\",
    \"TAU2_CURRICULUM_HARD_WEIGHT\": \"${TAU2_CURRICULUM_HARD_WEIGHT}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"HF_TOKEN\": \"${HF_TOKEN:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --rollout-num-gpus "${NUM_GPUS}" \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${CUSTOM_ARGS[@]} \
  ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"}
