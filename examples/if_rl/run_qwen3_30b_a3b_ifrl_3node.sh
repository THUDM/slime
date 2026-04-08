#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/ray_bootstrap_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/training_runner_utils.sh"

NUM_NODES=${NUM_NODES:-3}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-8}

MODEL_DIR=${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/models/Qwen3-30B-A3B}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/models/Qwen3-30B-A3B_torch_dist}
RAW_DATA=${RAW_DATA:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/data/raw_data/Nemotron-Cascade-2-RL-data/IF-RL/train.jsonl}
WORK_ROOT=${WORK_ROOT:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b}

SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
SCRIPT_QUERIES_PY="${SLIME_DIR}/examples/common/script_queries.py"

DATA_CACHE_DIR="${WORK_ROOT}/data_cache"
LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
NORMALIZED_DATA="${DATA_CACHE_DIR}/ifrl_train.normalized.jsonl"

# Stable defaults for this workspace. These match the successful 3-node run
# envelope that we can rerun without overloading SGLang workers.
#
# Author / paper reference (Nemotron-Cascade 2 IF-RL, Table 8 / §4.2.2):
#   IFRL_ROLLOUT_BATCH_SIZE=128
#   IFRL_SAMPLES_PER_PROMPT=16
#   IFRL_GLOBAL_BATCH_SIZE=2048
#   IFRL_MAX_CONTEXT_LEN=65536
#   IFRL_MAX_RESPONSE_LEN=49152   # paper reports 49K response length
#   IFRL_LR=2e-6
#   IFRL_ADAM_BETA2=0.95
#   IFRL_DYNAMIC_FILTER=slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
IFRL_ROLLOUT_BATCH_SIZE=${IFRL_ROLLOUT_BATCH_SIZE:-32}
IFRL_SAMPLES_PER_PROMPT=${IFRL_SAMPLES_PER_PROMPT:-8}
IFRL_GLOBAL_BATCH_SIZE=${IFRL_GLOBAL_BATCH_SIZE:-256}
IFRL_STEPS_PER_ROLLOUT=${IFRL_STEPS_PER_ROLLOUT:-1}
IFRL_MAX_CONTEXT_LEN=${IFRL_MAX_CONTEXT_LEN:-4096}
IFRL_MAX_RESPONSE_LEN=${IFRL_MAX_RESPONSE_LEN:-1024}
IFRL_LR=${IFRL_LR:-1e-6}
IFRL_ADAM_BETA2=${IFRL_ADAM_BETA2:-0.98}
IFRL_DYNAMIC_FILTER=${IFRL_DYNAMIC_FILTER:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}
IFRL_LOAD_DIR=${IFRL_LOAD_DIR:-}
IFRL_LOAD_NO_OPTIM=${IFRL_LOAD_NO_OPTIM:-0}
IFRL_LOAD_NO_RNG=${IFRL_LOAD_NO_RNG:-0}
IFRL_LOAD_FINETUNE=${IFRL_LOAD_FINETUNE:-0}
IFRL_WANDB_PROJECT=${IFRL_WANDB_PROJECT:-slime-ifrl}
IFRL_WANDB_GROUP=${IFRL_WANDB_GROUP:-qwen3-30b-a3b-ifrl-3node}
RAY_CLUSTER_WAIT_MAX_ATTEMPTS=${RAY_CLUSTER_WAIT_MAX_ATTEMPTS:-240}
RAY_CLUSTER_WAIT_SLEEP_SECONDS=${RAY_CLUSTER_WAIT_SLEEP_SECONDS:-15}
RAY_CLUSTER_STATUS_TIMEOUT_SECONDS=${RAY_CLUSTER_STATUS_TIMEOUT_SECONDS:-30}
RAY_WORKER_JOIN_MAX_ATTEMPTS=${RAY_WORKER_JOIN_MAX_ATTEMPTS:-180}
RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS=${RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS:-15}
RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS=${RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS:-1800}
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-${RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS}}"

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

get_local_ip() {
  hostname -I | awk '{print $1}'
}
HAS_NVLINK="$(detect_nvlink)"

ROLLOUT_ARGS=(
  --prompt-data "${NORMALIZED_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --rollout-shuffle
  --num-epoch 1
  --rollout-batch-size "${IFRL_ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${IFRL_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${IFRL_MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${IFRL_MAX_RESPONSE_LEN}"
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --global-batch-size "${IFRL_GLOBAL_BATCH_SIZE}"
  --num-steps-per-rollout "${IFRL_STEPS_PER_ROLLOUT}"
  --dynamic-sampling-filter-path "${IFRL_DYNAMIC_FILTER}"
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${IFRL_LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 "${IFRL_ADAM_BETA2}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 8
  --sglang-mem-fraction-static 0.7
  --sglang-ep-size 8
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-rm-path reward_ifrl.reward_func
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-host "${WANDB_BASE_URL:-https://wandb.ai}"
    --wandb-project "${IFRL_WANDB_PROJECT}"
    --wandb-group "${IFRL_WANDB_GROUP}"
    --wandb-key "${WANDB_API_KEY}"
    --disable-wandb-random-suffix
    --wandb-dir "${WORK_ROOT}/wandb"
  )
fi

prepare_data() {
  python3 "${SCRIPT_DIR}/prepare_ifrl_data.py" \
    --source "${RAW_DATA}" \
    --dest "${NORMALIZED_DATA}"
}

submit_ray_job() {
  # shellcheck disable=SC1091
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

  LOAD_ARGS=()
  if [[ -n "${IFRL_LOAD_DIR}" ]]; then
    LOAD_ARGS+=(--load "${IFRL_LOAD_DIR}")
    if [[ "${IFRL_LOAD_NO_OPTIM}" == "1" ]]; then
      LOAD_ARGS+=(--no-load-optim)
    fi
    if [[ "${IFRL_LOAD_NO_RNG}" == "1" ]]; then
      LOAD_ARGS+=(--no-load-rng)
    fi
    if [[ "${IFRL_LOAD_FINETUNE}" == "1" ]]; then
      LOAD_ARGS+=(--finetune)
    fi
  fi

  RUNTIME_ENV_JSON="{
    \"env_vars\": {
      \"PYTHONPATH\": \"${SCRIPT_DIR}:${MEGATRON_PATH}:${SLIME_DIR}\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"MASTER_ADDR\": \"${MASTER_ADDR}\",
      \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
      \"WANDB_BASE_URL\": \"${WANDB_BASE_URL:-}\"
    }
  }"

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --ref-load "${TORCH_DIST_DIR}" \
    "${LOAD_ARGS[@]}" \
    --save "${SAVE_DIR}" \
    --save-interval 50 \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(get_local_ip)}}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
NODE_IP="$(get_local_ip)"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cleanup_local_processes

if [[ "${NODE_RANK}" -eq 0 ]]; then
  prepare_data
  ensure_torch_dist_checkpoint
  ray start --head \
    --port="${MASTER_PORT}" \
    --node-ip-address "${MASTER_ADDR}" \
    --node-name "${WORKER_ID:-${HOSTNAME:-head-0}}" \
    --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}"
  wait_for_full_ray_cluster
  submit_ray_job
  ray stop --force || true
else
  # Keep non-zero ranks alive: each worker pod should start its own Ray worker.
  sleep 5
  start_ray_worker_with_retry
  while ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; do
    sleep 60
  done
fi
