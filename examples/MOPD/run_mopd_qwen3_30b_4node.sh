#!/bin/bash
# MOPD (Multi-Objective Policy Distillation) for Qwen3-30B-A3B on 4 nodes.
#
# Student: tool45/struct25/stem30 iter_479
# Teachers: tool50 for tool/stem/structured, plus code and math teachers
# Data: read directly from normalized pool sources via a .list manifest
#
# Usage: bash exp/mopd/run_mopd_qwen3_30b_4node.sh
#   (set env vars to override defaults, or export them in submit script)

set -euo pipefail

export PYTHONUNBUFFERED=1
pip install --break-system-packages math-verify 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/ray_bootstrap_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/data_cache_reuse_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/training_runner_utils.sh"

# ---- Cluster layout ----
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-16}

# ---- Student checkpoint ----
STUDENT_EXP_DIR="${AVALANCHE_ROOT}/experiments/qwen3_30b_a3b_mdv1_3node_tool45_struct25_stem30_resume0219_0331-1803-fix-cachefix-iter219-cachedatafix-waitfix"
MODEL_DIR=${MODEL_DIR:-${STUDENT_EXP_DIR}/hf_cache/iter_0000479_hf}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${STUDENT_EXP_DIR}/checkpoints}
STUDENT_CKPT_STEP=${STUDENT_CKPT_STEP:-479}

# ---- Work directory ----
WORK_ROOT=${WORK_ROOT:-${AVALANCHE_ROOT}/experiments/mopd_qwen3_30b_a3b_4node_pool_direct}

# ---- Data ----
TRAIN_POOL_ROOT=${TRAIN_POOL_ROOT:-${AVALANCHE_ROOT}/data/pool}
TRAIN_POOL_INCLUDE_DOMAINS=${TRAIN_POOL_INCLUDE_DOMAINS:-tool,stem,structured,code,math}
TRAIN_DATASETS=${TRAIN_DATASETS:-}
TRAIN_DATASETS_EXTRA=${TRAIN_DATASETS_EXTRA:-}
TRAIN_PATHS=${TRAIN_PATHS:-}
TRAIN_PATHS_EXTRA=${TRAIN_PATHS_EXTRA:-}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-}
TRAIN_POOL_EXCLUDE_PATTERNS=${TRAIN_POOL_EXCLUDE_PATTERNS:-stem/train/openbookqa,stem/train/scienceqa,stem/train/sciq,stem/train/ai2_arc,stem/train/aqua_rat,stem/train/mmlu_auxiliary,tool/train/xlam_function_calling_60k,structured/train/nemotron_structured_outputs,stem/train/medmcqa_data_,structured/train/jsonschemabench_train-}
TRAIN_SOURCE_LIST_BASENAME=${TRAIN_SOURCE_LIST_BASENAME:-mopd_train_sources.list}

# ---- Eval ----
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_CONFIG_PATH="${WORK_ROOT}/data_cache/eval_config.yaml"
EVAL_DATASETS=${EVAL_DATASETS:-}
EVAL_DATASETS_EXTRA=${EVAL_DATASETS_EXTRA:-}
EVAL_PATHS=${EVAL_PATHS:-}
EVAL_PATHS_EXTRA=${EVAL_PATHS_EXTRA:-}

# ---- SGLang multimodel config ----
SGLANG_MULTIMODEL_CONFIG=${SGLANG_MULTIMODEL_CONFIG:-${SCRIPT_DIR}/sglang_mopd_qwen3_30b.yaml}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.82}

# ---- Training hyperparameters ----
TOOLCALL_ROLLOUT_BATCH_SIZE=${TOOLCALL_ROLLOUT_BATCH_SIZE:-32}
TOOLCALL_SAMPLES_PER_PROMPT=${TOOLCALL_SAMPLES_PER_PROMPT:-16}
TOOLCALL_GLOBAL_BATCH_SIZE=${TOOLCALL_GLOBAL_BATCH_SIZE:-512}
TOOLCALL_STEPS_PER_ROLLOUT=${TOOLCALL_STEPS_PER_ROLLOUT:-1}
TOOLCALL_MAX_CONTEXT_LEN=${TOOLCALL_MAX_CONTEXT_LEN:-40960}
TOOLCALL_MAX_RESPONSE_LEN=${TOOLCALL_MAX_RESPONSE_LEN:-8192}
TOOLCALL_MAX_PROMPT_TOKENS=${TOOLCALL_MAX_PROMPT_TOKENS:-$((TOOLCALL_MAX_CONTEXT_LEN - TOOLCALL_MAX_RESPONSE_LEN))}
TOOLCALL_PARSER_TYPE=${TOOLCALL_PARSER_TYPE:-qwen3}
TOOLCALL_LR=${TOOLCALL_LR:-1e-6}
TOOLCALL_ADAM_BETA2=${TOOLCALL_ADAM_BETA2:-0.98}
TOOLCALL_USE_ROLLOUT_ROUTING_REPLAY=${TOOLCALL_USE_ROLLOUT_ROUTING_REPLAY:-0}

# ---- OPD hyperparameters ----
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.00}

# ---- Resume ----
TOOLCALL_RESUME_TRAINING=${TOOLCALL_RESUME_TRAINING:-1}
TOOLCALL_RESUME_NO_OPTIM=${TOOLCALL_RESUME_NO_OPTIM:-1}
TOOLCALL_RESUME_NO_RNG=${TOOLCALL_RESUME_NO_RNG:-1}
TOOLCALL_RESUME_FINETUNE=${TOOLCALL_RESUME_FINETUNE:-1}

# ---- W&B ----
TOOL_CALL_WANDB_PROJECT=${TOOL_CALL_WANDB_PROJECT:-slime-mopd}

# ---- Misc ----
SLIME_DIR=${SLIME_DIR:-${PROJECT_ROOT}/slime}
SCRIPT_QUERIES_PY="${SLIME_DIR}/examples/common/script_queries.py"
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

RAY_CLUSTER_WAIT_MAX_ATTEMPTS=${RAY_CLUSTER_WAIT_MAX_ATTEMPTS:-240}
RAY_CLUSTER_WAIT_SLEEP_SECONDS=${RAY_CLUSTER_WAIT_SLEEP_SECONDS:-15}
RAY_CLUSTER_STATUS_TIMEOUT_SECONDS=${RAY_CLUSTER_STATUS_TIMEOUT_SECONDS:-30}
RAY_WORKER_JOIN_MAX_ATTEMPTS=${RAY_WORKER_JOIN_MAX_ATTEMPTS:-180}
RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS=${RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS:-15}
RAY_HEAD_ADDR_WAIT_ATTEMPTS=${RAY_HEAD_ADDR_WAIT_ATTEMPTS:-900}
RAY_HEAD_ADDR_WAIT_SLEEP=${RAY_HEAD_ADDR_WAIT_SLEEP:-2}
RAY_HEAD_START_STATUS_MAX_ATTEMPTS=${RAY_HEAD_START_STATUS_MAX_ATTEMPTS:-120}
RAY_HEAD_START_STATUS_SLEEP_SECONDS=${RAY_HEAD_START_STATUS_SLEEP_SECONDS:-5}
RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS=${RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS:-1800}
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-${RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS}}"
RAY_HEAD_ADDR_FILE="${WORK_ROOT}/ray_head_addr.txt"
RAY_HEAD_LOCK_DIR="${WORK_ROOT}/ray_head_lock"

DATA_CACHE_DIR="${WORK_ROOT}/data_cache"
LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
TRAIN_SOURCE_LIST="${DATA_CACHE_DIR}/${TRAIN_SOURCE_LIST_BASENAME}"
TRACE_DIR="${WORK_ROOT}/rollout_traces"
TRACE_MAX_SAMPLES=${TRACE_MAX_SAMPLES:-8}
OPD_DOMAIN_MODEL_MAP=${OPD_DOMAIN_MODEL_MAP:-tool:tool,stem:tool,structured:tool,code:code,math:math}
export MULTIDOMAIN_V1_TRACE_DIR="${TRACE_DIR}"
export MULTIDOMAIN_V1_TRACE_MAX_SAMPLES="${TRACE_MAX_SAMPLES}"

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}" "${TRACE_DIR}"

MOPD_DOMAIN_SIGNATURE="$(python3 "${SCRIPT_QUERIES_PY}" domain-signature --domains "${TRAIN_POOL_INCLUDE_DOMAINS}")"
TOOL_CALL_WANDB_GROUP=${TOOL_CALL_WANDB_GROUP:-mopd-qwen3-30b-a3b-4node-${MOPD_DOMAIN_SIGNATURE}}

BOOTSTRAP_NODE_ID="${WORKER_ID:-${HOSTNAME:-node-${NODE_RANK:-unknown}}}"
BOOTSTRAP_LOG_FILE="${LOG_DIR}/bootstrap_${BOOTSTRAP_NODE_ID}.log"
exec > >(tee -a "${BOOTSTRAP_LOG_FILE}") 2>&1

echo "MOPD training domains: ${MOPD_DOMAIN_SIGNATURE}"

# ---- Data source manifest ----

prepare_training_source_list() {
  if [[ ! -d "${TRAIN_POOL_ROOT}" ]]; then
    echo "TRAIN_POOL_ROOT does not exist: ${TRAIN_POOL_ROOT}" >&2
    return 1
  fi

  local materialize_args=(
    "${TRAIN_POOL_ROOT}"
    "${DATA_CACHE_DIR}/materialized_train"
    "${TRAIN_SOURCE_LIST}"
    "${TRAIN_POOL_INCLUDE_DOMAINS}"
    "${TRAIN_POOL_EXCLUDE_PATTERNS}"
  )
  local item
  IFS=',' read -r -a _train_datasets <<< "${TRAIN_DATASETS}"
  for item in "${_train_datasets[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      materialize_args+=(--dataset "${item}")
    fi
  done
  IFS=',' read -r -a _train_dataset_extras <<< "${TRAIN_DATASETS_EXTRA}"
  for item in "${_train_dataset_extras[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      materialize_args+=(--dataset-extra "${item}")
    fi
  done
  IFS=',' read -r -a _train_paths <<< "${TRAIN_PATHS}"
  for item in "${_train_paths[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      materialize_args+=(--source "${item}")
    fi
  done
  IFS=',' read -r -a _train_path_extras <<< "${TRAIN_PATHS_EXTRA}"
  for item in "${_train_path_extras[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      materialize_args+=(--source-extra "${item}")
    fi
  done
  if [[ -n "${TRAIN_MANIFEST}" ]]; then
    materialize_args+=(--manifest "${TRAIN_MANIFEST}")
  fi

  # Materialize pool data: bridge top-level supervision_family fields -> metadata.ground_truth/reward_type
  # so that downstream reward functions can consume the data directly.
  PYTHONPATH="${SLIME_DIR}:${PYTHONPATH:-}" python3 "${SCRIPT_DIR}/materialize_train_pool.py" "${materialize_args[@]}"

  ensure_nonempty_file "${TRAIN_SOURCE_LIST}" "Training source manifest"
}

write_eval_config() {
  local eval_args=(
    --pool-root "${TRAIN_POOL_ROOT}"
    --output "${EVAL_CONFIG_PATH}"
    --max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
  )
  local item
  IFS=',' read -r -a _eval_datasets <<< "${EVAL_DATASETS}"
  for item in "${_eval_datasets[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--dataset "${item}")
    fi
  done
  IFS=',' read -r -a _eval_dataset_extras <<< "${EVAL_DATASETS_EXTRA}"
  for item in "${_eval_dataset_extras[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--dataset-extra "${item}")
    fi
  done
  IFS=',' read -r -a _eval_paths <<< "${EVAL_PATHS}"
  for item in "${_eval_paths[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--source "${item}")
    fi
  done
  IFS=',' read -r -a _eval_path_extras <<< "${EVAL_PATHS_EXTRA}"
  for item in "${_eval_path_extras[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      eval_args+=(--source-extra "${item}")
    fi
  done

  python3 "${SCRIPT_DIR}/write_eval_config.py" "${eval_args[@]}"
}

# ---- Checkpoint ----

ensure_torch_dist_checkpoint() {
  if [ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Found torch_dist checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi
  if [ -f "${TORCH_DIST_DIR}/common.pt" ] && [ -f "${TORCH_DIST_DIR}/metadata.json" ]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"
  PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}" torchrun \
    --nproc-per-node "${NUM_GPUS_PER_NODE}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
}

# ---- Submit Ray job ----

submit_ray_job() {
  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
  if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi

  # ---- Load args: resume from student checkpoint ----
  LOAD_ARGS=(
    --load "${TORCH_DIST_DIR}"
    --ckpt-step "${STUDENT_CKPT_STEP}"
    --no-load-optim
    --no-load-rng
    --finetune
  )

  # ---- Rollout args ----
  ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_SOURCE_LIST}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --tool-key tools
    --apply-chat-template
    --rollout-shuffle
    --num-epoch 1
    --rollout-batch-size "${TOOLCALL_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${TOOLCALL_SAMPLES_PER_PROMPT}"
    --rollout-max-context-len "${TOOLCALL_MAX_CONTEXT_LEN}"
    --rollout-max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --global-batch-size "${TOOLCALL_GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout "${TOOLCALL_STEPS_PER_ROLLOUT}"
    --balance-data
  )

  # ---- Megatron parallelism (training) ----
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

  # ---- GRPO + OPD args ----
  GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type sglang
    --opd-kl-coef "${OPD_KL_COEF}"
    --use-kl-loss
    --kl-loss-coef "${KL_LOSS_COEF}"
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
  )

  # ---- Optimizer ----
  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${TOOLCALL_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 "${TOOLCALL_ADAM_BETA2}"
  )

  # ---- SGLang: multimodel with per-engine 1 GPU ----
  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-ep-size 1
    --sglang-config "${SGLANG_MULTIMODEL_CONFIG}"
  )

  # ---- Misc ----
  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )
  if [[ "${TOOLCALL_USE_ROLLOUT_ROUTING_REPLAY}" == "1" ]]; then
    MISC_ARGS+=(--use-rollout-routing-replay)
  fi

  # ---- Eval ----
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-config "${EVAL_CONFIG_PATH}"
    --eval-function-path examples.MOPD.reward_mopd_eval_router.generate_eval_rollout
  )

  # ---- Reward: OPD route-by-domain ----
  CUSTOM_ARGS=(
    --custom-rm-path examples.MOPD.reward_func_mopd.reward_func_route_by_domain
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
    --custom-rollout-log-function-path log_rollout.log_rollout_data
    --custom-eval-rollout-log-function-path log_rollout.log_eval_rollout_data
  )

  # ---- W&B ----
  WANDB_ARGS=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_GROUP_NAME="$(normalize_wandb_group_name "${TOOL_CALL_WANDB_GROUP}")"
    WANDB_ARGS+=(
      --use-wandb
      --wandb-host "${WANDB_BASE_URL:-https://wandb.ai}"
      --wandb-project "${TOOL_CALL_WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP_NAME}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
    )
  fi

  MOPD_DIR="${SLIME_DIR}/examples/MOPD"
  RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${MOPD_DIR}:${SLIME_DIR}/examples:${MEGATRON_PATH}:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\",\"MULTIDOMAIN_V1_TRACE_DIR\":\"${MULTIDOMAIN_V1_TRACE_DIR}\",\"MULTIDOMAIN_V1_TRACE_MAX_SAMPLES\":\"${MULTIDOMAIN_V1_TRACE_MAX_SAMPLES}\",\"OPD_DOMAIN_MODEL_MAP\":\"${OPD_DOMAIN_MODEL_MAP}\"}}"

  TRAINING_RESOURCE_ARGS=(
    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}"
  )

  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    "${TRAINING_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --ref-load "${TORCH_DIST_DIR}" \
    "${LOAD_ARGS[@]}" \
    --save "${SAVE_DIR}" \
    --save-interval 10 \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

# ---- Main orchestration ----

NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
IS_RAY_HEAD=0

# Clean stale ray_head_lock/addr from previous runs so head election succeeds.
# Only rank 0 cleans to avoid a race where one node removes another's fresh lock.
if [[ "${NODE_RANK}" -eq 0 ]]; then
  if [[ -d "${RAY_HEAD_LOCK_DIR}" ]]; then
    echo "Rank 0: cleaning stale ray_head_lock from previous run."
    rm -rf "${RAY_HEAD_LOCK_DIR}"
  fi
  rm -f "${RAY_HEAD_ADDR_FILE}"
fi
sleep 3  # let rank 0 finish cleanup before all nodes race to elect

elect_ray_head_role

if [[ "${IS_RAY_HEAD}" -eq 1 ]]; then
  rm -f "${RAY_HEAD_ADDR_FILE}"
else
  MASTER_ADDR="$(resolve_worker_master_addr "${MASTER_ADDR}" "${RAY_HEAD_ADDR_FILE}" 2>/dev/null || true)"
  if [[ -z "${MASTER_ADDR}" ]]; then
    echo "Failed to determine Ray head address for worker rank ${NODE_RANK}." >&2
    exit 1
  fi
fi

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cleanup_local_processes

if [[ "${IS_RAY_HEAD}" -eq 1 ]]; then
  start_ray_head_and_persist_addr
  prepare_training_source_list
  write_eval_config
  ensure_torch_dist_checkpoint
  wait_for_full_ray_cluster
  submit_ray_job
  ray stop --force || true
else
  sleep 5
  start_ray_worker_with_retry
  while ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; do
    sleep 60
  done
fi
