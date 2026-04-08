#!/bin/bash
# MOPD (Multi-Objective Policy Distillation) for Qwen3-30B-A3B on 3 H200 nodes.
#
# Student: tool45/struct25/stem30 iter_479
# Teachers: mdv2_main_retry@279 for tool/stem/structured, plus code and math teachers
# Data: read directly from normalized pool sources via a .list manifest
#
# Usage: bash exp/mopd/run_mopd_qwen3_30b_3node_h200.sh
#   (set env vars to override defaults, or export them in submit script)

set -euo pipefail

export PYTHONUNBUFFERED=1
pip install --break-system-packages math-verify 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

# shellcheck source=/dev/null
if [[ -f "${AVALANCHE_ROOT}/login.sh" ]]; then
  source "${AVALANCHE_ROOT}/login.sh"
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/ray_bootstrap_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/data_cache_reuse_utils.sh"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/training_runner_utils.sh"

# ---- Cluster layout ----
NUM_NODES=${NUM_NODES:-3}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-7}

# ---- Student checkpoint ----
STUDENT_EXP_DIR="${AVALANCHE_ROOT}/experiments/qwen3_30b_a3b_mdv1_3node_tool45_struct25_stem30_resume0219_0331-1803-fix-cachefix-iter219-cachedatafix-waitfix"
MODEL_DIR=${MODEL_DIR:-${STUDENT_EXP_DIR}/hf_cache/iter_0000479_hf}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${STUDENT_EXP_DIR}/checkpoints}
STUDENT_CKPT_STEP=${STUDENT_CKPT_STEP:-479}

# ---- Teacher checkpoints ----
TOOL_TEACHER_EXP_DIR=${TOOL_TEACHER_EXP_DIR:-${AVALANCHE_ROOT}/experiments/multidomain_v2_main_retry}
TOOL_TEACHER_CKPT_STEP=${TOOL_TEACHER_CKPT_STEP:-279}
TOOL_TEACHER_TORCH_DIST_DIR=${TOOL_TEACHER_TORCH_DIST_DIR:-${TOOL_TEACHER_EXP_DIR}/checkpoints}
TOOL_TEACHER_ORIGIN_HF=${TOOL_TEACHER_ORIGIN_HF:-${TOOL_TEACHER_EXP_DIR}/hf_cache/iter_0000299_hf}
TOOL_TEACHER_HF_DIR=${TOOL_TEACHER_HF_DIR:-${TOOL_TEACHER_EXP_DIR}/hf_cache/iter_0000279_hf}
CODE_TEACHER_HF_DIR=${CODE_TEACHER_HF_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jl_workspace/experiment/code/output/qwen3-30b-a3b-code-3node/hf_cache/iter_0000099_hf}
MATH_TEACHER_HF_DIR=${MATH_TEACHER_HF_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jl_workspace/experiment/math/output/qwen3-30b-a3b-deepmath-3node/hf_cache/iter_0000339_hf}
TEACHER_STEP0_EVAL=${TEACHER_STEP0_EVAL:-1}
TEACHER_STEP0_EVAL_PORT_BASE=${TEACHER_STEP0_EVAL_PORT_BASE:-31000}
TEACHER_STEP0_EVAL_BATCH_SIZE=${TEACHER_STEP0_EVAL_BATCH_SIZE:-32}

# ---- Work directory ----
WORK_ROOT=${WORK_ROOT:-${AVALANCHE_ROOT}/experiments/mopd_qwen3_30b_a3b_3node_h200_mdv2_retry279}

# ---- Data ----
TRAIN_POOL_ROOT=${TRAIN_POOL_ROOT:-${AVALANCHE_ROOT}/data/pool}
TRAIN_POOL_INCLUDE_DOMAINS=${TRAIN_POOL_INCLUDE_DOMAINS:-stem,structured,code,math}
TRAIN_DATASETS=${TRAIN_DATASETS:-}
TRAIN_DATASETS_EXTRA=${TRAIN_DATASETS_EXTRA:-}
TRAIN_PATHS=${TRAIN_PATHS:-}
TRAIN_PATHS_EXTRA=${TRAIN_PATHS_EXTRA:-}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-}
MOPD_STEM_TRAIN_DATASETS=${MOPD_STEM_TRAIN_DATASETS:-nemotron_knowledge_mcqa}
MOPD_STRUCTURED_TRAIN_DATASETS=${MOPD_STRUCTURED_TRAIN_DATASETS:-nemotron_structured_outputs}
MOPD_MATH_TRAIN_DATASETS=${MOPD_MATH_TRAIN_DATASETS:-deepmath,dapo,bigmath}
MOPD_CODE_TRAIN_DATASETS=${MOPD_CODE_TRAIN_DATASETS:-apps,code_contests,taco,codeforces}
TRAIN_POOL_EXCLUDE_PATTERNS=${TRAIN_POOL_EXCLUDE_PATTERNS:-stem/train/openbookqa,stem/train/scienceqa,stem/train/sciq,stem/train/ai2_arc,stem/train/aqua_rat,stem/train/mmlu_auxiliary,tool/train/xlam_function_calling_60k,stem/train/medmcqa_data_,structured/train/jsonschemabench_train-}
TRAIN_SOURCE_LIST_BASENAME=${TRAIN_SOURCE_LIST_BASENAME:-mopd_train_sources.list}

# ---- Eval ----
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-0}
EVAL_CONFIG_PATH="${WORK_ROOT}/data_cache/eval_config.yaml"
OFFICIAL_EVAL_MANIFEST_PATH="${WORK_ROOT}/data_cache/eval_official_benchmarks.json"
TEACHER_STEP0_INCLUDE_BFCL=${TEACHER_STEP0_INCLUDE_BFCL:-1}
EVAL_DATASETS=${EVAL_DATASETS:-aime24,math500,mbppplus,gpqa,ifeval,ifbench_test}
EVAL_DATASETS_EXTRA=${EVAL_DATASETS_EXTRA:-}
EVAL_PATHS=${EVAL_PATHS:-}
EVAL_PATHS_EXTRA=${EVAL_PATHS_EXTRA:-}

# ---- SGLang multimodel config ----
SGLANG_MULTIMODEL_CONFIG=${SGLANG_MULTIMODEL_CONFIG:-${SCRIPT_DIR}/sglang_mopd_qwen3_30b_3node_h200.yaml}
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
TOOL_CALL_WANDB_GROUP=${TOOL_CALL_WANDB_GROUP:-mopd-qwen3-30b-a3b-3node-h200-${MOPD_DOMAIN_SIGNATURE}}

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
    --profile "mopd"
    --stem-train-datasets "${MOPD_STEM_TRAIN_DATASETS}"
    --structured-train-datasets "${MOPD_STRUCTURED_TRAIN_DATASETS}"
    --math-train-datasets "${MOPD_MATH_TRAIN_DATASETS}"
    --code-train-datasets "${MOPD_CODE_TRAIN_DATASETS}"
  )
  parse_csv_to_args "${TRAIN_DATASETS}" --dataset materialize_args
  parse_csv_to_args "${TRAIN_DATASETS_EXTRA}" --dataset-extra materialize_args
  parse_csv_to_args "${TRAIN_PATHS}" --source materialize_args
  parse_csv_to_args "${TRAIN_PATHS_EXTRA}" --source-extra materialize_args
  if [[ -n "${TRAIN_MANIFEST}" ]]; then
    materialize_args+=(--manifest "${TRAIN_MANIFEST}")
  fi

  # Materialize pool data: bridge top-level supervision_family fields -> metadata.ground_truth/reward_type
  # so that downstream reward functions can consume the data directly.
  PYTHONPATH="${SLIME_DIR}:${PYTHONPATH:-}" python3 "${SCRIPT_DIR}/../prepare_runtime_dataset.py" train \
    --pool-root "${TRAIN_POOL_ROOT}" \
    --cache-dir "${DATA_CACHE_DIR}/materialized_train" \
    --manifest-output "${TRAIN_SOURCE_LIST}" \
    --include-domains "${TRAIN_POOL_INCLUDE_DOMAINS}" \
    --exclude-patterns "${TRAIN_POOL_EXCLUDE_PATTERNS}" \
    "${materialize_args[@]}"

  ensure_nonempty_file "${TRAIN_SOURCE_LIST}" "Training source manifest"
}

write_eval_config() {
  local eval_args=(
    --pool-root "${TRAIN_POOL_ROOT}"
    --output "${EVAL_CONFIG_PATH}"
    --official-manifest-output "${OFFICIAL_EVAL_MANIFEST_PATH}"
    --max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
  )
  parse_csv_to_args "${EVAL_DATASETS}" --dataset eval_args
  parse_csv_to_args "${EVAL_DATASETS_EXTRA}" --dataset-extra eval_args
  parse_csv_to_args "${EVAL_PATHS}" --source eval_args
  parse_csv_to_args "${EVAL_PATHS_EXTRA}" --source-extra eval_args

  python3 "${SCRIPT_DIR}/../prepare_runtime_dataset.py" eval-config "${eval_args[@]}"
}

ensure_hf_checkpoint_from_torch_dist() {
  local iter_dir="$1"
  local hf_dir="$2"
  local origin_hf_dir="$3"

  if [[ -f "${hf_dir}/config.json" ]]; then
    echo "Found HF checkpoint at ${hf_dir}"
    return 0
  fi

  mkdir -p "$(dirname "${hf_dir}")"
  echo "Converting torch_dist checkpoint ${iter_dir} -> ${hf_dir}"
  python3 "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${iter_dir}" \
    --output-dir "${hf_dir}" \
    --origin-hf-dir "${origin_hf_dir}" \
    --force
}

ensure_tool_teacher_hf_checkpoint() {
  local iter_name
  iter_name="$(printf 'iter_%07d' "${TOOL_TEACHER_CKPT_STEP}")"
  ensure_hf_checkpoint_from_torch_dist \
    "${TOOL_TEACHER_TORCH_DIST_DIR}/${iter_name}" \
    "${TOOL_TEACHER_HF_DIR}" \
    "${TOOL_TEACHER_ORIGIN_HF}"
}

load_eval_data_specs() {
  mapfile -t EVAL_DATA_SPECS < <(
    python3 "${SCRIPT_QUERIES_PY}" load-eval-config --path "${EVAL_CONFIG_PATH}"
  )

  if [[ "${TEACHER_STEP0_INCLUDE_BFCL}" == "1" ]] && [[ -f "${OFFICIAL_EVAL_MANIFEST_PATH}" ]]; then
    mapfile -t official_specs < <(
      python3 "${SCRIPT_QUERIES_PY}" load-json-manifest --path "${OFFICIAL_EVAL_MANIFEST_PATH}"
    )
    if [[ ${#official_specs[@]} -gt 0 ]]; then
      EVAL_DATA_SPECS+=("${official_specs[@]}")
    fi
  fi
}

stop_local_sglang_eval_server() {
  local port="$1"
  if [[ -n "${SGLANG_EVAL_PID:-}" ]]; then
    kill "${SGLANG_EVAL_PID}" >/dev/null 2>&1 || true
    wait "${SGLANG_EVAL_PID}" 2>/dev/null || true
    SGLANG_EVAL_PID=""
  fi
  pkill -f "sglang.launch_server.*--port ${port}" >/dev/null 2>&1 || true
  sleep 2
}

start_local_sglang_eval_server() {
  local label="$1"
  local hf_dir="$2"
  local port="$3"
  local tp_size="$4"
  local mem_fraction="$5"
  local log_file="${LOG_DIR}/${label}_sglang_step0.log"

  stop_local_sglang_eval_server "${port}"
  echo "Starting local sglang for ${label} at ${hf_dir} on port ${port}"
  python3 -m sglang.launch_server \
    --model-path "${hf_dir}" \
    --port "${port}" \
    --tp "${tp_size}" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static "${mem_fraction}" >"${log_file}" 2>&1 &
  SGLANG_EVAL_PID=$!

  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "sglang is ready for ${label} on port ${port}"
      return 0
    fi
    sleep 5
  done

  echo "sglang did not become ready for ${label}" >&2
  return 1
}

run_teacher_step0_eval() {
  local label="$1"
  local hf_dir="$2"
  local port="$3"
  local tp_size="$4"
  local mem_fraction="$5"
  local run_id
  local run_name
  local wandb_group
  local log_file="${LOG_DIR}/${label}_step0_eval.log"
  local eval_args=()

  for spec in "${EVAL_DATA_SPECS[@]}"; do
    eval_args+=(--eval-data "${spec}")
  done

  run_id="$(python3 "${SCRIPT_QUERIES_PY}" random-run-id)"
  wandb_group="$(normalize_wandb_group_name "${TOOL_CALL_WANDB_GROUP}")"
  run_name="${wandb_group}-${label}-step0"

  start_local_sglang_eval_server "${label}" "${hf_dir}" "${port}" "${tp_size}" "${mem_fraction}"
  PYTHONPATH="${SCRIPT_DIR}:${SLIME_DIR}/examples:${SLIME_DIR}:${PYTHONPATH:-}" \
    python3 "${SLIME_DIR}/examples/eval_backfill.py" \
      --sglang-url "http://127.0.0.1:${port}" \
      --model-path "${hf_dir}" \
      --reward-module reward_mopd_eval_router.reward_func \
      "${eval_args[@]}" \
      --rollout-id 0 \
      --runtime-data-dir "${DATA_CACHE_DIR}/eval" \
      --wandb-run-id "${run_id}" \
      --wandb-run-name "${run_name}" \
      --wandb-project "${TOOL_CALL_WANDB_PROJECT}" \
      --wandb-group "${wandb_group}" \
      --wandb-host "${WANDB_BASE_URL:-}" \
      --wandb-key "${WANDB_API_KEY:-}" \
      --max-context-len "${TOOLCALL_MAX_PROMPT_TOKENS}" \
      --max-tokens "${TOOLCALL_MAX_RESPONSE_LEN}" \
      --batch-size "${TEACHER_STEP0_EVAL_BATCH_SIZE}" > >(tee -a "${log_file}") 2>&1
  stop_local_sglang_eval_server "${port}"
}

run_teacher_step0_evals() {
  local base_port
  if [[ "${TEACHER_STEP0_EVAL}" != "1" ]]; then
    return 0
  fi

  ensure_tool_teacher_hf_checkpoint
  load_eval_data_specs

  if [[ ${#EVAL_DATA_SPECS[@]} -eq 0 ]]; then
    echo "No eval datasets discovered from ${EVAL_CONFIG_PATH}" >&2
    return 1
  fi

  base_port="${TEACHER_STEP0_EVAL_PORT_BASE}"
  run_teacher_step0_eval tool "${TOOL_TEACHER_HF_DIR}" "${base_port}" 1 0.72
  run_teacher_step0_eval code "${CODE_TEACHER_HF_DIR}" "$((base_port + 1))" 1 0.76
  run_teacher_step0_eval math "${MATH_TEACHER_HF_DIR}" "$((base_port + 2))" 1 0.76
}

# ---- Submit Ray job ----

submit_ray_job() {
  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

  HAS_NVLINK="$(detect_nvlink)"

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
  if [[ "${SKIP_EVAL_BEFORE_TRAIN}" == "1" ]]; then
    EVAL_ARGS+=(--skip-eval-before-train)
  fi

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
      --wandb-dir "${WORK_ROOT}/wandb"
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
  run_teacher_step0_evals
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
