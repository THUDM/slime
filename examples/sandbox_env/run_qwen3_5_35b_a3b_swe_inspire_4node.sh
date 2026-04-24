#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_EXAMPLES_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SLIME_DIR_DEFAULT="$(cd -- "${SLIME_EXAMPLES_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SLIME_DIR_DEFAULT}/.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${WORKSPACE_ROOT}/.." && pwd)"

# shellcheck disable=SC1091
source "${WORKSPACE_ROOT}/login.sh"

NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-16}

MODEL_NAME=${MODEL_NAME:-Qwen3.5-35B-A3B}
MODEL_DIR=${MODEL_DIR:-${AVALANCHE_ROOT}/models/${MODEL_NAME}}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${AVALANCHE_ROOT}/models/${MODEL_NAME}_torch_dist}
WORK_ROOT_BASE=${WORK_ROOT_BASE:-${SCRIPT_DIR}/output_inspire_qwen3_5_35b_4node}

REBENCH_TASKS_JSON=${REBENCH_TASKS_JSON:-${AVALANCHE_ROOT}/data/raw_data/single/swe_rebench_v2/data/train-00000-of-00001.json}

SLIME_DIR=${SLIME_DIR:-${SLIME_DIR_DEFAULT}}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
ROCK_ROOT=${ROCK_ROOT:-${AVALANCHE_ROOT}/ROCK}
INSPIRE_SANDBOX_SITE_PACKAGES_DEFAULT="${AVALANCHE_ROOT}/.local/share/inspire_sandbox_site_packages"
INSPIRE_SANDBOX_SITE_PACKAGES=${INSPIRE_SANDBOX_SITE_PACKAGES:-${INSPIRE_SANDBOX_SITE_PACKAGES_DEFAULT}}
ROCK_SWE_SANDBOX_BACKEND=${ROCK_SWE_SANDBOX_BACKEND:-inspire}
ROCK_INSPIRE_SPEC=${ROCK_INSPIRE_SPEC:-G_C4}
ROCK_SWE_EVAL_SANDBOX_SOURCE=${ROCK_SWE_EVAL_SANDBOX_SOURCE:-template}
DEFAULT_ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST="${AVALANCHE_ROOT}/data/raw_data/single/swe_rebench_v2/data/prefetch_image_template_success.jsonl"
DEFAULT_ROCK_SWE_AGENT_CONFIG_PATH="${SCRIPT_DIR}/rock_agent_qwen_rebench_template.yaml"
ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST=${ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST:-${DEFAULT_ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST}}
ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES=${ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES:-1}
ROCK_SWE_AGENT_CONFIG_PATH=${ROCK_SWE_AGENT_CONFIG_PATH:-${DEFAULT_ROCK_SWE_AGENT_CONFIG_PATH}}
export ROCK_SWE_AGENT_CONFIG_PATH

SWE_TRAIN_MAX_PER_SOURCE=${SWE_TRAIN_MAX_PER_SOURCE:--1}
SWE_VAL_MAX_PER_SOURCE=${SWE_VAL_MAX_PER_SOURCE:-0}
SWE_SHUFFLE_DATA=${SWE_SHUFFLE_DATA:-0}
SWE_DATA_SEED=${SWE_DATA_SEED:-42}

SWE_NUM_ROLLOUT=${SWE_NUM_ROLLOUT:-200}
SWE_ROLLOUT_BATCH_SIZE=${SWE_ROLLOUT_BATCH_SIZE:-6}
SWE_SAMPLES_PER_PROMPT=${SWE_SAMPLES_PER_PROMPT:-4}
SWE_GLOBAL_BATCH_SIZE=${SWE_GLOBAL_BATCH_SIZE:-}
SWE_MICRO_BATCH_SIZE=${SWE_MICRO_BATCH_SIZE:-1}
SWE_STEPS_PER_ROLLOUT=${SWE_STEPS_PER_ROLLOUT:-1}
SWE_MAX_CONTEXT_LEN=${SWE_MAX_CONTEXT_LEN:-200000}
SWE_MAX_RESPONSE_LEN=${SWE_MAX_RESPONSE_LEN:-4096}
SWE_LR=${SWE_LR:-1e-6}
SWE_ADAM_BETA2=${SWE_ADAM_BETA2:-0.98}
SWE_KL_LOSS_COEF=${SWE_KL_LOSS_COEF:-0.0}
SWE_RESUME_TRAINING=${SWE_RESUME_TRAINING:-auto}
SWE_LOAD_DIR=${SWE_LOAD_DIR:-}
SWE_USE_WANDB=${SWE_USE_WANDB:-1}
SWE_WANDB_PROJECT=${SWE_WANDB_PROJECT:-slime-swe}
SWE_WANDB_GROUP=${SWE_WANDB_GROUP:-qwen3.5-35b-a3b-swe-inspire-4node}
SWE_WANDB_RUN_ID=${SWE_WANDB_RUN_ID:-qwen3.5-35b-a3b-swe-inspire-4node}
SWE_RUN_NAME=${SWE_RUN_NAME:-${SWE_WANDB_RUN_ID}}
SWE_ROLLOUT_NUM_GPUS_PER_ENGINE=${SWE_ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
SWE_SGLANG_EP_SIZE=${SWE_SGLANG_EP_SIZE:-8}
SWE_SGLANG_MEM_FRACTION_STATIC=${SWE_SGLANG_MEM_FRACTION_STATIC:-0.7}
SWE_MAX_TOKENS_PER_GPU=${SWE_MAX_TOKENS_PER_GPU:-1024}
SWE_TENSOR_MODEL_PARALLEL_SIZE=${SWE_TENSOR_MODEL_PARALLEL_SIZE:-2}
SWE_CONTEXT_PARALLEL_SIZE=${SWE_CONTEXT_PARALLEL_SIZE:-8}
SWE_EXPERT_MODEL_PARALLEL_SIZE=${SWE_EXPERT_MODEL_PARALLEL_SIZE:-8}
SWE_OPTIMIZER_CPU_OFFLOAD=${SWE_OPTIMIZER_CPU_OFFLOAD:-1}
SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D=${SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}
SWE_USE_PRECISION_AWARE_OPTIMIZER=${SWE_USE_PRECISION_AWARE_OPTIMIZER:-1}
SWE_DEBUG_ROLLOUT_ONLY=${SWE_DEBUG_ROLLOUT_ONLY:-0}
FORCE_REBUILD_TORCH_DIST=${FORCE_REBUILD_TORCH_DIST:-0}

ROCK_SWE_GROUP_CONCURRENCY=${ROCK_SWE_GROUP_CONCURRENCY:-${SWE_ROLLOUT_BATCH_SIZE}}
ROCK_SWE_SAMPLE_CONCURRENCY=${ROCK_SWE_SAMPLE_CONCURRENCY:-$(( SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT ))}
ROCK_SWE_OVER_SAMPLING_BATCH_SIZE=${ROCK_SWE_OVER_SAMPLING_BATCH_SIZE:-9}

RUN_DIR_NAME="${SWE_WANDB_GROUP//\//_}"
RUN_DIR_NAME="${RUN_DIR_NAME// /_}"
WORK_ROOT=${WORK_ROOT:-${WORK_ROOT_BASE}/${RUN_DIR_NAME}}
DATA_CACHE_DIR="${WORK_ROOT}/data_cache"
LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
NORMALIZED_TRAIN="${DATA_CACHE_DIR}/swe_train.normalized.jsonl"
NORMALIZED_VAL="${DATA_CACHE_DIR}/swe_val.normalized.jsonl"
JOB_ENTRYPOINT_SCRIPT="${WORK_ROOT}/run_train_async_job.sh"
ROCK_SWE_LOG_ROOT=${ROCK_SWE_LOG_ROOT:-${LOG_DIR}}

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

if [[ ! -d "${INSPIRE_SANDBOX_SITE_PACKAGES}" ]]; then
  echo "INSPIRE_SANDBOX_SITE_PACKAGES not found: ${INSPIRE_SANDBOX_SITE_PACKAGES}" >&2
  echo "Expected shared inspire-sandbox packages under ${INSPIRE_SANDBOX_SITE_PACKAGES_DEFAULT}" >&2
  exit 1
fi

DEFAULT_SWE_LOAD_DIR=""
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  DEFAULT_SWE_LOAD_DIR="${SAVE_DIR}"
fi
if [[ "${SWE_RESUME_TRAINING}" == "auto" ]]; then
  if [[ -n "${DEFAULT_SWE_LOAD_DIR}" ]]; then
    SWE_RESUME_TRAINING=1
  else
    SWE_RESUME_TRAINING=0
  fi
fi
if [[ -z "${SWE_LOAD_DIR}" && -n "${DEFAULT_SWE_LOAD_DIR}" ]]; then
  SWE_LOAD_DIR="${DEFAULT_SWE_LOAD_DIR}"
fi

RUN_LOG_PATH="${LOG_DIR}/run.log"
exec > >(tee "${RUN_LOG_PATH}") 2>&1

cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
}

get_local_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' || hostname
}

wait_for_full_ray_cluster() {
  local expected_gpus=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
  local expected_nodes=${NUM_NODES}
  local attempt

  for attempt in $(seq 1 300); do
    if python3 - <<PY
import json
import sys
import urllib.request

expected_gpus = ${expected_gpus}
expected_nodes = ${expected_nodes}

try:
    with urllib.request.urlopen("http://127.0.0.1:${DASHBOARD_PORT}/api/cluster_status", timeout=5) as response:
        payload = json.load(response)
except Exception:
    sys.exit(1)

report = payload.get("data", {}).get("clusterStatus", {}).get("autoscalerReport", {})
active_nodes = report.get("activeNodes") or {}
usage = payload.get("data", {}).get("clusterStatus", {}).get("loadMetricsReport", {}).get("usage", {})
gpu_total = (usage.get("GPU") or [0.0, 0.0])[1]

if len(active_nodes) >= expected_nodes and gpu_total >= expected_gpus:
    sys.exit(0)

sys.exit(1)
PY
    then
      echo "Ray cluster is ready with ${expected_nodes} nodes and ${expected_gpus} GPUs."
      return 0
    fi
    echo "Waiting for Ray workers to join (${attempt}/300)..."
    sleep 10
  done

  echo "Ray workers did not join the cluster in time." >&2
  ray status --address="${MASTER_ADDR}:${MASTER_PORT}" || true
  return 1
}

start_ray_worker_with_retry() {
  local attempt
  local rc
  local node_name="${WORKER_ID:-${HOSTNAME:-worker-${NODE_RANK}}}"

  for attempt in $(seq 1 150); do
    set +e
    ray start \
      --address="${MASTER_ADDR}:${MASTER_PORT}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --node-ip-address "${NODE_IP}" \
      --node-name "${node_name}" \
      --dashboard-port="${DASHBOARD_PORT}" \
      --disable-usage-stats
    rc=$?
    set -e

    if [[ "${rc}" -eq 0 ]]; then
      echo "Ray worker joined on attempt ${attempt}."
      return 0
    fi

    echo "Ray worker join failed on attempt ${attempt}, retrying..."
    ray stop --force 2>/dev/null || true
    sleep 5
  done

  echo "Ray worker failed to join cluster after retries." >&2
  return 1
}

detect_nvlink() {
  local count
  count="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
  if [[ "${count}" -gt 0 ]]; then
    printf '1'
  else
    printf '0'
  fi
}

override_qwen35_model_args() {
  local i
  for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[i]}" in
      --ffn-hidden-size)
        MODEL_ARGS[i+1]=5632
        ;;
      --num-query-groups)
        MODEL_ARGS[i+1]=2
        ;;
      --hidden-size)
        MODEL_ARGS[i+1]=2048
        ;;
      --num-experts)
        MODEL_ARGS[i+1]=256
        ;;
    esac
  done
}

prepare_data() {
  DATA_ARGS=(
    --tasks-json "${REBENCH_TASKS_JSON}"
    --train-dest "${NORMALIZED_TRAIN}"
    --val-dest "${NORMALIZED_VAL}"
    --train-max "${SWE_TRAIN_MAX_PER_SOURCE}"
    --val-max "${SWE_VAL_MAX_PER_SOURCE}"
    --seed "${SWE_DATA_SEED}"
    --conversation-prompt
  )
  if [[ "${SWE_SHUFFLE_DATA}" == "1" ]]; then
    DATA_ARGS+=(--shuffle)
  fi
  if [[ -n "${ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST}" ]]; then
    DATA_ARGS+=(--consumable-template-manifest "${ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST}")
  fi
  if [[ "${ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES}" == "1" ]]; then
    DATA_ARGS+=(--require-consumable-templates)
  fi
  python3 "${SCRIPT_DIR}/build_rebench_runtime_data.py" "${DATA_ARGS[@]}"
}

ensure_torch_dist_checkpoint() {
  if [[ "${FORCE_REBUILD_TORCH_DIST}" == "1" ]]; then
    echo "FORCE_REBUILD_TORCH_DIST=1, rebuilding ${TORCH_DIST_DIR}"
    rm -rf "${TORCH_DIST_DIR}"
  fi

  if [[ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Found torch_dist checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi
  if [[ -f "${TORCH_DIST_DIR}/common.pt" && -f "${TORCH_DIST_DIR}/metadata.json" ]]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${SLIME_DIR}"
  # shellcheck disable=SC1091
  source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
  override_qwen35_model_args
  PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}:${PYTHONPATH:-}" torchrun \
    --nproc-per-node "${NUM_GPUS_PER_NODE}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
}

validate_layout() {
  local rollout_product=$(( SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT ))
  local actor_total_gpus=$(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE ))
  local tensor_model_parallel_size=${SWE_TENSOR_MODEL_PARALLEL_SIZE}
  local expert_model_parallel_size=${SWE_EXPERT_MODEL_PARALLEL_SIZE}
  local num_query_groups=2
  local hidden_size=2048
  local num_experts=256

  if (( NUM_NODES <= 0 || NUM_GPUS_PER_NODE <= 0 )); then
    echo "NUM_NODES and NUM_GPUS_PER_NODE must be positive." >&2
    exit 1
  fi
  if (( ACTOR_NUM_NODES <= 0 || ACTOR_GPUS_PER_NODE <= 0 )); then
    echo "ACTOR_NUM_NODES and ACTOR_GPUS_PER_NODE must be positive." >&2
    exit 1
  fi
  if (( ROLLOUT_GPUS_TOTAL <= 0 )); then
    echo "ROLLOUT_GPUS_TOTAL must be positive." >&2
    exit 1
  fi
  if (( ACTOR_NUM_NODES > NUM_NODES )); then
    echo "ACTOR_NUM_NODES cannot exceed NUM_NODES." >&2
    exit 1
  fi
  if (( ACTOR_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
    echo "ACTOR_GPUS_PER_NODE cannot exceed NUM_GPUS_PER_NODE." >&2
    exit 1
  fi
  if (( actor_total_gpus + ROLLOUT_GPUS_TOTAL > NUM_NODES * NUM_GPUS_PER_NODE )); then
    echo "Actor GPUs plus rollout GPUs exceed the total cluster GPUs." >&2
    exit 1
  fi
  if (( SWE_STEPS_PER_ROLLOUT <= 0 )); then
    echo "SWE_STEPS_PER_ROLLOUT must be positive." >&2
    exit 1
  fi
  if (( rollout_product % SWE_STEPS_PER_ROLLOUT != 0 )); then
    echo "rollout_batch_size * n_samples_per_prompt must be divisible by num_steps_per_rollout." >&2
    exit 1
  fi
  if (( tensor_model_parallel_size <= 0 || expert_model_parallel_size <= 0 )); then
    echo "TP and EP must be positive." >&2
    exit 1
  fi
  if (( num_query_groups % tensor_model_parallel_size != 0 )); then
    echo "Qwen3.5-35B-A3B num_query_groups=2 must be divisible by TP." >&2
    exit 1
  fi
  if (( hidden_size % tensor_model_parallel_size != 0 )); then
    echo "Qwen3.5-35B-A3B hidden_size=2048 must be divisible by TP." >&2
    exit 1
  fi
  if (( num_experts % expert_model_parallel_size != 0 )); then
    echo "Qwen3.5-35B-A3B num_experts=256 must be divisible by EP." >&2
    exit 1
  fi
  if (( actor_total_gpus % tensor_model_parallel_size != 0 )); then
    echo "Actor GPU topology must be divisible by TP." >&2
    exit 1
  fi
  if (( actor_total_gpus % SWE_CONTEXT_PARALLEL_SIZE != 0 )); then
    echo "Actor GPU topology must be divisible by CP." >&2
    exit 1
  fi
  if (( actor_total_gpus % (tensor_model_parallel_size * SWE_CONTEXT_PARALLEL_SIZE) != 0 )); then
    echo "Actor GPU topology must be divisible by TP*CP." >&2
    exit 1
  fi
  if (( actor_total_gpus % expert_model_parallel_size != 0 )); then
    echo "Actor GPU topology must be divisible by EP." >&2
    exit 1
  fi
  if (( ROLLOUT_GPUS_TOTAL % SWE_ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "ROLLOUT_GPUS_TOTAL must be divisible by SWE_ROLLOUT_NUM_GPUS_PER_ENGINE." >&2
    exit 1
  fi
  if (( SWE_ROLLOUT_NUM_GPUS_PER_ENGINE % SWE_SGLANG_EP_SIZE != 0 )); then
    echo "SWE_ROLLOUT_NUM_GPUS_PER_ENGINE must be divisible by SWE_SGLANG_EP_SIZE." >&2
    exit 1
  fi
  if (( num_experts % SWE_SGLANG_EP_SIZE != 0 )); then
    echo "Qwen3.5-35B-A3B num_experts=256 must be divisible by SGLang EP." >&2
    exit 1
  fi
  if [[ -z "${SWE_GLOBAL_BATCH_SIZE}" ]]; then
    SWE_GLOBAL_BATCH_SIZE=$(( rollout_product / SWE_STEPS_PER_ROLLOUT ))
  fi
  local expected_global_batch_size=$(( rollout_product / SWE_STEPS_PER_ROLLOUT ))
  if (( SWE_GLOBAL_BATCH_SIZE != expected_global_batch_size )); then
    echo "SWE_GLOBAL_BATCH_SIZE=${SWE_GLOBAL_BATCH_SIZE} is inconsistent; expected ${expected_global_batch_size}." >&2
    exit 1
  fi
  local actor_data_parallel_size=$(( actor_total_gpus / (tensor_model_parallel_size * SWE_CONTEXT_PARALLEL_SIZE) ))
  if (( actor_data_parallel_size <= 0 || SWE_GLOBAL_BATCH_SIZE % actor_data_parallel_size != 0 )); then
    echo "SWE_GLOBAL_BATCH_SIZE (${SWE_GLOBAL_BATCH_SIZE}) must be divisible by actor DP (${actor_data_parallel_size})." >&2
    exit 1
  fi
  if (( ROCK_SWE_GROUP_CONCURRENCY <= 0 || ROCK_SWE_GROUP_CONCURRENCY > SWE_ROLLOUT_BATCH_SIZE )); then
    echo "ROCK_SWE_GROUP_CONCURRENCY must be in (0, SWE_ROLLOUT_BATCH_SIZE]." >&2
    exit 1
  fi
  if (( ROCK_SWE_SAMPLE_CONCURRENCY < SWE_SAMPLES_PER_PROMPT )); then
    echo "ROCK_SWE_SAMPLE_CONCURRENCY must be >= SWE_SAMPLES_PER_PROMPT." >&2
    exit 1
  fi
}

submit_ray_job() {
  cd "${SLIME_DIR}"
  # shellcheck disable=SC1091
  source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
  override_qwen35_model_args

  local has_nvlink
  has_nvlink="$(detect_nvlink)"

  LOAD_ARGS=()
  if [[ "${SWE_RESUME_TRAINING}" == "1" && -n "${SWE_LOAD_DIR}" ]]; then
    LOAD_ARGS+=(--load "${SWE_LOAD_DIR}")
  fi

  ROLLOUT_ARGS=(
    --prompt-data "${NORMALIZED_TRAIN}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --rollout-shuffle
    --num-rollout "${SWE_NUM_ROLLOUT}"
    --rollout-batch-size "${SWE_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${SWE_SAMPLES_PER_PROMPT}"
    --rollout-max-context-len "${SWE_MAX_CONTEXT_LEN}"
    --rollout-max-response-len "${SWE_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 0.95
    --rollout-stop "</tool_call>" "</tool_call>\n" "\n</tool_call>\n" "\n</function>"
    --global-batch-size "${SWE_GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${SWE_MICRO_BATCH_SIZE}"
    --num-steps-per-rollout "${SWE_STEPS_PER_ROLLOUT}"
    --loss-mask-type qwen3_5
    --rollout-function-path examples.sandbox_env.rock_swe_rollout.generate_rollout
  )

  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${SWE_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 "${SWE_ADAM_BETA2}"
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef "${SWE_KL_LOSS_COEF}"
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
  )

  PERF_ARGS=(
    --tensor-model-parallel-size "${SWE_TENSOR_MODEL_PARALLEL_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${SWE_CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${SWE_EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${SWE_MAX_TOKENS_PER_GPU}"
    --log-probs-chunk-size 512
    --recompute-loss-function
  )

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${SWE_ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SWE_SGLANG_MEM_FRACTION_STATIC}"
    --sglang-ep-size "${SWE_SGLANG_EP_SIZE}"
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
  )

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )

  WANDB_ARGS=()
  if [[ "${SWE_USE_WANDB}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS+=(
      --use-wandb
      --wandb-host "${WANDB_BASE_URL:-https://wandb.ai}"
      --wandb-project "${SWE_WANDB_PROJECT}"
      --wandb-group "${SWE_WANDB_GROUP}"
      --wandb-run-id "${SWE_WANDB_RUN_ID}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
    )
  fi

  DEBUG_ARGS=()
  if [[ "${SWE_DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
    DEBUG_ARGS+=(--debug-rollout-only)
  fi

  OFFLOAD_ARGS=()
  if [[ "${SWE_OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
    OFFLOAD_ARGS+=(--optimizer-cpu-offload)
  fi
  if [[ "${SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D}" == "1" ]]; then
    OFFLOAD_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
  fi
  if [[ "${SWE_USE_PRECISION_AWARE_OPTIMIZER}" == "1" ]]; then
    OFFLOAD_ARGS+=(--use-precision-aware-optimizer)
  fi

  RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${INSPIRE_SANDBOX_SITE_PACKAGES}:${WORKSPACE_ROOT}:${AVALANCHE_ROOT}:${MEGATRON_PATH}:${SLIME_DIR}:${ROCK_ROOT}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${has_nvlink}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\",\"SBX_API_KEY\":\"${SBX_API_KEY:-}\",\"SBX_API_URL\":\"${SBX_API_URL:-}\",\"INSP_GITHUB_TOKEN\":\"${INSP_GITHUB_TOKEN:-}\",\"ROCK_SWE_SANDBOX_BACKEND\":\"${ROCK_SWE_SANDBOX_BACKEND}\",\"ROCK_SWE_AGENT_CONFIG_PATH\":\"${ROCK_SWE_AGENT_CONFIG_PATH:-${SCRIPT_DIR}/rock_agent_iflow_swe_train.yaml}\",\"ROCK_SWE_ROCK_ROOT\":\"${ROCK_SWE_ROCK_ROOT:-${ROCK_ROOT}}\",\"ROCK_SWE_MAX_TURNS\":\"${ROCK_SWE_MAX_TURNS:-50}\",\"ROCK_SWE_AGENT_FINISH_TIMEOUT\":\"${ROCK_SWE_AGENT_FINISH_TIMEOUT:-10800}\",\"ROCK_SWE_WAIT_TIMEOUT\":\"${ROCK_SWE_WAIT_TIMEOUT:-10800}\",\"ROCK_SWE_WAIT_INTERVAL\":\"${ROCK_SWE_WAIT_INTERVAL:-500}\",\"ROCK_SWE_KEEP_CONTAINERS\":\"${ROCK_SWE_KEEP_CONTAINERS:-0}\",\"ROCK_SWE_SANDBOX_START_RETRY_TIMES\":\"${ROCK_SWE_SANDBOX_START_RETRY_TIMES:-10}\",\"ROCK_SWE_SANDBOX_START_RETRY_INTERVAL\":\"${ROCK_SWE_SANDBOX_START_RETRY_INTERVAL:-5}\",\"ROCK_SWE_AGENT_INSTALL_RETRY_TIMES\":\"${ROCK_SWE_AGENT_INSTALL_RETRY_TIMES:-10}\",\"ROCK_SWE_AGENT_INSTALL_RETRY_INTERVAL\":\"${ROCK_SWE_AGENT_INSTALL_RETRY_INTERVAL:-1}\",\"ROCK_INSPIRE_TEMPLATE_WAIT_SECONDS\":\"${ROCK_INSPIRE_TEMPLATE_WAIT_SECONDS:-1800}\",\"ROCK_INSPIRE_SANDBOX_TIMEOUT\":\"${ROCK_INSPIRE_SANDBOX_TIMEOUT:-3600}\",\"ROCK_INSPIRE_SPEC\":\"${ROCK_INSPIRE_SPEC}\",\"ROCK_SWE_LOG_ROOT\":\"${ROCK_SWE_LOG_ROOT}\",\"ROCK_SWE_GROUP_CONCURRENCY\":\"${ROCK_SWE_GROUP_CONCURRENCY}\",\"ROCK_SWE_SAMPLE_CONCURRENCY\":\"${ROCK_SWE_SAMPLE_CONCURRENCY}\",\"ROCK_SWE_OVER_SAMPLING_BATCH_SIZE\":\"${ROCK_SWE_OVER_SAMPLING_BATCH_SIZE}\"}}"

  JOB_CMD=(
    python3 "${SLIME_DIR}/train_async.py"
    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}"
    "${OFFLOAD_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${TORCH_DIST_DIR}"
    "${LOAD_ARGS[@]}"
    --save "${SAVE_DIR}"
    --save-interval 20
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${DEBUG_ARGS[@]}"
  )

  {
    printf '#!/bin/bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "${SLIME_DIR}"
    printf 'exec '
    printf '%q ' "${JOB_CMD[@]}"
    printf '\n'
  } > "${JOB_ENTRYPOINT_SCRIPT}"
  chmod +x "${JOB_ENTRYPOINT_SCRIPT}"

  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- bash "${JOB_ENTRYPOINT_SCRIPT}"
}

NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-10.244.68.85}}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
NODE_IP="$(get_local_ip)"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cleanup_local_processes
validate_layout

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
  sleep 5
  start_ray_worker_with_retry
  while ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; do
    sleep 60
  done
fi
