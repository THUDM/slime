#!/bin/bash
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

MODEL_NAME=${MODEL_NAME:-Qwen3.5-0.8B}
MODEL_DIR=${MODEL_DIR:-${AVALANCHE_ROOT}/models/${MODEL_NAME}}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-${AVALANCHE_ROOT}/models/${MODEL_NAME}_torch_dist}
WORK_ROOT_BASE=${WORK_ROOT_BASE:-${SCRIPT_DIR}/output_inspire_qwen3_5_0_8b_debug}

REBENCH_TASKS_JSON=${REBENCH_TASKS_JSON:-${AVALANCHE_ROOT}/data/raw_data/single/swe_rebench_v2/data/train-00000-of-00001.json}

SLIME_DIR=${SLIME_DIR:-${SLIME_DIR_DEFAULT}}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
ROCK_ROOT=${ROCK_ROOT:-${AVALANCHE_ROOT}/ROCK}
INSPIRE_SANDBOX_SITE_PACKAGES_DEFAULT="${AVALANCHE_ROOT}/.local/share/inspire_sandbox_site_packages"
INSPIRE_SANDBOX_SITE_PACKAGES=${INSPIRE_SANDBOX_SITE_PACKAGES:-${INSPIRE_SANDBOX_SITE_PACKAGES_DEFAULT}}
ROCK_SWE_SANDBOX_BACKEND=${ROCK_SWE_SANDBOX_BACKEND:-inspire}
ROCK_INSPIRE_SPEC=${ROCK_INSPIRE_SPEC:-G_C4}
ROCK_SWE_EVAL_SANDBOX_SOURCE=template #template/image
DEFAULT_ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST="${AVALANCHE_ROOT}/data/raw_data/single/swe_rebench_v2/data/prefetch_image_template_success.jsonl"
DEFAULT_ROCK_SWE_AGENT_CONFIG_PATH="${SCRIPT_DIR}/rock_agent_qwen_rebench_template.yaml"
ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST=${ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST:-${DEFAULT_ROCK_SWE_CONSUMABLE_TEMPLATE_MANIFEST}}
ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES=${ROCK_SWE_REQUIRE_CONSUMABLE_TEMPLATES:-1}
ROCK_SWE_AGENT_CONFIG_PATH=${ROCK_SWE_AGENT_CONFIG_PATH:-${DEFAULT_ROCK_SWE_AGENT_CONFIG_PATH}}
export ROCK_SWE_AGENT_CONFIG_PATH

SWE_DEBUG_MAX_TRAIN_PER_SOURCE=${SWE_DEBUG_MAX_TRAIN_PER_SOURCE:-1024}
SWE_DEBUG_MAX_VAL_PER_SOURCE=${SWE_DEBUG_MAX_VAL_PER_SOURCE:-0}
SWE_NUM_ROLLOUT=${SWE_NUM_ROLLOUT:-200}
SWE_ROLLOUT_BATCH_SIZE=${SWE_ROLLOUT_BATCH_SIZE:-32}
SWE_SAMPLES_PER_PROMPT=${SWE_SAMPLES_PER_PROMPT:-2}
# Cap concurrent samples to avoid inspire envd gateway saturation (see diagnose_concurrency*.py).
ROCK_SWE_GROUP_CONCURRENCY=${ROCK_SWE_GROUP_CONCURRENCY:-${SWE_ROLLOUT_BATCH_SIZE}}
ROCK_SWE_SAMPLE_CONCURRENCY=${ROCK_SWE_SAMPLE_CONCURRENCY:-$(( SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT ))}
SWE_GLOBAL_BATCH_SIZE=${SWE_GLOBAL_BATCH_SIZE:-}
SWE_MICRO_BATCH_SIZE=${SWE_MICRO_BATCH_SIZE:-1}
SWE_STEPS_PER_ROLLOUT=${SWE_STEPS_PER_ROLLOUT:-1}
SWE_MAX_CONTEXT_LEN=${SWE_MAX_CONTEXT_LEN:-32768}
SWE_MAX_RESPONSE_LEN=${SWE_MAX_RESPONSE_LEN:-4096}
SWE_LR=${SWE_LR:-1e-6}
SWE_ADAM_BETA2=${SWE_ADAM_BETA2:-0.98}
SWE_RESUME_TRAINING=${SWE_RESUME_TRAINING:-0}
SWE_LOAD_DIR=${SWE_LOAD_DIR:-}
SWE_WANDB_PROJECT=${SWE_WANDB_PROJECT:-slime-swe}
SWE_WANDB_GROUP=${SWE_WANDB_GROUP:-debug}
SWE_WANDB_RUN_ID=${SWE_WANDB_RUN_ID:-qwen3.5-0.8b-swe-8gpu-debug}
SWE_USE_WANDB=${SWE_USE_WANDB:-0}
SWE_DEBUG_ROLLOUT_ONLY=${SWE_DEBUG_ROLLOUT_ONLY:-0}
FORCE_REBUILD_TORCH_DIST=${FORCE_REBUILD_TORCH_DIST:-0}

NUM_GPUS_TOTAL=${NUM_GPUS_TOTAL:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-4}
SWE_TENSOR_MODEL_PARALLEL_SIZE=${SWE_TENSOR_MODEL_PARALLEL_SIZE:-1}
SWE_EXPERT_MODEL_PARALLEL_SIZE=${SWE_EXPERT_MODEL_PARALLEL_SIZE:-1}
SWE_ROLLOUT_NUM_GPUS_PER_ENGINE=${SWE_ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SWE_SGLANG_EP_SIZE=${SWE_SGLANG_EP_SIZE:-1}
SWE_SGLANG_MEM_FRACTION_STATIC=${SWE_SGLANG_MEM_FRACTION_STATIC:-0.6}
SWE_MAX_TOKENS_PER_GPU=${SWE_MAX_TOKENS_PER_GPU:-8192}

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

if [[ -z "${NUM_GPUS_TOTAL}" ]] || (( NUM_GPUS_TOTAL <= 0 )); then
  echo "No GPUs detected." >&2
  exit 1
fi
if (( ACTOR_GPUS + ROLLOUT_GPUS_TOTAL > NUM_GPUS_TOTAL )); then
  echo "ACTOR_GPUS (${ACTOR_GPUS}) + ROLLOUT_GPUS_TOTAL (${ROLLOUT_GPUS_TOTAL}) exceeds NUM_GPUS_TOTAL (${NUM_GPUS_TOTAL})." >&2
  exit 1
fi

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

if [[ ! -d "${INSPIRE_SANDBOX_SITE_PACKAGES}" ]]; then
  echo "INSPIRE_SANDBOX_SITE_PACKAGES not found: ${INSPIRE_SANDBOX_SITE_PACKAGES}" >&2
  echo "Expected shared inspire-sandbox packages under ${INSPIRE_SANDBOX_SITE_PACKAGES_DEFAULT}" >&2
  exit 1
fi

if [[ "${SWE_RESUME_TRAINING}" == "1" ]] && [[ -z "${SWE_LOAD_DIR}" ]] && [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  SWE_LOAD_DIR="${SAVE_DIR}"
fi

RUN_LOG_PATH="${LOG_DIR}/run.log"
exec > >(tee "${RUN_LOG_PATH}") 2>&1

cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
}

prepare_data() {
  DATA_ARGS=(
    --tasks-json "${REBENCH_TASKS_JSON}"
    --train-dest "${NORMALIZED_TRAIN}"
    --val-dest "${NORMALIZED_VAL}"
    --train-max "${SWE_DEBUG_MAX_TRAIN_PER_SOURCE}"
    --val-max "${SWE_DEBUG_MAX_VAL_PER_SOURCE}"
    --conversation-prompt
  )
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
  if [[ -f "${TORCH_DIST_DIR}/common.pt" ]] && [[ -f "${TORCH_DIST_DIR}/metadata.json" ]]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3.5-0.8B.sh"
  export PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}:${PYTHONPATH:-}"
  python -m torch.distributed.run \
    --nproc-per-node 1 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
}

validate_layout() {
  local rollout_product=$(( SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT ))
  local expected_global_batch_size=$(( rollout_product / SWE_STEPS_PER_ROLLOUT ))

  if (( SWE_STEPS_PER_ROLLOUT <= 0 )); then
    echo "SWE_STEPS_PER_ROLLOUT must be positive." >&2
    exit 1
  fi
  if (( rollout_product % SWE_STEPS_PER_ROLLOUT != 0 )); then
    echo "rollout_batch_size * n_samples_per_prompt must be divisible by num_steps_per_rollout." >&2
    exit 1
  fi
  if (( ACTOR_GPUS % SWE_TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "ACTOR_GPUS must be divisible by SWE_TENSOR_MODEL_PARALLEL_SIZE." >&2
    exit 1
  fi
  if (( 2 % SWE_TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "Qwen3.5-0.8B num_query_groups=2 must be divisible by TP." >&2
    exit 1
  fi
  if (( 1024 % SWE_TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "Qwen3.5-0.8B hidden_size=1024 must be divisible by TP." >&2
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
  if [[ -z "${SWE_GLOBAL_BATCH_SIZE}" ]]; then
    SWE_GLOBAL_BATCH_SIZE=${expected_global_batch_size}
  fi
  if (( SWE_GLOBAL_BATCH_SIZE != expected_global_batch_size )); then
    echo "SWE_GLOBAL_BATCH_SIZE=${SWE_GLOBAL_BATCH_SIZE} is inconsistent; expected ${expected_global_batch_size}." >&2
    exit 1
  fi
  local actor_data_parallel_size=$(( ACTOR_GPUS / SWE_TENSOR_MODEL_PARALLEL_SIZE ))
  if (( actor_data_parallel_size <= 0 || SWE_GLOBAL_BATCH_SIZE % actor_data_parallel_size != 0 )); then
    echo "SWE_GLOBAL_BATCH_SIZE (${SWE_GLOBAL_BATCH_SIZE}) must be divisible by actor DP (${actor_data_parallel_size})." >&2
    exit 1
  fi
}

cleanup_local_processes
prepare_data
ensure_torch_dist_checkpoint
validate_layout

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3.5-0.8B.sh"

LOAD_ARGS=()
if [[ "${SWE_RESUME_TRAINING}" == "1" ]] && [[ -n "${SWE_LOAD_DIR}" ]]; then
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
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

PERF_ARGS=(
  --tensor-model-parallel-size "${SWE_TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size "${SWE_EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${SWE_MAX_TOKENS_PER_GPU}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${SWE_ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SWE_SGLANG_MEM_FRACTION_STATIC}"
  --sglang-ep-size "${SWE_SGLANG_EP_SIZE}"
  --sglang-cuda-graph-max-bs 32
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

WANDB_ARGS=()
if [[ "${SWE_USE_WANDB}" == "1" ]] && [[ -n "${WANDB_API_KEY:-}" ]]; then
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

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS_TOTAL}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${INSPIRE_SANDBOX_SITE_PACKAGES}:${WORKSPACE_ROOT}:${AVALANCHE_ROOT}:${MEGATRON_PATH}:${SLIME_DIR}:${ROCK_ROOT}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\",\"SBX_API_KEY\":\"${SBX_API_KEY:-}\",\"SBX_API_URL\":\"${SBX_API_URL:-}\",\"INSP_GITHUB_TOKEN\":\"${INSP_GITHUB_TOKEN:-}\",\"ROCK_SWE_BASE_URL\":\"${ROCK_SWE_BASE_URL:-}\",\"ROCK_SWE_SANDBOX_BACKEND\":\"${ROCK_SWE_SANDBOX_BACKEND}\",\"ROCK_SWE_EVAL_SANDBOX_SOURCE\":\"${ROCK_SWE_EVAL_SANDBOX_SOURCE}\",\"ROCK_SWE_AGENT_CONFIG_PATH\":\"${ROCK_SWE_AGENT_CONFIG_PATH:-${SCRIPT_DIR}/rock_agent_iflow_swe_train.yaml}\",\"ROCK_SWE_ROCK_ROOT\":\"${ROCK_SWE_ROCK_ROOT:-${ROCK_ROOT}}\",\"ROCK_SWE_MAX_TURNS\":\"${ROCK_SWE_MAX_TURNS:-50}\",\"ROCK_SWE_AGENT_FINISH_TIMEOUT\":\"${ROCK_SWE_AGENT_FINISH_TIMEOUT:-10800}\",\"ROCK_SWE_WAIT_TIMEOUT\":\"${ROCK_SWE_WAIT_TIMEOUT:-10800}\",\"ROCK_SWE_WAIT_INTERVAL\":\"${ROCK_SWE_WAIT_INTERVAL:-500}\",\"ROCK_SWE_MEMORY\":\"${ROCK_SWE_MEMORY:-4g}\",\"ROCK_SWE_KEEP_CONTAINERS\":\"${ROCK_SWE_KEEP_CONTAINERS:-0}\",\"ROCK_SWE_SANDBOX_START_RETRY_TIMES\":\"${ROCK_SWE_SANDBOX_START_RETRY_TIMES:-10}\",\"ROCK_SWE_SANDBOX_START_RETRY_INTERVAL\":\"${ROCK_SWE_SANDBOX_START_RETRY_INTERVAL:-5}\",\"ROCK_SWE_AGENT_INSTALL_RETRY_TIMES\":\"${ROCK_SWE_AGENT_INSTALL_RETRY_TIMES:-10}\",\"ROCK_SWE_AGENT_INSTALL_RETRY_INTERVAL\":\"${ROCK_SWE_AGENT_INSTALL_RETRY_INTERVAL:-1}\",\"ROCK_INSPIRE_TEMPLATE_WAIT_SECONDS\":\"${ROCK_INSPIRE_TEMPLATE_WAIT_SECONDS:-1800}\",\"ROCK_INSPIRE_SANDBOX_TIMEOUT\":\"${ROCK_INSPIRE_SANDBOX_TIMEOUT:-3600}\",\"ROCK_INSPIRE_SPEC\":\"${ROCK_INSPIRE_SPEC}\",\"ROCK_SWE_LOG_ROOT\":\"${ROCK_SWE_LOG_ROOT}\",\"ROCK_SWE_GROUP_CONCURRENCY\":\"${ROCK_SWE_GROUP_CONCURRENCY}\",\"ROCK_SWE_SAMPLE_CONCURRENCY\":\"${ROCK_SWE_SAMPLE_CONCURRENCY}\"}}"

JOB_CMD=(
  python3 "${SLIME_DIR}/train_async.py"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${ACTOR_GPUS}"
  --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}"
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

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- bash "${JOB_ENTRYPOINT_SCRIPT}"

ray stop --force || true
