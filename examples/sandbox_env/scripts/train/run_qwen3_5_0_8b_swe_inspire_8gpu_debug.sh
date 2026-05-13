#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SANDBOX_ENV_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SLIME_EXAMPLES_DIR="$(cd -- "${SANDBOX_ENV_DIR}/.." && pwd)"
SLIME_DIR_DEFAULT="$(cd -- "${SLIME_EXAMPLES_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${SLIME_DIR_DEFAULT}/.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${WORKSPACE_ROOT}/.." && pwd)"

# shellcheck disable=SC1091
source "${WORKSPACE_ROOT}/login.sh"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME=Qwen3.5-0.8B
MODEL_DIR="${AVALANCHE_ROOT}/models/${MODEL_NAME}"
TORCH_DIST_DIR="${AVALANCHE_ROOT}/models/${MODEL_NAME}_torch_dist"

# ── Paths ─────────────────────────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR_DEFAULT}"
MEGATRON_PATH=/root/Megatron-LM
INSPIRE_SANDBOX_SITE_PACKAGES="${AVALANCHE_ROOT}/.local/share/inspire_sandbox_site_packages"
SHARE_WORKSPACE="${AVALANCHE_ROOT}/share_workspace"
REBENCH_TASKS_JSON="${AVALANCHE_ROOT}/data/raw_data/single/swe_rebench_v2/data/train-00000-of-00001.json"

SWE_CONSUMABLE_TEMPLATE_MANIFEST="${SWE_CONSUMABLE_TEMPLATE_MANIFEST:-${SANDBOX_ENV_DIR}/data_output/swe_rebench_scaffold_template_success.respec_gc2_all_clean.jsonl}"

# ── WandB (set before output paths to derive RUN_DIR_NAME) ───────────────────
SWE_USE_WANDB=0
SWE_WANDB_PROJECT=slime-swe
SWE_WANDB_GROUP=debug
SWE_WANDB_RUN_ID=qwen3.5-0.8b-swe-8gpu-debug

# ── Output ────────────────────────────────────────────────────────────────────
WORK_ROOT_BASE="${SANDBOX_ENV_DIR}/output/qwen3_5_0_8b_debug"
RUN_DIR_NAME="${SWE_WANDB_GROUP//\//_}"
RUN_DIR_NAME="${RUN_DIR_NAME// /_}"
WORK_ROOT="${WORK_ROOT_BASE}/${RUN_DIR_NAME}"
DATA_CACHE_DIR="${WORK_ROOT}/data_cache"
LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
NORMALIZED_TRAIN="${DATA_CACHE_DIR}/swe_train.normalized.jsonl"
NORMALIZED_VAL="${DATA_CACHE_DIR}/swe_val.normalized.jsonl"
JOB_ENTRYPOINT_SCRIPT="${WORK_ROOT}/run_train_async_job.sh"
SWE_LOG_ROOT="${LOG_DIR}"

# ── Data ──────────────────────────────────────────────────────────────────────
SWE_DEBUG_MAX_TRAIN_PER_SOURCE=1024
SWE_DEBUG_MAX_VAL_PER_SOURCE=0
SWE_REQUIRE_CONSUMABLE_TEMPLATES=1

# ── Scaffold ──────────────────────────────────────────────────────────────────
SWE_AGENT_HARNESS=qwen_code
SWE_MODEL_PROXY_PORT=30001
SWE_WSTUNNEL_SERVER_PORT=19090
# Protocol root and tool paths come from agentic_protocol's shared layout
# (default /__avaeval_agentic_protocol_v1__). To use a non-default root, export
# AGENTIC_PROTOCOL_ROOT before launching this script and rebuild templates
# against that root.

# ── Cluster ───────────────────────────────────────────────────────────────────
ACTOR_GPUS=4
ROLLOUT_GPUS_TOTAL=4

# ── Rollout ───────────────────────────────────────────────────────────────────
SWE_NUM_ROLLOUT=200
SWE_ROLLOUT_BATCH_SIZE=4
SWE_SAMPLES_PER_PROMPT=2
SWE_GLOBAL_BATCH_SIZE=""
SWE_MICRO_BATCH_SIZE=1
SWE_STEPS_PER_ROLLOUT=1
SWE_MAX_CONTEXT_LEN=32768
SWE_MAX_RESPONSE_LEN=4096
SWE_GROUP_CONCURRENCY=4
SWE_SAMPLE_CONCURRENCY=8

# ── Optimizer ─────────────────────────────────────────────────────────────────
SWE_LR=1e-6
SWE_ADAM_BETA2=0.98

# ── Parallelism ───────────────────────────────────────────────────────────────
SWE_TENSOR_MODEL_PARALLEL_SIZE=1
SWE_EXPERT_MODEL_PARALLEL_SIZE=1
SWE_ROLLOUT_NUM_GPUS_PER_ENGINE=1
SWE_SGLANG_EP_SIZE=1
SWE_SGLANG_MEM_FRACTION_STATIC=0.6
SWE_MAX_TOKENS_PER_GPU=1024

# ── Sandbox runtime ───────────────────────────────────────────────────────────
SWE_MAX_TURNS=20
SWE_AGENT_FINISH_TIMEOUT=10800
SWE_WAIT_TIMEOUT=10800
SWE_KEEP_CONTAINERS=0
SWE_SANDBOX_START_RETRY_TIMES=10
SWE_SANDBOX_START_RETRY_INTERVAL=5

# ── Misc ──────────────────────────────────────────────────────────────────────
SWE_RESUME_TRAINING=0
SWE_LOAD_DIR=""
SWE_DEBUG_ROLLOUT_ONLY=0
FORCE_REBUILD_TORCH_DIST=0

# ── Init ──────────────────────────────────────────────────────────────────────
NUM_GPUS_TOTAL=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

if [[ ! -d "${INSPIRE_SANDBOX_SITE_PACKAGES}" ]]; then
  echo "INSPIRE_SANDBOX_SITE_PACKAGES not found: ${INSPIRE_SANDBOX_SITE_PACKAGES}" >&2
  exit 1
fi

if [[ "${SWE_RESUME_TRAINING}" == "1" ]] && [[ -z "${SWE_LOAD_DIR}" ]] && [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  SWE_LOAD_DIR="${SAVE_DIR}"
fi

exec > >(tee "${LOG_DIR}/run.log") 2>&1

# ── Functions ─────────────────────────────────────────────────────────────────

# Stop local Ray/SGLang processes before starting the debug job.
cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
}

# Build the small normalized SWE-rebench debug dataset.
prepare_data() {
  DATA_ARGS=(
    --tasks-json "${REBENCH_TASKS_JSON}"
    --train-dest "${NORMALIZED_TRAIN}"
    --val-dest "${NORMALIZED_VAL}"
    --train-max "${SWE_DEBUG_MAX_TRAIN_PER_SOURCE}"
    --val-max "${SWE_DEBUG_MAX_VAL_PER_SOURCE}"
    --conversation-prompt
  )
  [[ -n "${SWE_CONSUMABLE_TEMPLATE_MANIFEST}" ]] && DATA_ARGS+=(--consumable-template-manifest "${SWE_CONSUMABLE_TEMPLATE_MANIFEST}")
  [[ "${SWE_REQUIRE_CONSUMABLE_TEMPLATES}" == "1" ]] && DATA_ARGS+=(--require-consumable-templates)
  python3 "${SANDBOX_ENV_DIR}/build_rebench_runtime_data.py" "${DATA_ARGS[@]}"
}

# Convert the Qwen3.5-0.8B HF checkpoint to torch_dist when needed.
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

# Validate 8-GPU debug rollout, optimizer, and parallelism settings.
validate_layout() {
  local rollout_product=$(( SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT ))
  local expected_global_batch_size=$(( rollout_product / SWE_STEPS_PER_ROLLOUT ))

  if (( SWE_STEPS_PER_ROLLOUT <= 0 )); then
    echo "SWE_STEPS_PER_ROLLOUT must be positive." >&2; exit 1
  fi
  if (( rollout_product % SWE_STEPS_PER_ROLLOUT != 0 )); then
    echo "rollout_batch_size * n_samples_per_prompt must be divisible by num_steps_per_rollout." >&2; exit 1
  fi
  if (( ACTOR_GPUS % SWE_TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "ACTOR_GPUS must be divisible by SWE_TENSOR_MODEL_PARALLEL_SIZE." >&2; exit 1
  fi
  if (( 2 % SWE_TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "Qwen3.5-0.8B num_query_groups=2 must be divisible by TP." >&2; exit 1
  fi
  if (( 1024 % SWE_TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
    echo "Qwen3.5-0.8B hidden_size=1024 must be divisible by TP." >&2; exit 1
  fi
  if (( ROLLOUT_GPUS_TOTAL % SWE_ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "ROLLOUT_GPUS_TOTAL must be divisible by SWE_ROLLOUT_NUM_GPUS_PER_ENGINE." >&2; exit 1
  fi
  if (( SWE_ROLLOUT_NUM_GPUS_PER_ENGINE % SWE_SGLANG_EP_SIZE != 0 )); then
    echo "SWE_ROLLOUT_NUM_GPUS_PER_ENGINE must be divisible by SWE_SGLANG_EP_SIZE." >&2; exit 1
  fi
  if [[ -z "${SWE_GLOBAL_BATCH_SIZE}" ]]; then
    SWE_GLOBAL_BATCH_SIZE=${expected_global_batch_size}
  fi
  if (( SWE_GLOBAL_BATCH_SIZE != expected_global_batch_size )); then
    echo "SWE_GLOBAL_BATCH_SIZE=${SWE_GLOBAL_BATCH_SIZE} is inconsistent; expected ${expected_global_batch_size}." >&2; exit 1
  fi
  local actor_data_parallel_size=$(( ACTOR_GPUS / SWE_TENSOR_MODEL_PARALLEL_SIZE ))
  if (( actor_data_parallel_size <= 0 || SWE_GLOBAL_BATCH_SIZE % actor_data_parallel_size != 0 )); then
    echo "SWE_GLOBAL_BATCH_SIZE (${SWE_GLOBAL_BATCH_SIZE}) must be divisible by actor DP (${actor_data_parallel_size})." >&2; exit 1
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

if [[ -z "${NUM_GPUS_TOTAL}" ]] || (( NUM_GPUS_TOTAL <= 0 )); then
  echo "No GPUs detected." >&2; exit 1
fi
if (( ACTOR_GPUS + ROLLOUT_GPUS_TOTAL > NUM_GPUS_TOTAL )); then
  echo "ACTOR_GPUS (${ACTOR_GPUS}) + ROLLOUT_GPUS_TOTAL (${ROLLOUT_GPUS_TOTAL}) exceeds NUM_GPUS_TOTAL (${NUM_GPUS_TOTAL})." >&2; exit 1
fi

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
  # --apply-chat-template
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
  --rollout-function-path examples.sandbox_env.swe_rollout.generate_rollout
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
  --context-parallel-size 4
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
[[ "${SWE_DEBUG_ROLLOUT_ONLY}" == "1" ]] && DEBUG_ARGS+=(--debug-rollout-only)

export MASTER_ADDR=127.0.0.1
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS_TOTAL}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${SHARE_WORKSPACE}:${INSPIRE_SANDBOX_SITE_PACKAGES}:${WORKSPACE_ROOT}:${AVALANCHE_ROOT}:${MEGATRON_PATH}:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\",\"SBX_API_KEY\":\"${SBX_API_KEY:-}\",\"SBX_API_URL\":\"${SBX_API_URL:-}\",\"INSP_GITHUB_TOKEN\":\"${INSP_GITHUB_TOKEN:-}\",\"SWE_AGENT_HARNESS\":\"${SWE_AGENT_HARNESS}\",\"AGENTIC_PROTOCOL_ROOT\":\"${AGENTIC_PROTOCOL_ROOT:-}\",\"SWE_MODEL_PROXY_PORT\":\"${SWE_MODEL_PROXY_PORT}\",\"SWE_WSTUNNEL_SERVER_PORT\":\"${SWE_WSTUNNEL_SERVER_PORT}\",\"SWE_MAX_TURNS\":\"${SWE_MAX_TURNS}\",\"SWE_AGENT_FINISH_TIMEOUT\":\"${SWE_AGENT_FINISH_TIMEOUT}\",\"SWE_WAIT_TIMEOUT\":\"${SWE_WAIT_TIMEOUT}\",\"SWE_KEEP_CONTAINERS\":\"${SWE_KEEP_CONTAINERS}\",\"SWE_SANDBOX_START_RETRY_TIMES\":\"${SWE_SANDBOX_START_RETRY_TIMES}\",\"SWE_SANDBOX_START_RETRY_INTERVAL\":\"${SWE_SANDBOX_START_RETRY_INTERVAL}\",\"SWE_LOG_ROOT\":\"${SWE_LOG_ROOT}\",\"SWE_GROUP_CONCURRENCY\":\"${SWE_GROUP_CONCURRENCY}\",\"SWE_SAMPLE_CONCURRENCY\":\"${SWE_SAMPLE_CONCURRENCY}\"}}"

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
