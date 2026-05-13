#!/usr/bin/env bash
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

# ── Cluster topology ──────────────────────────────────────────────────────────
NUM_NODES=8
NUM_GPUS_PER_NODE=8
ACTOR_NUM_NODES=6
ACTOR_GPUS_PER_NODE=8
ROLLOUT_GPUS_TOTAL=16

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME=Qwen3.5-35B-A3B
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
SWE_USE_WANDB=1
SWE_WANDB_PROJECT=slime-swe
SWE_WANDB_GROUP=qwen3.5-35b-a3b-swe-inspire-8node
SWE_WANDB_RUN_ID=qwen3.5-35b-a3b-swe-inspire-8node

# ── Output ────────────────────────────────────────────────────────────────────
WORK_ROOT_BASE="${SANDBOX_ENV_DIR}/output/qwen3_5_35b_8node"
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
SWE_TRAIN_MAX_PER_SOURCE=-1
SWE_VAL_MAX_PER_SOURCE=0
SWE_SHUFFLE_DATA=0
SWE_DATA_SEED=42
SWE_REQUIRE_CONSUMABLE_TEMPLATES=1

# ── Scaffold ──────────────────────────────────────────────────────────────────
SWE_AGENT_HARNESS=qwen_code
# Protocol root and tool paths come from agentic_protocol shared layout
# (default /__avaeval_agentic_protocol_v1__). Override via AGENTIC_PROTOCOL_ROOT.
SWE_MODEL_PROXY_PORT=30001
SWE_WSTUNNEL_SERVER_PORT=19090

# ── Rollout ───────────────────────────────────────────────────────────────────
SWE_NUM_ROLLOUT=200
SWE_ROLLOUT_BATCH_SIZE=6
SWE_SAMPLES_PER_PROMPT=4
SWE_MICRO_BATCH_SIZE=1
SWE_STEPS_PER_ROLLOUT=1
SWE_GLOBAL_BATCH_SIZE=""
SWE_MAX_CONTEXT_LEN=128000
SWE_MAX_RESPONSE_LEN=8192
SWE_OVER_SAMPLING_BATCH_SIZE=9
SWE_GROUP_CONCURRENCY=6
SWE_SAMPLE_CONCURRENCY=24

# ── Optimizer ─────────────────────────────────────────────────────────────────
SWE_LR=1e-6
SWE_ADAM_BETA2=0.98

# ── GRPO ──────────────────────────────────────────────────────────────────────
SWE_KL_LOSS_COEF=0.0

# ── Parallelism ───────────────────────────────────────────────────────────────
SWE_TENSOR_MODEL_PARALLEL_SIZE=1
SWE_CONTEXT_PARALLEL_SIZE=16
SWE_EXPERT_MODEL_PARALLEL_SIZE=8
SWE_ROLLOUT_NUM_GPUS_PER_ENGINE=8
SWE_SGLANG_EP_SIZE=8
SWE_SGLANG_MEM_FRACTION_STATIC=0.7
SWE_MAX_TOKENS_PER_GPU=8192
SWE_USE_DYNAMIC_BATCH_SIZE=0

# ── Optimizer offload ─────────────────────────────────────────────────────────
SWE_OPTIMIZER_CPU_OFFLOAD=1
SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D=1
SWE_USE_PRECISION_AWARE_OPTIMIZER=1

# ── Sandbox runtime ───────────────────────────────────────────────────────────
SWE_MAX_TURNS=80
SWE_AGENT_FINISH_TIMEOUT=10800
SWE_WAIT_TIMEOUT=10800
SWE_KEEP_CONTAINERS=0
SWE_SANDBOX_START_RETRY_TIMES=10
SWE_SANDBOX_START_RETRY_INTERVAL=5

# ── Misc ──────────────────────────────────────────────────────────────────────
SWE_RESUME_TRAINING=auto
SWE_LOAD_DIR=""
SWE_DEBUG_ROLLOUT_ONLY=0
FORCE_REBUILD_TORCH_DIST=0

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/swe_train_lib.sh"
