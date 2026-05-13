#!/usr/bin/env bash
# Production Qwen3.5-35B-A3B + SWE-Rebench training on G_C2 sandbox spec.
#
# Models on the proven 6-node config but with:
#  - G_C2 consumable manifest (re-spec + from-image merged, dedup by instance_id)
#  - SAMPLE_CONCURRENCY pushed to 32 (sandbox pool benchmark 2026-05-11 shows
#    32 is safe; 48 is reachable with create retries; 64+ exceeds burst cap
#    "too many sandboxes starting on this node")
#  - Sandbox start retries already wired (10 × 5s = 50s window for burst recovery)
#  - Topology env-overridable; default = 6 nodes (4 actor + 2 rollout × 8 GPU)
#
# GPU budget options (H200, 8 per node):
#  - 96 GPUs (12 nodes): ACTOR_NUM_NODES=8 + ROLLOUT_GPUS_TOTAL=32 (4 engines)
#  - 128 GPUs (16 nodes): ACTOR_NUM_NODES=12 + ROLLOUT_GPUS_TOTAL=32   ← single-run sweet spot
#  - 256 GPUs (32 nodes): split into 2× 128-GPU runs (different harness/LR)
#    Actor side diminishing returns >96 GPU for 3B-active MoE; rollout side
#    bounded by sandbox pool (~32-52 active). Don't put 256 into one run.
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

# ── Cluster topology (override via env for 96/128/256 GPU layouts) ────────────
NUM_NODES="${NUM_NODES:-6}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
ROLLOUT_GPUS_TOTAL="${ROLLOUT_GPUS_TOTAL:-16}"

# ── Model (defaults to avalanche v0.1 SFT 1epoch ckpt; override via env) ──────
MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-Nemotron-Cascade-2-sft-1epoch}"
MODEL_DIR="${MODEL_DIR:-${AVALANCHE_ROOT}/models/avalanche_ckpts/v0.1/${MODEL_NAME}}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${AVALANCHE_ROOT}/models/avalanche_ckpts/v0.1/${MODEL_NAME}_torch_dist_mtp}"
# models/avalanche_ckpts/v0.1/Qwen3.5-35B-A3B-Nemotron-Cascade-2-sft-1epoch_torch_dist_mtp
# ── Paths ─────────────────────────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR_DEFAULT}"
MEGATRON_PATH=/root/Megatron-LM
INSPIRE_SANDBOX_SITE_PACKAGES="${AVALANCHE_ROOT}/.local/share/inspire_sandbox_site_packages"
SHARE_WORKSPACE="${AVALANCHE_ROOT}/share_workspace"
REBENCH_TASKS_JSON="${REBENCH_TASKS_JSON:-${AVALANCHE_ROOT}/data/raw_data/single/swe_rebench_v2/data/train-00000-of-00001.json}"

SWE_CONSUMABLE_TEMPLATE_MANIFEST="${SWE_CONSUMABLE_TEMPLATE_MANIFEST:-${SANDBOX_ENV_DIR}/data_output/swe_rebench_scaffold_template_success.respec_gc2_all_clean.jsonl}"

# ── WandB ─────────────────────────────────────────────────────────────────────
SWE_USE_WANDB="${SWE_USE_WANDB:-1}"
SWE_WANDB_PROJECT=slime-swe
SWE_WANDB_GROUP="${SWE_WANDB_GROUP:-qwen3.5-35b-a3b-swe-inspire-gc2}"
SWE_WANDB_RUN_ID="${SWE_WANDB_RUN_ID:-${SWE_WANDB_GROUP}}"

# ── Output ────────────────────────────────────────────────────────────────────
WORK_ROOT_BASE="${WORK_ROOT_BASE:-${SANDBOX_ENV_DIR}/output/qwen3_5_35b_gc2_prod}"
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
SWE_TRAIN_MAX_PER_SOURCE="${SWE_TRAIN_MAX_PER_SOURCE:--1}"
SWE_VAL_MAX_PER_SOURCE="${SWE_VAL_MAX_PER_SOURCE:-0}"
SWE_SHUFFLE_DATA="${SWE_SHUFFLE_DATA:-1}"
SWE_DATA_SEED="${SWE_DATA_SEED:-42}"
SWE_REQUIRE_CONSUMABLE_TEMPLATES="${SWE_REQUIRE_CONSUMABLE_TEMPLATES:-1}"

# ── Scaffold ──────────────────────────────────────────────────────────────────
SWE_AGENT_HARNESS="${SWE_AGENT_HARNESS:-qwen_code}"
SWE_MODEL_PROXY_PORT=30001
SWE_WSTUNNEL_SERVER_PORT=19090

# ── Rollout (concurrency tuned for G_C2 sandbox pool, 2026-05-11 benchmark) ──
# Active-sandbox steady-state ~32-52 (CPU bound 400/2=200, burst cap ~32-48).
# Push beyond 48 only with sandbox warm-pool / reuse engineering.
SWE_NUM_ROLLOUT="${SWE_NUM_ROLLOUT:-200}"
SWE_ROLLOUT_BATCH_SIZE="${SWE_ROLLOUT_BATCH_SIZE:-8}"
SWE_SAMPLES_PER_PROMPT="${SWE_SAMPLES_PER_PROMPT:-4}"
SWE_MICRO_BATCH_SIZE="${SWE_MICRO_BATCH_SIZE:-1}"
SWE_STEPS_PER_ROLLOUT="${SWE_STEPS_PER_ROLLOUT:-1}"
SWE_GLOBAL_BATCH_SIZE="${SWE_GLOBAL_BATCH_SIZE:-}"
SWE_MAX_CONTEXT_LEN="${SWE_MAX_CONTEXT_LEN:-128000}"
SWE_MAX_RESPONSE_LEN="${SWE_MAX_RESPONSE_LEN:-4096}"
SWE_OVER_SAMPLING_BATCH_SIZE="${SWE_OVER_SAMPLING_BATCH_SIZE:-16}"
SWE_GROUP_CONCURRENCY="${SWE_GROUP_CONCURRENCY:-8}"
SWE_SAMPLE_CONCURRENCY="${SWE_SAMPLE_CONCURRENCY:-50}"

# ── Dynamic sampling filter (drop all-same-reward groups, save grad cost) ────
# Oversamples SWE_OVER_SAMPLING_BATCH_SIZE prompts then keeps only groups whose
# rewards have non-zero std (i.e. advantage signal exists). Aligned with DAPO.
SWE_DYNAMIC_SAMPLING_FILTER_PATH="${SWE_DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"

# ── Optimizer ─────────────────────────────────────────────────────────────────
SWE_LR="${SWE_LR:-1e-6}"
SWE_ADAM_BETA2="${SWE_ADAM_BETA2:-0.98}"

# ── GRPO ──────────────────────────────────────────────────────────────────────
SWE_KL_LOSS_COEF="${SWE_KL_LOSS_COEF:-0.0}"

# ── Parallelism (matches 6-node config; works on TP=1, CP=ACTOR/2, EP=ACTOR/2)
SWE_TENSOR_MODEL_PARALLEL_SIZE="${SWE_TENSOR_MODEL_PARALLEL_SIZE:-1}"
SWE_CONTEXT_PARALLEL_SIZE="${SWE_CONTEXT_PARALLEL_SIZE:-32}"
SWE_EXPERT_MODEL_PARALLEL_SIZE="${SWE_EXPERT_MODEL_PARALLEL_SIZE:-32}"
SWE_ROLLOUT_NUM_GPUS_PER_ENGINE="${SWE_ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
SWE_SGLANG_EP_SIZE="${SWE_SGLANG_EP_SIZE:-8}"
SWE_SGLANG_MEM_FRACTION_STATIC="${SWE_SGLANG_MEM_FRACTION_STATIC:-0.85}"
SWE_MAX_TOKENS_PER_GPU="${SWE_MAX_TOKENS_PER_GPU:-1024}"
SWE_USE_DYNAMIC_BATCH_SIZE="${SWE_USE_DYNAMIC_BATCH_SIZE:-1}"

# ── SGLang H200 max-throughput template (partner's settings, 2026-05-11) ──────
# Skipped (architectural, needs --sglang-config YAML): tp_size=2 + dp_size=4
SWE_SGLANG_SCHEDULE_CONSERVATIVENESS="${SWE_SGLANG_SCHEDULE_CONSERVATIVENESS:-1.2}"
SWE_SGLANG_MAX_RUNNING_REQUESTS="${SWE_SGLANG_MAX_RUNNING_REQUESTS:-192}"
# 当前 SGLang 配置没有启用 enable_dp_attention，所以 SGLang scheduler 自己断言失败，随后 Ray 看到
#   SGLang server 异常退出，就报：
#   Exception: Server process terminated unexpectedly.
# so set enable_dp_attention=0 来避免这个问题（虽然理论上应该无影响，因为当前模型并不支持 dp_attention）：
SWE_SGLANG_ENABLE_PREFILL_DELAYER="${SWE_SGLANG_ENABLE_PREFILL_DELAYER:-0}"
SWE_SGLANG_PREFILL_DELAYER_LOW_WATERMARK="${SWE_SGLANG_PREFILL_DELAYER_LOW_WATERMARK:-0.9}"
SWE_SGLANG_SCHEDULE_POLICY="${SWE_SGLANG_SCHEDULE_POLICY:-lpm}"
SWE_SGLANG_ALLOW_AUTO_TRUNCATE="${SWE_SGLANG_ALLOW_AUTO_TRUNCATE:-1}"
SWE_SGLANG_REASONING_PARSER="${SWE_SGLANG_REASONING_PARSER:-qwen3}"
SWE_SGLANG_TOOL_CALL_PARSER="${SWE_SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
SWE_SGLANG_MAMBA_SCHEDULER_STRATEGY="${SWE_SGLANG_MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"

# ── MTP (Multi-Token Prediction) — SFT ckpt 自带 mtp_num_hidden_layers=1 ──────
# 两侧都开:rollout 用 EAGLE-style speculative decoding 加速 inference,
# actor 同步训 MTP head(辅助 loss × 0.2)保持 draft 准确率,否则 head 会随
# actor 漂移、accept rate 下降。
SGLANG_ENABLE_MTP_ROLLOUT="${SGLANG_ENABLE_MTP_ROLLOUT:-1}"
SGLANG_SPECULATIVE_ALGORITHM="${SGLANG_SPECULATIVE_ALGORITHM:-EAGLE}"
SGLANG_SPECULATIVE_NUM_STEPS="${SGLANG_SPECULATIVE_NUM_STEPS:-3}"
SGLANG_SPECULATIVE_EAGLE_TOPK="${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS="${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
ENABLE_MTP_TRAINING="${ENABLE_MTP_TRAINING:-1}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-1}"
MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.2}"

# ── R3 (Rollout Routing Replay) — 强制 actor 重放 rollout 的 MoE 专家路由 ────
# 在 actor/rollout TP/EP 拓扑不同(如我们 actor EP=32 / rollout EP=8)时,
# 保证训练 100% on-policy(包括 router 决策)。需要 cherry-picked upstream
# f8879db2 (已应用) 以兼容 MTP-only routers,否则 assert 会 fire。
USE_R3="${USE_R3:-0}"

# ── Optimizer offload ─────────────────────────────────────────────────────────
SWE_OPTIMIZER_CPU_OFFLOAD="${SWE_OPTIMIZER_CPU_OFFLOAD:-1}"
SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D="${SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D:-1}"
SWE_USE_PRECISION_AWARE_OPTIMIZER="${SWE_USE_PRECISION_AWARE_OPTIMIZER:-1}"

# ── Sandbox runtime ───────────────────────────────────────────────────────────
SWE_MAX_TURNS="${SWE_MAX_TURNS:-50}"
SWE_AGENT_FINISH_TIMEOUT="${SWE_AGENT_FINISH_TIMEOUT:-10800}"
SWE_WAIT_TIMEOUT="${SWE_WAIT_TIMEOUT:-10800}"
SWE_KEEP_CONTAINERS="${SWE_KEEP_CONTAINERS:-0}"
# Tuned for G_C2 burst create failures (benchmark 2026-05-11):
SWE_SANDBOX_START_RETRY_TIMES="${SWE_SANDBOX_START_RETRY_TIMES:-15}"
SWE_SANDBOX_START_RETRY_INTERVAL="${SWE_SANDBOX_START_RETRY_INTERVAL:-8}"

# ── Misc ──────────────────────────────────────────────────────────────────────
SWE_RESUME_TRAINING="${SWE_RESUME_TRAINING:-auto}"
SWE_LOAD_DIR="${SWE_LOAD_DIR:-}"
SWE_DEBUG_ROLLOUT_ONLY="${SWE_DEBUG_ROLLOUT_ONLY:-0}"
FORCE_REBUILD_TORCH_DIST="${FORCE_REBUILD_TORCH_DIST:-0}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/swe_train_lib.sh"
