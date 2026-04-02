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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/slime/examples/multidomain_v1/ray_bootstrap_utils.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/slime/examples/multidomain_v1/data_cache_reuse_utils.sh"

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
TRAIN_POOL_EXCLUDE_PATTERNS=${TRAIN_POOL_EXCLUDE_PATTERNS:-stem/train/openbookqa,stem/train/scienceqa,stem/train/sciq,stem/train/ai2_arc,stem/train/aqua_rat,tool/train/xlam_function_calling_60k,structured/train/nemotron_structured_outputs}
TRAIN_SOURCE_LIST_BASENAME=${TRAIN_SOURCE_LIST_BASENAME:-mopd_train_sources.list}

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
TOOL_CALL_WANDB_GROUP=${TOOL_CALL_WANDB_GROUP:-${JOB_NAME:-mopd-qwen3-30b-a3b-4node}}

# ---- Misc ----
SLIME_DIR=${SLIME_DIR:-${PROJECT_ROOT}/slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

RAY_CLUSTER_WAIT_MAX_ATTEMPTS=${RAY_CLUSTER_WAIT_MAX_ATTEMPTS:-60}
RAY_CLUSTER_WAIT_SLEEP_SECONDS=${RAY_CLUSTER_WAIT_SLEEP_SECONDS:-10}
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

BOOTSTRAP_NODE_ID="${WORKER_ID:-${HOSTNAME:-node-${NODE_RANK:-unknown}}}"
BOOTSTRAP_LOG_FILE="${LOG_DIR}/bootstrap_${BOOTSTRAP_NODE_ID}.log"
exec > >(tee -a "${BOOTSTRAP_LOG_FILE}") 2>&1

# ---- Helpers (from mdv1) ----

cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
}

normalize_wandb_group_name() {
  local candidate="$1"
  local suffix
  if (( ${#candidate} <= 128 )); then
    printf '%s\n' "${candidate}"
    return 0
  fi
  suffix=$(python3 - "$candidate" <<'PY'
import hashlib
import sys

print(hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest()[:8])
PY
)
  printf '%.119s-%s\n' "$candidate" "$suffix"
}

resolve_hostname_to_ip() {
  local host="$1"
  python3 - "$host" <<'PY'
import socket
import sys

host = sys.argv[1]
try:
    print(socket.gethostbyname(host))
except OSError:
    sys.exit(1)
PY
}

is_ip_address() {
  local candidate="$1"
  python3 - "$candidate" <<'PY'
import ipaddress
import sys

candidate = sys.argv[1]
try:
    ipaddress.ip_address(candidate)
except ValueError:
    sys.exit(1)
PY
}

normalize_master_addr() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return 1
  fi
  if is_ip_address "${candidate}" 2>/dev/null; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  resolve_hostname_to_ip "${candidate}"
}

wait_for_master_addr_file() {
  local path="$1"
  local attempt
  for attempt in $(seq 1 180); do
    if [[ -s "${path}" ]]; then
      cat "${path}"
      return 0
    fi
    sleep 2
  done
  return 1
}

elect_ray_head_role() {
  if mkdir "${RAY_HEAD_LOCK_DIR}" 2>/dev/null; then
    IS_RAY_HEAD=1
    printf '%s\n' "${WORKER_ID:-${HOSTNAME:-node-${NODE_RANK}}}" > "${RAY_HEAD_LOCK_DIR}/owner"
    echo "Elected this node as Ray head via shared lock."
  else
    IS_RAY_HEAD=0
    echo "This node will join as Ray worker."
  fi
}

ensure_nonempty_jsonl() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${label} file was not created: ${path}" >&2
    return 1
  fi
  if [[ ! -s "${path}" ]]; then
    echo "${label} file is empty: ${path}" >&2
    return 1
  fi
}

ensure_nonempty_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${label} file was not created: ${path}" >&2
    return 1
  fi
  if [[ ! -s "${path}" ]]; then
    echo "${label} file is empty: ${path}" >&2
    return 1
  fi
}

wait_for_full_ray_cluster() {
  local expected_gpus=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
  local expected_nodes=${NUM_NODES}
  local attempt

  for attempt in $(seq 1 "${RAY_CLUSTER_WAIT_MAX_ATTEMPTS}"); do
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
    echo "Waiting for Ray workers to join (${attempt}/${RAY_CLUSTER_WAIT_MAX_ATTEMPTS})..."
    sleep "${RAY_CLUSTER_WAIT_SLEEP_SECONDS}"
  done

  echo "Ray workers did not join the cluster in time." >&2
  ray status --address="${MASTER_ADDR}:${MASTER_PORT}" || true
  return 1
}

filter_jsonl_by_prompt_budget() {
  local input_path="$1"
  local label="$2"

  if [[ ! -f "${input_path}" ]]; then
    return 0
  fi

  python3 - "${input_path}" "${label}" "${MODEL_DIR}" "${TOOLCALL_MAX_PROMPT_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

path = Path(sys.argv[1])
label = sys.argv[2]
model_dir = sys.argv[3]
max_prompt_tokens = int(sys.argv[4])

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tmp_path = path.with_suffix(path.suffix + ".tmp")
kept = 0
skipped = 0
worst_tokens = -1
worst_record = ""

with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        sample = json.loads(line)
        prompt_messages = sample.get("prompt")
        tools = sample.get("tools")
        if isinstance(prompt_messages, list):
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif isinstance(prompt_messages, str):
            prompt_text = prompt_messages
        else:
            prompt_text = json.dumps(prompt_messages, ensure_ascii=False)
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if prompt_tokens <= max_prompt_tokens:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
            continue
        skipped += 1
        if prompt_tokens > worst_tokens:
            metadata = sample.get("metadata") or {}
            worst_tokens = prompt_tokens
            worst_record = str(metadata.get("dataset_name") or metadata.get("record_id") or "")

if kept == 0:
    tmp_path.unlink(missing_ok=True)
    raise SystemExit(
        f"Filtered every sample from {label} for exceeding {max_prompt_tokens} prompt tokens"
    )

tmp_path.replace(path)
print(
    f"Filtered {skipped} samples exceeding {max_prompt_tokens} prompt tokens for {label}: "
    f"kept={kept} worst_tokens={worst_tokens} worst_record={worst_record}"
)
PY
}

start_ray_worker_with_retry() {
  local attempt
  local rc
  local node_name="${WORKER_ID:-${HOSTNAME:-worker-${NODE_RANK}}}"

  for attempt in $(seq 1 30); do
    set +e
    ray start \
      --address="${MASTER_ADDR}:${MASTER_PORT}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --node-name "${node_name}" \
      --dashboard-port="${DASHBOARD_PORT}" \
      --disable-usage-stats
    rc=$?
    set -e

    if [ "${rc}" -eq 0 ]; then
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

start_ray_head_and_persist_addr() {
  local node_name="${WORKER_ID:-${HOSTNAME:-head-0}}"
  local output
  local rc
  local detected_addr

  set +e
  output="$(
    ray start --head \
      --port="${MASTER_PORT}" \
      --node-name "${node_name}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port="${DASHBOARD_PORT}" 2>&1
  )"
  rc=$?
  set -e

  printf '%s\n' "${output}"
  if [[ "${rc}" -ne 0 ]]; then
    return "${rc}"
  fi

  detected_addr="$(
    RAY_START_OUTPUT="${output}" python3 - <<'PY'
import os
import re

text = os.environ["RAY_START_OUTPUT"]
patterns = [
    r"--address='([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):[0-9]+'",
    r"ray\.init\(_node_ip_address='([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)'\)",
    r"http://([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):8265",
    r"Local node IP.*?([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)",
]
for pattern in patterns:
    match = re.search(pattern, text)
    if match:
        print(match.group(1))
        raise SystemExit(0)
raise SystemExit(1)
PY
  )" || true

  if [[ -z "${detected_addr}" ]]; then
    echo "Failed to determine Ray head IP from ray start output." >&2
    return 1
  fi

  MASTER_ADDR="${detected_addr}"
  printf '%s\n' "${MASTER_ADDR}" > "${RAY_HEAD_ADDR_FILE}"
  echo "Persisted Ray head address: ${MASTER_ADDR}"

  local attempt
  for attempt in $(seq 1 10); do
    if ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  echo "Ray head address ${MASTER_ADDR}:${MASTER_PORT} was not reachable after startup." >&2
  return 1
}

# ---- Data source manifest ----

prepare_training_source_list() {
  if [[ ! -d "${TRAIN_POOL_ROOT}" ]]; then
    echo "TRAIN_POOL_ROOT does not exist: ${TRAIN_POOL_ROOT}" >&2
    return 1
  fi

  python3 - "${TRAIN_POOL_ROOT}" "${TRAIN_SOURCE_LIST}" "${TRAIN_POOL_INCLUDE_DOMAINS}" "${TRAIN_POOL_EXCLUDE_PATTERNS}" <<'PY'
import sys
from pathlib import Path

pool_root = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2])
include_domains = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
exclude_patterns = [item.strip() for item in sys.argv[4].split(",") if item.strip()]

paths = []
for domain in include_domains:
    domain_root = pool_root / domain
    if not domain_root.exists():
        continue
    if domain in {"tool", "stem", "structured"}:
        candidates = sorted((domain_root / "train").glob("*.jsonl"))
    else:
        candidates = sorted(domain_root.glob("*.jsonl"))
    for path in candidates:
        rel = path.relative_to(pool_root).as_posix()
        if any(pattern in rel for pattern in exclude_patterns):
            continue
        paths.append(path.resolve())

dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open("w", encoding="utf-8") as handle:
    for path in paths:
        handle.write(f"{path}\n")

print(f"wrote {len(paths)} training sources to {dest}")
for path in paths:
    print(path)
PY

  ensure_nonempty_file "${TRAIN_SOURCE_LIST}" "Training source manifest"
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

  # ---- Reward: OPD route-by-domain ----
  CUSTOM_ARGS=(
    --custom-rm-path examples.MOPD.reward_func_mopd.reward_func_route_by_domain
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
    --custom-rollout-log-function-path examples.MOPD.log_mopd_rollout.log_rollout_data
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

  RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${SLIME_DIR}/examples/multidomain_v1:${MEGATRON_PATH}:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\",\"MULTIDOMAIN_V1_TRACE_DIR\":\"${MULTIDOMAIN_V1_TRACE_DIR}\",\"MULTIDOMAIN_V1_TRACE_MAX_SAMPLES\":\"${MULTIDOMAIN_V1_TRACE_MAX_SAMPLES}\",\"OPD_DOMAIN_MODEL_MAP\":\"${OPD_DOMAIN_MODEL_MAP}\"}}"

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
    --save-interval 20 \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
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
