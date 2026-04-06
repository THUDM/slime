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
EVAL_CONFIG_PATH="${WORK_ROOT}/data_cache/eval_config.yaml"
OFFICIAL_EVAL_MANIFEST_PATH="${WORK_ROOT}/data_cache/eval_official_benchmarks.json"
TEACHER_STEP0_INCLUDE_BFCL=${TEACHER_STEP0_INCLUDE_BFCL:-1}
EVAL_DATASETS=${EVAL_DATASETS:-}
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

MOPD_DOMAIN_SIGNATURE="$(
  PYTHONPATH="${SLIME_DIR}/examples:${PYTHONPATH:-}" python3 - "${TRAIN_POOL_INCLUDE_DOMAINS}" <<'PY'
import sys

from multidomain_shared import domain_signature

domains = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(domain_signature(domains))
PY
)"
TOOL_CALL_WANDB_GROUP=${TOOL_CALL_WANDB_GROUP:-mopd-qwen3-30b-a3b-3node-h200-${MOPD_DOMAIN_SIGNATURE}}

BOOTSTRAP_NODE_ID="${WORKER_ID:-${HOSTNAME:-node-${NODE_RANK:-unknown}}}"
BOOTSTRAP_LOG_FILE="${LOG_DIR}/bootstrap_${BOOTSTRAP_NODE_ID}.log"
exec > >(tee -a "${BOOTSTRAP_LOG_FILE}") 2>&1

echo "MOPD training domains: ${MOPD_DOMAIN_SIGNATURE}"

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
    with urllib.request.urlopen("http://127.0.0.1:${DASHBOARD_PORT}/api/cluster_status", timeout=${RAY_CLUSTER_STATUS_TIMEOUT_SECONDS}) as response:
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

  for attempt in $(seq 1 "${RAY_WORKER_JOIN_MAX_ATTEMPTS}"); do
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
    sleep "${RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS}"
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
  for attempt in $(seq 1 "${RAY_HEAD_START_STATUS_MAX_ATTEMPTS}"); do
    if ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${RAY_HEAD_START_STATUS_SLEEP_SECONDS}"
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

  local materialize_args=(
    "${TRAIN_POOL_ROOT}"
    "${DATA_CACHE_DIR}/materialized_train"
    "${TRAIN_SOURCE_LIST}"
    "${TRAIN_POOL_INCLUDE_DOMAINS}"
    "${TRAIN_POOL_EXCLUDE_PATTERNS}"
    --stem-train-datasets "${MOPD_STEM_TRAIN_DATASETS}"
    --structured-train-datasets "${MOPD_STRUCTURED_TRAIN_DATASETS}"
    --math-train-datasets "${MOPD_MATH_TRAIN_DATASETS}"
    --code-train-datasets "${MOPD_CODE_TRAIN_DATASETS}"
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
    --official-manifest-output "${OFFICIAL_EVAL_MANIFEST_PATH}"
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
    python3 - "${EVAL_CONFIG_PATH}" <<'INNERPY'
import sys
import yaml

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    cfg = yaml.safe_load(handle)

for dataset in (cfg.get('eval') or {}).get('datasets', []):
    print(f"{dataset['name']}:{dataset['path']}")
INNERPY
  )

  if [[ "${TEACHER_STEP0_INCLUDE_BFCL}" == "1" ]] && [[ -f "${OFFICIAL_EVAL_MANIFEST_PATH}" ]]; then
    mapfile -t official_specs < <(
      python3 - "${OFFICIAL_EVAL_MANIFEST_PATH}" <<'INNERPY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    manifest = json.load(handle)

for dataset in manifest.get('datasets', []):
    print(f"{dataset['name']}:{dataset['path']}")
INNERPY
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

  run_id="$(python3 - <<'INNERPY'
import uuid
print(uuid.uuid4().hex[:8])
INNERPY
)"
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
