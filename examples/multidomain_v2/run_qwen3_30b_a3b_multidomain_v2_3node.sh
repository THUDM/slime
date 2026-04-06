#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ray_bootstrap_utils.sh"

NUM_NODES=${NUM_NODES:-3}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-8}

MODEL_DIR=${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints/iter_0001432_hf}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints}
WORK_ROOT=${WORK_ROOT:-${AVALANCHE_ROOT}/experiments/multidomain_v2_3node}
TRAIN_POOL_ROOT=${TRAIN_POOL_ROOT:-${AVALANCHE_ROOT}/data/pool}
TRAIN_POOL_GROUP=${TRAIN_POOL_GROUP:-${TRAIN_POOL_DOMAIN:-tool_call}}
TRAIN_POOL_DATASETS=${TRAIN_POOL_DATASETS:-}
TRAIN_DATASETS=${TRAIN_DATASETS:-}
TRAIN_DATASETS_EXTRA=${TRAIN_DATASETS_EXTRA:-}
TRAIN_PATHS=${TRAIN_PATHS:-}
TRAIN_PATHS_EXTRA=${TRAIN_PATHS_EXTRA:-}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-}
EVAL_DATASETS=${EVAL_DATASETS:-}
EVAL_DATASETS_EXTRA=${EVAL_DATASETS_EXTRA:-}
EVAL_PATHS=${EVAL_PATHS:-}
EVAL_PATHS_EXTRA=${EVAL_PATHS_EXTRA:-}

SLIME_DIR=${SLIME_DIR:-${PROJECT_ROOT}/slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}

EVAL_TOOL_BFCL_V3=${EVAL_TOOL_BFCL_V3:-${TRAIN_POOL_ROOT}/tool/eval/bfcl_v3_train-00000-of-00001.jsonl}
EVAL_TOOL_BFCL_MULTI_TURN=${EVAL_TOOL_BFCL_MULTI_TURN:-${TRAIN_POOL_ROOT}/tool/eval/bfcl_v3_multi_turn_base_train-00000-of-00001.jsonl}
EVAL_STRUCTURED_IFEVAL=${EVAL_STRUCTURED_IFEVAL:-${TRAIN_POOL_ROOT}/structured/eval/ifeval_ifeval_input_data.jsonl}
EVAL_STRUCTURED_JSONSCHEMABENCH=${EVAL_STRUCTURED_JSONSCHEMABENCH:-${TRAIN_POOL_ROOT}/structured/eval/jsonschemabench_test-00000-of-00001.jsonl}
EVAL_STRUCTURED_IFBENCH_TEST=${EVAL_STRUCTURED_IFBENCH_TEST:-${TRAIN_POOL_ROOT}/structured/eval/ifbench_test_data_train-00000-of-00001.jsonl}
EVAL_STEM_MMLU_PRO=${EVAL_STEM_MMLU_PRO:-${TRAIN_POOL_ROOT}/stem/eval/mmlu_pro_test-00000-of-00001.jsonl}
EVAL_STEM_GPQA_MAIN=${EVAL_STEM_GPQA_MAIN:-${TRAIN_POOL_ROOT}/stem/eval/gpqa_gpqa_main.jsonl}

EVAL_BFCL_V3_SAMPLES=${EVAL_BFCL_V3_SAMPLES:-4441}
EVAL_BFCL_MULTI_TURN_SAMPLES=${EVAL_BFCL_MULTI_TURN_SAMPLES:-200}
EVAL_IFEVAL_SAMPLES=${EVAL_IFEVAL_SAMPLES:-541}
EVAL_JSONSCHEMABENCH_SAMPLES=${EVAL_JSONSCHEMABENCH_SAMPLES:-5722}
EVAL_IFBENCH_TEST_SAMPLES=${EVAL_IFBENCH_TEST_SAMPLES:-300}
EVAL_MMLU_PRO_SAMPLES=${EVAL_MMLU_PRO_SAMPLES:-12032}
EVAL_GPQA_MAIN_SAMPLES=${EVAL_GPQA_MAIN_SAMPLES:-448}

LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
RUNTIME_DATA_DIR="${WORK_ROOT}/runtime_data"
TRAIN_SOURCE_LIST="${LOG_DIR}/train_pool_sources.list"
BFCL_V3_EVAL="${RUNTIME_DATA_DIR}/bfcl_v3_eval.normalized.jsonl"
BFCL_MULTI_TURN_EVAL="${RUNTIME_DATA_DIR}/bfcl_multi_turn_eval.normalized.jsonl"
IFEVAL_EVAL="${RUNTIME_DATA_DIR}/ifeval_eval.normalized.jsonl"
JSONSCHEMABENCH_EVAL="${RUNTIME_DATA_DIR}/jsonschemabench_eval.normalized.jsonl"
IFBENCH_TEST_EVAL="${RUNTIME_DATA_DIR}/ifbench_test_eval.normalized.jsonl"
MMLU_PRO_EVAL="${RUNTIME_DATA_DIR}/mmlu_pro_eval.normalized.jsonl"
GPQA_MAIN_EVAL="${RUNTIME_DATA_DIR}/gpqa_main_eval.normalized.jsonl"
CUSTOM_EVAL_PROMPT_DATA_FILE="${RUNTIME_DATA_DIR}/custom_eval_prompt_data.txt"
TRACE_DIR="${WORK_ROOT}/rollout_traces"
TRACE_MAX_SAMPLES=${TRACE_MAX_SAMPLES:-8}
export MULTIDOMAIN_V1_TRACE_DIR="${TRACE_DIR}"
export MULTIDOMAIN_V1_TRACE_MAX_SAMPLES="${TRACE_MAX_SAMPLES}"

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
TOOLCALL_COLOCATE=${TOOLCALL_COLOCATE:-0}
TOOLCALL_DYNAMIC_FILTER=${TOOLCALL_DYNAMIC_FILTER:-}
TOOLCALL_USE_ROUTING_REPLAY=${TOOLCALL_USE_ROUTING_REPLAY:-0}
TOOLCALL_USE_ROLLOUT_ROUTING_REPLAY=${TOOLCALL_USE_ROLLOUT_ROUTING_REPLAY:-0}
TOOLCALL_RESUME_TRAINING=${TOOLCALL_RESUME_TRAINING:-0}
TOOLCALL_RESUME_NO_OPTIM=${TOOLCALL_RESUME_NO_OPTIM:-1}
TOOLCALL_RESUME_NO_RNG=${TOOLCALL_RESUME_NO_RNG:-1}
TOOLCALL_RESUME_FINETUNE=${TOOLCALL_RESUME_FINETUNE:-1}
TOOL_CALL_LOAD_DIR=${TOOL_CALL_LOAD_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints}
TOOL_CALL_WANDB_PROJECT=${TOOL_CALL_WANDB_PROJECT:-slime-multidomain-v2}
RAY_CLUSTER_WAIT_MAX_ATTEMPTS=${RAY_CLUSTER_WAIT_MAX_ATTEMPTS:-120}
RAY_CLUSTER_WAIT_SLEEP_SECONDS=${RAY_CLUSTER_WAIT_SLEEP_SECONDS:-10}
RAY_CLUSTER_STATUS_TIMEOUT_SECONDS=${RAY_CLUSTER_STATUS_TIMEOUT_SECONDS:-30}
RAY_WORKER_JOIN_MAX_ATTEMPTS=${RAY_WORKER_JOIN_MAX_ATTEMPTS:-60}
RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS=${RAY_WORKER_JOIN_RETRY_SLEEP_SECONDS:-10}
RAY_HEAD_ADDR_WAIT_ATTEMPTS=${RAY_HEAD_ADDR_WAIT_ATTEMPTS:-300}
RAY_HEAD_ADDR_WAIT_SLEEP=${RAY_HEAD_ADDR_WAIT_SLEEP:-2}
RAY_HEAD_START_STATUS_MAX_ATTEMPTS=${RAY_HEAD_START_STATUS_MAX_ATTEMPTS:-30}
RAY_HEAD_START_STATUS_SLEEP_SECONDS=${RAY_HEAD_START_STATUS_SLEEP_SECONDS:-2}
RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS=${RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS:-300}
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-${RAY_GCS_RPC_SERVER_RECONNECT_TIMEOUT_SECONDS}}"
RAY_HEAD_ADDR_FILE="${WORK_ROOT}/ray_head_addr.txt"
RAY_HEAD_LOCK_DIR="${WORK_ROOT}/ray_head_lock"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}" "${RUNTIME_DATA_DIR}" "${TRACE_DIR}"

if [[ -z "${TRAIN_POOL_DATASETS}" ]]; then
  TRAIN_POOL_DATASETS="$(
    PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}" python3 - "${TRAIN_POOL_GROUP}" <<'PY'
import sys

from multidomain_shared import default_train_datasets_for_group

print(",".join(default_train_datasets_for_group(sys.argv[1])))
PY
  )"
fi

if [[ -z "${TRAIN_DATASETS}" ]]; then
  TRAIN_DATASETS="${TRAIN_POOL_DATASETS}"
fi

TRAIN_GROUP_SIGNATURE="$(
  PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}" python3 - "${TRAIN_DATASETS}" "${TRAIN_DATASETS_EXTRA}" "${TRAIN_PATHS}" "${TRAIN_PATHS_EXTRA}" <<'PY'
import sys

from multidomain_shared import group_signature_for_train_datasets

dataset_names = [item.strip() for item in ",".join(sys.argv[1:3]).split(",") if item.strip()]
path_names = [item.strip() for item in ",".join(sys.argv[3:]).split(",") if item.strip()]
if dataset_names:
    print(group_signature_for_train_datasets(dataset_names))
elif path_names:
    print("custom")
else:
    print("unknown")
PY
)"

DEFAULT_WANDB_GROUP_FROM_DOMAINS="qwen3-30b-a3b-mdv2-3node-${TRAIN_GROUP_SIGNATURE}"
TOOL_CALL_WANDB_GROUP=${TOOL_CALL_WANDB_GROUP:-${DEFAULT_WANDB_GROUP_FROM_DOMAINS:-${JOB_NAME:-qwen3-30b-a3b-mdv2-3node}}}

BOOTSTRAP_NODE_ID="${WORKER_ID:-${HOSTNAME:-node-${NODE_RANK:-unknown}}}"
BOOTSTRAP_LOG_FILE="${LOG_DIR}/bootstrap_${BOOTSTRAP_NODE_ID}.log"
exec > >(tee -a "${BOOTSTRAP_LOG_FILE}") 2>&1

if [[ "${TRAIN_GROUP_SIGNATURE}" == mixed-* ]]; then
  echo "Training mixes multiple groups: ${TRAIN_GROUP_SIGNATURE}"
else
  echo "Training group: ${TRAIN_GROUP_SIGNATURE}"
fi

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

prepare_training_source_list() {
  if [[ ! -d "${TRAIN_POOL_ROOT}" ]]; then
    echo "TRAIN_POOL_ROOT does not exist: ${TRAIN_POOL_ROOT}" >&2
    return 1
  fi

  local train_pool_prep_args=("--pool-root" "${TRAIN_POOL_ROOT}")
  if [[ -n "${TRAIN_DATASETS}" ]]; then
    local dataset_name
    local train_pool_dataset_array=()
    IFS=',' read -r -a train_pool_dataset_array <<< "${TRAIN_DATASETS}"
    for dataset_name in "${train_pool_dataset_array[@]}"; do
      dataset_name="${dataset_name//[[:space:]]/}"
      if [[ -n "${dataset_name}" ]]; then
        train_pool_prep_args+=("--dataset" "${dataset_name}")
      fi
    done
  fi
  if [[ -n "${TRAIN_DATASETS_EXTRA}" ]]; then
    local dataset_extra
    local train_pool_dataset_extra_array=()
    IFS=',' read -r -a train_pool_dataset_extra_array <<< "${TRAIN_DATASETS_EXTRA}"
    for dataset_extra in "${train_pool_dataset_extra_array[@]}"; do
      dataset_extra="${dataset_extra//[[:space:]]/}"
      if [[ -n "${dataset_extra}" ]]; then
        train_pool_prep_args+=("--dataset-extra" "${dataset_extra}")
      fi
    done
  fi
  if [[ -n "${TRAIN_PATHS}" ]]; then
    local train_path
    local train_pool_path_array=()
    IFS=',' read -r -a train_pool_path_array <<< "${TRAIN_PATHS}"
    for train_path in "${train_pool_path_array[@]}"; do
      train_path="${train_path//[[:space:]]/}"
      if [[ -n "${train_path}" ]]; then
        train_pool_prep_args+=("--source" "${train_path}")
      fi
    done
  fi
  if [[ -n "${TRAIN_PATHS_EXTRA}" ]]; then
    local train_path_extra
    local train_pool_path_extra_array=()
    IFS=',' read -r -a train_pool_path_extra_array <<< "${TRAIN_PATHS_EXTRA}"
    for train_path_extra in "${train_pool_path_extra_array[@]}"; do
      train_path_extra="${train_path_extra//[[:space:]]/}"
      if [[ -n "${train_path_extra}" ]]; then
        train_pool_prep_args+=("--source-extra" "${train_path_extra}")
      fi
    done
  fi
  if [[ -n "${TRAIN_MANIFEST}" ]]; then
    train_pool_prep_args+=("--manifest" "${TRAIN_MANIFEST}")
  fi

  python3 "${SCRIPT_DIR}/prepare_multidomain_v2_data.py" \
    "${train_pool_prep_args[@]}" \
    --dest /dev/null \
    --print-sources > "${TRAIN_SOURCE_LIST}"

  ensure_nonempty_file "${TRAIN_SOURCE_LIST}" "Training source manifest"
  echo "Prepared training source manifest at ${TRAIN_SOURCE_LIST}:"
  sed 's/^/  /' "${TRAIN_SOURCE_LIST}"
}

prepare_eval_data() {
  rm -f "${CUSTOM_EVAL_PROMPT_DATA_FILE}"
  if [[ -n "${EVAL_DATASETS}" || -n "${EVAL_DATASETS_EXTRA}" || -n "${EVAL_PATHS}" || -n "${EVAL_PATHS_EXTRA}" ]]; then
    : > "${CUSTOM_EVAL_PROMPT_DATA_FILE}"
    while IFS=$'\t' read -r name source_path sample_count; do
      [[ -n "${name}" ]] || continue
      local_dest="${RUNTIME_DATA_DIR}/${name}_eval.normalized.jsonl"
      python3 "${SCRIPT_DIR}/../materialize_eval_dataset.py" \
        --pool-root "${TRAIN_POOL_ROOT}" \
        --source "${source_path}" \
        --dest "${local_dest}" \
        --max-samples "${sample_count}"
      filter_jsonl_by_prompt_budget "${local_dest}" "${name}_eval"
      printf '%s\t%s\n' "${name}_eval" "${local_dest}" >> "${CUSTOM_EVAL_PROMPT_DATA_FILE}"
    done < <(
      PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH:-}" python3 - "${TRAIN_POOL_ROOT}" "${EVAL_DATASETS}" "${EVAL_DATASETS_EXTRA}" "${EVAL_PATHS}" "${EVAL_PATHS_EXTRA}" <<'PY'
import sys
from pathlib import Path

from dataset_selection import resolve_eval_datasets

pool_root = Path(sys.argv[1])
datasets = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
dataset_extras = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
paths = [item.strip() for item in sys.argv[4].split(",") if item.strip()]
path_extras = [item.strip() for item in sys.argv[5].split(",") if item.strip()]

resolved = resolve_eval_datasets(
    pool_root=pool_root,
    datasets=datasets or None,
    dataset_extras=dataset_extras or None,
    paths=paths or None,
    path_extras=path_extras or None,
)
for item in resolved:
    print(f"{item.name}\t{item.source}\t{item.n_samples_per_eval_prompt}")
PY
    )
    return 0
  fi

  if (( EVAL_IFEVAL_SAMPLES > 0 )); then
    python3 "${SCRIPT_DIR}/prepare_multidomain_v2_data.py" \
      --source "${EVAL_STRUCTURED_IFEVAL}" \
      --dest "${IFEVAL_EVAL}" \
      --max-samples "${EVAL_IFEVAL_SAMPLES}"
    filter_jsonl_by_prompt_budget "${IFEVAL_EVAL}" ifeval_eval
  fi

  if (( EVAL_JSONSCHEMABENCH_SAMPLES > 0 )); then
    python3 "${SCRIPT_DIR}/prepare_multidomain_v2_data.py" \
      --source "${EVAL_STRUCTURED_JSONSCHEMABENCH}" \
      --dest "${JSONSCHEMABENCH_EVAL}" \
      --max-samples "${EVAL_JSONSCHEMABENCH_SAMPLES}"
    filter_jsonl_by_prompt_budget "${JSONSCHEMABENCH_EVAL}" jsonschemabench_eval
  fi

  if (( EVAL_IFBENCH_TEST_SAMPLES > 0 )); then
    python3 "${SCRIPT_DIR}/prepare_multidomain_v2_data.py" \
      --source "${EVAL_STRUCTURED_IFBENCH_TEST}" \
      --dest "${IFBENCH_TEST_EVAL}" \
      --max-samples "${EVAL_IFBENCH_TEST_SAMPLES}"
    filter_jsonl_by_prompt_budget "${IFBENCH_TEST_EVAL}" ifbench_test_eval
  fi

  if (( EVAL_MMLU_PRO_SAMPLES > 0 )); then
    python3 "${SCRIPT_DIR}/prepare_multidomain_v2_data.py" \
      --source "${EVAL_STEM_MMLU_PRO}" \
      --dest "${MMLU_PRO_EVAL}" \
      --max-samples "${EVAL_MMLU_PRO_SAMPLES}"
    filter_jsonl_by_prompt_budget "${MMLU_PRO_EVAL}" mmlu_pro_eval
  fi

  if (( EVAL_GPQA_MAIN_SAMPLES > 0 )); then
    python3 "${SCRIPT_DIR}/prepare_multidomain_v2_data.py" \
      --source "${EVAL_STEM_GPQA_MAIN}" \
      --dest "${GPQA_MAIN_EVAL}" \
      --max-samples "${EVAL_GPQA_MAIN_SAMPLES}"
    filter_jsonl_by_prompt_budget "${GPQA_MAIN_EVAL}" gpqa_main_eval
  fi
}

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

submit_ray_job() {
  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
  if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi

  LOAD_ARGS=()
  if [[ "${TOOLCALL_RESUME_TRAINING}" == "1" ]] && [[ -n "${TOOL_CALL_LOAD_DIR}" ]]; then
    if [[ "${TOOL_CALL_LOAD_DIR}" == */checkpoints/iter_* ]]; then
      local resume_load_dir
      local resume_ckpt_step
      resume_load_dir="$(dirname "${TOOL_CALL_LOAD_DIR}")"
      resume_ckpt_step="${TOOL_CALL_LOAD_DIR##*/}"
      resume_ckpt_step="${resume_ckpt_step#iter_}"
      resume_ckpt_step=$((10#${resume_ckpt_step}))
      LOAD_ARGS+=(--load "${resume_load_dir}" --ckpt-step "${resume_ckpt_step}")
    else
      LOAD_ARGS+=(--load "${TOOL_CALL_LOAD_DIR}")
    fi
    if [[ "${TOOLCALL_RESUME_NO_OPTIM}" == "1" ]]; then
      LOAD_ARGS+=(--no-load-optim)
    fi
    if [[ "${TOOLCALL_RESUME_NO_RNG}" == "1" ]]; then
      LOAD_ARGS+=(--no-load-rng)
    fi
    if [[ "${TOOLCALL_RESUME_FINETUNE}" == "1" ]]; then
      LOAD_ARGS+=(--finetune)
    fi
  fi

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
    --rollout-max-prompt-len "${TOOLCALL_MAX_PROMPT_TOKENS}"
    --rollout-max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --global-batch-size "${TOOLCALL_GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout "${TOOLCALL_STEPS_PER_ROLLOUT}"
    --balance-data
  )
  if [[ -n "${TOOLCALL_DYNAMIC_FILTER}" ]]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${TOOLCALL_DYNAMIC_FILTER}")
  fi

  EVAL_ARGS=()
  EVAL_PROMPT_DATA_ARGS=()
  if [[ -s "${CUSTOM_EVAL_PROMPT_DATA_FILE}" ]]; then
    while IFS=$'\t' read -r eval_name eval_path; do
      [[ -n "${eval_name}" ]] || continue
      EVAL_PROMPT_DATA_ARGS+=("${eval_name}" "${eval_path}")
    done < "${CUSTOM_EVAL_PROMPT_DATA_FILE}"
  else
    if (( EVAL_IFEVAL_SAMPLES > 0 )); then
      EVAL_PROMPT_DATA_ARGS+=(ifeval_eval "${IFEVAL_EVAL}")
    fi
    if (( EVAL_JSONSCHEMABENCH_SAMPLES > 0 )); then
      EVAL_PROMPT_DATA_ARGS+=(jsonschemabench_eval "${JSONSCHEMABENCH_EVAL}")
    fi
    if (( EVAL_IFBENCH_TEST_SAMPLES > 0 )); then
      EVAL_PROMPT_DATA_ARGS+=(ifbench_test_eval "${IFBENCH_TEST_EVAL}")
    fi
    if (( EVAL_MMLU_PRO_SAMPLES > 0 )); then
      EVAL_PROMPT_DATA_ARGS+=(mmlu_pro_eval "${MMLU_PRO_EVAL}")
    fi
    if (( EVAL_GPQA_MAIN_SAMPLES > 0 )); then
      EVAL_PROMPT_DATA_ARGS+=(gpqa_main_eval "${GPQA_MAIN_EVAL}")
    fi
  fi
  if [[ "${#EVAL_PROMPT_DATA_ARGS[@]}" -gt 0 ]]; then
    EVAL_ARGS+=(
      --eval-interval 20
      --eval-input-key prompt
      --eval-label-key label
      --eval-tool-key tools
      --n-samples-per-eval-prompt 1
      --eval-max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
    )
    EVAL_ARGS+=(--eval-prompt-data "${EVAL_PROMPT_DATA_ARGS[@]}")
  fi

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
    --eps-clip-high 0.2
  )

  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${TOOLCALL_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 "${TOOLCALL_ADAM_BETA2}"
  )

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-mem-fraction-static 0.7
    --sglang-ep-size 8
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
  )

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )
  if [[ "${TOOLCALL_USE_ROUTING_REPLAY}" == "1" ]]; then
    MISC_ARGS+=(--use-routing-replay)
  fi
  if [[ "${TOOLCALL_USE_ROLLOUT_ROUTING_REPLAY}" == "1" ]]; then
    MISC_ARGS+=(--use-rollout-routing-replay)
  fi

  CUSTOM_ARGS=(
    --custom-rm-path "multidomain_shared.reward_func"
    --custom-rollout-log-function-path log_rollout.log_rollout_data
    --custom-eval-rollout-log-function-path log_rollout.log_eval_rollout_data
  )

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

  RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${SCRIPT_DIR}:${SCRIPT_DIR}/..:${MEGATRON_PATH}:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\",\"MULTIDOMAIN_V1_TRACE_DIR\":\"${MULTIDOMAIN_V1_TRACE_DIR}\",\"MULTIDOMAIN_V1_TRACE_MAX_SAMPLES\":\"${MULTIDOMAIN_V1_TRACE_MAX_SAMPLES}\"}}"

  TRAINING_RESOURCE_ARGS=(
    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
  )
  if [[ "${TOOLCALL_COLOCATE}" == "1" ]]; then
    TRAINING_RESOURCE_ARGS+=(--colocate)
  else
    TRAINING_RESOURCE_ARGS+=(--rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}")
  fi

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
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
IS_RAY_HEAD=0

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
  prepare_eval_data
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
