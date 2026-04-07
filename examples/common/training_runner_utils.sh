#!/usr/bin/env bash

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
  local ray_args=(
    --address="${MASTER_ADDR}:${MASTER_PORT}"
    --num-gpus "${NUM_GPUS_PER_NODE}"
    --node-name "${node_name}"
    --dashboard-port="${DASHBOARD_PORT}"
    --disable-usage-stats
  )

  if [[ -n "${NODE_IP:-}" ]]; then
    ray_args+=(--node-ip-address "${NODE_IP}")
  fi

  for attempt in $(seq 1 "${RAY_WORKER_JOIN_MAX_ATTEMPTS}"); do
    set +e
    ray start "${ray_args[@]}"
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
  local ray_args=(
    --head
    --port="${MASTER_PORT}"
    --node-name "${node_name}"
    --num-gpus "${NUM_GPUS_PER_NODE}"
    --disable-usage-stats
    --dashboard-host=0.0.0.0
    --dashboard-port="${DASHBOARD_PORT}"
  )

  if [[ -n "${NODE_IP:-}" ]]; then
    ray_args+=(--node-ip-address "${NODE_IP}")
  fi

  set +e
  output="$(ray start "${ray_args[@]}" 2>&1)"
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
