#!/usr/bin/env bash

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_QUERIES_PY="${SCRIPT_QUERIES_PY:-${COMMON_DIR}/script_queries.py}"

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
  suffix=$(python3 "${SCRIPT_QUERIES_PY}" short-sha1 --text "${candidate}")
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
    if python3 "${SCRIPT_QUERIES_PY}" ray-cluster-ready \
      --dashboard-port "${DASHBOARD_PORT}" \
      --timeout-seconds "${RAY_CLUSTER_STATUS_TIMEOUT_SECONDS}" \
      --expected-gpus "${expected_gpus}" \
      --expected-nodes "${expected_nodes}"
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

  python3 "${SCRIPT_QUERIES_PY}" filter-jsonl-by-prompt-budget \
    --input "${input_path}" \
    --label "${label}" \
    --model-dir "${MODEL_DIR}" \
    --max-prompt-tokens "${TOOLCALL_MAX_PROMPT_TOKENS}"
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

  detected_addr="$(python3 "${SCRIPT_QUERIES_PY}" extract-ray-head-addr --text "${output}")" || true

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
