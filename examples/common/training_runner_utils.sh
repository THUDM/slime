#!/usr/bin/env bash

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_QUERIES_PY="${SCRIPT_QUERIES_PY:-${COMMON_DIR}/system_queries.py}"

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

cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
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

# run_head_worker_loop HEAD_PREPARE_FN
#
# Runs the standard multi-node Ray bootstrap loop.  The optional
# HEAD_PREPARE_FN is called on the head node after Ray starts and before
# wait_for_full_ray_cluster.  It should contain all data / checkpoint
# preparation steps (e.g. prepare_training_source_list, write_eval_config,
# ensure_torch_dist_checkpoint, etc.).  submit_ray_job is always called after
# the cluster is ready.
#
# Globals used (must be set by the caller):
#   NODE_RANK, MASTER_ADDR, MASTER_PORT, DASHBOARD_PORT
#   RAY_HEAD_LOCK_DIR, RAY_HEAD_ADDR_FILE
run_head_worker_loop() {
  local head_prepare_fn="${1:-}"

  NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
  MASTER_ADDR="${MASTER_ADDR:-}"
  MASTER_PORT=${MASTER_PORT:-6379}
  DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
  IS_RAY_HEAD=0

  # Rank 0 cleans stale lock/addr files from previous runs so head election
  # succeeds cleanly.  Brief sleep lets rank 0 finish before others race.
  if [[ "${NODE_RANK}" -eq 0 ]]; then
    if [[ -d "${RAY_HEAD_LOCK_DIR}" ]]; then
      echo "Rank 0: cleaning stale ray_head_lock from previous run."
      rm -rf "${RAY_HEAD_LOCK_DIR}"
    fi
    rm -f "${RAY_HEAD_ADDR_FILE}"
  fi
  sleep 3

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
    [[ -n "${head_prepare_fn}" ]] && "${head_prepare_fn}"
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
}
