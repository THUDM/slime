#!/usr/bin/env bash

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
  local max_attempts="${RAY_HEAD_ADDR_WAIT_ATTEMPTS:-180}"
  local sleep_seconds="${RAY_HEAD_ADDR_WAIT_SLEEP:-2}"
  for attempt in $(seq 1 "${max_attempts}"); do
    if [[ -s "${path}" ]]; then
      cat "${path}"
      return 0
    fi
    sleep "${sleep_seconds}"
  done
  return 1
}

resolve_worker_master_addr() {
  local inherited_addr="${1:-}"
  local head_addr_file="${2:-}"
  local shared_addr=""
  local normalized_shared=""
  local normalized_inherited=""

  if [[ -n "${head_addr_file}" ]]; then
    shared_addr="$(wait_for_master_addr_file "${head_addr_file}" 2>/dev/null || true)"
    normalized_shared="$(normalize_master_addr "${shared_addr}" 2>/dev/null || true)"
    if [[ -n "${normalized_shared}" ]]; then
      printf '%s\n' "${normalized_shared}"
      return 0
    fi
  fi

  normalized_inherited="$(normalize_master_addr "${inherited_addr}" 2>/dev/null || true)"
  if [[ -n "${normalized_inherited}" ]]; then
    printf '%s\n' "${normalized_inherited}"
    return 0
  fi

  return 1
}
