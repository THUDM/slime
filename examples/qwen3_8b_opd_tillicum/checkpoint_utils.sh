#!/usr/bin/env bash

checkpoint_latest_iteration() {
  local save_dir="$1"
  local tracker="${save_dir}/latest_checkpointed_iteration.txt"
  if [[ ! -f "${tracker}" ]]; then
    return 1
  fi
  tr -d '[:space:]' <"${tracker}"
}

checkpoint_iter_name() {
  local iteration="$1"
  printf "iter_%07d" "${iteration}"
}

checkpoint_latest_dir() {
  local save_dir="$1"
  local iteration
  iteration="$(checkpoint_latest_iteration "${save_dir}")" || return 1
  printf "%s/%s" "${save_dir}" "$(checkpoint_iter_name "${iteration}")"
}

checkpoint_is_complete() {
  local checkpoint_dir="$1"
  [[ -d "${checkpoint_dir}" && -f "${checkpoint_dir}/.metadata" ]]
}

checkpoint_prune_old() {
  local stage="$1"
  local save_dir="$2"
  local mode="${3:-background}"
  local latest_dir
  latest_dir="$(checkpoint_latest_dir "${save_dir}")" || return 0

  if [[ ! -d "${latest_dir}" ]]; then
    echo "CHECKPOINT_PRUNE_WAIT stage=${stage} latest_dir_missing=${latest_dir}"
    return 0
  fi

  if ! checkpoint_is_complete "${latest_dir}"; then
    if [[ "${mode}" == "final" ]]; then
      echo "CHECKPOINT_PRUNE_FAIL stage=${stage} latest_incomplete=${latest_dir}" >&2
      return 1
    fi
    echo "CHECKPOINT_PRUNE_WAIT stage=${stage} latest_incomplete=${latest_dir}"
    return 0
  fi

  local dir
  find "${save_dir}" -maxdepth 1 -type d -name 'iter_*' -print | while IFS= read -r dir; do
    if [[ "${dir}" != "${latest_dir}" ]]; then
      if [[ "${mode}" != "final" ]] && ! checkpoint_is_complete "${dir}"; then
        echo "CHECKPOINT_PRUNE_SKIP_INCOMPLETE stage=${stage} dir=${dir}"
        continue
      fi
      echo "CHECKPOINT_PRUNE_REMOVE stage=${stage} dir=${dir}"
      rm -rf -- "${dir}"
    fi
  done
}

checkpoint_report() {
  local stage="$1"
  local save_dir="$2"
  local report_dir="$3"
  mkdir -p "${report_dir}"

  local iteration latest_dir bytes report
  iteration="$(checkpoint_latest_iteration "${save_dir}")"
  latest_dir="${save_dir}/$(checkpoint_iter_name "${iteration}")"
  if [[ ! -d "${latest_dir}" ]]; then
    echo "Missing latest checkpoint dir: ${latest_dir}" >&2
    return 1
  fi
  if ! checkpoint_is_complete "${latest_dir}"; then
    echo "Latest checkpoint is incomplete: ${latest_dir}" >&2
    return 1
  fi

  bytes="$(du -sb "${latest_dir}" | awk '{print $1}')"
  report="${report_dir}/${stage}_checkpoint_storage.tsv"
  if [[ ! -f "${report}" ]]; then
    printf "stage\titeration\tcheckpoint_dir\tbytes\testimated_full_optim_bytes\n" >"${report}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${stage}" \
    "${iteration}" \
    "${latest_dir}" \
    "${bytes}" \
    "${ESTIMATED_FULL_OPTIM_CKPT_BYTES:-}" >>"${report}"

  echo "CHECKPOINT_SIZE_BYTES stage=${stage} iter=${iteration} bytes=${bytes} dir=${latest_dir}"
}

checkpoint_verify_single_latest() {
  local stage="$1"
  local save_dir="$2"
  local latest_dir
  latest_dir="$(checkpoint_latest_dir "${save_dir}")"

  local count
  count="$(find "${save_dir}" -maxdepth 1 -type d -name 'iter_*' | wc -l)"
  if [[ "${count}" -ne 1 ]]; then
    echo "CHECKPOINT_PRUNE_FAIL stage=${stage} expected_count=1 actual_count=${count}" >&2
    find "${save_dir}" -maxdepth 1 -type d -name 'iter_*' -print >&2
    return 1
  fi

  if [[ ! -d "${latest_dir}" ]]; then
    echo "CHECKPOINT_PRUNE_FAIL stage=${stage} missing_latest=${latest_dir}" >&2
    return 1
  fi
  if ! checkpoint_is_complete "${latest_dir}"; then
    echo "CHECKPOINT_PRUNE_FAIL stage=${stage} incomplete_latest=${latest_dir}" >&2
    return 1
  fi

  echo "CHECKPOINT_PRUNE_OK stage=${stage} remaining_iter=$(basename "${latest_dir}")"
}

checkpoint_start_pruner() {
  local stage="$1"
  local save_dir="$2"
  local interval_seconds="$3"
  local report_dir="${4:-}"

  (
    set +e
    last_reported_iteration=""
    while true; do
      checkpoint_prune_old "${stage}" "${save_dir}"
      if [[ -n "${report_dir}" ]]; then
        iteration="$(checkpoint_latest_iteration "${save_dir}" 2>/dev/null)"
        if [[ -n "${iteration}" && "${iteration}" != "${last_reported_iteration}" ]]; then
          latest_dir="${save_dir}/$(checkpoint_iter_name "${iteration}")"
          if checkpoint_is_complete "${latest_dir}"; then
            checkpoint_report "${stage}" "${save_dir}" "${report_dir}"
            last_reported_iteration="${iteration}"
          fi
        fi
      fi
      sleep "${interval_seconds}"
    done
  ) &
  CHECKPOINT_PRUNER_PID="$!"
  export CHECKPOINT_PRUNER_PID
  echo "CHECKPOINT_PRUNER_STARTED stage=${stage} pid=${CHECKPOINT_PRUNER_PID}"
}

checkpoint_stop_pruner() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}
