#!/usr/bin/env bash

build_run_env_exports() {
  local run_env_exports=""
  local var_name=""
  for var_name in "${FORWARDED_ENV_VARS[@]}"; do
    if [[ -n "${!var_name+x}" ]]; then
      run_env_exports+=" export ${var_name}=$(printf '%q' "${!var_name}");"
    fi
  done
  printf '%s' "${run_env_exports}"
}

submit_inspire_job() {
  local inspire_bin="${INSPIRE_BIN:-${INSPIRE_CLI:-inspire}}"
  local run_env_exports
  run_env_exports="$(build_run_env_exports)"

  if declare -F customize_run_env_exports >/dev/null 2>&1; then
    run_env_exports="$(customize_run_env_exports "${run_env_exports}")"
  fi

  local run_cmd
  run_cmd="cd ${REMOTE_ROOT} &&${run_env_exports} if [[ -f \"${RUN_SCRIPT}\" ]]; then bash ${RUN_SCRIPT}; elif [[ -f \"${RUN_SCRIPT_FALLBACK}\" ]]; then bash ${RUN_SCRIPT_FALLBACK}; else echo \"Run script not found: ${RUN_SCRIPT} or ${RUN_SCRIPT_FALLBACK}\" >&2; exit 1; fi"

  local cmd=(
    "${inspire_bin}" job create
    --name "${JOB_NAME}"
    --resource "${RESOURCE}"
    --nodes "${SUBMIT_NODES}"
    --priority "${PRIORITY}"
    --max-time "${MAX_TIME}"
    --no-auto
    --command "${run_cmd}"
  )

  if [[ "${FAULT_TOLERANT:-0}" == "1" ]]; then
    cmd+=(--fault-tolerant)
  fi
  if [[ -n "${IMAGE:-}" ]]; then
    cmd+=(--image "${IMAGE}")
  fi
  if [[ -n "${LOCATION:-}" ]]; then
    cmd+=(--location "${LOCATION}")
  fi
  if [[ -n "${WORKSPACE_ID:-}" ]]; then
    cmd+=(--workspace-id "${WORKSPACE_ID}")
  fi

  local submit_cwd="${SUBMIT_CWD:-${TMPDIR:-/tmp}}"
  (
    cd "${submit_cwd}"
    export INSPIRE_TARGET_DIR="${JOB_LOG_ROOT}"
    "${cmd[@]}"
  )
}
