#!/usr/bin/env bash

required_data_cache_files() {
  printf '%s\n' "${NORMALIZED_TRAIN}"
  if (( ${EVAL_BFCL_V3_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${BFCL_V3_EVAL:-}"
  fi
  if (( ${EVAL_BFCL_MULTI_TURN_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${BFCL_MULTI_TURN_EVAL:-}"
  fi
  if (( ${EVAL_IFEVAL_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${IFEVAL_EVAL:-}"
  fi
  if (( ${EVAL_JSONSCHEMABENCH_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${JSONSCHEMABENCH_EVAL:-}"
  fi
  if (( ${EVAL_IFBENCH_TEST_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${IFBENCH_TEST_EVAL:-}"
  fi
  if (( ${EVAL_MMLU_PRO_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${MMLU_PRO_EVAL:-}"
  fi
  if (( ${EVAL_GPQA_MAIN_SAMPLES:-0} > 0 )); then
    printf '%s\n' "${GPQA_MAIN_EVAL:-}"
  fi
}

data_cache_dir_has_required_files() {
  local source_dir="$1"
  local path
  while IFS= read -r path; do
    if [[ -z "${path}" ]]; then
      return 1
    fi
    local candidate="${source_dir}/$(basename "${path}")"
    if [[ ! -s "${candidate}" ]]; then
      return 1
    fi
  done < <(required_data_cache_files)
  return 0
}

resume_source_data_cache_dir() {
  local load_dir="$1"
  if [[ -z "${load_dir}" ]]; then
    return 1
  fi
  load_dir="${load_dir%/}"
  if [[ "${load_dir}" == */checkpoints/iter_* ]]; then
    printf '%s/data_cache\n' "$(dirname "$(dirname "${load_dir}")")"
    return 0
  fi
  printf '%s/data_cache\n' "$(dirname "${load_dir}")"
}

copy_required_data_cache_files() {
  local source_dir="$1"
  local dest_dir="$2"
  mkdir -p "${dest_dir}"
  local path
  while IFS= read -r path; do
    cp -f "${source_dir}/$(basename "${path}")" "${path}"
  done < <(required_data_cache_files)
}

reuse_resume_data_cache_if_available() {
  if data_cache_dir_has_required_files "${DATA_CACHE_DIR}"; then
    echo "Using existing prepared data_cache from ${DATA_CACHE_DIR}"
    return 0
  fi

  if [[ "${TOOLCALL_RESUME_TRAINING:-0}" != "1" ]]; then
    return 1
  fi

  local source_dir
  source_dir="$(resume_source_data_cache_dir "${TOOL_CALL_LOAD_DIR:-}")" || return 1
  if ! data_cache_dir_has_required_files "${source_dir}"; then
    return 1
  fi

  copy_required_data_cache_files "${source_dir}" "${DATA_CACHE_DIR}"
  echo "Reused prepared data_cache from ${source_dir}"
  return 0
}
