#!/usr/bin/env bash

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_QUERIES_PY="${SCRIPT_QUERIES_PY:-${COMMON_DIR}/system_queries.py}"

parse_csv_to_args() {
  local csv="$1"
  local flag="$2"
  local out_name="$3"
  local -n out_ref="${out_name}"
  local item
  local items=()

  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      out_ref+=("${flag}" "${item}")
    fi
  done
}

detect_nvlink() {
  local count
  count=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
  if [[ "${count}" -gt 0 ]]; then
    printf '1\n'
  else
    printf '0\n'
  fi
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

ensure_torch_dist_checkpoint() {
  if [[ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Found torch_dist checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi
  if [[ -f "${TORCH_DIST_DIR}/common.pt" ]] && [[ -f "${TORCH_DIST_DIR}/metadata.json" ]]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${SLIME_DIR}"
  # shellcheck disable=SC1091
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"
  PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}" torchrun \
    --nproc-per-node "${NUM_GPUS_PER_NODE}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
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
