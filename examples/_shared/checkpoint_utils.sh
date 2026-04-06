#!/usr/bin/env bash

list_checkpoint_iterations() {
  local ckpt_dir="$1"
  local iter_dir=""
  local iter_name=""
  local iter_num=""

  for iter_dir in "${ckpt_dir}"/iter_*; do
    [[ -d "${iter_dir}" ]] || continue
    iter_name="$(basename "${iter_dir}")"
    iter_num="${iter_name#iter_}"
    printf '%d\n' "$((10#${iter_num}))"
  done | sort -n
}

convert_torch_dist_checkpoint_to_hf() {
  local iter_dir="$1"
  local hf_out="$2"

  if [[ -f "${hf_out}/config.json" ]] && [[ "${FORCE_RECONVERT:-0}" != "1" ]]; then
    echo "  HF checkpoint already exists: ${hf_out}"
    return 0
  fi

  rm -rf "${hf_out}"
  echo "  Converting torch_dist -> HF: ${iter_dir} -> ${hf_out}"
  python3 "${PROJECT_ROOT}/slime/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${iter_dir}" \
    --output-dir "${hf_out}" \
    --origin-hf-dir "${MODEL_DIR}" \
    --force
}
