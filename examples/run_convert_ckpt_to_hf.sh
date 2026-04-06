#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the experiment path}"
MODEL_DIR="${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints/iter_0001432_hf}"
HF_CACHE="${HF_CACHE:-${EXPERIMENT_DIR}/hf_cache}"
FORCE_RECONVERT="${FORCE_RECONVERT:-0}"

CKPT_DIR="${EXPERIMENT_DIR}/checkpoints"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

mkdir -p "${HF_CACHE}" "${EXPERIMENT_DIR}/logs"

ITERATIONS=()
for iter_dir in "${CKPT_DIR}"/iter_*; do
  [[ -d "${iter_dir}" ]] || continue
  iter_name="$(basename "${iter_dir}")"
  iter_num="${iter_name#iter_}"
  iter_num="$((10#${iter_num}))"
  ITERATIONS+=("${iter_num}")
done

if [[ ${#ITERATIONS[@]} -eq 0 ]]; then
  echo "ERROR: no checkpoints found under ${CKPT_DIR}" >&2
  exit 1
fi

IFS=$'\n' ITERATIONS=($(sort -n <<<"${ITERATIONS[*]}"))
unset IFS

echo "=== Convert checkpoints to HF ==="
echo "Experiment: ${EXPERIMENT_DIR}"
echo "Origin HF:  ${MODEL_DIR}"
echo "Output:     ${HF_CACHE}"
echo "Found ${#ITERATIONS[@]} checkpoints: ${ITERATIONS[*]}"

converted=0
skipped=0
failed=0

for iter_num in "${ITERATIONS[@]}"; do
  iter_name="$(printf 'iter_%07d' "${iter_num}")"
  iter_dir="${CKPT_DIR}/${iter_name}"
  hf_out="${HF_CACHE}/${iter_name}_hf"

  echo ""
  echo "------ ${iter_name} ------"
  if [[ -f "${hf_out}/config.json" ]] && [[ "${FORCE_RECONVERT}" != "1" ]]; then
    echo "Skip existing HF checkpoint: ${hf_out}"
    skipped=$((skipped + 1))
    continue
  fi

  rm -rf "${hf_out}"
  if python3 "${PROJECT_ROOT}/slime/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${iter_dir}" \
    --output-dir "${hf_out}" \
    --origin-hf-dir "${MODEL_DIR}" \
    --force; then
    converted=$((converted + 1))
  else
    echo "ERROR: conversion failed for ${iter_name}" >&2
    failed=$((failed + 1))
  fi
done

echo ""
echo "=== Conversion complete ==="
echo "Converted: ${converted}"
echo "Skipped:   ${skipped}"
echo "Failed:    ${failed}"

if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
