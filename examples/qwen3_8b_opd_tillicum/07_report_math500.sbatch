#!/bin/bash
#SBATCH --job-name=slime-qwen3-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${COMBINED_EVAL_OUTPUT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"

echo "Starting MATH-500 report aggregation on $(hostname) at $(date)"
echo "Base eval dir: ${BASE_EVAL_OUTPUT_DIR}"
echo "SFT eval dir: ${SFT_EVAL_OUTPUT_DIR}"
echo "OPD eval dir: ${OPD_EVAL_OUTPUT_DIR}"
echo "Combined report dir: ${COMBINED_EVAL_OUTPUT_DIR}"

aggregate_if_available() {
  local dir="$1"
  local label="$2"
  local summary_path=""
  if [[ -d "${dir}" ]]; then
    summary_path="$(find "${dir}" -mindepth 2 -maxdepth 2 -name summary.json -print -quit)"
  fi

  if [[ -n "${summary_path}" ]]; then
    echo "Aggregating ${label}: ${dir}"
    python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
      --aggregate-dir "${dir}" \
      --out-json "${dir}/summary_all.json"
  else
    echo "No per-stage summaries to aggregate for ${label}: ${dir}"
  fi
}

has_any_stage_summary() {
  local report_dir summary_path
  for report_dir in "${BASE_EVAL_OUTPUT_DIR}" "${SFT_EVAL_OUTPUT_DIR}" "${OPD_EVAL_OUTPUT_DIR}"; do
    summary_path=""
    if [[ -d "${report_dir}" ]]; then
      summary_path="$(find "${report_dir}" -mindepth 2 -maxdepth 2 -name summary.json -print -quit)"
    fi
    if [[ -n "${summary_path}" ]]; then
      return 0
    fi
  done
  return 1
}

aggregate_if_available "${BASE_EVAL_OUTPUT_DIR}" base
aggregate_if_available "${SFT_EVAL_OUTPUT_DIR}" sft
aggregate_if_available "${OPD_EVAL_OUTPUT_DIR}" opd

if has_any_stage_summary; then
  echo "Writing combined eval report: ${COMBINED_EVAL_OUTPUT_DIR}"
  python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
    --combined-output-dir "${COMBINED_EVAL_OUTPUT_DIR}" \
    --base-dir "${BASE_EVAL_OUTPUT_DIR}" \
    --sft-dir "${SFT_EVAL_OUTPUT_DIR}" \
    --opd-dir "${OPD_EVAL_OUTPUT_DIR}" \
    --sft-total-samples "$((SFT_NUM_ROLLOUT * SFT_ROLLOUT_BATCH_SIZE))" \
    --sft-rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}" \
    --opd-rollout-batch-size "${OPD_ROLLOUT_BATCH_SIZE}" \
    --out-json "${COMBINED_EVAL_OUTPUT_DIR}/summary_all.json"
else
  echo "No per-stage summaries available for combined report yet."
fi

echo "Finished MATH-500 report aggregation at $(date)"
