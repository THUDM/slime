#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

SBATCH_COMMON=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")
SBATCH_REPORT_COMMON=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")
if [[ -n "${GPU_GRES:-}" ]]; then
  SBATCH_COMMON+=(--gres "${GPU_GRES}")
fi

PREV_SFT_EVAL_JOB_ID="${PREV_SFT_EVAL_JOB_ID:-152211}"
submit_log="${SLURM_LOG_DIR}/submit_resume_sft_eval_then_opd_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting SFT eval continuation / OPD / base / backfill chain"
echo "previous SFT eval job: ${PREV_SFT_EVAL_JOB_ID}"
echo "account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "submit log: ${submit_log}"

jid_sft_eval_resume="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterany:${PREV_SFT_EVAL_JOB_ID} \
    --time=08:00:00 \
    --job-name=slime-qwen3-sft-math500-resume \
    --export=ALL,EVAL_TARGETS=sft,EVAL_OUTPUT_DIR="${SFT_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_sft_report="$(
  sbatch --parsable "${SBATCH_REPORT_COMMON[@]}" \
    --dependency=afterany:${jid_sft_eval_resume} \
    --time=00:30:00 \
    --job-name=slime-qwen3-sft-report \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"
jid_opd="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_sft_eval_resume} \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd_eval_final="$(
  OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}" \
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_opd} \
    --time=03:00:00 \
    --job-name=slime-qwen3-opd-final-math500 \
    --export=ALL,EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_base_eval="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_opd_eval_final} \
    --time=03:00:00 \
    --job-name=slime-qwen3-base-math500 \
    --export=ALL,EVAL_TARGETS=base,EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_interim_report="$(
  sbatch --parsable "${SBATCH_REPORT_COMMON[@]}" \
    --dependency=afterok:${jid_base_eval} \
    --time=00:30:00 \
    --job-name=slime-qwen3-interim-report \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"
jid_opd_eval_backfill="$(
  OPD_MILESTONE_ROLLOUT_IDS="47 23" \
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_interim_report} \
    --time=05:00:00 \
    --job-name=slime-qwen3-opd-backfill-math500 \
    --export=ALL,EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_final_report="$(
  sbatch --parsable "${SBATCH_REPORT_COMMON[@]}" \
    --dependency=afterany:${jid_opd_eval_backfill} \
    --time=00:30:00 \
    --job-name=slime-qwen3-final-report \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "prev_sft_eval=${PREV_SFT_EVAL_JOB_ID}"
  echo "sft_eval_resume=${jid_sft_eval_resume}"
  echo "sft_report=${jid_sft_report}"
  echo "opd=${jid_opd}"
  echo "opd_eval_final=${jid_opd_eval_final}"
  echo "base_eval=${jid_base_eval}"
  echo "interim_report=${jid_interim_report}"
  echo "opd_eval_backfill=${jid_opd_eval_backfill}"
  echo "final_report=${jid_final_report}"
  echo "SFT_EVAL_OUTPUT_DIR=${SFT_EVAL_OUTPUT_DIR}"
  echo "OPD_EVAL_OUTPUT_DIR=${OPD_EVAL_OUTPUT_DIR}"
  echo "BASE_EVAL_OUTPUT_DIR=${BASE_EVAL_OUTPUT_DIR}"
  echo "COMBINED_EVAL_OUTPUT_DIR=${COMBINED_EVAL_OUTPUT_DIR}"
  echo "SFT_SAVE_DIR=${SFT_SAVE_DIR}"
  echo "OPD_SAVE_DIR=${OPD_SAVE_DIR}"
  echo "CHECKPOINT_REPORT_DIR=${CHECKPOINT_REPORT_DIR}"
} | tee "${submit_log}"
