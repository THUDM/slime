#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

SBATCH_COMMON=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")
if [[ -n "${GPU_GRES:-}" ]]; then
  SBATCH_COMMON+=(--gres "${GPU_GRES}")
fi

submit_log="${SLURM_LOG_DIR}/submit_25k_10k_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting 25k SFT / 10k OPD chain"
echo "account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "submit log: ${submit_log}"

jid_data="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    examples/qwen3_8b_opd_tillicum/02_prepare_data_25k_10k.sbatch
)"
jid_convert="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch
)"
jid_sft="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_data}:${jid_convert} \
    examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
)"
jid_sft_eval="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_sft} \
    --time=05:00:00 \
    --job-name=slime-qwen3-sft-math500 \
    --export=ALL,EVAL_TARGETS=sft,EVAL_OUTPUT_DIR="${SFT_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_opd="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_sft_eval} \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd_eval="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_opd} \
    --time=05:00:00 \
    --job-name=slime-qwen3-opd-math500 \
    --export=ALL,EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_base_eval="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency=afterok:${jid_opd_eval} \
    --time=02:00:00 \
    --job-name=slime-qwen3-base-math500 \
    --export=ALL,EVAL_TARGETS=base,EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"

{
  echo "data=${jid_data}"
  echo "convert=${jid_convert}"
  echo "sft=${jid_sft}"
  echo "sft_eval=${jid_sft_eval}"
  echo "opd=${jid_opd}"
  echo "opd_eval=${jid_opd_eval}"
  echo "base_eval=${jid_base_eval}"
  echo "SFT_SAVE_DIR=${SFT_SAVE_DIR}"
  echo "OPD_SAVE_DIR=${OPD_SAVE_DIR}"
  echo "SFT_EVAL_OUTPUT_DIR=${SFT_EVAL_OUTPUT_DIR}"
  echo "OPD_EVAL_OUTPUT_DIR=${OPD_EVAL_OUTPUT_DIR}"
  echo "CHECKPOINT_REPORT_DIR=${CHECKPOINT_REPORT_DIR}"
} | tee "${submit_log}"
