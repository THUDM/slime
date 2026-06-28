#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SLIME_REPO_ROOT}"

echo "Dry check environment"
echo "  repo: ${SLIME_REPO_ROOT}"
echo "  account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "  scratch: ${SCRATCH_ROOT}"
echo "  container (${SLIME_CONTAINER_FORMAT}): ${SLIME_SIF}"

SHELL_FILES=(
  examples/qwen3_8b_opd_tillicum/env.sh
  examples/qwen3_8b_opd_tillicum/checkpoint_utils.sh
  examples/qwen3_8b_opd_tillicum/container_exec.sh
  examples/qwen3_8b_opd_tillicum/00_pull_or_load_container.sh
  examples/qwen3_8b_opd_tillicum/01_prepare_env.sh
  examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh
  examples/qwen3_8b_opd_tillicum/submit_25k_10k_chain.sh
  examples/qwen3_8b_opd_tillicum/submit_resume_sft_eval_then_opd_chain.sh
  examples/qwen3_8b_opd_tillicum/02_prepare_data_25k_10k.sbatch
  examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch
  examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
  examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)

PYTHON_FILES=(
  examples/qwen3_8b_opd_tillicum/02_prepare_openthoughts3_math_sample.py
  examples/qwen3_8b_opd_tillicum/summarize_eval.py
)

echo "Checking shell syntax"
for file in "${SHELL_FILES[@]}"; do
  bash -n "${file}"
done

echo "Checking Python syntax"
python3 -m py_compile "${PYTHON_FILES[@]}"

SBATCH_FILES=(
  examples/qwen3_8b_opd_tillicum/02_prepare_data_25k_10k.sbatch
  examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch
  examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
  examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
  examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)

echo "Checking Slurm scripts with sbatch --test-only"
for file in "${SBATCH_FILES[@]}"; do
  sbatch --test-only -A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" "${file}"
done

if [[ "${RUN_CONTAINER_CHECKS:-0}" == "1" ]]; then
  if [[ -e "${SLIME_SIF}" ]]; then
    echo "Checking imports inside container"
    "${SCRIPT_DIR}/container_exec.sh" python3 -c "import encodings, slime, sglang, torch, transformers, datasets; print('container imports ok')"
  else
    echo "RUN_CONTAINER_CHECKS=1 but SLIME_SIF does not exist; skipping import check."
  fi
else
  echo "Skipping container import check. Set RUN_CONTAINER_CHECKS=1 after the SIF exists."
fi

echo "git diff --stat"
git diff --stat

echo "git diff --name-only"
git diff --name-only

echo "git status --short"
git status --short

echo "Dry checks completed. No real jobs were submitted."
