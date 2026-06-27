#!/bin/bash
#SBATCH --job-name=slime-qwen3-convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting model prep on $(hostname) at $(date)"
echo "Student HF: ${STUDENT_HF_REPO} -> ${STUDENT_HF_DIR}"
echo "Teacher HF: ${TEACHER_HF_REPO} -> ${TEACHER_HF_DIR}"
echo "Student torch_dist: ${STUDENT_TORCH_DIST_DIR}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
mkdir -p "${MODEL_ROOT}" "${DATA_ROOT}"

if [[ ! -d "${STUDENT_HF_DIR}" || -z "$(ls -A "${STUDENT_HF_DIR}" 2>/dev/null)" ]]; then
  hf download "${STUDENT_HF_REPO}" --local-dir "${STUDENT_HF_DIR}"
else
  echo "Student HF snapshot already exists: ${STUDENT_HF_DIR}"
fi

if [[ ! -d "${TEACHER_HF_DIR}" || -z "$(ls -A "${TEACHER_HF_DIR}" 2>/dev/null)" ]]; then
  hf download "${TEACHER_HF_REPO}" --local-dir "${TEACHER_HF_DIR}"
else
  echo "Teacher HF snapshot already exists: ${TEACHER_HF_DIR}"
fi

hf download --repo-type dataset "${MATH500_DATASET}" --local-dir "${DATA_ROOT}/math500_hf_snapshot" || true

if [[ -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "Student torch_dist conversion already exists: ${STUDENT_TORCH_DIST_DIR}"
  exit 0
fi

source scripts/models/qwen3-8B.sh
mkdir -p "${STUDENT_TORCH_DIST_DIR}"
PYTHONPATH="/root/Megatron-LM:${SLIME_REPO_ROOT}:${PYTHONPATH:-}" \
torchrun --nproc_per_node="${CONVERT_NPROC:-8}" tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${STUDENT_HF_DIR}" \
  --save "${STUDENT_TORCH_DIST_DIR}"
'

echo "Finished model prep at $(date)"
