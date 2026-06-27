#!/usr/bin/env bash

# Source this file from the repository root or from any script in this
# directory. All generated state is kept under scrubbed/scratch storage.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it:"
  echo "  source ${BASH_SOURCE[0]}"
  exit 2
fi

export TILLICUM_EXAMPLE_DIR
TILLICUM_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export SLIME_REPO_ROOT
SLIME_REPO_ROOT="$(cd -- "${TILLICUM_EXAMPLE_DIR}/../.." >/dev/null 2>&1 && pwd)"

export ACCOUNT="${ACCOUNT:-raivn}"
export PARTITION="${PARTITION:-gpu-h200}"
export QOS="${QOS:-normal}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd}"
export DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/data}"
export MODEL_ROOT="${MODEL_ROOT:-${SCRATCH_ROOT}/models}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/outputs}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH_ROOT}/ray_tmp}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_ROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_ROOT}/apptainer_tmp}"
export SLIME_IMAGE_URI="${SLIME_IMAGE_URI:-docker://slimerl/slime:latest}"
export SLIME_CONTAINER_FORMAT="${SLIME_CONTAINER_FORMAT:-sandbox}"
if [[ -z "${SLIME_SIF:-}" ]]; then
  if [[ "${SLIME_CONTAINER_FORMAT}" == "sandbox" ]]; then
    export SLIME_SIF="${SCRATCH_ROOT}/containers/slime_latest.sandbox"
  else
    export SLIME_SIF="${SCRATCH_ROOT}/containers/slime_latest.sif"
  fi
else
  export SLIME_SIF
fi
export CONTAINER_BIND_ROOTS="${CONTAINER_BIND_ROOTS:-/gpfs/scrubbed/suryadv,/tmp}"
export CONTAINER_HOME="${CONTAINER_HOME:-${SCRATCH_ROOT}/container_home}"
export CONTAINER_HOME_INNER="${CONTAINER_HOME_INNER:-/home/${USER:-slime}}"

export STUDENT_HF_REPO="${STUDENT_HF_REPO:-Qwen/Qwen3-8B-Base}"
export TEACHER_HF_REPO="${TEACHER_HF_REPO:-Qwen/Qwen3-32B}"
export OT3_DATASET="${OT3_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
export MATH500_DATASET="${MATH500_DATASET:-HuggingFaceH4/MATH-500}"

export STUDENT_HF_DIR="${STUDENT_HF_DIR:-${MODEL_ROOT}/Qwen3-8B-Base}"
export TEACHER_HF_DIR="${TEACHER_HF_DIR:-${MODEL_ROOT}/Qwen3-32B}"
export STUDENT_TORCH_DIST_DIR="${STUDENT_TORCH_DIST_DIR:-${MODEL_ROOT}/Qwen3-8B-Base_torch_dist}"

export SFT_SIZE="${SFT_SIZE:-100000}"
export OPD_SIZE="${OPD_SIZE:-50000}"
export DATA_SEED="${DATA_SEED:-1234}"
export DATA_MATH_FIELD="${DATA_MATH_FIELD:-domain}"
export DATA_MATH_VALUE="${DATA_MATH_VALUE:-math}"
export OT3_SPLIT="${OT3_SPLIT:-train}"

export SFT_PARQUET="${SFT_PARQUET:-${DATA_ROOT}/openthoughts3_math_sft_${SFT_SIZE}.parquet}"
export OPD_JSONL="${OPD_JSONL:-${DATA_ROOT}/openthoughts3_math_opd_${OPD_SIZE}.jsonl}"
export SPLIT_METADATA="${SPLIT_METADATA:-${DATA_ROOT}/openthoughts3_math_split_metadata.json}"
export MATH500_JSONL="${MATH500_JSONL:-${DATA_ROOT}/math500_deepscaler.jsonl}"
export MATH500_CONFIG="${MATH500_CONFIG:-${DATA_ROOT}/math500_eval.yaml}"

export SFT_SAVE_DIR="${SFT_SAVE_DIR:-${OUTPUT_ROOT}/qwen3_8b_sft_100k}"
export OPD_SAVE_DIR="${OPD_SAVE_DIR:-${OUTPUT_ROOT}/qwen3_8b_sft_100k_opd_50k}"
export SFT_DETAILS_DIR="${SFT_DETAILS_DIR:-${OUTPUT_ROOT}/sft_details}"
export OPD_ROLLOUT_LOG_DIR="${OPD_ROLLOUT_LOG_DIR:-${OUTPUT_ROOT}/opd_rollout_logs}"
export TEACHER_LOG_DIR="${TEACHER_LOG_DIR:-${OUTPUT_ROOT}/teacher_logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/math500_eval_1x}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${OUTPUT_ROOT}/slurm_logs}"

export SFT_NUM_EPOCH="${SFT_NUM_EPOCH:-1}"
export SFT_ROLLOUT_BATCH_SIZE="${SFT_ROLLOUT_BATCH_SIZE:-256}"
export SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-256}"
export SFT_MAX_TOKENS_PER_GPU="${SFT_MAX_TOKENS_PER_GPU:-16384}"
export SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-50}"

export OPD_NUM_ROLLOUT="${OPD_NUM_ROLLOUT:-391}"
export OPD_ROLLOUT_BATCH_SIZE="${OPD_ROLLOUT_BATCH_SIZE:-128}"
export OPD_GLOBAL_BATCH_SIZE="${OPD_GLOBAL_BATCH_SIZE:-128}"
export OPD_N_SAMPLES_PER_PROMPT="${OPD_N_SAMPLES_PER_PROMPT:-1}"
export OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN:-16384}"
export OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU:-16384}"
export OPD_SAVE_INTERVAL="${OPD_SAVE_INTERVAL:-50}"
export OPD_ACTOR_GPUS="${OPD_ACTOR_GPUS:-2}"
export OPD_ROLLOUT_GPUS="${OPD_ROLLOUT_GPUS:-5}"
export OPD_RAY_GPUS="${OPD_RAY_GPUS:-7}"
export OPD_TEACHER_GPU="${OPD_TEACHER_GPU:-7}"
export OPD_TEACHER_PORT="${OPD_TEACHER_PORT:-13141}"
export OPD_TEACHER_MEM_FRACTION="${OPD_TEACHER_MEM_FRACTION:-0.6}"

export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32768}"
export EVAL_ROLLOUT_BATCH_SIZE="${EVAL_ROLLOUT_BATCH_SIZE:-64}"
export EVAL_ROLLOUT_NUM_GPUS="${EVAL_ROLLOUT_NUM_GPUS:-8}"
export EVAL_NUM_REPEATS="${EVAL_NUM_REPEATS:-1}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
