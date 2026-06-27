#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]"
  exit 2
fi

if [[ ! -e "${SLIME_SIF}" ]]; then
  cat >&2 <<EOF
Missing container image/sandbox:
  ${SLIME_SIF}

Create it with:
  bash ${SCRIPT_DIR}/00_pull_or_load_container.sh
EOF
  exit 1
fi

APPTAINER_BIN="${APPTAINER_BIN:-}"
if [[ -z "${APPTAINER_BIN}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    APPTAINER_BIN=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    APPTAINER_BIN=singularity
  else
    echo "Neither apptainer nor singularity is available on PATH." >&2
    exit 1
  fi
fi

mkdir -p \
  "${APPTAINER_CACHEDIR}" \
  "${APPTAINER_TMPDIR}" \
  "${CONTAINER_HOME}" \
  "${TMPDIR}" \
  "${RAY_TMPDIR}"

export APPTAINER_CACHEDIR APPTAINER_TMPDIR

IFS=',' read -r -a BIND_ROOTS <<< "${CONTAINER_BIND_ROOTS}"
APPTAINER_ARGS=(
  exec
  --nv
  --cleanenv
  --ipc
  --writable-tmpfs
  --cwd "${SLIME_REPO_ROOT}"
  --home "${CONTAINER_HOME}:${CONTAINER_HOME_INNER}"
)

for bind_root in "${BIND_ROOTS[@]}"; do
  if [[ -e "${bind_root}" ]]; then
    APPTAINER_ARGS+=(--bind "${bind_root}:${bind_root}")
  fi
done

PASS_ENV=(
  ACCOUNT
  PARTITION
  QOS
  SCRATCH_ROOT
  DATA_ROOT
  MODEL_ROOT
  OUTPUT_ROOT
  HF_HOME
  HF_DATASETS_CACHE
  TRANSFORMERS_CACHE
  WANDB_MODE
  WANDB_DIR
  TMPDIR
  RAY_TMPDIR
  SLIME_REPO_ROOT
  TILLICUM_EXAMPLE_DIR
  STUDENT_HF_REPO
  TEACHER_HF_REPO
  OT3_DATASET
  MATH500_DATASET
  STUDENT_HF_DIR
  TEACHER_HF_DIR
  STUDENT_TORCH_DIST_DIR
  SFT_PARQUET
  OPD_JSONL
  SPLIT_METADATA
  MATH500_JSONL
  MATH500_CONFIG
  SFT_SAVE_DIR
  OPD_SAVE_DIR
  SFT_DETAILS_DIR
  OPD_ROLLOUT_LOG_DIR
  TEACHER_LOG_DIR
  EVAL_OUTPUT_DIR
  SLURM_LOG_DIR
  PYTHONUNBUFFERED
  TOKENIZERS_PARALLELISM
  NCCL_DEBUG
)

for name in "${PASS_ENV[@]}"; do
  if [[ -n "${!name-}" ]]; then
    APPTAINER_ARGS+=(--env "${name}=${!name}")
  fi
done

APPTAINER_ARGS+=(
  --env "HOME=${CONTAINER_HOME_INNER}"
  --env "PYTHONPATH=${SLIME_REPO_ROOT}:/root/Megatron-LM:${PYTHONPATH:-}"
  --env "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
)

"${APPTAINER_BIN}" "${APPTAINER_ARGS[@]}" "${SLIME_SIF}" "$@"
