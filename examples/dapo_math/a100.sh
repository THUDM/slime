#!/bin/bash
# ── Cluster: ib-a100 (A100 80GB, SM80) ──────────────────────────────────────

export SLIME_CLUSTER=a100
export BASE_DIR=../../slime
source ${BASE_DIR}/envs/.env

# ── Common ───────────────────────────────────────────────────────────────────
export PYTHONBUFFERED=16
export OMP_NUM_THREADS=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLIME_FAST_CLEAR_MEMORY=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

# ── NCCL ─────────────────────────────────────────────────────────────────────
export NCCL_NVLS_ENABLE=0  # A100: no NVLS

# ── Wandb ────────────────────────────────────────────────────────────────────
export WANDB_API_KEY=-
export WANDB_ENTITY=-

# ── Paths ──────────────────────────────────────────────────────────────────
export REPO_ROOT="/root"
export UPSTREAM="/root/slime"
export MEGATRON_PATH="/root/Megatron-LM"

# ── Exclude problematic nodes ───────────────────────────────────────────────
export SLIME_EXCLUDE_NODES=""
export RAY_DEDUP_LOGS=0

# ── Container ────────────────────────────────────────────────────────────────
CONTAINER_IMAGE="${BASE_DIR}/images/slimerl+sglang+v0.5.9-slime-autoenv2.sqsh"
# nvidia+pytorch+25.06-py3.sqsh"
CONTAINER_MOUNTS="/tmp:/tmp"
CONTAINER_WORKDIR="${REPO_ROOT}"
SRUN_CONTAINER=(
    --container-image="${CONTAINER_IMAGE}"
    --container-mounts="${CONTAINER_MOUNTS}"
    --container-workdir="${CONTAINER_WORKDIR}"
    --container-writable
    --export="ALL"
)
