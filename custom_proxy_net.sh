# Copyright (c) Kanana LLM Team - Kakao Corp.

#!/bin/bash
# Run this on a machine with network access to download conda packages
# for offline installation on the target machine.
#
# Usage: bash custom_proxy_net.sh
# Output: .downloads/conda-env.tar.gz, .downloads/pip-pkgs/

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="$SCRIPT_DIR/.envs"
MAMBA_BIN="$ENVS_DIR/bin"
TMPDIR="$ENVS_DIR/tmp"
export TMPDIR
mkdir -p "$MAMBA_BIN" "$TMPDIR"

DOWNLOADS="$SCRIPT_DIR/.downloads"
TARBALL="$DOWNLOADS/conda-env.tar.gz"
PIP_DL="$DOWNLOADS/pip-pkgs"

# install micromamba if not present
if ! command -v micromamba &>/dev/null; then
  BIN_FOLDER="$MAMBA_BIN" INIT_YES=no CONDA_FORGE_YES=no \
    bash "$SCRIPT_DIR/.downloads/mamba_install.sh" < /dev/null
fi
export PATH="$MAMBA_BIN:$PATH"
eval "$(micromamba shell hook -s bash)"

# --- conda env tarball ---
if [ -f "$TARBALL" ]; then
  echo "SKIP: $TARBALL already exists"
else
  # Use a fresh root to ensure all packages are actually downloaded
  DL_ROOT=$(mktemp -d -p "$TMPDIR")
  trap "rm -rf $DL_ROOT" EXIT

  MAMBA_ROOT_PREFIX="$DL_ROOT" micromamba create -n slime \
    python=3.12 pip -c conda-forge -y
  MAMBA_ROOT_PREFIX="$DL_ROOT" micromamba install -n slime \
    cuda cuda-nvtx cuda-nvtx-dev nccl \
    -c nvidia/label/cuda-12.9.1 -c conda-forge -y
  MAMBA_ROOT_PREFIX="$DL_ROOT" micromamba install -n slime \
    cudnn -c conda-forge -y

  # Verify actual package files exist (not just metadata)
  PKG_COUNT=$(find "$DL_ROOT/pkgs" -maxdepth 1 \( -name "*.tar.bz2" -o -name "*.conda" \) | wc -l)
  echo "Downloaded $PKG_COUNT package files"
  if [ "$PKG_COUNT" -lt 5 ]; then
    echo "ERROR: Too few packages downloaded, something went wrong"
    exit 1
  fi

  mkdir -p "$DOWNLOADS"
  tar -czf "$TARBALL" -C "$DL_ROOT" envs/ pkgs/
  echo "conda-env.tar.gz ($(du -h "$TARBALL" | cut -f1)) created."
fi

# --- torch wheels ---
mkdir -p "$PIP_DL"
TORCH_PKGS=(torch-2.9.1 torchvision-0.24.1 torchaudio-2.9.1)
NEED_TORCH=0
for pkg in "${TORCH_PKGS[@]}"; do
  if ! ls "$PIP_DL"/${pkg}* &>/dev/null; then
    NEED_TORCH=1
    break
  fi
done

if [ "$NEED_TORCH" -eq 1 ]; then
  pip download -d "$PIP_DL" \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu129 --proxy ''
  echo "Downloaded $(ls "$PIP_DL" | wc -l) pip wheel(s)"
else
  echo "SKIP: torch wheels already exist in $PIP_DL"
fi

echo ""
echo "Done. Copy to the target machine, then run:"
echo "  bash custom_proxy.sh && bash build_conda.sh"
