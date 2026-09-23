# Copyright (c) Kanana LLM Team - Kakao Corp.

#!/bin/bash
# Run this on the target machine (no network required).
# Installs micromamba and restores the conda package cache
# so that build_conda.sh can run with --offline.
#
# Usage: bash custom_proxy.sh

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hardcoded to /root to match the previous build_conda.sh (Docker HOME=/root)
MAMBA_BIN="/root/.local/bin"
MAMBA_ROOT="/root/micromamba"
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"

if ! command -v micromamba &>/dev/null; then
  BIN_FOLDER="$MAMBA_BIN" INIT_YES=yes CONDA_FORGE_YES=yes \
    PREFIX_LOCATION="$MAMBA_ROOT" \
    bash "$SCRIPT_DIR/.downloads/mamba_install.sh" < /dev/null
fi
export PATH="$MAMBA_BIN:$PATH"
ENV_TARBALL="$SCRIPT_DIR/.downloads/conda-env.tar.gz"
if [ -f "$ENV_TARBALL" ]; then
  echo "Restoring conda environment to $MAMBA_ROOT ..."
  mkdir -p "$MAMBA_ROOT"
  tar -xzf "$ENV_TARBALL" -C "$MAMBA_ROOT"

  # Fix hardcoded shebangs from the build machine's temp directory
  SLIME_BIN="$MAMBA_ROOT/envs/slime/bin"
  TARGET_PYTHON="$SLIME_BIN/python"
  if [ -d "$SLIME_BIN" ]; then
    echo "Fixing shebangs in $SLIME_BIN ..."
    find "$SLIME_BIN" -maxdepth 1 -type f | while read -r f; do
      if head -c 2 "$f" | grep -q '^#!'; then
        sed -i "1s|^#!.*/envs/slime/bin/python|#!${TARGET_PYTHON}|" "$f"
      fi
    done
  fi

  echo "Done. slime env restored to $MAMBA_ROOT/envs/slime/"
else
  echo "WARNING: $ENV_TARBALL not found. Run custom_proxy_net.sh on a networked machine first."
  exit 1
fi

source ~/.bashrc
