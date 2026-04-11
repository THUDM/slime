#!/bin/bash
#SBATCH --job-name=K-kevin-build
#SBATCH --nodelist=ib-a100-cluster-a-n[001-256]
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=build_log/build_conda_%j.log
#SBATCH --error=build_log/build_conda_%j.log

set -ex

SCRIPT_DIR="/mnt/datafs/ib-a100-cluster-a-pri/lmt/users/kevin/slime"
IMAGE_DIR="/mnt/datafs/ib-a100-cluster-a-pri/lmt/images"
BASE_SQSH="$IMAGE_DIR/slimerl+sglang+v0.5.9.sqsh"
OUTPUT_SQSH="$IMAGE_DIR/slimerl+sglang+v0.5.9-slime.sqsh"
CONTAINER_NAME="slime_build_$$"
DATA_MOUNT="/mnt/datafs:/mnt/datafs"

# mkdir -p "$SCRIPT_DIR/build_log"

# Create container from base image
enroot create --name "$CONTAINER_NAME" "$BASE_SQSH"

# Run build inside the container
enroot start --rw --mount "$DATA_MOUNT" "$CONTAINER_NAME" \
  bash -c "
    set -ex
    export PATCH_VERSION=v0.5.9.a100
    cd $SCRIPT_DIR
    bash custom_proxy.sh
    bash build_conda.sh
  "

# Export the modified container as a new sqsh image
enroot export -o "$OUTPUT_SQSH" "$CONTAINER_NAME"

# Cleanup
enroot remove -- "$CONTAINER_NAME"

echo ""
echo "Done. Image saved to $OUTPUT_SQSH"
