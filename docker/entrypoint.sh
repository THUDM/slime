#!/bin/bash
set -e
source /opt/slime/cuda-compat.sh
exec /opt/nvidia/nvidia_entrypoint.sh "$@"
