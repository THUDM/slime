#!/usr/bin/env bash
set -euo pipefail
TASK_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${TASK_EXAMPLE_DIR}/run.sh" grpo "$@"
