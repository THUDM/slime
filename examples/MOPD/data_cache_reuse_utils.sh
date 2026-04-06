#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../_shared/data_cache_reuse_utils.sh"
