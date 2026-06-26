#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

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

mkdir -p "$(dirname "${SLIME_SIF}")" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR

cat <<EOF
Container setup
  image uri: ${SLIME_IMAGE_URI}
  SIF path : ${SLIME_SIF}
  cache    : ${APPTAINER_CACHEDIR}
  tmp      : ${APPTAINER_TMPDIR}
EOF

if [[ -f "${SLIME_SIF}" && "${FORCE_PULL:-0}" != "1" ]]; then
  echo "Container already exists. Set FORCE_PULL=1 to replace it."
  exit 0
fi

"${APPTAINER_BIN}" pull --force "${SLIME_SIF}" "${SLIME_IMAGE_URI}"
echo "Wrote ${SLIME_SIF}"
