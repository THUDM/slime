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

validate_container() {
  echo "Validating container Python/imports"
  "${APPTAINER_BIN}" exec --cleanenv "${SLIME_SIF}" \
    python3 -c "import encodings, sglang, torch; print('container ok')"
}

mkdir -p "$(dirname "${SLIME_SIF}")" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
export APPTAINER_CACHEDIR APPTAINER_TMPDIR

cat <<EOF
Container setup
  image uri: ${SLIME_IMAGE_URI}
  format   : ${SLIME_CONTAINER_FORMAT}
  path     : ${SLIME_SIF}
  cache    : ${APPTAINER_CACHEDIR}
  tmp      : ${APPTAINER_TMPDIR}
EOF

if [[ -e "${SLIME_SIF}" && "${FORCE_PULL:-0}" != "1" ]]; then
  echo "Container already exists. Validating it now."
  if validate_container; then
    echo "Existing container is valid: ${SLIME_SIF}"
  else
    cat >&2 <<EOF
Existing container failed validation:
  ${SLIME_SIF}

Move it aside or remove it manually, then rerun this script to rebuild it.
EOF
    exit 1
  fi
  exit 0
fi

if [[ -e "${SLIME_SIF}" && "${FORCE_PULL:-0}" == "1" ]]; then
  echo "Refusing to overwrite existing ${SLIME_SIF} automatically." >&2
  echo "Move it aside or remove it manually, then rerun this script." >&2
  exit 1
fi

if [[ "${SLIME_CONTAINER_FORMAT}" == "sandbox" ]]; then
  "${APPTAINER_BIN}" build --sandbox "${SLIME_SIF}" "${SLIME_IMAGE_URI}"
else
  "${APPTAINER_BIN}" pull --force "${SLIME_SIF}" "${SLIME_IMAGE_URI}"
fi

validate_container
echo "Wrote and validated ${SLIME_SIF}"
