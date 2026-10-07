#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOC_LANG="${1:-}"

if [ "$DOC_LANG" != "en" ] && [ "$DOC_LANG" != "zh" ]; then
    echo "Usage: $0 [en|zh] [sphinx-build options...]" >&2
    exit 1
fi
shift

cd "$SCRIPT_DIR"
SLIME_DOC_LANG="$DOC_LANG" sphinx-build -b html -D language="$DOC_LANG" --conf-dir . \
    "$@" "./$DOC_LANG" "./build/$DOC_LANG"
