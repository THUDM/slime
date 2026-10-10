#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOC_LANG="${1:-}"

if [ "$DOC_LANG" != "en" ] && [ "$DOC_LANG" != "zh" ]; then
    echo "Usage: $0 [en|zh] [sphinx-build options...]" >&2
    exit 1
fi
shift

SPHINX_BUILD="$(command -v sphinx-build || true)"
if [ -z "$SPHINX_BUILD" ] && [ -x "$SCRIPT_DIR/../.venv-docs/bin/sphinx-build" ]; then
    SPHINX_BUILD="$SCRIPT_DIR/../.venv-docs/bin/sphinx-build"
fi
if [ -z "$SPHINX_BUILD" ]; then
    cat >&2 <<'EOF'
[slime-docs] sphinx-build was not found. From the repository root, run:
  python3 -m venv .venv-docs
  .venv-docs/bin/python -m pip install -r docs/requirements.txt
Then retry: bash docs/build_all.sh
EOF
    exit 1
fi

cd "$SCRIPT_DIR"
SLIME_DOC_LANG="$DOC_LANG" "$SPHINX_BUILD" -b html -D language="$DOC_LANG" --conf-dir . \
    "$@" "./$DOC_LANG" "./build/$DOC_LANG"
