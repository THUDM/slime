#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOC_LANG="${1:-all}"
PORT="${PORT:-8000}"

cd "$SCRIPT_DIR"

if [ "$DOC_LANG" = "all" ]; then
    # Expect both builds present
    if [ ! -d build/en ] || [ ! -d build/zh ]; then
        echo "[serve] Missing build/en or build/zh. Run ./build_all.sh first." >&2
        exit 1
    fi
    echo "[serve] Serving multi-language docs root on http://localhost:$PORT (en/, zh/)"
    python -m http.server -d ./build "$PORT"
    exit $?
fi

if [ "$DOC_LANG" != "en" ] && [ "$DOC_LANG" != "zh" ]; then
    echo "Usage: $0 [en|zh|all]" >&2
    exit 1
fi

if [ ! -d "build/$DOC_LANG" ]; then
    echo "[serve] build/$DOC_LANG not found. Run ./build.sh $DOC_LANG first." >&2
    exit 1
fi
echo "[serve] Serving $DOC_LANG docs on http://localhost:$PORT"
python -m http.server -d "./build/$DOC_LANG" "$PORT"
