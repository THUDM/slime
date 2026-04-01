#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LANG="${1:-all}"
PORT="${PORT:-8000}"

cd "$SCRIPT_DIR"

if [ "$LANG" = "all" ]; then
    # Expect both builds present
    if [ ! -d build/en ] || [ ! -d build/zh ]; then
        echo "[serve] Missing build/en or build/zh. Run ./build_all.sh first." >&2
    fi
    echo "[serve] Serving multi-language docs root on http://localhost:$PORT (en/, zh/)"
    python -m http.server -d ./build "$PORT"
    exit $?
fi

if [ "$LANG" != "en" ] && [ "$LANG" != "zh" ]; then
    echo "Usage: $0 [en|zh|all]" >&2
    exit 1
fi

if [ ! -d "build/$LANG" ]; then
    echo "[serve] build/$LANG not found. Run ./build.sh $LANG first." >&2
    exit 1
fi
echo "[serve] Serving $LANG docs on http://localhost:$PORT" 
python -m http.server -d ./build/$LANG "$PORT"