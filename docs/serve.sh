#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LANG=$1

# make sure language is only en or zh
if [ "$LANG" != "en" ] && [ "$LANG" != "zh" ]; then
    echo "Language must be en or zh"
    exit 1
fi

cd $SCRIPT_DIR
python -m http.server -d ./build/$LANG 8000