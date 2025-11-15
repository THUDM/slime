#!/bin/bash

# Helper script to start retrieval server in tmux
# This avoids complex escaping issues with tmux send-keys

set -e

# Parse arguments
INDEX_PATH="$1"
CORPUS_PATH="$2"
TOPK="$3"
RETRIEVER_NAME="$4"
RETRIEVER_MODEL="$5"
LOG_FILE="$6"

# Log startup
echo "Starting retrieval server at $(date)" | tee "${LOG_FILE}"
echo "Index: ${INDEX_PATH}" | tee -a "${LOG_FILE}"
echo "Corpus: ${CORPUS_PATH}" | tee -a "${LOG_FILE}"
echo "Top-K: ${TOPK}" | tee -a "${LOG_FILE}"
echo "Retriever: ${RETRIEVER_NAME}" | tee -a "${LOG_FILE}"
echo "Model: ${RETRIEVER_MODEL}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Activate venv and run server
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Use the venv python directly (more reliable than sourcing activate)
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

echo "Starting Python server..." | tee -a "${LOG_FILE}"
exec "${PYTHON_BIN}" retrieval_server.py \
    --index_path "${INDEX_PATH}" \
    --corpus_path "${CORPUS_PATH}" \
    --topk "${TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    2>&1 | tee -a "${LOG_FILE}"
