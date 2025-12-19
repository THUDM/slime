#!/bin/bash
# Start SGLang server for a tau2 user simulator
#
# This server provides low-latency user simulation for multi-turn RL rollouts.
# Run this BEFORE starting GRPO training (run_grpo.sh).
#
# GPU allocation:
#   - User sim (this script): GPU 2,3 (TP=2)
#   - Policy model: GPU 0,1 (managed by slime)
#
# For SFT evaluation only (4 GPUs available):
#   - User sim: GPU 2,3
#   - Policy SGLang: GPU 0,1

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TAU_BENCH_OUT_DIR="${TAU_BENCH_OUT_DIR:-${SCRIPT_DIR}/../outputs}"

USER_SIM_MODEL="${USER_SIM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
USER_SIM_PORT="${USER_SIM_PORT:-30001}"
USER_SIM_GPUS="${USER_SIM_GPUS:-2,3}"
USER_SIM_TP="${USER_SIM_TP:-2}"
USER_SIM_MEM_FRACTION="${USER_SIM_MEM_FRACTION:-0.85}"

# Check if model is downloaded
MODEL_DIR="${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507"
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Downloading ${USER_SIM_MODEL} to ${MODEL_DIR}..."
    huggingface-cli download "${USER_SIM_MODEL}" --local-dir "${MODEL_DIR}"
fi

LOG_DIR="${TAU_BENCH_OUT_DIR}/tau2/logs"
mkdir -p "${LOG_DIR}"

echo "Starting user simulator server..."
echo "  Model: ${MODEL_DIR}"
echo "  Port: ${USER_SIM_PORT}"
echo "  GPUs: ${USER_SIM_GPUS} (TP=${USER_SIM_TP})"
echo "  Memory fraction: ${USER_SIM_MEM_FRACTION}"

# Kill any existing server on this port
pkill -f "sglang.*--port ${USER_SIM_PORT}" || true
sleep 2

# Start SGLang server
CUDA_VISIBLE_DEVICES="${USER_SIM_GPUS}" python3 -m sglang.launch_server \
    --model-path "${MODEL_DIR}" \
    --host 0.0.0.0 \
    --port "${USER_SIM_PORT}" \
    --tp "${USER_SIM_TP}" \
    --mem-fraction-static "${USER_SIM_MEM_FRACTION}" \
    --disable-flashinfer \
    --chat-template chatml \
    2>&1 | tee "${LOG_DIR}/user_sim_server.log" &

SERVER_PID=$!
echo "Server starting with PID ${SERVER_PID}..."

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s "http://127.0.0.1:${USER_SIM_PORT}/health" > /dev/null 2>&1; then
        echo "User simulator server is ready!"
        echo ""
        echo "Test with:"
        echo "  curl http://127.0.0.1:${USER_SIM_PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"${USER_SIM_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
        exit 0
    fi
    sleep 2
    echo "  Still waiting... (${i}/60)"
done

echo "ERROR: Server failed to start within 120 seconds"
echo "Check logs at ${LOG_DIR}/user_sim_server.log"
exit 1
