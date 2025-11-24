#!/bin/bash

# Wrapper script to run Qwen2.5-3B training with local dense retriever
# This script starts the retrieval server in a tmux window and runs training in another

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RETRIEVER_DIR="${SCRIPT_DIR}/local_dense_retriever"
TMUX_SESSION="slime-search-r1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Slime Search-R1 Training with Local Dense Retriever ===${NC}"
python3 -m uv pip install tenacity

# Helper function: Check if a port is free
wait_for_port_free() {
    local port=$1
    local max_wait=${2:-30}  # Default 30 seconds
    local wait_count=0

    echo "Checking if port ${port} is free..."

    while [ $wait_count -lt $max_wait ]; do
        # Check using lsof (works on macOS and Linux)
        if command -v lsof &> /dev/null; then
            if ! lsof -i :${port} -sTCP:LISTEN -t > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Port ${port} is free${NC}"
                return 0
            fi
        # Fallback to ss (Linux-only, but more common in Docker)
        elif command -v ss &> /dev/null; then
            if ! ss -tuln | grep -q ":${port} "; then
                echo -e "${GREEN}✓ Port ${port} is free${NC}"
                return 0
            fi
        # Fallback to netstat (less reliable but widely available)
        elif command -v netstat &> /dev/null; then
            if ! netstat -tuln 2>/dev/null | grep -q ":${port} "; then
                echo -e "${GREEN}✓ Port ${port} is free${NC}"
                return 0
            fi
        else
            echo -e "${YELLOW}Warning: No port checking tool available (lsof/ss/netstat), skipping check${NC}"
            return 0
        fi

        if [ $wait_count -eq 0 ]; then
            echo -n "  Port ${port} still in use, waiting"
        else
            echo -n "."
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done

    echo ""
    echo -e "${RED}Error: Port ${port} is still in use after ${max_wait} seconds${NC}"
    echo ""
    echo "To manually check and free the port:"
    echo "  lsof -i :${port}  # Find the process"
    echo "  kill -9 <PID>     # Kill the process"
    return 1
}

# Helper function: Gracefully kill a process by pattern
graceful_kill_process() {
    local pattern=$1
    local process_name=$2  # For display purposes

    # Find PIDs matching the pattern
    local pids=$(pgrep -f "${pattern}")

    if [ -z "${pids}" ]; then
        echo "No ${process_name} process found"
        return 0
    fi

    echo "Found ${process_name} process(es): ${pids}"
    echo "Attempting graceful shutdown (SIGTERM)..."
    pkill -f "${pattern}" || true

    # Wait up to 5 seconds for graceful shutdown
    local wait_count=0
    while [ $wait_count -lt 5 ]; do
        sleep 1
        pids=$(pgrep -f "${pattern}")
        if [ -z "${pids}" ]; then
            echo -e "${GREEN}✓ ${process_name} stopped gracefully${NC}"
            return 0
        fi
        wait_count=$((wait_count + 1))
    done

    # If still running, force kill
    echo "${process_name} did not stop gracefully, forcing shutdown (SIGKILL)..."
    pkill -9 -f "${pattern}" || true
    sleep 1

    pids=$(pgrep -f "${pattern}")
    if [ -z "${pids}" ]; then
        echo -e "${GREEN}✓ ${process_name} stopped (forced)${NC}"
        return 0
    else
        echo -e "${YELLOW}Warning: ${process_name} may still be running${NC}"
        return 1
    fi
}

# 1. Check prerequisites
echo "Checking prerequisites..."

# Check tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    echo "Please install tmux: apt install tmux -y"
    exit 1
fi

# Check if .venv exists
if [ ! -d "${RETRIEVER_DIR}/.venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${RETRIEVER_DIR}/.venv${NC}"
    echo ""
    echo "Please set up the local retriever first:"
    echo "  cd ${RETRIEVER_DIR}"
    echo "  uv venv"
    echo "  source .venv/bin/activate"
    echo "  uv pip install torch faiss-cpu==1.12.0 uvicorn fastapi huggingface_hub datasets transformers"
    echo "  python download.py --save_path ./corpus_and_index"
    echo "  cd corpus_and_index"
    echo "  cat part_* > e5_Flat.index"
    echo "  rm part_*"
    echo "  gzip -d wiki-18.jsonl.gz"
    exit 1
fi

# Check if corpus and index files exist
if [ ! -f "${RETRIEVER_DIR}/corpus_and_index/e5_Flat.index" ]; then
    echo -e "${RED}Error: FAISS index not found at ${RETRIEVER_DIR}/corpus_and_index/e5_Flat.index${NC}"
    echo "Please run the download and setup steps first (see README.md)"
    exit 1
fi

if [ ! -f "${RETRIEVER_DIR}/corpus_and_index/wiki-18.jsonl" ]; then
    echo -e "${RED}Error: Corpus not found at ${RETRIEVER_DIR}/corpus_and_index/wiki-18.jsonl${NC}"
    echo "Please run the download and setup steps first (see README.md)"
    exit 1
fi

# Check GPU availability (required for model.cuda())
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
    echo -e "${YELLOW}The retrieval server requires GPU to run the E5 encoder model.${NC}"
elif ! nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi failed. GPU may not be available.${NC}"
    echo -e "${YELLOW}The retrieval server requires GPU to run the E5 encoder model.${NC}"
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓ GPU available (${GPU_COUNT} GPU(s) detected)${NC}"
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"

# 2. Configuration
export RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-"http://localhost:8000"}

# Parse port from RETRIEVAL_SERVER_URL
RETRIEVER_PORT=$(echo "${RETRIEVAL_SERVER_URL}" | sed -n 's/.*:\([0-9]\{4,5\}\).*/\1/p')
if [ -z "${RETRIEVER_PORT}" ]; then
    RETRIEVER_PORT="8000"  # Fallback to default if parsing fails
fi

RETRIEVER_TOPK=3
RETRIEVER_MODEL="intfloat/e5-base-v2"
RETRIEVER_LOG="/root/workspace/slime-open/retrieval_server_$(date +%Y%m%d_%H%M%S).log"
TRAINING_LOG="/root/workspace/slime-open/training_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}

echo "Retrieval server configuration:"
echo "  URL: ${RETRIEVAL_SERVER_URL}"
echo "  Port: ${RETRIEVER_PORT}"
echo "  Top-K: ${RETRIEVER_TOPK}"
echo "  Model: ${RETRIEVER_MODEL}"
echo "  Log file: ${RETRIEVER_LOG}"
echo ""
echo "Training configuration:"
echo "  Log file: ${TRAINING_LOG}"

# 3. Kill any existing retrieval server
echo "Cleaning up any existing retrieval server..."
graceful_kill_process "retrieval_server.py" "retrieval server"

# Wait for port to be released
if ! wait_for_port_free "${RETRIEVER_PORT}" 30; then
    echo -e "${RED}Failed to free port ${RETRIEVER_PORT}. Please manually kill the process and retry.${NC}"
    exit 1
fi

# 4. Kill existing tmux session if it exists
if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
    echo "Killing existing tmux session: ${TMUX_SESSION}"
    tmux kill-session -t ${TMUX_SESSION}
    sleep 1  # Wait for tmux to clean up resources
fi

# 5. Create cleanup trap
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    graceful_kill_process "retrieval_server.py" "retrieval server"
    if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
        tmux kill-session -t ${TMUX_SESSION}
    fi
    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# 6. Create tmux session with retrieval server
echo "Creating tmux session: ${TMUX_SESSION}"
echo "Starting retrieval server (this may take 40-60 seconds to load 61GB index + 14GB corpus + model)..."

# Create new tmux session (detached) with retrieval server window
tmux new-session -d -s ${TMUX_SESSION} -n retrieval-server

# Start retrieval server using helper script (avoids complex escaping issues)
INDEX_PATH="${RETRIEVER_DIR}/corpus_and_index/e5_Flat.index"
CORPUS_PATH="${RETRIEVER_DIR}/corpus_and_index/wiki-18.jsonl"
START_SCRIPT="${RETRIEVER_DIR}/start_server.sh"

tmux send-keys -t ${TMUX_SESSION}:0 "bash ${START_SCRIPT} ${INDEX_PATH} ${CORPUS_PATH} ${RETRIEVER_TOPK} e5 ${RETRIEVER_MODEL} ${RETRIEVER_LOG} ${RETRIEVER_PORT}" C-m

# Give the server a moment to start initializing
sleep 5

# 7. Wait for retrieval server to be ready
echo "Waiting for retrieval server to start (loading index and model, may take up to 120 seconds)..."
MAX_WAIT=120
WAIT_COUNT=0
START_TIME=$(date +%s)

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s ${RETRIEVAL_SERVER_URL}/docs > /dev/null 2>&1; then
        ELAPSED=$(($(date +%s) - START_TIME))
        echo ""
        echo -e "${GREEN}✓ Retrieval server is ready (started in ${ELAPSED} seconds)${NC}"
        break
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    # Show progress every 10 seconds
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo ""
        echo -n "  ${WAIT_COUNT}s elapsed, still waiting"
    else
        echo -n "."
    fi
done
echo ""

if [ $WAIT_COUNT -eq $MAX_WAIT ]; then
    echo -e "${RED}Error: Retrieval server failed to start within ${MAX_WAIT} seconds${NC}"
    echo ""
    echo "=== Last 30 lines from retrieval server output ==="
    if [ -f "${RETRIEVER_LOG}" ]; then
        tail -30 "${RETRIEVER_LOG}"
    else
        echo "(Log file not found at ${RETRIEVER_LOG})"
    fi
    echo "=== End of log ==="
    echo ""
    echo "Debugging steps:"
    echo "  1. Check the log file: cat ${RETRIEVER_LOG}"
    echo "  2. Attach to tmux session: tmux attach -t ${TMUX_SESSION}"
    echo "  3. Switch to retrieval-server window: Ctrl+b, 0"
    echo "  4. Check GPU availability: nvidia-smi"
    echo ""
    exit 1
fi

# 8. Create training window and run training script
echo "Creating training window..."
tmux new-window -t ${TMUX_SESSION}:1 -n training

# Set environment variable for the training session and cd to project root
# (train.py is at /workspace/slime-open/train.py, not in examples/search-r1-oai/)
PROJECT_ROOT="${SCRIPT_DIR}/../.."
tmux send-keys -t ${TMUX_SESSION}:1 "export RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL}" C-m
tmux send-keys -t ${TMUX_SESSION}:1 "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" C-m
tmux send-keys -t ${TMUX_SESSION}:1 "cd ${PROJECT_ROOT}" C-m
tmux send-keys -t ${TMUX_SESSION}:1 "bash examples/search-r1-oai/run_qwen2.5-3B-it.sh |& tee ${TRAINING_LOG}" C-m

echo ""
echo -e "${GREEN}=== Training started ===${NC}"
echo ""
echo "Tmux session '${TMUX_SESSION}' created with two windows:"
echo "  Window 0 (retrieval-server): Retrieval server running"
echo "  Window 1 (training): Training job"
echo ""
echo "Log files:"
echo "  Retrieval server: ${RETRIEVER_LOG}"
echo "  Training: ${TRAINING_LOG}"
echo ""
echo "To view the session:"
echo "  tmux attach -t ${TMUX_SESSION}"
echo ""
echo "To switch between windows in tmux:"
echo "  Ctrl+b, 0  -> Switch to retrieval-server window"
echo "  Ctrl+b, 1  -> Switch to training window"
echo "  Ctrl+b, d  -> Detach from session (keeps running)"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t ${TMUX_SESSION}"
echo ""
echo -e "${YELLOW}Attaching to training window...${NC}"

# Attach to training window (user can switch between windows)
tmux attach -t ${TMUX_SESSION}:1
