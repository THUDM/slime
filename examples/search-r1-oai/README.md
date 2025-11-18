# Search-R1 with OpenAI Tool Calling

This example demonstrates training LLMs with retrieval-augmented generation using OpenAI-compatible tool calling interface and a local dense retriever.

## Overview

The implementation includes:
- **Local Dense Retriever**: E5-based FAISS retriever serving Wikipedia-2018 corpus
- **OpenAI Tool Calling**: Multi-turn generation with `search` function calling
- **GRPO Training**: Group Relative Policy Optimization for RL training
- **Custom Generation**: Token-accurate generation via `/v1/chat/completions` API with logprobs

## Setup

### 1. Prepare Model Checkpoints

```bash
# Download HuggingFace checkpoint
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/workspace/Qwen2.5-3B-Instruct

# Convert to Megatron format
cd /root/workspace/slime-open
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/workspace/Qwen2.5-3B-Instruct \
    --save /root/workspace/Qwen2.5-3B-Instruct_torch_dist
```

### 2. Prepare Training Data

```bash
cd /root/workspace/slime-open/examples/search-r1-oai
python data_preprocess.py
```

### 3. Setup Local Dense Retriever

Follow instructions in [`local_dense_retriever/README.md`](local_dense_retriever/README.md):

```bash
cd local_dense_retriever

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch faiss-cpu==1.12.0 uvicorn fastapi huggingface_hub datasets transformers

# Download corpus and index (~75GB)
python download.py --save_path ./corpus_and_index

# Merge index files
cd corpus_and_index
cat part_* > e5_Flat.index
rm part_*

# Decompress corpus
gzip -d wiki-18.jsonl.gz
```

## Running Training

### Quick Start (Default Port 8000)

```bash
cd /root/workspace/slime-open
CUDA_VISIBLE_DEVICES="0,1,2,3" bash examples/search-r1-oai/run_qwen2.5-3B-it-with-retrieval.sh
```

### Custom Port Configuration

You can specify a custom port for the retrieval server using the `RETRIEVAL_SERVER_URL` environment variable:

```bash
# Use port 9000 instead of default 8000
export RETRIEVAL_SERVER_URL="http://localhost:9000"
CUDA_VISIBLE_DEVICES="0,1,2,3" bash examples/search-r1-oai/run_qwen2.5-3B-it-with-retrieval.sh

# Use a remote retrieval server
export RETRIEVAL_SERVER_URL="http://192.168.1.100:8500"
CUDA_VISIBLE_DEVICES="0,1,2,3" bash examples/search-r1-oai/run_qwen2.5-3B-it-with-retrieval.sh
```

**How it works:**
- The wrapper script automatically parses the port from `RETRIEVAL_SERVER_URL`
- The port is passed through `start_server.sh` to `retrieval_server.py`
- Training scripts use the same `RETRIEVAL_SERVER_URL` to connect to the retriever
- Default port is 8000 if `RETRIEVAL_SERVER_URL` is not set

### What Happens

The wrapper script (`run_qwen2.5-3B-it-with-retrieval.sh`) will:

1. **Check Prerequisites**: Verify tmux, virtual environment, corpus, and index files
2. **Start Retrieval Server**: Launch in a tmux window (takes 40-60s to load ~75GB data)
3. **Wait for Server**: Health check via `/docs` endpoint (max 120s)
4. **Start Training**: Launch training in another tmux window
5. **Cleanup**: Automatically kill retrieval server on exit

### Monitoring

The script creates a tmux session named `slime-search-r1` with two windows:

```bash
# Attach to the session
tmux attach -t slime-search-r1

# Switch between windows
# Ctrl+b, 0  -> Retrieval server window
# Ctrl+b, 1  -> Training window
# Ctrl+b, d  -> Detach (keeps running)

# Kill the session
tmux kill-session -t slime-search-r1
```

**Log Files:**
- Retrieval server: `/tmp/retrieval_server_YYYYMMDD_HHMMSS.log`
- Training: `/tmp/training_YYYYMMDD_HHMMSS.log`

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Tmux Session                            │
├──────────────────────────┬──────────────────────────────────┤
│  Window 0                │  Window 1                        │
│  Retrieval Server        │  Training Job                    │
│                          │                                  │
│  • E5 Encoder Model      │  • Ray Cluster                   │
│  • FAISS Index (61GB)    │  • SGLang Rollout Engine         │
│  • Wiki-18 Corpus (14GB) │  • Megatron Training Backend     │
│  • Port: 8000 (default)  │  • GRPO Algorithm                │
│                          │                                  │
│  FastAPI Server          │  Generate Function               │
│  └─ /retrieve            │  └─ POST /v1/chat/completions    │
│  └─ /docs                │     (with logprobs)              │
└──────────────────────────┴──────────────────────────────────┘
```

### Custom Generation Flow

The generation function (`generate_with_search_oai.py`) implements:

1. **Multi-turn Conversation**: Up to 3 turns with tool calling
2. **Token Extraction**: Parse `token_id` from logprobs for accurate token tracking
3. **Log Probability Tracking**: Extract `logprob` for each token for GRPO training
4. **Loss Mask Construction**:
   - Assistant content: `loss_mask=1` (train on responses)
   - Tool messages: `loss_mask=0` (don't train on retrieval results)
5. **Tool Execution**: Call local retriever via `/retrieve` endpoint
6. **Incremental Tokenization**: Use `apply_chat_template` delta method for tool messages

### Key Parameters

**Training (run_qwen2.5-3B-it.sh):**
- `--advantage-estimator grpo`: Use GRPO algorithm
- `--rollout-batch-size 32`: 32 prompts per rollout
- `--n-samples-per-prompt 8`: 8 responses per prompt (for GRPO group)
- `--global-batch-size 256`: Total training batch size
- `--balance-data`: Balance positive/negative samples

**Rollout:**
- `--rollout-temperature 0.8`: Sampling temperature
- `--rollout-max-response-len 512`: Max tokens per response
- `--custom-generate-function-path generate_with_search_oai.generate`
- `--custom-rm-path generate_with_search_oai.reward_func`

**Retriever:**
- `--topk 3`: Retrieve top-3 documents per query
- Model: `intfloat/e5-base-v2`
- Index: FAISS Flat index (exact search)

## Troubleshooting

### Retrieval Server Fails to Start

**Symptom**: Timeout waiting for retrieval server

**Solutions:**
1. Check GPU availability: `nvidia-smi`
2. Verify index file exists: `ls -lh local_dense_retriever/corpus_and_index/e5_Flat.index`
3. Check server log: `cat /tmp/retrieval_server_*.log`
4. Manually start server to see errors:
   ```bash
   cd local_dense_retriever
   source .venv/bin/activate
   python retrieval_server.py \
       --index_path corpus_and_index/e5_Flat.index \
       --corpus_path corpus_and_index/wiki-18.jsonl \
       --port 8000
   ```

### Port Already in Use

**Symptom**: `Address already in use` error

**Solutions:**
1. Kill existing process:
   ```bash
   pkill -9 -f retrieval_server.py
   # or
   lsof -ti:8000 | xargs kill -9
   ```
2. Use a different port:
   ```bash
   export RETRIEVAL_SERVER_URL="http://localhost:9000"
   ```

### Training Fails to Connect to Retriever

**Symptom**: Connection refused errors in training log

**Solutions:**
1. Verify server is running: `curl http://localhost:8000/docs`
2. Check `RETRIEVAL_SERVER_URL` is set correctly
3. Ensure retrieval server finished loading (check log for "Application startup complete")
4. Verify firewall allows connections on the port

### Out of Memory (OOM)

**Symptom**: CUDA OOM during loading or generation

**Solutions:**
1. Reduce `--rollout-batch-size` (default 32 → try 16)
2. Reduce `--n-samples-per-prompt` (default 8 → try 4)
3. Reduce `--sglang-mem-fraction-static` (default 0.7 → try 0.6)
4. Use CPU-based FAISS index (slower but saves GPU memory)
