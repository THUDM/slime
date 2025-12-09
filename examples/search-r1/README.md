# Search-R1: Retrieval-Augmented Reasoning with Multi-Turn Conversations

This example demonstrates training LLMs with retrieval-augmented generation (RAG) using both XML-based format and OpenAI-compatible tool calling. It is a minimal reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) showcasing multi-turn conversations and tool-calling in Slime.

## Features

- **Dual API Modes**: XML format (`generate` mode) or OpenAI tool calling (`chat` mode)
- **Flexible Search Backends**: Google Search API or local dense retriever (E5-based FAISS)
- **Token Tracking Strategies**: Manual extraction, Router RadixTree cache, or Router ChatHandler
- **GRPO Training**: Group Relative Policy Optimization for RL training
- **Custom Generation**: Token-accurate generation with logprobs tracking

## Quick Start

### 1. Environment Setup

Use the `slimerl/slime:latest` Docker image:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .

# For Search-R1
pip install chardet tenacity
```

### 2. Model Preparation

```bash
# Download HuggingFace checkpoint
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct

# Convert to Megatron format
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B-Instruct \
    --save /root/Qwen2.5-3B-Instruct_torch_dist
```

### 3. Data Preparation

**For XML format (generate mode):**

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/
python scripts/data_process/nq_search.py --local_dir /root/nq_search/
```

**For OpenAI format (chat mode):**

```bash
cd /root/slime/examples/search-r1
python oai_format_data_preprocess.py
```

## Configuration

Search-R1 supports flexible configuration via environment variables:

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LLM_API_MODE` | `generate` / `chat` | `generate` | API format (XML or OpenAI tool calling) |
| `SEARCH_BACKEND` | `google` / `local` | `google` | Search provider (Google API or local retriever) |
| `TOKEN_TRACKING_MODE` | `manual` / `router_radix` / `router_handler` | `manual` | Token extraction strategy |
| `TRACK_LOG_PROBS` | `true` / `false` | `true` | Track log probabilities (required for PPO/GRPO) |
| `LOSS_MASK_MODE` | `strict` / `simple` | `strict` | Loss mask strategy (strict: assistant=1, tool=0; simple: all=1) |
| `RETRIEVAL_SERVER_URL` | URL | `http://localhost:8000` | Local retriever server URL (for `local` backend) |
| `GOOGLE_API_KEY` | string | `YOUR_API_KEY` | Google Search API key (for `google` backend) |

### Mode Combinations

| LLM API Mode | Token Tracking | Compatible | Use Case |
|--------------|----------------|-----------|----------|
| `generate` | `manual` | ✅ | Original XML format with API logprobs extraction |
| `generate` | `router_radix` | ✅ | XML format with Router cache (faster) |
| `chat` | `manual` | ✅ | OpenAI format with API logprobs extraction |
| `chat` | `router_handler` | ✅ | OpenAI format with Router messages cache (optimal) |
| `chat` | `router_radix` | ⚠️ | Works but loses OpenAI messages semantics |
| `generate` | `router_handler` | ❌ | **Invalid** (strict validation will raise error) |

### Configuration Examples

#### Mode 1: Original (XML + Google Search)

```bash
# Default settings (no env vars needed)
cd /root/slime
bash examples/search-r1/run_qwen2.5_3B.sh
```

#### Mode 2: OpenAI Format + Local Retriever

```bash
export LLM_API_MODE=chat
export SEARCH_BACKEND=local
export TOKEN_TRACKING_MODE=manual
export RETRIEVAL_SERVER_URL="http://localhost:8000"

cd /root/slime
bash examples/search-r1/run_qwen2.5_3B_with_retrieval.sh
```

#### Mode 3: Router-Accelerated (Generate Mode)

```bash
export LLM_API_MODE=generate
export SEARCH_BACKEND=local
export TOKEN_TRACKING_MODE=router_radix

cd /root/slime
bash examples/search-r1/run_qwen2.5_3B_with_retrieval.sh
```

#### Mode 4: Router-Accelerated (Chat Mode)

```bash
export LLM_API_MODE=chat
export SEARCH_BACKEND=local
export TOKEN_TRACKING_MODE=router_handler

cd /root/slime
bash examples/search-r1/run_qwen2.5_3B_with_retrieval.sh
```

## Running Training

### Option 1: Google Search Backend (Original)

Configure your Google API key:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
cd /root/slime
bash examples/search-r1/run_qwen2.5_3B.sh
```

### Option 2: Local Dense Retriever

#### Setup Local Retriever

Follow [`local_dense_retriever/README.md`](local_dense_retriever/README.md):

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

#### Run Training with Local Retriever

The wrapper script automatically starts the retrieval server:

```bash
cd /root/slime

# Use default port 8000
export SEARCH_BACKEND=local
bash examples/search-r1/run_qwen2.5_3B_with_retrieval.sh

# Use custom port 9000
export RETRIEVAL_SERVER_URL="http://localhost:9000"
bash examples/search-r1/run_qwen2.5_3B_with_retrieval.sh

# Use remote retrieval server
export RETRIEVAL_SERVER_URL="http://192.168.1.100:8500"
bash examples/search-r1/run_qwen2.5_3B_with_retrieval.sh
```

**What the wrapper script does:**

1. Check prerequisites (tmux, venv, corpus, index files)
2. Start retrieval server in tmux window (40-60s to load ~75GB data)
3. Wait for server health check via `/docs` endpoint (max 120s)
4. Start training in another tmux window
5. Cleanup on exit (automatically kill retrieval server)

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
│  └─ /retrieve            │  └─ POST /generate or            │
│  └─ /docs                │     /v1/chat/completions         │
└──────────────────────────┴──────────────────────────────────┘
```

### Custom Generation Flow

The generation function (`generate_with_search.py`) implements:

1. **Multi-turn Conversation**: Up to 3 turns with search/answer actions
2. **Token Extraction**:
   - **Manual mode**: Parse from API logprobs (`output_token_logprobs` or `logprobs.content`)
   - **Router_radix mode**: Query Router `/retrieve_from_text` endpoint (text-based)
   - **Router_handler mode**: Query Router `/retrieve_from_messages_template` endpoint (messages-based)
3. **Log Probability Tracking**: Extract logprobs for GRPO/PPO training
4. **Loss Mask Construction**:
   - **Strict mode** (`LOSS_MASK_MODE=strict`): Assistant content=1 (trainable), tool messages=0 (skip)
   - **Simple mode** (`LOSS_MASK_MODE=simple`): All content=1 (train on everything)
5. **Tool Execution**: Call search backend (Google API or local retriever)
6. **Mode Compatibility Validation**: Strict validation prevents invalid mode combinations (e.g., `generate + router_handler`)

## Code Structure

To implement multi-turn conversation + tool-calling in Slime, you only need to implement:

1. **Custom Generation Function** (`generate_with_search.py:generate`): Handles multi-turn logic, search execution, token tracking
2. **Reward Model** (`generate_with_search.py:reward_func`): Evaluates response quality (e.g., exact match with ground truth)

These correspond to:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
```

## Key Parameters

### Training Script Arguments

```bash
# Algorithm
--advantage-estimator grpo       # GRPO algorithm
--use-kl-loss                    # Enable KL divergence loss
--kl-loss-coef 0.00              # KL loss coefficient
--eps-clip 0.2                   # PPO clipping parameter

# Data
--rollout-batch-size 32          # 32 prompts per rollout
--n-samples-per-prompt 8         # 8 responses per prompt (GRPO group)
--global-batch-size 256          # Total training batch size
--balance-data                   # Balance positive/negative samples

# Generation
--rollout-temperature 0.8        # Sampling temperature (generate mode default)
                                 # or 1.0 (chat mode default)
--rollout-max-response-len 512   # Max tokens per response
```

### Generation Configuration (in `generate_with_search.py`)

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 3,                    # Max search/answer turns
    "topk": 3,                         # Retrieve top-3 documents
    "search_concurrency": 256,         # Concurrent search requests
    "format_score": 0.2,               # Reward bonus for correct format
    "max_context_length": 16384,       # Max context length
}
```

## Monitoring

### Tmux Session Management

The wrapper script creates a tmux session named `slime-search-r1`:

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

### Log Files

- Retrieval server: `/root/workspace/slime-open/retrieval_server_YYYYMMDD_HHMMSS.log`
- Training: `/root/workspace/slime-open/training_YYYYMMDD_HHMMSS.log`

## Troubleshooting

### Retrieval Server Fails to Start

**Symptom**: Timeout waiting for retrieval server

**Solutions**:
1. Check GPU availability: `nvidia-smi`
2. Verify index file exists: `ls -lh local_dense_retriever/corpus_and_index/e5_Flat.index`
3. Check server log: `cat /root/workspace/slime-open/retrieval_server_*.log`
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

**Solutions**:
```bash
# Kill existing process
pkill -9 -f retrieval_server.py
# or
lsof -ti:8000 | xargs kill -9

# Use a different port
export RETRIEVAL_SERVER_URL="http://localhost:9000"
```

### Training Fails to Connect to Retriever

**Symptom**: Connection refused errors in training log

**Solutions**:
1. Verify server is running: `curl http://localhost:8000/docs`
2. Check `RETRIEVAL_SERVER_URL` is set correctly
3. Ensure retrieval server finished loading (check log for "Application startup complete")
4. Verify firewall allows connections on the port

### Out of Memory (OOM)

**Symptom**: CUDA OOM during loading or generation

**Solutions**:
1. Reduce `--rollout-batch-size` (default 32 → try 16)
2. Reduce `--n-samples-per-prompt` (default 8 → try 4)
3. Reduce `--sglang-mem-fraction-static` (default 0.7 → try 0.6)
4. Use CPU-based FAISS index (slower but saves GPU memory)

### Invalid Mode Configuration

**Symptom**: `ValueError: Invalid configuration: token_tracking_mode='router_handler' is NOT compatible with llm_api_mode='generate'`

**Solution**: Check the Mode Combinations table above. Example fix:

```bash
# Invalid (will raise error)
export LLM_API_MODE=generate
export TOKEN_TRACKING_MODE=router_handler  # ❌ Incompatible

# Valid alternatives
export LLM_API_MODE=generate
export TOKEN_TRACKING_MODE=manual          # ✅ Works

# or
export LLM_API_MODE=chat
export TOKEN_TRACKING_MODE=router_handler  # ✅ Works
```

## Performance Tuning

### Memory Optimization

```bash
# Disable log probs if not needed (for SFT)
export TRACK_LOG_PROBS=false

# Reduce batch sizes
--rollout-batch-size 16
--n-samples-per-prompt 4
```

### Speed Optimization

```bash
# Use Router tracking (reduces per-turn overhead)
export TOKEN_TRACKING_MODE=router_radix    # For generate mode
# or
export TOKEN_TRACKING_MODE=router_handler  # For chat mode

# Use local retriever (faster than Google API)
export SEARCH_BACKEND=local
```

### Quality Optimization

```bash
# Strict loss mask for better RL signal
export LOSS_MASK_MODE=strict

# Chat mode for exact token tracking
export LLM_API_MODE=chat
```

## Citation

If you use this example, please cite the original Search-R1 paper:

```bibtex
@article{jin2025search,
  title={Search-R1: Self-Improving Reasoning with Multi-Turn Retrieval},
  author={Jin, Bowen and others},
  journal={arXiv preprint arXiv:2502.xxxxx},
  year={2025}
}
```
