# Search-R1 lite

This is a minimal reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and an example of using multi-turn conversation and tool-calling in slime.

## Environment Setup

Use the `slimerl/slime:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
pip install -e .
# for Search R1
pip install chardet
```

Download and prepare the training data:

```bash
cd /root/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/

# Set your working directory
WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# Process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# (Optional) Process multiple dataset search format test file
# Note: the final file is not shuffled
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

**Note:** If you plan to use local search backend, see the [Appendix](#appendix-setting-up-local-retriever) for instructions on setting up the local retrieval server.

Initialize the Qwen2.5-3B model:

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

## Configuration

### Search Backend Configuration

The `generate_with_search.py` file supports both **local search** and **Google search** backends. Configure via the `SEARCH_R1_CONFIGS` dictionary:

```python
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,

    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"

    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,
    },

    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual serper.dev API key
        "snippet_only": True,
        "proxy": None,
    },

    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities (required for TIS)

    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}
```

#### Using Local Search

1. Set `"search_backend": "local"`
2. Configure `"local"` section with your local retrieval server URL
3. Start your local search server before running the training script

#### Using Google Search

1. Set `"search_backend": "google"`
2. Configure `"google"` section with your serper.dev API key
3. Get your API key from [serper.dev](https://serper.dev)

### Enabling TIS (Trajectory Importance Sampling)

TIS requires log probability collection. To enable TIS:

**1. In `generate_with_search.py`:**
```python
SEARCH_R1_CONFIGS = {
    # ... other configs
    "return_logprob": True,  # Must be True for TIS
}
```

**2. In `run_qwen2.5_3B.sh`:**

Uncomment the TIS-related arguments in `GRPO_ARGS`:
```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # Uncomment to enable TIS
   --use-tis
)
```

And uncomment the TIS configuration paths in `CUSTOM_ARGS`:
```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # Uncomment to enable TIS
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)
```

**Important Notes:**
- TIS requires `return_logprob=True` in `SEARCH_R1_CONFIGS`
- When collecting log probabilities, response postprocessing is automatically disabled to maintain token/logp alignment
- TIS adds computational overhead but can improve training efficiency

## Running the Script

```bash
cd slime/
bash examples/search-r1/run_qwen2.5_3B.sh
```

## Code Structure

To implement multi-turn conversation + tool-calling in slime, you only need to implement a custom data generation function and a reward model for the task. These correspond to the following 2 configuration items in the startup script:

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)
```

These are the `generate` and `reward_func` functions in `generate_with_search.py`.

## Appendix: Setting up Local Retriever

This section provides detailed instructions for setting up the local dense retriever for use with the local search backend.

### Prerequisites

The local retriever requires a separate conda environment to avoid conflicts with the training environment. It uses GPU for efficient retrieval.

### Step 1: Install Conda

If you don't have conda installed, run the following commands:

```bash
# Download and install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init
source ~/.bashrc

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Step 2: Create Retriever Environment

Create and activate a conda environment with Python 3.10:

```bash
# Create environment
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install required packages
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

### Step 3: Download Index and Corpus

**Note:** The local retrieval files are large. You'll need approximately **60-70 GB** for download and **132 GB** after extraction. Make sure you have sufficient disk space.

```bash
# Set your save path
save_path=/root/Index

# Download the index and corpus files
python /root/slime/examples/search-r1/local_dense_retriever/download.py --save_path $save_path

# Combine split index files
cat $save_path/part_* > $save_path/e5_Flat.index

# Decompress the corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

### Step 4: Start Local Retrieval Server

```bash
# If you encounter "conda not found" error, run:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda init
# source ~/.bashrc

# Activate retriever environment
conda activate retriever

# Set paths
save_path=/root/Index
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Start the retrieval server
python /root/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
```

**Important Notes:**
- First startup will download the model and load the index, which may take a few minutes
- Normal startup time (excluding downloads): 1-2 minutes
- GPU memory usage per GPU: approximately 5-7 GB
- The local search engine's Python process will not terminate when the shell closes
- To restart the server: `lsof -i :8000` to find the PID, then kill it and restart

### Step 5: Start Training

Make sure you're **NOT** in the retriever conda environment. If you are, run `conda deactivate`.

```bash
cd /root/slime

# Set your wandb key (optional)
export WANDB_KEY="your_wandb_key_here"

# If ray process is stuck, try:
# rm -rf /root/.cache
# rm -rf /root/.*

# Run the training script
bash /root/slime/examples/search-r1/run_qwen2.5_3B.sh
```

### Troubleshooting

**Ray process stuck:**
```bash
rm -rf /root/.cache
# If still stuck:
rm -rf /root/.*
```

**Conda environment issues:**
- Make sure you deactivate the retriever environment before running training
- Verify you're using the base Python environment for training

**Retrieval server not responding:**
- Check if the server is running: `lsof -i :8000`
- Verify GPU availability: `nvidia-smi`
- Check logs for any error messages
