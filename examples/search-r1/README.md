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

Please refer to the script provided in Search-R1 to download the data:

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/
python scripts/data_process/nq_search.py --local_dir /root/nq_search/
```

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
