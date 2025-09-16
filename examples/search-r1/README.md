# Search-R1 lite

This is a minimal reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and an example of using multi-turn conversation and tool-calling in slime.

## Environment Setup

Use the `zhuzilin/slime:latest` image and initialize the environment required for Search-R1:

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

## Running the Script

You need to configure your serper.dev API in `generate_with_search.py`:

```python
SEARCH_R1_CONFIGS = {
    "max_turns": 3,
    "topk": 3,
    "google_api_key": "YOUR_API_KEY",  # Replace with your actual API key
    "snippet_only": True,  # Set to True to only return snippets
    "proxy": None,  # Set to your proxy if needed
    "search_concurrency": 256,
    # rm
    "format_score": 0.2,
}
```

And run:

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
