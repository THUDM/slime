# Tau bench 
This example shows slime training in an agentic multi-turn tool use environment. 


## Environment Setup 
Use the `zhuzilin/slime:latest` image and initialize the environment required for Search-R1:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
# for tau bench 
cd /root/
git clone https://github.com:JD-ETH/tau-bench.git
cd tau-bench
git checkout feature/litellm-retry
pip install -e . 
```

Use the following script to generate mock data for slime training. 

```bash
cd /root/slime/examples/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

Initialize the Qwen2.5-3B-Instruct model needed for tool use:

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct
huggingface-cli download Qwen/Qwen3-14B --local-dir /root/Qwen3-14B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B-Instruct \
    --save /root/Qwen2.5-3B-Instruct_torch_dist


source scripts/models/qwen3-14B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-14B \
    --save /root/Qwen3-14B_torch_dist
```

## Running the Script

You need to configure your litellm API in `generate_with_tau.py` for user simulation:

```python
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"], only tool-calling implemented for now
    "user_model": "gemini-2.0-flash-lite",  # Cheap Model for user simulator
    "user_model_provider": "gemini",
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail, ["test"] for airline
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router", # Unused, required
    "model": "qwen2.5-3b", # Unused, reqired
}
# Replace with your actual API key for user sim    
GEMINI_API_KEY = "YOUR KEY" 
```

And run:

```bash
cd /root/slime
bash examples/tau-bench/run_qwen2.5_3B_Instruct.sh
```

```bash
cd /root/slime
bash examples/tau-bench/run_qwen3_14B.sh
```
