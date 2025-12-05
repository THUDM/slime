# Tau2 bench with slime

This example mirrors `examples/tau-bench`, but plugs the newer tau2 gym environment into slime rollouts.

## Setup

Use the `zhuzilin/slime:latest` image and initialize the environment required for Tau2-Bench:
```bash
cd /root/
git clone https://github.com/slimerl/slime.git
cd slime
pip install -e .
# for tau2 bench 
cd /root/
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
pip install -e .
```

Use the following script to generate mock data for slime training. 

```bash
cd /root/slime
python examples/tau2-bench/tau2_mock.py \
  --output-dir /root/tau2-bench/data/tau2
```
Initialize the Qwen2.5-3B-Instruct model needed for tool use:

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

## Running the Script

The custom rollout entrypoint is `examples.tau2-bench.generate_with_tau2.generate`. A sample launcher is provided in `examples/tau2-bench/run_tau2_qwen3_4B.sh`; the important CLI flags are:

```bash
--prompt-data /root/tau2-bench/data/tau2/airline_train_tasks.jsonl
--input-key task_id
--custom-generate-function-path examples.tau2-bench.generate_with_tau2.generate
```

You need to configure your litellm API in `generate_with_tau2.py` for user simulation:

```python
TAU2_CONFIGS = {
    "domain": "airline",  # tau2 domain: airline | retail | telecom | mock
    "task_split": "train",  # task split within the domain
    "max_steps": 100,  # safety cap on interaction steps
    "user_llm": "gpt-4.1-mini",  # LiteLLM model name for user simulator
    "solo_mode": False,  # set True to disable user simulator
}
# Replace with your actual API key for user sim
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
```

And run:

```bash
cd /root/slime
bash examples/tau2-bench/run_tau2_qwen3_4B.sh
```
