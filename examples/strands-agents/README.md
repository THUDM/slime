# Slime x Strands-Agents

This is a running example that connects the [Strands-Agents](https://github.com/strands-agents/sdk-python) agent scaffolding framework with Slime for RL training.

## Install Dependencies

1. Pull the `slimerl/slime:latest` image and enter it
2. Goes to slime folder: `cd /root/slime`
3. Install Slime: `pip install -e .`
4. Goes to the example folder: `cd /root/slime/examples/strands_agent`
5. Install other dependencies: `pip install -r requirements.txt`

> NOTE: we use camel-ai's subprocess code interpreter for python code execution, which is NOT a good practice; it's just for convenience of this example and the dependencies for solving math problems are usually ready in `slime`'s docker

## Prepare Model

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-8B --local-dir /root/models/qwen3-8B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/qwen3-8B \
    --save /root/models/qwen3-8B_torch_dist
```

## Prepare Dataset

We used `dapo-math-17k` as training data:

```
from datasets import load_dataset
ds = load_dataset("zhuzilin/dapo-math-17k", split="train")
ds.to_json("/root/data/dapo-math-17k.jsonl", orient="records", lines=True)
```

and `aime-2024` as eval data:

```
from datasets import load_dataset
ds = load_dataset("zhuzilin/aime-2024", split="train")
ds.to_json("/root/data/aime-2024.jsonl", orient="records", lines=True)
```

## Run Training

Assuming `/root/slime` is up-to-date (if this PR is not merged you may need to switch branch):

```
cd /root/slime
export WANDB_KEY=$your_wandb_key
bash examples/strands-agents/run_qwen3_8B.sh
```

## Quick Testing - Optional

The `test_generate.py` script is used for quick testing of the `generate` function.

First, we launch a SGLang server using:

```
nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8000 \
    --host 0.0.0.0 \
    --tool-call-parser qwen \
    --tp-size 8 \
    --mem-fraction-static 0.9 &
```

> Remember to change `tp-size` and `mem-fraction-static` to match your machine

Then, we can directly run `test_generate.py` with:

```
python test_generate.py
```
