# Slime x Strands-Agents

This is a running example that connects the [Strands-Agents](https://github.com/strands-agents/sdk-python) agent scaffolding framework with Slime for RL training.

## Install Dependencies

1. Pull the `slimerl/slime:latest` image and enter it
2. Goes to slime folder: `cd /root/slime` (Clone the repository if not already there: `cd /root && git clone https://github.com/THUDM/slime.git`)
3. Install Slime: `pip install -e .`
4. Goes to the example folder: `cd /root/slime/examples/strands-agents`
5. Install other dependencies: `pip install -r requirements.txt`

> NOTE: we use camel-ai's subprocess code interpreter for python code execution, which is NOT a good practice; it's just for convenience of this example and the dependencies for solving math problems are usually ready in `slime`'s docker

## Prepare Model

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/models/Qwen/Qwen3-4B-Instruct-2507

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen/Qwen3-4B-Instruct-2507 \
    --save /root/models/Qwen/Qwen3-4B-Instruct-2507_torch_dist
```

## Prepare Dataset

Following [Retool](https://arxiv.org/abs/2504.11536), we used `dapo-math-17k` as training data:

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
bash examples/strands-agents/strands_qwen3_4b.sh
```
