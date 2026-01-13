# slime x Strands-SGLang

This example connects `slime` with [`strands-sglang`](https://github.com/horizon-rl/strands-sglang) (SGLang extension for the agentic scaffolding [`strands`](https://github.com/strands-agents/sdk-python)) for agentic RL training.

## Why `strands-sglang`?

| Component                                                          | Agent Loop                          | TITO Support                           |
| ------------------------------------------------------------------ | ----------------------------------- | -------------------------------------- |
| [Strands-Agents](https://github.com/strands-agents/sdk-python)     | ✅ Handles agent loop, custom hooks | ❌ text-based, requires retokenization |
| [SGLang](https://github.com/sgl-project/sglang)                    | ❌ Single generation only           | ✅ Native `input_ids` in/out           |
| **[strands-sglang](https://github.com/horizon-rl/strands-sglang)** | ✅ Via Strands                      | ✅ Via SGLang's native API             |

`strands-sglang` bridges the gap by extending `strands` with SGLang's native `/generate` endpoint:

- Captures exact token IDs during generation (no retokenization drift)
- Automatically tracks `loss_mask` via `token_manager`
- Provides `ToolIterationLimiter` for clean trajectory truncation

## Install Dependencies

1. Pull the `slimerl/slime:latest` image and enter it
2. Go to slime folder: `cd /root/slime`
3. Install slime: `pip install -e . --no-deps`
4. Go to the example folder: `cd /root/slime/examples/strands_sglang`
5. Install other dependencies: `pip install -r requirements.txt`

> NOTE: `strands-sglang` is under rapid development, so we recommend using the GitHub repo version: `strands-sglang @ git+https://github.com/horizon-rl/strands-sglang.git`

> NOTE: We use camel-ai's subprocess code interpreter for python code execution, which is NOT a good practice; it's just for convenience of this example.

## Prepare Model

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-8B --local-dir /root/models/Qwen/Qwen3-8B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen/Qwen3-8B \
    --save /root/models/Qwen/Qwen3-8B_torch_dist
```

## Prepare Dataset

Following [Retool](https://arxiv.org/abs/2504.11536), we use `dapo-math-17k` as training data:

```python
from datasets import load_dataset
ds = load_dataset("zhuzilin/dapo-math-17k", split="train")
ds.to_json("/root/data/dapo-math-17k.jsonl", orient="records", lines=True)
```

and `aime-2024` as eval data:

```python
from datasets import load_dataset
ds = load_dataset("zhuzilin/aime-2024", split="train")
ds.to_json("/root/data/aime-2024.jsonl", orient="records", lines=True)
```

## Run Training

```bash
cd /root/slime
export WANDB_KEY=$your_wandb_key
bash examples/strands_sglang/strands_qwen3_8b.sh
```
