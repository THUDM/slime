# VLM Single-Turn RL (FSDP & Megatron)

Training VLMs with FSDP or Megatron on single-turn reasoning task using GRPO on the [GEO3K dataset](https://huggingface.co/datasets/hiyouga/geometry3k). We used processed version [here](https://huggingface.co/datasets/chenhegu/geo3k_imgurl).

Note: Please make sure the cudnn version in the environment is 9.16.0.29 to prevent severe performance regression in conv3d in torch 2.9 mentioned in https://github.com/pytorch/pytorch/issues/168167. Otherwise, you can reinstall cudnn with:
```bash
pip install nvidia-cudnn-cu12==9.16.0.29
```

<p align="center">
  <img src="fsdp_vs_megatron.png" alt="FSDP vs Megatron Reward Plot" width="800">
</p>

## Data Preparation (For SFT Training)

The [geo3k_imgurl](https://huggingface.co/datasets/chenhegu/geo3k_imgurl) dataset contains:
- `problem`: The math problem text (string)
- `answer`: The answer (string, e.g., "270")
- `images`: Image data (list)

For SFT training, we need to format the `answer` field for `\boxed{}` format and the messages. You can use the following script to format the answer field:

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("chenhegu/geo3k_imgurl", split="train")

def format_answer(answer: str) -> str:
    """Format answer to include \\boxed{} format."""
    return f"Answer: \\boxed{{{answer}}}"

def process_sample(sample):
    formatted_answer = f"Answer: \\boxed{{{sample['answer']}}}"
    
    sample["messages"] = [
        {"role": "user", "content": sample["problem"]},
        {"role": "assistant", "content": formatted_answer}
    ]
    return sample

ds = ds.map(process_sample)
ds.to_parquet("/root/datasets/geo3k_imgurl/train_formatted.parquet")
```

## Reproduce

```bash
export WANDB_API_KEY=your_wandb_api_key

# Megatron backend (default -> Qwen3-VL-8B-Instruct + Megatron)
./examples/geo3k_vlm/run_geo3k_vlm.sh

# FSDP backend
SLIME_SCRIPT_TRAIN_BACKEND=fsdp ./examples/geo3k_vlm/run_geo3k_vlm.sh

# With different model
SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct ./examples/geo3k_vlm/run_geo3k_vlm.sh

# SFT
./examples/geo_3k_vlm/run_geo3k_vlm_sft.sh
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SLIME_SCRIPT_TRAIN_BACKEND` | `megatron` | Training backend (`megatron` or `fsdp`) |
| `SLIME_SCRIPT_MODEL_NAME` | `Qwen3-VL-8B-Instruct` | Model name |
| `SLIME_SCRIPT_DATASET_NAME` | `chenhegu/geo3k_imgurl` | HuggingFace dataset name |
| `SLIME_SCRIPT_NUM_GPUS` | `8` | Number of GPUs |
| `SLIME_SCRIPT_EXTERNAL_RAY` | `0` | Use external Ray cluster (`1` to enable) |

### Supported Models

- `Qwen3-VL-2B-Instruct`
- `Qwen3-VL-4B-Instruct`
- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-30B-A3B-Instruct`
- `Qwen3-VL-235B-A22B-Instruct`
- `Qwen3-VL-2B-Thinking`
- `Qwen3-VL-4B-Thinking`
- `Qwen3-VL-8B-Thinking`
- `Qwen3-VL-30B-A3B-Thinking`
- `Qwen3-VL-235B-A22B-Thinking`

## Notes

### Reward Model Configuration

We experimented with three reward model configurations:
1. A geo3k-specific RM with tolerance=0.05 (to handle rounding in ground truth labels)
2. A geo3k-specific RM with tolerance=0.0 (strict matching)
3. The default math RM

All three performed similarly, so we use the default math RM for simplicity.

### Numerical Precision with Non-Binary Rewards

Our initial geo3k-specific verifier produced "format scores" (**0 and 0.9**) instead of clean binary rewards. Under **fp32**, fractional values like 0.9 can't be exactly represented, so when all samples in a group have the same reward, `reward - mean` doesn't equal zero—creating spurious gradient signal.

We fixed this by switching to the default math RM with clean **binary 0/1 rewards**. If you encounter similar precision issues with non-binary rewards, you can change the reward tensor dtype from `torch.float` to `torch.float16` in `slime/ray/rollout.py` (`_post_process_rewards` method) to truncate precision artifacts.

## B200
Blackwell currently does not support fa3, we need to use  `--sglang-mm-attention-backend sdpa` and `--attn-implementation flash_attention_2`
